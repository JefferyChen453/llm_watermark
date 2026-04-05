import argparse
import json
import logging
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from vllm import LLM, SamplingParams

from dataset import load_generation_dataset, make_multitask_prompt_mapper
from gptwm import _get_english_token_ids, _make_green_list_mask_numpy
from gptwm_vllm_config import GPTWatermarkAdapterLogitsProcessor, set_watermark_config

os.environ["VLLM_LOG_LEVEL"] = "ERROR"
logging.getLogger("vllm").setLevel(logging.ERROR)


def build_base_sampling_kwargs(args) -> dict:
    base = dict(min_tokens=args.min_new_tokens, max_tokens=args.max_new_tokens)
    if args.beam_size is not None:
        base.update(n=args.beam_size, use_beam_search=True)
    else:
        base.update(
            temperature=1.0,
            top_k=args.top_k if args.top_k is not None else -1,
            top_p=args.top_p,
        )
    return base


def load_tokenizer(model_name: str):
    if "decapoda-research-llama-7B-hf" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def make_output_paths(args):
    base_name = (
        f"{args.model_name.replace('/', '-')}_"
        f"strength_{args.strength}_"
        f"frac_{args.fraction}_"
        f"len_{args.max_new_tokens}_"
        f"num_{args.num_test if args.num_test is not None else 'all'}_"
        f"train_greenprob_vllm"
    )
    if args.only_English:
        base_name += "_only_English"
    sample_file = os.path.join(args.output_dir, f"{base_name}.jsonl")
    summary_file = os.path.join(args.output_dir, f"{base_name}_summary.json")
    return sample_file, summary_file


def generate_records(args, tokenizer, prompt_file: str) -> List[dict]:
    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    set_watermark_config(
        fraction=args.fraction,
        strength=args.strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        only_English=args.only_English,
        tokenizer=tokenizer,
    )

    seed_list = list(range(1, args.seed_num + 1))
    ds = load_generation_dataset(prompt_file, args.num_test)
    ds = ds.map(
        make_multitask_prompt_mapper(tokenizer),
        batched=False,
        with_indices=True,
    )

    llm_kwargs = {
        "model": args.model_name,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "hf_overrides": {},
    }

    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    if args.yarn:
        target_max_position_embeddings = args.max_model_len or 262144
        llm_kwargs["max_model_len"] = target_max_position_embeddings
        llm_kwargs["hf_overrides"]["max_position_embeddings"] = target_max_position_embeddings
        llm_kwargs["hf_overrides"]["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": 32768,
        }

    llm = LLM(**llm_kwargs, logits_processors=[GPTWatermarkAdapterLogitsProcessor])
    base_sampling_kwargs = build_base_sampling_kwargs(args)

    records = []
    for batch in tqdm(ds.iter(batch_size=args.gen_batch_size), desc="Generating"):
        input_prompts = batch["input_prompt"]
        indices = batch["idx"]
        batch_seeds = [seed_list[idx % len(seed_list)] for idx in indices]
        sampling_params_list = [
            SamplingParams(
                **base_sampling_kwargs,
                detokenize=False,
                extra_args={"watermark_key": seed},
            )
            for seed in batch_seeds
        ]

        outputs = llm.generate(input_prompts, sampling_params_list)
        for i, out in enumerate(outputs):
            records.append(
                {
                    "idx": int(indices[i]),
                    "seed": int(batch_seeds[i]),
                    "prefix": batch["prefix"][i],
                    "gold_completion": batch["gold_completion"][i],
                    "input_prompt": input_prompts[i],
                    "gen_token_ids": out.outputs[0].token_ids,
                    "gen_completion": tokenizer.decode(out.outputs[0].token_ids, skip_special_tokens=True),
                }
            )

    return records


def score_green_probs(args, tokenizer, records: List[dict]) -> dict:
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()
    model.to(args.score_device)

    english_token_ids = None
    if args.only_English:
        english_token_ids = _get_english_token_ids(tokenizer, tokenizer.vocab_size)

    sample_metrics = []
    total_green_prob_before_bias = 0.0
    total_green_prob_after_bias = 0.0
    total_tokens = 0

    for start in tqdm(range(0, len(records), args.score_batch_size), desc="Scoring"):
        batch_records = records[start:start + args.score_batch_size]
        full_sequences = []
        prompt_lens = []
        green_masks = []
        generated_lens = []

        for record in batch_records:
            prompt_ids = tokenizer.encode(record["input_prompt"], add_special_tokens=False)
            gen_ids = record["gen_token_ids"]
            full_sequences.append(prompt_ids + gen_ids)
            prompt_lens.append(len(prompt_ids))
            generated_lens.append(len(gen_ids))
            green_masks.append(
                _make_green_list_mask_numpy(
                    watermark_key=record["seed"],
                    fraction=args.fraction,
                    vocab_size=tokenizer.vocab_size,
                    model_emb_length=config.vocab_size,
                    only_English=args.only_English,
                    tokenizer=tokenizer,
                    english_token_ids=english_token_ids,
                )
            )

        encoded = tokenizer.pad(
            {"input_ids": full_sequences},
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(args.score_device)
        attention_mask = encoded["attention_mask"].to(args.score_device)
        green_masks_tensor = torch.tensor(np.stack(green_masks), device=args.score_device, dtype=torch.bool)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]

        for i, record in enumerate(batch_records):
            gen_len = generated_lens[i]
            if gen_len == 0:
                sample_metrics.append(
                    {
                        **record,
                        "avg_green_prob_before_bias": None,
                        "avg_green_prob_after_bias": None,
                        "num_generated_tokens": 0,
                    }
                )
                continue

            start_idx = prompt_lens[i] - 1
            end_idx = start_idx + gen_len
            sample_logits = logits[i, start_idx:end_idx].float()
            total_logsumexp_unbiased = torch.logsumexp(sample_logits, dim=-1)
            green_logsumexp_unbiased = torch.logsumexp(sample_logits[:, green_masks_tensor[i]], dim=-1)
            green_prob_before_bias = torch.exp(green_logsumexp_unbiased - total_logsumexp_unbiased)
            avg_green_prob_before_bias = green_prob_before_bias.mean().item()

            green_bias = args.strength * green_masks_tensor[i].float().unsqueeze(0)
            biased_logits = sample_logits + green_bias
            total_logsumexp_biased = torch.logsumexp(biased_logits, dim=-1)
            green_logsumexp_biased = torch.logsumexp(biased_logits[:, green_masks_tensor[i]], dim=-1)
            green_prob_after_bias = torch.exp(green_logsumexp_biased - total_logsumexp_biased)
            avg_green_prob_after_bias = green_prob_after_bias.mean().item()

            total_green_prob_before_bias += green_prob_before_bias.sum().item()
            total_green_prob_after_bias += green_prob_after_bias.sum().item()
            total_tokens += gen_len
            sample_metrics.append(
                {
                    **record,
                    "avg_green_prob_before_bias": avg_green_prob_before_bias,
                    "avg_green_prob_after_bias": avg_green_prob_after_bias,
                    "num_generated_tokens": gen_len,
                }
            )

    dataset_avg_green_prob_before_bias = (
        total_green_prob_before_bias / total_tokens if total_tokens > 0 else None
    )
    dataset_avg_green_prob_after_bias = (
        total_green_prob_after_bias / total_tokens if total_tokens > 0 else None
    )
    return {
        "sample_metrics": sample_metrics,
        "dataset_avg_green_prob_before_bias": dataset_avg_green_prob_before_bias,
        "dataset_avg_green_prob_after_bias": dataset_avg_green_prob_after_bias,
        "total_generated_tokens": total_tokens,
        "num_samples": len(records),
    }


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    sample_file, summary_file = make_output_paths(args)
    tokenizer = load_tokenizer(args.model_name)

    records = generate_records(args, tokenizer, args.prompt_file)
    metrics = score_green_probs(args, tokenizer, records)

    with open(sample_file, "w") as f:
        for row in metrics["sample_metrics"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "model_name": args.model_name,
        "prompt_file": args.prompt_file,
        "fraction": args.fraction,
        "strength": args.strength,
        "green_prob_definition": {
            "before_bias": "softmax(forward(prompt+response logits on response))",
            "after_bias": "softmax(forward(prompt+response logits on response) + strength * green_mask)",
        },
        "seed_num": args.seed_num,
        "only_English": args.only_English,
        "num_samples": metrics["num_samples"],
        "total_generated_tokens": metrics["total_generated_tokens"],
        "avg_green_prob_before_bias": metrics["dataset_avg_green_prob_before_bias"],
        "avg_green_prob_after_bias": metrics["dataset_avg_green_prob_after_bias"],
        "sample_metrics_file": sample_file,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    gen = parser.add_argument_group("Generation")
    gen.add_argument("--model_name", type=str, required=True)
    gen.add_argument("--min_new_tokens", type=int, default=500)
    gen.add_argument("--max_new_tokens", type=int, default=600)
    gen.add_argument("--gen_batch_size", type=int, default=256)
    gen.add_argument("--beam_size", type=int, default=None)
    gen.add_argument("--top_k", type=int, default=None)
    gen.add_argument("--top_p", type=float, default=0.9)
    gen.add_argument("--tensor_parallel_size", type=int, default=8)
    gen.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    gen.add_argument("--yarn", action="store_true")
    gen.add_argument("--yarn_factor", type=float, default=4.0)
    gen.add_argument("--max_model_len", type=int, default=None)

    wm = parser.add_argument_group("Watermark")
    wm.add_argument("--fraction", type=float, required=True)
    wm.add_argument("--strength", type=float, default=3.0)
    wm.add_argument("--seed_num", type=int, default=500)
    wm.add_argument("--only_English", action="store_true")

    score = parser.add_argument_group("Scoring")
    score.add_argument("--score_batch_size", type=int, default=4)
    score.add_argument("--score_device", type=str, default="cuda")
    score.add_argument("--flash_attn", action="store_true")

    data = parser.add_argument_group("Data")
    data.add_argument("--prompt_file", type=str, required=True)
    data.add_argument("--output_dir", type=str, required=True)
    data.add_argument("--num_test", type=int, default=None)

    args = parser.parse_args()
    main(args)
