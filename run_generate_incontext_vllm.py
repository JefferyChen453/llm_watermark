"""
Generate text with in-context watermarking: green word list in system prompt.
Using vLLM for efficient inference.
"""
import argparse
import json
import logging
import os

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer
from vllm import LLM, SamplingParams

from dataset import apply_chat_template, load_generation_dataset, load_jsonl
from gptwm_incontext import InContextWatermarkGenerator
from prompt import get_incontext_system_prompt
from gptwm_vllm_config import set_watermark_config, GPTWatermarkAdapterLogitsProcessor


os.environ["VLLM_LOG_LEVEL"] = "ERROR"

logging.getLogger("vllm").setLevel(logging.ERROR)


def main(args):
    model_name = args.model_name.split('/')[-3] + '_' + args.model_name.split('/')[-2] if os.path.exists(args.model_name) else args.model_name
    output_file = (
        f"{args.output_dir}/"
        f"{model_name.replace('/', '-')}_"
        f"strength_{args.strength}_"
        f"frac_{args.fraction}_"
        f"len_{args.max_new_tokens}_"
        f"num_{args.num_test if args.num_test else len(load_jsonl(args.prompt_file))}_incontext_vllm.jsonl"
    )
    if args.only_English:
        output_file = output_file.replace('.jsonl', '_only_English.jsonl')

    # tokenizer (for watermark generation and tokenization)
    if 'decapoda-research-llama-7B-hf' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(args.model_name)

    # seed_list = list(range(1, args.seed_num + 1))
    seed_list = list(range(0, 1))

    # Load dataset and add idx for seed assignment
    ds = load_generation_dataset(args.prompt_file, args.num_test)
    ds = ds.map(lambda example, idx: {**example, "idx": idx}, with_indices=True)

    # Load vLLM model with YaRN support if enabled
    llm_kwargs = {
        "model": args.model_name,
        "dtype": "bfloat16",

        # parallel parameters
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.95,
        "hf_overrides": {},
    }

    # Set max_model_len
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    else:
        llm_kwargs["max_model_len"] = 262144 #! Qwen3 with yarn only supports 131072

    if args.yarn:
        # Configure YaRN for long context
        print("Using YaRN for long context")
        target_max_position_embeddings = llm_kwargs["max_model_len"]
        print(f"Overriding max_position_embeddings to {target_max_position_embeddings}")
        llm_kwargs["hf_overrides"]["max_position_embeddings"] = target_max_position_embeddings
        llm_kwargs["hf_overrides"]["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": 32768
        }

    print("Loading vLLM model...")
    if args.add_logits_wm:
        set_watermark_config(
            fraction=args.fraction,
            strength=args.strength,
            vocab_size=tokenizer.vocab_size,
            model_emb_length=model_config.vocab_size,
            only_English=args.only_English,
            tokenizer=tokenizer,
        )
        llm = LLM(
            **llm_kwargs,
            logits_processors=[GPTWatermarkAdapterLogitsProcessor]
        )
    else:
        llm = LLM(**llm_kwargs)
    print("vLLM model loaded successfully")
    print("="*100)

    base_sampling_kwargs = dict(min_tokens=args.min_new_tokens, max_tokens=args.max_new_tokens)
    if args.beam_size is not None:
        base_sampling_kwargs.update(n=args.beam_size, use_beam_search=True)
    else:
        base_sampling_kwargs.update(
            temperature=1.0,
            top_k=args.top_k if args.top_k is not None else -1,
            top_p=args.top_p,
        )

    for batch in tqdm(ds.iter(batch_size=args.batch_size), desc="Generating"):
        indices = batch["idx"]
        batch_seeds = [seed_list[idx % len(seed_list)] for idx in indices]

        # Build per-sample incontext prompts (prompt depends on seed)
        input_prompts = []
        for i, seed in enumerate(batch_seeds):
            wm_gen = InContextWatermarkGenerator(
                fraction=args.fraction,
                vocab_size=tokenizer.vocab_size,
                model_emb_length=model_config.vocab_size,
                watermark_key=seed,
                only_English=args.only_English,
                tokenizer=tokenizer,
            )
            green_token_string = wm_gen.get_green_token_string(shuffle=args.shuffle_green_tokens)
            dataset_type = batch.get("dataset_type", ["lfqa"] * len(indices))[i]
            system_prompt = get_incontext_system_prompt(dataset_type, green_token_string)
            input_prompts.append(apply_chat_template(tokenizer, system_prompt, batch["prefix"][i]))

        if args.add_logits_wm:
            sampling_params_list = [
                SamplingParams(**base_sampling_kwargs, extra_args={"watermark_key": seed})
                for seed in batch_seeds
            ]
        else:
            sampling_params_list = [SamplingParams(**base_sampling_kwargs)] * len(batch_seeds)

        outputs_vllm = llm.generate(input_prompts, sampling_params_list)

        outputs = []
        for i, out in enumerate(outputs_vllm):
            outputs.append(json.dumps({
                "prefix": batch["prefix"][i],
                "input_prompt": input_prompts[i],
                "gold_completion": batch["gold_completion"][i],
                "gen_completion": out.outputs[0].text,
                "seed": batch_seeds[i],
            }, ensure_ascii=False))

        with open(output_file, "a") as f:
            f.write("\n".join(outputs) + "\n")

    print("Results saved to:", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model generation parameters
    parser.add_argument_group("Model Generation")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--min_new_tokens", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--yarn", action="store_true", help="Enable YaRN for long context")
    parser.add_argument("--yarn_factor", type=float, default=4.0, help="YaRN scaling factor")
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum model length for vLLM")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for vLLM generation")

    # Watermark parameters
    parser.add_argument_group("Watermark")
    parser.add_argument("--add_logits_wm", action="store_true", help="Add logits-based watermarking")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--seed_num", type=int, default=100)
    parser.add_argument("--only_English", action="store_true")
    parser.add_argument("--shuffle_green_tokens", action="store_true",
                        help="Shuffle green token order per sample (use for training; omit for val/test to enable vLLM prefix caching)")

    # Data parameters
    parser.add_argument_group("Data")
    parser.add_argument("--prompt_file", type=str, default="./UnigramWatermark/data/LFQA/inputs.jsonl")
    parser.add_argument("--output_dir", type=str, default="./test")
    parser.add_argument("--num_test", type=int, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)
