import argparse
import json
import logging
import os

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer
from vllm import LLM, SamplingParams

from dataset import load_generation_dataset, load_jsonl, make_multitask_prompt_mapper
from gptwm_vllm_config import set_watermark_config, GPTWatermarkAdapterLogitsProcessor

os.environ["VLLM_LOG_LEVEL"] = "ERROR"
logging.getLogger("vllm").setLevel(logging.ERROR)

def main(args):
    output_file = (
        f"{args.output_dir}/"
        f"{args.model_name.replace('/', '-')}_"
        f"strength_{args.strength}_"
        f"frac_{args.fraction}_"
        f"len_{args.max_new_tokens}_"
        f"num_{args.num_test if args.num_test else len(load_jsonl(args.prompt_file))}_vllm.jsonl"
    )
    if args.only_English:
        output_file = output_file.replace('.jsonl', '_only_English.jsonl')

    # tokenizer
    if 'decapoda-research-llama-7B-hf' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(args.model_name)
    set_watermark_config(
        fraction=args.fraction,
        strength=args.strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        only_English=args.only_English,
        tokenizer=tokenizer,
    )

    seed_list = list(range(1, args.seed_num + 1))

    # Load dataset: apply chat template per example using dataset_type
    ds = load_generation_dataset(args.prompt_file, args.num_test)
    ds = ds.map(
        make_multitask_prompt_mapper(tokenizer),
        batched=False,
        with_indices=True,
    )

    print("Loading vLLM model...")
    llm_kwargs = {
        "model": args.model_name,
        "dtype": "bfloat16",
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
    }
    llm = LLM(**llm_kwargs, logits_processors=[GPTWatermarkAdapterLogitsProcessor])
    print("vLLM model loaded successfully")
    print("=" * 100)

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
        input_prompt = batch["input_prompt"]
        indices = batch["idx"]
        batch_seeds = [seed_list[idx % len(seed_list)] for idx in indices]

        sampling_params_list = [
            SamplingParams(
                **base_sampling_kwargs,
                extra_args={"watermark_key": seed},
            )
            for seed in batch_seeds
        ]

        outputs_vllm = llm.generate(input_prompt, sampling_params_list)

        outputs = []
        for i, out in enumerate(outputs_vllm):
            outputs.append(json.dumps({
                "prefix": batch["prefix"][i],
                "input_prompt": input_prompt[i],
                "gold_completion": batch["gold_completion"][i],
                "gen_completion": out.outputs[0].text,
                "seed": batch_seeds[i],
                "dataset_type": batch["dataset_type"][i],
            }, ensure_ascii=False))

        with open(output_file, "a") as f:
            f.write("\n".join(outputs) + "\n")

    print("Results saved to:", output_file)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model generation parameters
    parser.add_argument_group("Generation")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--min_new_tokens", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Watermark parameters
    parser.add_argument_group("Watermark")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--seed_num", type=int, default=100)
    parser.add_argument("--only_English", action="store_true")

    # Data parameters
    parser.add_argument_group("Data")
    parser.add_argument("--prompt_file", type=str, default="./UnigramWatermark/data/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="./UnigramWatermark/data/LFQA/")
    parser.add_argument("--num_test", type=int, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)