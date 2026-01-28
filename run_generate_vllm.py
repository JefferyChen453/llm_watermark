"""
Generate text with logits-based watermarking (GPTWatermarkLogitsWarper).
Using vLLM for efficient inference. Mirrors run_generate.py data flow and output format.
"""
import argparse
import json
import os
from tqdm import tqdm

from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer
from vllm import LLM, SamplingParams

from dataset import load_generation_dataset
from gptwm import GPTWatermarkLogitsWarper


def make_vllm_logits_processor(warper: GPTWatermarkLogitsWarper):
    watermark = warper.strength * warper.green_list_mask

    def _fn(_output_ids: list, logits):
        return logits + watermark.to(logits.device)

    return _fn


def create_sampling_params(args, logits_processors=None):
    """Create SamplingParams based on arguments."""
    base = dict(max_tokens=args.max_new_tokens)
    if logits_processors:
        base["logits_processors"] = logits_processors
    if args.beam_size is not None:
        return SamplingParams(
            n=args.beam_size,
            use_beam_search=True,
            **base,
        )
    return SamplingParams(
        temperature=1.0,
        top_k=args.top_k if args.top_k is not None else -1,
        top_p=args.top_p,
        **base,
    )


def main(args):
    output_file = (
        f"{args.output_dir}/"
        f"{args.model_name.replace('/', '-')}_"
        f"strength_{args.strength}_"
        f"frac_{args.fraction}_"
        f"len_{args.max_new_tokens}_"
        f"num_{args.num_test}_vllm.jsonl"
    )
    if args.only_English:
        output_file = output_file.replace(".jsonl", "_only_English.jsonl")

    if "decapoda-research-llama-7B-hf" in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(args.model_name)
    warper = GPTWatermarkLogitsWarper(
        fraction=args.fraction,
        strength=args.strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        watermark_key=args.wm_key,
        only_English=args.only_English,
        tokenizer=tokenizer,
    )
    logits_processors = [make_vllm_logits_processor(warper)]

    # vLLM model (structure aligned with run_generate_incontext_vllm)
    llm_kwargs = {
        "model": args.model_name,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
    }
    llm_kwargs["hf_overrides"] = {}

    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    else:
        llm_kwargs["max_model_len"] = 262144

    if args.yarn:
        print("Using YaRN for long context")
        target_max = llm_kwargs["max_model_len"]
        llm_kwargs["hf_overrides"]["max_position_embeddings"] = target_max
        llm_kwargs["hf_overrides"]["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": 32768,
        }

    print("Loading vLLM model...")
    llm = LLM(**llm_kwargs)
    print("vLLM model loaded successfully")
    print("=" * 100)

    ds = load_generation_dataset(args.prompt_file, args.num_test)
    sampling_params = create_sampling_params(args, logits_processors)

    for batch in tqdm(ds.iter(batch_size=args.batch_size), desc="Generating"):
        prefixes = batch["prefix"]
        gold_completions = batch["gold_completion"]
        prompts = prefixes

        outputs_vllm = llm.generate(prompts, sampling_params)
        outputs = []
        for i, out in enumerate(outputs_vllm):
            gen_text = out.outputs[0].text
            outputs.append(
                json.dumps(
                    {
                        "prefix": prefixes[i],
                        "gold_completion": gold_completions[i],
                        "gen_completion": gen_text,
                    },
                    ensure_ascii=False,
                )
            )
        with open(output_file, "a") as f:
            f.write("\n".join(outputs) + "\n")

    print("Results saved to:", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    gen = parser.add_argument_group("Generation")
    gen.add_argument("--model_name", type=str, required=True)
    gen.add_argument("--max_new_tokens", type=int, default=200)
    gen.add_argument("--beam_size", type=int, default=None)
    gen.add_argument("--top_k", type=int, default=None)
    gen.add_argument("--top_p", type=float, default=0.9)
    gen.add_argument("--yarn", action="store_true", help="Enable YaRN for long context")
    gen.add_argument("--yarn_factor", type=float, default=4.0)
    gen.add_argument("--max_model_len", type=int, default=None)
    gen.add_argument("--tensor_parallel_size", type=int, default=8)
    gen.add_argument("--batch_size", type=int, default=32)

    wm = parser.add_argument_group("Watermark")
    wm.add_argument("--fraction", type=float, default=0.5)
    wm.add_argument("--strength", type=float, default=2.0)
    wm.add_argument("--wm_key", type=int, default=0)
    wm.add_argument("--only_English", action="store_true")

    data = parser.add_argument_group("Data")
    data.add_argument("--prompt_file", type=str, default="./UnigramWatermark/data/LFQA/inputs.jsonl")
    data.add_argument("--output_dir", type=str, default="./UnigramWatermark/data/LFQA/")
    data.add_argument("--num_test", type=int, default=500)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
