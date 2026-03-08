"""
Generate text with logits-based watermarking (GPTWatermarkLogitsWarper).
Using vLLM for efficient inference. Mirrors run_generate.py data flow and output format.
"""
import argparse
import json
import logging
import os
from tqdm import tqdm

from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer
from vllm import LLM, SamplingParams

from dataset import load_generation_dataset, load_jsonl, make_prompt_mapper
from prompt import get_system_prompt
from gptwm import GPTWatermarkBase
from gptwm_vllm_config import vLLMGPTWatermarkLogitsWarper, set_watermark_base


os.environ["VLLM_LOG_LEVEL"] = "ERROR"

logging.getLogger("vllm").setLevel(logging.ERROR)


def create_sampling_params(args):
    """Create SamplingParams based on arguments."""
    base = dict(max_tokens=args.max_new_tokens)
    if args.beam_size is not None:
        return SamplingParams(
            n=args.beam_size,
            use_beam_search=True,
            **base
        )
    return SamplingParams(
        temperature=1.0,
        top_k=args.top_k if args.top_k is not None else -1,
        top_p=args.top_p,
        **base
    )

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
        output_file = output_file.replace(".jsonl", "_only_English.jsonl")

    if "decapoda-research-llama-7B-hf" in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(args.model_name)
    wm_base = GPTWatermarkBase(
        fraction=args.fraction,
        strength=args.strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        watermark_key=args.wm_key,
        only_English=args.only_English,
        tokenizer=tokenizer,
    )
    sampling_params = create_sampling_params(args)

    # Load dataset and apply chat template with dataset-specific system prompt
    ds = load_generation_dataset(args.prompt_file, args.num_test).to_iterable_dataset()
    ds = ds.map(
        make_prompt_mapper(tokenizer, get_system_prompt(args.dataset_type)),
        batched=True,
    )

    # vLLM model (structure aligned with run_generate_incontext_vllm)
    llm_kwargs = {
        "model": args.model_name,
        "dtype": "bfloat16",
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
    }
    llm_kwargs["hf_overrides"] = {}
    if args.yarn:
        # Configure YaRN for long context
        print("Using YaRN for long context")

        # Override max_position_embeddings in model config to match max_model_len
        llm_kwargs["max_model_len"] = args.max_model_len
        llm_kwargs["hf_overrides"]["max_position_embeddings"] = args.max_model_len
        llm_kwargs["hf_overrides"]["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": 32768
        }

    print("Loading vLLM model...")
    set_watermark_base(wm_base)
    llm = LLM(
        **llm_kwargs,
        logits_processors=[vLLMGPTWatermarkLogitsWarper]
    )
    print("vLLM model loaded successfully")
    print("=" * 100)

    for batch in tqdm(ds.iter(batch_size=args.batch_size), desc="Generating"):
        input_prompts = batch["input_prompts"]
        prefixes = batch["prefix"]
        gold_completions = batch["gold_completion"]

        outputs_vllm = llm.generate(input_prompts, sampling_params)
        outputs = []
        for i, out in enumerate(outputs_vllm):
            gen_text = out.outputs[0].text
            outputs.append(
                json.dumps(
                    {
                        "input_prompts": input_prompts[i],
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
    gen.add_argument("--max_model_len", type=int, default=131072)
    gen.add_argument("--batch_size", type=int, default=64)

    wm = parser.add_argument_group("Watermark")
    wm.add_argument("--fraction", type=float, default=0.5)
    wm.add_argument("--strength", type=float, default=2.0)
    wm.add_argument("--wm_key", type=int, default=0)
    wm.add_argument("--only_English", action="store_true")

    data = parser.add_argument_group("Data")
    data.add_argument("--prompt_file", type=str, default="./UnigramWatermark/data/LFQA/inputs.jsonl")
    data.add_argument("--output_dir", type=str, default="./UnigramWatermark/data/LFQA/")
    data.add_argument("--num_test", type=int, default=None)
    data.add_argument(
        "--dataset_type",
        type=str,
        default="lfqa",
        choices=["lfqa", "opengen"],
        help="Dataset type: lfqa or opengen, determines system prompt from prompt.py",
    )

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
