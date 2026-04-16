"""Generate text with Initials ICW watermarking (optional ICW prompt + optional logit bias).

Variants:
    --add_icw_prompt       include <green>/<red> letter lists in the system prompt
    --add_logits_wm        add per-step logit bias to green-initial leading-space tokens
    --strength <float>     bias strength (ignored when --add_logits_wm is off)
    --seed <int>           partition key (single value for pilot evaluation)

Output JSONL records: prefix, input_prompt, gold_completion, gen_completion, seed.
"""

import argparse
import json
import logging
import os

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from dataset import apply_chat_template, load_generation_dataset, load_jsonl
from gptwm_initials import partition_letters
from prompt import get_initials_incontext_prompt
from gptwm_vllm_config import set_initials_config, InitialsAdapterLogitsProcessor


os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
logging.getLogger("vllm").setLevel(logging.ERROR)


def main(args):
    model_tag = args.model_name.replace("/", "-")
    suffix_bits = []
    suffix_bits.append("prompt" if args.add_icw_prompt else "noprompt")
    suffix_bits.append(f"s{args.strength}" if args.add_logits_wm else "nobias")
    suffix_bits.append(f"seed{args.seed}")
    output_file = (
        f"{args.output_dir}/{model_tag}_initials_"
        + "_".join(suffix_bits)
        + f"_len_{args.max_new_tokens}_num_{args.num_test}.jsonl"
    )
    if os.path.exists(output_file) and not args.overwrite:
        print(f"[skip] {output_file} exists")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config = AutoConfig.from_pretrained(args.model_name)

    green_letters, red_letters = partition_letters(seed=args.seed)
    # Keep inference-time prompt alphabetical for reproducibility.
    green_letters = sorted(green_letters)
    red_letters = sorted(red_letters)
    print(f"[seed={args.seed}] green: {green_letters}")
    print(f"[seed={args.seed}] red:   {red_letters}")

    ds = load_generation_dataset(args.prompt_file, args.num_test)

    llm_kwargs = {
        "model": args.model_name,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": 0.9,
        "max_model_len": args.max_model_len,
    }

    print("Loading vLLM model...")
    if args.add_logits_wm:
        set_initials_config(
            strength=args.strength,
            vocab_size=tokenizer.vocab_size,
            model_emb_length=model_config.vocab_size,
            tokenizer=tokenizer,
            default_seed=args.seed,
        )
        llm = LLM(**llm_kwargs, logits_processors=[InitialsAdapterLogitsProcessor])
    else:
        llm = LLM(**llm_kwargs)
    print("vLLM model loaded.")
    print("=" * 80)

    base_sampling_kwargs = dict(
        min_tokens=args.min_new_tokens,
        max_tokens=args.max_new_tokens,
        temperature=1.0,
        top_k=args.top_k if args.top_k is not None else -1,
        top_p=args.top_p,
    )

    # Build system prompt (same across all samples when single-seed)
    system_prompt = get_initials_incontext_prompt(
        args.dataset_type, green_letters, red_letters,
    ) if args.add_icw_prompt else ""
    print(f"system_prompt ({'ICW' if args.add_icw_prompt else 'plain'}):")
    print(repr(system_prompt[:200]))

    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        for batch in tqdm(ds.iter(batch_size=args.batch_size), desc="Generating"):
            input_prompts = [
                apply_chat_template(tokenizer, system_prompt, p) for p in batch["prefix"]
            ]
            if args.add_logits_wm:
                sp_list = [
                    SamplingParams(**base_sampling_kwargs, extra_args={"initials_seed": args.seed})
                    for _ in range(len(input_prompts))
                ]
            else:
                sp_list = [SamplingParams(**base_sampling_kwargs) for _ in range(len(input_prompts))]

            outs = llm.generate(input_prompts, sp_list)
            records = []
            for i, out in enumerate(outs):
                records.append(json.dumps({
                    "prefix": batch["prefix"][i],
                    "input_prompt": input_prompts[i],
                    "gold_completion": batch["gold_completion"][i],
                    "gen_completion": out.outputs[0].text,
                    "seed": args.seed,
                    "strength": args.strength if args.add_logits_wm else 0.0,
                    "has_icw_prompt": args.add_icw_prompt,
                }, ensure_ascii=False))
            f.write("\n".join(records) + "\n")

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--min_new_tokens", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    # Initials-specific
    parser.add_argument("--add_icw_prompt", action="store_true",
                        help="Include <green>/<red> letter lists in system prompt")
    parser.add_argument("--add_logits_wm", action="store_true",
                        help="Apply per-step logit bias on green-initial leading-space tokens")
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0, help="Partition key")
    parser.add_argument("--dataset_type", type=str, default="lfqa_initials")
    # Data
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    main(args)
