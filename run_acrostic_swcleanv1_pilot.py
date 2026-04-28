"""Acrostic pilot — clean_v1 prompt + ICW-style secret strings, single
generation pass on test_477.

For each LFQA prefix in test_477.json:
  1. sample a per-prompt secret string via sample_target_icw (20-letter pool,
     length 18, uniform).
  2. build the clean_v1 acrostic prompt.
  3. vLLM generate one response.
  4. write JSONL row with prefix / target / response / generation metadata.

Detection (3-extractor SW shuffle-S) is done in a separate analysis script.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from acrostics_icw import (
    build_acrostic_prompt,
    build_acrostic_chat_messages,
    sample_target_icw,
)
from dataset import (
    apply_chat_template,
    apply_chat_template_messages,
    load_generation_dataset,
)


# Multi-turn variants need a different code path
MULTI_TURN_VARIANTS = {"clean_v2_chat"}


os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
logging.getLogger("vllm").setLevel(logging.ERROR)


def main(args):
    ds = load_generation_dataset(args.prompt_file, args.num_test)
    print(f"Loaded {len(ds)} prompts from {args.prompt_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Per-sample target sampling (ICW pool)
    targets = [
        sample_target_icw(seed=args.seed_base + i, length=args.target_length, uppercase=True)
        for i in range(len(ds))
    ]

    # Build chat-templated prompts (single-turn or multi-turn per variant)
    is_multi_turn = args.variant in MULTI_TURN_VARIANTS
    jobs = []
    for i, ex in enumerate(ds):
        prefix = ex["prefix"]
        target = targets[i]
        if is_multi_turn:
            messages = build_acrostic_chat_messages(prefix, target, variant=args.variant)
            chat = apply_chat_template_messages(tokenizer, messages)
        else:
            system, user = build_acrostic_prompt(prefix, target, variant=args.variant)
            chat = apply_chat_template(tokenizer, system_prompt=system, user_prompt=user)
        jobs.append({
            "idx": i,
            "prefix": prefix,
            "target": target,
            "input_prompt": chat,
        })
    print(f"Built {len(jobs)} prompts (variant={args.variant}, target_len={args.target_length}, "
          f"multi_turn={is_multi_turn})")

    # vLLM init
    llm_kwargs = dict(
        model=args.model_name,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=args.tp,
    )
    if args.yarn:
        llm_kwargs["max_model_len"] = args.max_model_len
        llm_kwargs.setdefault("hf_overrides", {})
        llm_kwargs["hf_overrides"]["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
    print(f"vLLM init: model={args.model_name}, tp={args.tp}")
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=1,
    )

    prompts_batch = [j["input_prompt"] for j in jobs]
    outs = llm.generate(prompts_batch, sp)
    assert len(outs) == len(jobs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for j, o in zip(jobs, outs):
            out0 = o.outputs[0]
            response = out0.text
            rec = {
                "idx": j["idx"],
                "prefix": j["prefix"],
                "target": j["target"],
                "response": response,
                "finish_reason": out0.finish_reason,
                "n_input_tokens": len(o.prompt_token_ids),
                "n_output_tokens": len(out0.token_ids),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(jobs)} rows -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-14B")
    p.add_argument("--prompt_file",
                   default="/home/tianyichen/llm_watermark/data/processed_data/vblagoje_lfqa/test_477.json")
    p.add_argument("--num_test", type=int, default=-1, help="-1 = all rows in prompt_file")
    p.add_argument("--target_length", type=int, default=18)
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--yarn", action="store_true")
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--variant", default="clean_v1",
                   help="Prompt variant: clean_v1 / clean_v2_noex / clean_v2_1ex / clean_v2_chat")
    p.add_argument("--output", required=True,
                   help="Output JSONL path, e.g. outputs/acrostic_swcleanv1_pilot/wm.jsonl")
    args = p.parse_args()

    if args.num_test == -1:
        # Resolve to actual file size
        import json as _json
        with open(args.prompt_file) as f:
            args.num_test = sum(1 for line in f if line.strip())
        print(f"num_test=-1 → using all {args.num_test} rows")

    main(args)
