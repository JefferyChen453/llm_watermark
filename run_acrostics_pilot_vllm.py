"""Acrostics ICW pilot — compare {none, weak, strong} privileged prompts.

Goal: establish that
  - No constraint → ~0% hit rate (model can't do acrostics w/o hint)
  - Strong privileged prompt → high hit rate (teacher can do it)
  - Weak privileged prompt → middling (less specific = less useful)

This demonstrates (1) Acrostics isn't achievable via simple logit-bias like
Lexical/Initials, and (2) privileged-prompt teacher is a viable teacher for KD.
"""

import argparse
import json
import logging
import os
import random
import string
from pathlib import Path

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from acrostics_icw import build_acrostic_prompt, sample_target, verify_acrostic
from dataset import apply_chat_template, load_generation_dataset


os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
logging.getLogger("vllm").setLevel(logging.ERROR)


def main(args):
    ds = load_generation_dataset(args.prompt_file, args.num_test)
    print(f"Loaded {len(ds)} prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Per-prompt target (deterministic over runs with same seed_base)
    pool = string.ascii_uppercase if args.target_uppercase else string.ascii_lowercase
    if args.fixed_target:
        # All prompts share one target — used for evaluation matrix where we
        # want a single canonical X across the whole test set.
        targets = [args.fixed_target] * len(ds)
    else:
        targets = [
            sample_target(seed=args.seed_base + i, length=args.target_length, pool=pool)
            for i in range(len(ds))
        ]

    # Build (prompt_text, target, variant, idx) tuples for all conditions
    jobs = []
    for variant in args.variants:
        for i, ex in enumerate(ds):
            prefix = ex["prefix"]
            target = targets[i]
            system, user = build_acrostic_prompt(prefix, target, variant=variant)
            chat = apply_chat_template(tokenizer, system_prompt=system, user_prompt=user)
            jobs.append({
                "idx": i,
                "variant": variant,
                "target": target,
                "prefix": prefix,
                "input_prompt": chat,
            })
    print(f"Total jobs: {len(jobs)}  ({len(args.variants)} variants × {len(ds)} prompts)")

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

    # Collate + verify
    tag_suffix = f"_{args.output_tag}" if args.output_tag else ""
    output_path = Path(args.output_dir) / f"acrostics_pilot_{args.model_tag}_n{len(ds)}_len{args.target_length}{tag_suffix}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for j, o in zip(jobs, outs):
            response = o.outputs[0].text
            verdict = verify_acrostic(response, j["target"])
            rec = {
                **{k: j[k] for k in ("idx", "variant", "target", "prefix")},
                "response": response,
                "first_letters": verdict.first_letters,
                "is_subsequence": verdict.is_subsequence,
                "is_contiguous": verdict.is_contiguous,
                "levenshtein": verdict.levenshtein_to_substring,
                "n_sentences": verdict.n_sentences,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-14B")
    p.add_argument("--model_tag", default="qwen3-14b")
    p.add_argument("--prompt_file", default="/home/tianyichen/llm_watermark/data/processed_data/vblagoje_lfqa/test_477.json")
    p.add_argument("--num_test", type=int, default=100)
    p.add_argument("--target_length", type=int, default=4)
    p.add_argument("--seed_base", type=int, default=20260417)
    p.add_argument("--variants", nargs="+", default=["none", "weak", "strong"])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=600)
    p.add_argument("--yarn", action="store_true")
    p.add_argument("--max_model_len", type=int, default=131072)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--output_dir", default="/home/tianyichen/llm_watermark/outputs/acrostics_pilot")
    p.add_argument("--output_tag", default=None, help="Optional suffix for output filename")
    p.add_argument("--target_uppercase", action="store_true", help="Sample target from A-Z instead of a-z")
    p.add_argument("--fixed_target", default=None, help="If set, use this target for ALL prompts (eval mode)")
    args = p.parse_args()
    main(args)
