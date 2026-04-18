"""Rebuild validation parquet for Initials v3 prompt (no vLLM needed).

Strategy: load the existing v1 validation parquet, keep green + neg rows
unchanged, regenerate initials rows' ``prompt`` with the new v3 template
(same prefix and seed=0 as v1), write new parquet.

Schema preserved: {prompt, response, prefix, seed, dataset_type,
positive_or_negative, task}. For initials rows, ``dataset_type`` switches
to ``lfqa_initials_v3`` and ``prompt`` uses the new system prompt.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from dataset import apply_chat_template
from gptwm_initials import partition_letters
from prompt import get_initials_incontext_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_val", required=True,
                   help="Existing v1 validation parquet "
                        "(e.g. validation_mixed_green177_initials177_neg177.parquet)")
    p.add_argument("--output_val", required=True)
    p.add_argument("--dataset_type", default="lfqa_initials_v3")
    p.add_argument("--model_name", default="Qwen/Qwen3-14B")
    p.add_argument("--eval_initials_seed", type=int, default=0,
                   help="Fixed wm_seed for all val initials rows (sorted letter order)")
    args = p.parse_args()

    df = pq.read_table(args.input_val).to_pandas()
    print(f"Loaded {len(df)} rows from {args.input_val}")
    print(f"  task breakdown: {df['task'].value_counts().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # ---- Regenerate initials prompt with v3 template ----
    green, red = partition_letters(args.eval_initials_seed)
    green_sorted = sorted(green)
    red_sorted = sorted(red)
    icw_sys = get_initials_incontext_prompt(args.dataset_type, green_sorted, red_sorted)
    print(f"v3 system prompt (first 300 chars):")
    print(icw_sys[:300])

    init_mask = df["task"] == "initials"
    n_init = int(init_mask.sum())
    print(f"\nRebuilding {n_init} initials prompts...")

    new_prompts = []
    for _, row in df[init_mask].iterrows():
        new_p = apply_chat_template(tokenizer, icw_sys, row["prefix"])
        new_prompts.append(new_p)

    df.loc[init_mask, "prompt"] = new_prompts
    df.loc[init_mask, "dataset_type"] = args.dataset_type
    # seed already = args.eval_initials_seed in v1; assert
    seeds = df.loc[init_mask, "seed"].unique()
    print(f"  initials seeds present: {seeds}  (expected [{args.eval_initials_seed}])")

    out = Path(args.output_val)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")
    print(f"  task breakdown: {df['task'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
