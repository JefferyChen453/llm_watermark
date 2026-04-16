"""Assemble mixed-task validation parquet.

- Takes the existing posneg val parquet (177 green pos + 177 neg), adds a
  "task" column to distinguish the two.
- Adds 177 Initials positive rows, one per prefix, using a fixed
  eval_initials_seed and the Initials ICW system prompt. Response is a
  placeholder — vLLM regenerates at validation time.

Schema matches the RLHFDataset val loader:
  prompt, response, prefix, seed, dataset_type, positive_or_negative, task
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent))

from dataset import apply_chat_template
from gptwm_initials import partition_letters
from prompt import get_initials_incontext_prompt


REQUIRED_COLS = [
    "prompt", "response", "prefix", "seed", "dataset_type",
    "positive_or_negative", "task",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--posneg_val_parquet",
                   default="verl/data/sft_modified_loss/vblagoje_lfqa/validation_pos177_neg177_seed0_frac0.25_new_sys_prompt.parquet")
    p.add_argument("--output_parquet", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-14B")
    p.add_argument("--eval_initials_seed", type=int, default=0)
    p.add_argument("--dataset_type_initials", default="lfqa_initials")
    args = p.parse_args()

    # ---- Load existing val, tag green + neg ----
    df = pq.read_table(args.posneg_val_parquet).to_pandas()
    print(f"posneg val: {len(df)} rows")
    df = df.copy()

    if "positive_or_negative" not in df.columns:
        raise KeyError("posneg val parquet missing 'positive_or_negative' column")
    df["task"] = df["positive_or_negative"].apply(
        lambda x: "green" if x == "positive" else "neg"
    )
    # Pos = green, Neg = neg
    green_df = df[df["task"] == "green"].reset_index(drop=True)
    neg_df = df[df["task"] == "neg"].reset_index(drop=True)
    print(f"  green pos: {len(green_df)} | neg: {len(neg_df)}")

    # Ensure schema
    for sdf in (green_df, neg_df):
        for col in REQUIRED_COLS:
            if col not in sdf.columns:
                raise KeyError(f"val parquet missing column {col!r}")
    green_df = green_df[REQUIRED_COLS]
    neg_df = neg_df[REQUIRED_COLS]

    # ---- Build 177 Initials pos rows (same prefixes as green pos) ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    green_letters, red_letters = partition_letters(args.eval_initials_seed)
    g_sorted = sorted(green_letters); r_sorted = sorted(red_letters)
    icw_system = get_initials_incontext_prompt(
        args.dataset_type_initials, g_sorted, r_sorted,
    )
    print(f"eval_initials_seed={args.eval_initials_seed}")
    print(f"  green: {g_sorted}")
    print(f"  red:   {r_sorted}")

    initials_rows = []
    for _, row in green_df.iterrows():
        prefix = row["prefix"]
        prompt_icw = apply_chat_template(tokenizer, icw_system, prefix)
        initials_rows.append({
            "prompt": prompt_icw,
            "response": " ",  # placeholder; vLLM regenerates at val time
            "prefix": prefix,
            "seed": args.eval_initials_seed,
            "dataset_type": args.dataset_type_initials,
            "positive_or_negative": "positive",
            "task": "initials",
        })
    initials_df = pd.DataFrame(initials_rows, columns=REQUIRED_COLS)
    print(f"initials pos: {len(initials_df)}")

    merged = pd.concat([green_df, initials_df, neg_df], ignore_index=True)
    print(f"\nmerged: {len(merged)} total | tasks: {merged['task'].value_counts().to_dict()}")

    out = Path(args.output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
