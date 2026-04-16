"""Assemble mixed-task training parquet: green + initials + neg.

Emits the unified schema:
    prompt, prompt_ref, response, prefix, seed, z_score, fraction, dataset_type, task

``prompt_ref`` is the **per-sample** reference-model input, chosen by task:
  - green    : clean prompt (no ICW) — biased teacher = clean_ref + bias
  - neg      : clean prompt (no ICW) — anchor to base distribution via clean-ref KL
  - initials : ICW prompt (identical to ``prompt``) — biased teacher = ICW_ref + bias
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


NEG_SENTINEL = -99999.0
REQUIRED_COLS = [
    "prompt", "prompt_ref", "response", "prefix",
    "seed", "z_score", "fraction", "dataset_type", "task",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--posneg_parquet", required=True,
                   help="Existing green + neg parquet (5931 pos + 1000 neg)")
    p.add_argument("--initials_filtered_jsonl", required=True,
                   help="Filtered Initials positives (from filter_initials_syn.py)")
    p.add_argument("--output_parquet", required=True)
    args = p.parse_args()

    # ---- Green + Neg from posneg parquet (legacy schema uses "prompt_no_incontext_wm") ----
    posneg_df = pq.read_table(args.posneg_parquet).to_pandas()
    print(f"posneg parquet: {len(posneg_df)} rows")
    posneg_df = posneg_df.copy()
    posneg_df["task"] = posneg_df["fraction"].apply(lambda f: "green" if float(f) > 0 else "neg")
    # Rename the legacy column to unified ``prompt_ref`` (green + neg both use clean)
    if "prompt_no_incontext_wm" not in posneg_df.columns:
        raise KeyError("posneg parquet missing 'prompt_no_incontext_wm' — expected clean prompt column")
    posneg_df["prompt_ref"] = posneg_df["prompt_no_incontext_wm"]

    green_df = posneg_df[posneg_df["task"] == "green"].reset_index(drop=True)
    neg_df = posneg_df[posneg_df["task"] == "neg"].reset_index(drop=True)
    print(f"  green pos: {len(green_df)} | neg: {len(neg_df)}")

    # ---- Initials pos from filtered JSONL ----
    # For initials, ref sees the SAME ICW prompt as actor (prompt_ref = prompt)
    initials_records = []
    with open(args.initials_filtered_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            initials_records.append({
                "prompt": r["prompt"],
                "prompt_ref": r["prompt"],                       # ICW prompt shared with actor
                "response": r["response"],
                "prefix": r["prefix"],
                "seed": int(r["seed"]),
                "z_score": float(r["z_score"]),
                "fraction": float(r["fraction"]),                # γ
                "dataset_type": r.get("dataset_type", "lfqa_initials"),
                "task": "initials",
            })
    initials_df = pd.DataFrame(initials_records, columns=REQUIRED_COLS)
    print(f"initials filtered: {len(initials_df)}")

    # ---- Align schemas to REQUIRED_COLS ----
    green_df = green_df[REQUIRED_COLS].reset_index(drop=True)
    neg_df = neg_df[REQUIRED_COLS].reset_index(drop=True)

    merged = pd.concat([green_df, initials_df, neg_df], ignore_index=True)
    print(f"\nmerged: {len(merged)} total")
    print(f"  task breakdown: {merged['task'].value_counts().to_dict()}")
    print(f"  z_score ranges:")
    for t in ("green", "initials", "neg"):
        sub = merged[merged["task"] == t]["z_score"]
        print(f"    {t}: min={sub.min():.2f} mean={sub.mean():.2f} max={sub.max():.2f}  n={len(sub)}")

    # Sanity: for initials rows, prompt_ref == prompt
    init_mismatch = (merged[merged["task"] == "initials"]["prompt"]
                     != merged[merged["task"] == "initials"]["prompt_ref"]).sum()
    print(f"  initials prompt_ref mismatch rows: {init_mismatch} (should be 0)")
    # Sanity: for green/neg rows, prompt_ref != prompt
    gn_same = (merged[merged["task"].isin(["green", "neg"])]["prompt"]
               == merged[merged["task"].isin(["green", "neg"])]["prompt_ref"]).sum()
    print(f"  green/neg rows where prompt == prompt_ref: {gn_same} (neg-only; green should be 0)")

    out = Path(args.output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
