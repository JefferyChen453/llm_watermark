#!/usr/bin/env python3
"""
1. 将 14B_lfqa jsonl 转为 parquet：prompt=input_prompt, response=gold_completion，并保留 gen_completion 供分析。
2. 用 Qwen3-14B tokenizer 对 input_prompt+gen_completion 做 token 数分析（同 analyze_parquet_tokens）。
"""
import argparse
import json
import os
import numpy as np

try:
    import pandas as pd
except ImportError:
    raise ImportError("Need pandas. Run: pip install pandas pyarrow")


def load_jsonl_simple(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default="/home/tianyichen/llm_watermark/temp/sft_train/14B_lfqa_filtered_data_pos_3872_neg_1129.jsonl",
        help="Input JSONL (keys: input_prompt, prefix, gold_completion, gen_completion)",
    )
    parser.add_argument(
        "--output_parquet",
        default=None,
        help="Output parquet path (default: same dir as input, .parquet)",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-14B",
        help="Tokenizer for analysis",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only analyze first N samples (default: all)",
    )
    args = parser.parse_args()

    # ----- 1. Convert jsonl -> parquet -----
    rows = load_jsonl_simple(args.input_jsonl)
    parquet_rows = []
    for r in rows:
        prompt = r.get("input_prompt") or r.get("input_prompts") or ""
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        response = r.get("gold_completion", "")
        gen = r.get("gen_completion", "")
        parquet_rows.append({"prompt": prompt, "response": response, "gen_completion": gen})

    out_path = args.output_parquet
    if out_path is None:
        base = args.input_jsonl.rstrip(".jsonl")
        out_path = base + ".parquet"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    pd.DataFrame(parquet_rows).to_parquet(out_path, index=False)
    print(f"Saved parquet: {out_path} ({len(parquet_rows)} rows)\n")

    # ----- 2. Token analysis on prompt + gen_completion -----
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    df = pd.read_parquet(out_path)
    if args.sample is not None:
        df = df.head(args.sample)

    prompts = (df["prompt"].fillna("")).astype(str).tolist()
    gen_completions = (df["gen_completion"].fillna("")).astype(str).tolist()
    full_texts = [p + g for p, g in zip(prompts, gen_completions)]

    batch_size = 128
    prompt_lens_list = []
    gen_lens_list = []
    token_lens_list = []

    print("Tokenizing (input_prompt + gen_completion)...")
    for i in range(0, len(prompts), batch_size):
        p_batch = prompts[i : i + batch_size]
        g_batch = gen_completions[i : i + batch_size]
        f_batch = full_texts[i : i + batch_size]

        pe = tokenizer(p_batch, add_special_tokens=False, truncation=False, padding=False)
        ge = tokenizer(g_batch, add_special_tokens=False, truncation=False, padding=False)
        fe = tokenizer(f_batch, add_special_tokens=False, truncation=False, padding=False)

        prompt_lens_list.extend(len(ids) for ids in pe["input_ids"])
        gen_lens_list.extend(len(ids) for ids in ge["input_ids"])
        token_lens_list.extend(len(ids) for ids in fe["input_ids"])

    prompt_lens = np.array(prompt_lens_list)
    gen_lens = np.array(gen_lens_list)
    token_lens = np.array(token_lens_list)

    def _stats(arr):
        return {
            "min": arr.min(),
            "max": arr.max(),
            "mean": arr.mean(),
            "median": np.median(arr),
            "std": arr.std(),
            "p50": np.percentile(arr, 50),
            "p90": np.percentile(arr, 90),
            "p95": np.percentile(arr, 95),
            "p99": np.percentile(arr, 99),
        }

    print("=" * 60)
    print("Input_prompt + Gen_completion token counts (Qwen3-14B tokenizer)")
    print("=" * 60)
    for k, v in _stats(token_lens).items():
        print(f"  {k}: {v:.2f}")
    print()

    print("Input_prompt token counts")
    print("-" * 40)
    for k, v in _stats(prompt_lens).items():
        print(f"  {k}: {v:.2f}")
    print()

    print("Gen_completion token counts")
    print("-" * 40)
    for k, v in _stats(gen_lens).items():
        print(f"  {k}: {v:.2f}")
    print()

    print("(Input_prompt + Gen_completion) length distribution (bins)")
    print("-" * 50)
    bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, float("inf")]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if hi == float("inf"):
            label = f">={lo}"
            n = (token_lens >= lo).sum()
        else:
            label = f"{lo}-{hi}"
            n = ((token_lens >= lo) & (token_lens < hi)).sum()
        pct = n / len(token_lens) * 100
        print(f"  {label:>12}: {n:>6} ({pct:5.2f}%)")
    print()

    diff = token_lens - (prompt_lens + gen_lens)
    print("Sanity: full - prompt - gen_completion")
    print(f"  Mean diff: {diff.mean():.2f}, Max abs diff: {np.abs(diff).max()}")


if __name__ == "__main__":
    main()
