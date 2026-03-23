#!/usr/bin/env python3
"""
分析 parquet 数据中 prompt+response 的 token 数（使用 Qwen3-14B tokenizer）。
"""
import argparse
import numpy as np

try:
    import pandas as pd
except ImportError:
    raise ImportError("Need pandas. Run: pip install pandas pyarrow")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_path",
        default="/home/tianyichen/llm_watermark/verl/data/sft_modified_loss/vblagoje_lfqa/validation_177.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-14B",
        help="Tokenizer model name (Qwen3-14B)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=64,
        help="Only analyze first N samples (default: all)",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    df = pd.read_parquet(args.parquet_path)

    if args.sample is not None:
        df = df.head(args.sample)

    # Tokenize prompt, response, and prompt+response (batch for speed)
    print(f"Loading parquet: {args.parquet_path}")
    print(f"Samples: {len(df)}, Columns: {df.columns.tolist()}\n")

    prompts = (df["prompt"].fillna("")).astype(str).tolist()
    responses = (df["response"].fillna("")).astype(str).tolist()
    full_texts = [p + r for p, r in zip(prompts, responses)]

    batch_size = 128
    prompt_lens_list = []
    response_lens_list = []
    token_lens_list = []

    print("Tokenizing (batch)...")
    for i in range(0, len(prompts), batch_size):
        p_batch = prompts[i : i + batch_size]
        r_batch = responses[i : i + batch_size]
        f_batch = full_texts[i : i + batch_size]

        pe = tokenizer(p_batch, add_special_tokens=False, truncation=False, padding=False)
        re = tokenizer(r_batch, add_special_tokens=False, truncation=False, padding=False)
        fe = tokenizer(f_batch, add_special_tokens=False, truncation=False, padding=False)

        prompt_lens_list.extend(len(ids) for ids in pe["input_ids"])
        response_lens_list.extend(len(ids) for ids in re["input_ids"])
        token_lens_list.extend(len(ids) for ids in fe["input_ids"])

    prompt_lens = np.array(prompt_lens_list)
    response_lens = np.array(response_lens_list)
    token_lens = np.array(token_lens_list)

    # Summary stats
    def _stats(arr, name):
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
    print("Prompt + Response (full) token counts")
    print("=" * 60)
    s = _stats(token_lens, "prompt+response")
    for k, v in s.items():
        print(f"  {k}: {v:.2f}")
    print()

    print("Prompt token counts")
    print("-" * 40)
    s = _stats(prompt_lens, "prompt")
    for k, v in s.items():
        print(f"  {k}: {v:.2f}")
    print()

    print("Response token counts")
    print("-" * 40)
    s = _stats(response_lens, "response")
    for k, v in s.items():
        print(f"  {k}: {v:.2f}")
    print()

    # Histogram bins for full length
    print("Prompt+Response token length distribution (bins)")
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

    # Sanity: prompt_lens + response_lens vs full
    # Tokenizer may merge differently when concatenating, so full may differ slightly
    diff = token_lens - (prompt_lens + response_lens)
    print("Sanity check: (prompt+response) vs full text tokenization")
    print(f"  Mean diff (full - prompt - response): {diff.mean():.2f}")
    print(f"  Max abs diff: {np.abs(diff).max()}")


if __name__ == "__main__":
    main()
