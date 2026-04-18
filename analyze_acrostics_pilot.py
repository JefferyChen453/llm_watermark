"""Analyze Acrostics pilot output: per-variant hit rate + Levenshtein stats + plots."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(args):
    rows = _load(args.input_file)
    print(f"Loaded {len(rows)} rows")

    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    print()
    print(f"{'Variant':<10} | {'N':>4} | {'SubseqHit%':>11} | {'ContigHit%':>11} | {'Lev.mean':>9} | {'#Sent.mean':>10}")
    print("-" * 80)

    summary = {}
    for v, rs in by_variant.items():
        n = len(rs)
        sub = sum(r["is_subsequence"] for r in rs) / n * 100
        con = sum(r["is_contiguous"] for r in rs) / n * 100
        lev = np.mean([r["levenshtein"] for r in rs])
        sents = np.mean([r["n_sentences"] for r in rs])
        print(f"{v:<10} | {n:>4} | {sub:>11.1f} | {con:>11.1f} | {lev:>9.2f} | {sents:>10.2f}")
        summary[v] = {"n": n, "subseq_hit_pct": sub, "contig_hit_pct": con, "lev_mean": lev, "n_sent_mean": sents}

    # Plot
    try:
        import matplotlib.pyplot as plt
        variants = list(by_variant.keys())
        sub_vals = [summary[v]["subseq_hit_pct"] for v in variants]
        con_vals = [summary[v]["contig_hit_pct"] for v in variants]
        lev_vals = [summary[v]["lev_mean"] for v in variants]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(len(variants))
        axes[0].bar(x - 0.2, sub_vals, 0.4, label="Subsequence")
        axes[0].bar(x + 0.2, con_vals, 0.4, label="Contiguous")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(variants)
        axes[0].set_ylabel("Hit rate (%)")
        axes[0].set_title("Acrostics hit rate by variant")
        axes[0].legend()

        axes[1].bar(x, lev_vals, color="tab:orange")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(variants)
        axes[1].set_ylabel("Mean Levenshtein to nearest |T|-window")
        axes[1].set_title("Mean Lev distance by variant")

        plt.tight_layout()
        out = Path(args.input_file).with_suffix(".png")
        plt.savefig(out, dpi=100)
        print(f"\nSaved plot: {out}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")

    # Dump summary as JSON
    out_json = Path(args.input_file).with_suffix(".summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {out_json}")

    # Qualitative: print 2 examples per variant
    print("\n=== Qualitative examples ===")
    for v, rs in by_variant.items():
        hits = [r for r in rs if r["is_subsequence"]]
        misses = [r for r in rs if not r["is_subsequence"]]
        print(f"\n--- {v}: {len(hits)} hits / {len(misses)} misses ---")
        if hits:
            r = hits[0]
            print(f"  [HIT] target={r['target']!r}, first_letters={r['first_letters']!r}")
            print(f"        response: {r['response'][:260]}...")
        if misses:
            r = misses[0]
            print(f"  [MISS] target={r['target']!r}, first_letters={r['first_letters']!r}")
            print(f"         response: {r['response'][:260]}...")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    args = p.parse_args()
    main(args)
