"""Extractor diff analysis: where do strict / loose / nltk disagree?

Reads analyzed jsonl (from analyze_acrostic_swcleanv1.py) and reports:
  - per-pair disagreement rate
  - per-extractor mean / median sw_z, p<0.05 rate, hit_rate (z >= threshold)
  - top-K samples where extractors disagree the most (large diff in fl)
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_extractor_stats(rows, ext, z_threshold=4.0):
    zs = [r.get(f"sw_z_{ext}", 0.0) for r in rows if f"sw_z_{ext}" in r]
    ps = [r.get(f"sw_p_{ext}", 1.0) for r in rows if f"sw_p_{ext}" in r]
    nsent = [r.get(f"n_sentences_{ext}", 0) for r in rows if f"n_sentences_{ext}" in r]

    return {
        "n": len(zs),
        "z_mean": round(statistics.mean(zs), 3) if zs else 0.0,
        "z_median": round(statistics.median(zs), 3) if zs else 0.0,
        "z_p25": round(statistics.quantiles(zs, n=4)[0], 3) if len(zs) >= 4 else None,
        "z_p75": round(statistics.quantiles(zs, n=4)[2], 3) if len(zs) >= 4 else None,
        "z_max": round(max(zs), 3) if zs else 0.0,
        "z_min": round(min(zs), 3) if zs else 0.0,
        "p_lt_0.05": round(sum(1 for p in ps if p < 0.05) / len(ps), 4) if ps else 0.0,
        "p_lt_0.01": round(sum(1 for p in ps if p < 0.01) / len(ps), 4) if ps else 0.0,
        "p_lt_0.001": round(sum(1 for p in ps if p < 0.001) / len(ps), 4) if ps else 0.0,
        f"z_ge_{z_threshold}_rate": round(sum(1 for z in zs if z >= z_threshold) / len(zs), 4) if zs else 0.0,
        "n_sent_median": int(statistics.median(nsent)) if nsent else 0,
        "n_sent_p95": int(statistics.quantiles(nsent, n=20)[18]) if len(nsent) >= 20 else None,
    }


def disagreement_pairs(rows, exts):
    """For each pair (e1, e2), count how often fl_{e1} != fl_{e2}."""
    out = {}
    n = len(rows)
    for i, e1 in enumerate(exts):
        for e2 in exts[i+1:]:
            diff = sum(1 for r in rows
                       if r.get(f"fl_{e1}", "") != r.get(f"fl_{e2}", ""))
            out[f"{e1}_vs_{e2}"] = {
                "diff_count": diff,
                "diff_rate": round(diff / n, 4) if n else 0.0,
            }
    return out


def show_top_diffs(rows, e1, e2, k=5):
    """Return top-K rows with largest |len(fl_e1) - len(fl_e2)| difference."""
    diffs = []
    for r in rows:
        fl1 = r.get(f"fl_{e1}", "")
        fl2 = r.get(f"fl_{e2}", "")
        if fl1 != fl2:
            length_diff = abs(len(fl1) - len(fl2))
            z1 = r.get(f"sw_z_{e1}", 0.0)
            z2 = r.get(f"sw_z_{e2}", 0.0)
            diffs.append((length_diff, abs(z1 - z2), r["idx"], fl1, fl2, z1, z2))
    diffs.sort(reverse=True)
    return diffs[:k]


def main(args):
    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}\n")

    # Detect extractors present
    exts = sorted({k[3:] for r in rows for k in r if k.startswith("fl_")})
    print(f"Extractors in data: {exts}\n")

    # Per-extractor stats
    print("=" * 72)
    print("Per-extractor SW z-stat distribution")
    print("=" * 72)
    for ext in exts:
        s = per_extractor_stats(rows, ext, z_threshold=args.z_threshold)
        print(f"\n[{ext}]")
        for k, v in s.items():
            print(f"  {k:>22s}: {v}")

    # Tag leak rate
    print("\n" + "=" * 72)
    print("Tag leak rate (clean_v1 sanity)")
    print("=" * 72)
    leak_keys = ["tag_response", "tag_query", "tag_task", "tag_example", "label_walkthrough"]
    for k in leak_keys + ["tag_leak_any"]:
        cnt = sum(1 for r in rows if r.get(k))
        print(f"  {k:>20s}: {cnt:>4d} / {len(rows)}  ({cnt / len(rows) * 100:.1f}%)")

    # Truncation rate
    print("\n" + "=" * 72)
    print("Generation finish_reason")
    print("=" * 72)
    fr = Counter(r.get("finish_reason", "?") for r in rows)
    for k, v in fr.most_common():
        print(f"  {k:>20s}: {v:>4d}  ({v / len(rows) * 100:.1f}%)")

    # Pairwise disagreement
    print("\n" + "=" * 72)
    print("Extractor pairwise disagreement (fl_e1 != fl_e2)")
    print("=" * 72)
    dis = disagreement_pairs(rows, exts)
    for pair, st in dis.items():
        print(f"  {pair:>30s}: diff={st['diff_count']}/{len(rows)}  ({st['diff_rate']*100:.1f}%)")

    # Top diffs case study
    if len(exts) >= 2:
        print("\n" + "=" * 72)
        print(f"Top {args.top_k} biggest fl-length diffs (regex_strict vs nltk)")
        print("=" * 72)
        if "regex_strict" in exts and "nltk" in exts:
            top = show_top_diffs(rows, "regex_strict", "nltk", k=args.top_k)
            for ld, zd, idx, fl1, fl2, z1, z2 in top:
                print(f"\n  idx={idx}: |len_diff|={ld}, |z_diff|={zd:.2f}")
                print(f"    strict ({len(fl1)}): {fl1!r}  z={z1:+.2f}")
                print(f"    nltk   ({len(fl2)}): {fl2!r}  z={z2:+.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--z_threshold", type=float, default=4.0,
                   help="Threshold for hit_rate (z >= threshold)")
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    main(args)
