"""Pretty-print N random samples from an analyzed jsonl for human review.

Each sample shows:
  - prefix (truncated)
  - secret string + length
  - generated response (full)
  - per-extractor extracted first-letter sequence
  - per-extractor SW stats (obs / mu / sigma / z / p / target_eff)
  - tag-leak flags (if any)
"""

import argparse
import json
import random
from pathlib import Path


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " […]"


def fmt_sample(row: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(f"Sample idx={row.get('idx')} | finish={row.get('finish_reason', '?')} "
                 f"| n_out_tok={row.get('n_output_tokens', '?')}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("[QUERY]")
    lines.append(truncate(row.get("prefix", ""), 600))
    lines.append("")
    target = row.get("target", "")
    lines.append(f"[SECRET]  {target}  (len {len(target)})")
    lines.append("")
    lines.append("[GENERATION]")
    lines.append(row.get("response", ""))
    lines.append("")

    # Per-extractor block
    extractors = []
    for k in row:
        if k.startswith("fl_"):
            extractors.append(k[3:])  # strip "fl_"
    lines.append("[EXTRACTORS + SW]")
    for ext in extractors:
        fl = row.get(f"fl_{ext}", "")
        n_sent = row.get(f"n_sentences_{ext}", "?")
        tgt_eff = row.get(f"target_eff_{ext}", "")
        obs = row.get(f"sw_obs_{ext}", "?")
        mu = row.get(f"sw_mu_{ext}", "?")
        sigma = row.get(f"sw_sigma_{ext}", "?")
        z = row.get(f"sw_z_{ext}", "?")
        p = row.get(f"sw_p_{ext}", "?")
        lines.append(f"  {ext:>14s}: fl={fl!r}  n_sent={n_sent}  tgt_eff={tgt_eff!r}")
        lines.append(f"  {'':>14s}  SW obs={obs} μ={mu} σ={sigma} | zE={z} | p={p}")

    # Tag leaks
    leak_keys = [k for k in row if k.startswith("tag_") or k == "label_walkthrough"]
    leaks = [k for k in leak_keys if row.get(k) is True]
    lines.append("")
    if leaks:
        lines.append(f"[TAG LEAK]  ⚠️  {leaks}")
    else:
        lines.append("[TAG LEAK]  none")
    lines.append("")
    return "\n".join(lines)


def main(args):
    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows; sampling {args.n} with seed={args.seed}")

    rng = random.Random(args.seed)
    sample_idxs = rng.sample(range(len(rows)), min(args.n, len(rows)))
    sample_idxs.sort()

    out_lines = []
    for ix in sample_idxs:
        out_lines.append(fmt_sample(rows[ix]))

    out_text = "\n".join(out_lines)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(out_text)
        print(f"Wrote inspection to {args.output}")
    else:
        print(out_text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None,
                   help="If unset, print to stdout")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
