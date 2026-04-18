"""Pick the best candidate per (prefix, seed) group for reject-sampled
Initials synthesis output.

Input JSONL is produced by ``run_generate_initials_syn_vllm.py --n_candidates N``
followed by ``filter_initials_syn.py`` (which adds ``z_score``, ``gen_len``,
``rep4``, ``regex_meta_leak``).

Selection rule (per group):
  1. Keep candidates passing ALL of:
       - gen_len >= min_gen_len
       - rep4 < max_ngram_rep
       - z_score >= min_z
       - not regex_meta_leak
  2. Of those, keep the one with the HIGHEST z_score.
  3. If no candidate passes, the entire prompt is dropped.

Output: one record per surviving (prefix, seed) group.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True,
                   help="JSONL with per-candidate metrics (output of filter_initials_syn.py "
                        "with regex meta-leak flag; typically *_with_z.jsonl)")
    p.add_argument("--output_file", required=True)
    p.add_argument("--min_gen_len", type=int, default=200)
    p.add_argument("--max_ngram_rep", type=float, default=0.15)
    p.add_argument("--min_z", type=float, default=6.0)
    p.add_argument("--allow_regex_meta_leak", action="store_true",
                   help="If set, do not drop candidates flagged by regex meta-leak filter")
    args = p.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_file)

    recs = [json.loads(l) for l in in_path.open() if l.strip()]
    print(f"Loaded {len(recs)} candidate records from {in_path}")

    # Group by (prefix, seed)
    groups = defaultdict(list)
    for r in recs:
        key = (r["prefix"], int(r["seed"]))
        groups[key].append(r)
    print(f"Unique (prefix, seed) groups: {len(groups)}")

    group_sizes = [len(v) for v in groups.values()]
    print(f"Candidates per group: min={min(group_sizes)} med={sorted(group_sizes)[len(group_sizes)//2]} "
          f"max={max(group_sizes)}")

    def passes(r):
        if r["gen_len"] < args.min_gen_len:
            return False
        if r["rep4"] >= args.max_ngram_rep:
            return False
        if r["z_score"] < args.min_z:
            return False
        if not args.allow_regex_meta_leak and r.get("regex_meta_leak", False):
            return False
        return True

    picked = []
    dropped_groups = 0
    reason_counts = {"no_survivor": 0}
    for key, candidates in groups.items():
        survivors = [c for c in candidates if passes(c)]
        if not survivors:
            dropped_groups += 1
            reason_counts["no_survivor"] += 1
            continue
        best = max(survivors, key=lambda r: r["z_score"])
        picked.append(best)

    with out_path.open("w") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nPicked: {len(picked)} groups")
    print(f"Dropped (no candidate passed): {dropped_groups}")
    print(f"Pass rate: {len(picked) / len(groups) * 100:.1f}%")
    if picked:
        zs = sorted([r["z_score"] for r in picked])
        print(f"Z-score of picked: min={zs[0]:.2f} med={zs[len(zs)//2]:.2f} max={zs[-1]:.2f}")
    print(f"Wrote -> {out_path}")

    stats = {
        "input": str(in_path),
        "output": str(out_path),
        "n_candidates_total": len(recs),
        "n_groups": len(groups),
        "n_picked": len(picked),
        "n_groups_dropped": dropped_groups,
        "thresholds": {
            "min_gen_len": args.min_gen_len,
            "max_ngram_rep": args.max_ngram_rep,
            "min_z": args.min_z,
            "allow_regex_meta_leak": args.allow_regex_meta_leak,
        },
    }
    stats_path = out_path.with_suffix(".pick_stats.json")
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats -> {stats_path}")


if __name__ == "__main__":
    main()
