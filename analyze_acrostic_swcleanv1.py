"""Analyze the swcleanv1 pilot output:
For each generation, run SW shuffle-S detection with 3 extractors:
  - regex_strict (RL-reward style; blocks heading/list cheats)
  - regex_loose  (our regex, no strict filter)
  - nltk         (ICW paper's NLTK Punkt + first-alpha)

Also flag tag-leak patterns (`<response`, `<query`, `<task`, `<example`,
`Walkthrough`).

Each input row is augmented with per-extractor SW stats and written to a new
JSONL.
"""

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from acrostics_zstat import compute_sw_zstat


TAG_PATTERNS = {
    "tag_response": re.compile(r"<\s*response\b", re.IGNORECASE),
    "tag_query": re.compile(r"<\s*query\b", re.IGNORECASE),
    "tag_task": re.compile(r"<\s*task\b", re.IGNORECASE),
    "tag_example": re.compile(r"<\s*example\b", re.IGNORECASE),
    "label_walkthrough": re.compile(r"\bwalkthrough\b", re.IGNORECASE),
}


def detect_tag_leaks(text: str) -> dict:
    return {name: bool(rx.search(text)) for name, rx in TAG_PATTERNS.items()}


def run_one_extractor(text, target, extractor, n_resample, seed):
    r = compute_sw_zstat(
        text=text, target=target,
        n_resample=n_resample, seed=seed,
        truncate_target=True, extractor=extractor,
    )
    return {
        f"fl_{extractor}": r.fl,
        f"target_eff_{extractor}": r.target_eff,
        f"n_sentences_{extractor}": r.n_sentences,
        f"sw_obs_{extractor}": r.obs,
        f"sw_mu_{extractor}": round(r.mu, 4),
        f"sw_sigma_{extractor}": round(r.sigma, 4),
        f"sw_z_{extractor}": round(r.z, 4),
        f"sw_p_{extractor}": round(r.p, 6),
    }


def main(args):
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_in = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows_in.append(json.loads(line))
    print(f"Loaded {len(rows_in)} rows from {in_path}")
    print(f"Extractors: {args.extractors}, n_resample={args.n_resample}")

    out_f = open(out_path, "w")
    for row in tqdm(rows_in, desc="detect"):
        text = row["response"]
        target = row["target"]
        out = dict(row)  # copy original fields

        # Tag-leak flags (extractor-independent)
        leak = detect_tag_leaks(text)
        out["tag_leak_any"] = any(leak.values())
        out.update(leak)

        # Per-extractor SW stats
        for extractor in args.extractors:
            try:
                out.update(run_one_extractor(
                    text, target, extractor,
                    n_resample=args.n_resample, seed=args.seed,
                ))
            except Exception as e:
                # Don't crash the whole run on one bad row
                out[f"error_{extractor}"] = str(e)

        out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
    out_f.close()
    print(f"Wrote {len(rows_in)} rows -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input jsonl from run_acrostic_swcleanv1_pilot.py")
    p.add_argument("--output", required=True, help="Output jsonl with sw stats")
    p.add_argument("--extractors", nargs="+",
                   default=["regex_strict", "regex_loose", "nltk"],
                   help="Which extractors to run")
    p.add_argument("--n_resample", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
