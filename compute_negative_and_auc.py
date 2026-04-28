"""Compute SW z-stat on test_477 gold_completion (negatives) using the same
per-sample targets as the clean_v2_noex positives. Then compute AUC-ROC and
TPR @ FPR for the 3 extractors.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from acrostics_icw import sample_target_icw
from acrostics_zstat import compute_sw_zstat


def main(args):
    # Load gold_completion as negatives
    rows = []
    with open(args.prompt_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {args.prompt_file}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extractors = ["regex_strict", "regex_loose", "nltk"]

    out_f = open(out_path, "w")
    for i, row in enumerate(tqdm(rows, desc="negatives")):
        target = sample_target_icw(
            seed=args.seed_base + i, length=args.target_length, uppercase=True
        )
        rec = {
            "idx": i,
            "prefix": row["prefix"],
            "target": target,
            "response": row["gold_completion"],
        }
        for ext in extractors:
            try:
                r = compute_sw_zstat(
                    text=row["gold_completion"], target=target,
                    n_resample=args.n_resample, seed=args.seed,
                    truncate_target=True, extractor=ext,
                )
                rec[f"fl_{ext}"] = r.fl
                rec[f"target_eff_{ext}"] = r.target_eff
                rec[f"n_sentences_{ext}"] = r.n_sentences
                rec[f"sw_obs_{ext}"] = r.obs
                rec[f"sw_z_{ext}"] = round(r.z, 4)
                rec[f"sw_p_{ext}"] = round(r.p, 6)
            except Exception as e:
                rec[f"error_{ext}"] = str(e)
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_f.close()
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt_file",
                   default="/home/tianyichen/llm_watermark/data/processed_data/vblagoje_lfqa/test_477.json")
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--target_length", type=int, default=18)
    p.add_argument("--n_resample", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    main(args)
