"""Detect Initials ICW watermark on a generation JSONL.

For each record:
  - Tokenize ``gen_completion`` with the same tokenizer used for generation.
  - Compute z-score and hit-rate on leading-space English tokens whose first
    letter is in the green set (determined by ``seed``).
  - Write per-sample stats to a sibling ``*_z.jsonl`` + summary.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from gptwm_initials import (
    InitialsDetector,
    build_token_first_letter_map,
    compute_gamma_from_stats,
    partition_letters,
)
from gptwm import _get_english_token_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default=None,
                        help="Per-sample output (default: input_file with _z suffix)")
    parser.add_argument("--summary_file", default=None)
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B")
    parser.add_argument("--stats_file", default="data/initials_icw/leading_space_first_letter_stats.json")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override per-record seed (else read from record)")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_file) if args.output_file else in_path.with_name(
        in_path.stem + "_z.jsonl"
    )
    summary_path = Path(args.summary_file) if args.summary_file else in_path.with_name(
        in_path.stem + "_summary.json"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name)
    vocab_size = tokenizer.vocab_size
    english_ids = _get_english_token_ids(tokenizer, vocab_size)
    first_letter_map = build_token_first_letter_map(tokenizer, vocab_size, english_ids)

    # Cache detectors per seed
    detector_cache = {}
    def get_detector(seed: int) -> InitialsDetector:
        if seed not in detector_cache:
            green, _ = partition_letters(seed)
            gamma = compute_gamma_from_stats(green, args.stats_file)
            detector_cache[seed] = InitialsDetector(
                gamma=gamma, seed=seed, strength=0.0,
                vocab_size=vocab_size,
                model_emb_length=config.vocab_size,
                tokenizer=tokenizer,
                english_token_ids=english_ids,
                first_letter_map=first_letter_map,
            )
        return detector_cache[seed]

    records = [json.loads(l) for l in in_path.open() if l.strip()]
    print(f"Loaded {len(records)} records from {in_path}")

    z_scores = []
    hit_rates = []
    unidetect_z = []
    n_leading_space = []
    gen_lens = []

    with out_path.open("w") as f:
        for rec in records:
            seed = args.seed if args.seed is not None else int(rec.get("seed", 0))
            det = get_detector(seed)
            ids = tokenizer(rec["gen_completion"], add_special_tokens=False)["input_ids"]
            n_green, n_total = det.hits(ids)
            z = det._z_score(n_green, n_total, det.gamma)
            uni_z = det.unidetect(ids)
            hit = n_green / n_total if n_total > 0 else 0.0

            z_scores.append(z)
            hit_rates.append(hit)
            unidetect_z.append(uni_z)
            n_leading_space.append(n_total)
            gen_lens.append(len(ids))

            out_rec = {
                **{k: rec[k] for k in ("prefix", "seed") if k in rec},
                "gen_len": len(ids),
                "n_leading_space_eng": n_total,
                "n_green_initial": n_green,
                "hit_rate": hit,
                "z_score": z,
                "unidetect_z": uni_z,
                "gamma": det.gamma,
                "green_letters": det.green_letters,
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    def stats(xs):
        arr = np.array(xs, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    seed_used = args.seed if args.seed is not None else int(records[0].get("seed", 0))
    green, red = partition_letters(seed_used)
    gamma = compute_gamma_from_stats(green, args.stats_file)

    summary = {
        "input_file": str(in_path),
        "n_samples": len(records),
        "seed": seed_used,
        "green_letters": green,
        "red_letters": red,
        "gamma": gamma,
        "z_score": stats(z_scores),
        "unidetect_z": stats(unidetect_z),
        "hit_rate": stats(hit_rates),
        "n_leading_space": stats(n_leading_space),
        "gen_len": stats(gen_lens),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"per-sample -> {out_path}")
    print(f"summary    -> {summary_path}")
    print(f"  gamma={gamma:.4f}  mean z={summary['z_score']['mean']:.3f}  mean hit_rate={summary['hit_rate']['mean']:.3f}  mean len={summary['gen_len']['mean']:.1f}")


if __name__ == "__main__":
    main()
