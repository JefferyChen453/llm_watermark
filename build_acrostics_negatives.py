"""Build negative (human gold_completion) dataset for Acrostics paper DTS pilot.

Pairs each ELI5 query with the same deterministic target X used for positive
generation, producing an apples-to-apples negative set for ROC-AUC eval.
"""

import argparse
import json
import string
from pathlib import Path

from acrostics_icw import sample_target, verify_acrostic


def main(args):
    with open(args.prompt_file) as f:
        rows = [json.loads(l) for l in f]
    rows = rows[: args.num_test]
    print(f"Loaded {len(rows)} rows from {args.prompt_file}")

    pool = string.ascii_uppercase if args.target_uppercase else string.ascii_lowercase
    targets = [
        sample_target(seed=args.seed_base + i, length=args.target_length, pool=pool)
        for i in range(len(rows))
    ]

    output_path = Path(args.output_dir) / (
        f"acrostics_pilot_{args.model_tag}_n{len(rows)}_len{args.target_length}"
        f"_{args.output_tag}_negatives.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing = 0
    written = 0
    with open(output_path, "w") as f:
        for i, r in enumerate(rows):
            target = targets[i]
            gold = r.get("gold_completion", "") or ""
            if not gold:
                missing += 1
                continue
            verdict = verify_acrostic(gold, target)
            rec = {
                "idx": i,
                "variant": "human_gold",
                "target": target,
                "prefix": r["prefix"],
                "response": gold,
                "first_letters": verdict.first_letters,
                "is_subsequence": verdict.is_subsequence,
                "is_contiguous": verdict.is_contiguous,
                "levenshtein": verdict.levenshtein_to_substring,
                "n_sentences": verdict.n_sentences,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {output_path} ({written} rows, {missing} skipped for missing gold_completion)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_tag", default="qwen3-14b")
    p.add_argument("--prompt_file", default="/home/tianyichen/llm_watermark/data/processed_data/vblagoje_lfqa/test_477.json")
    p.add_argument("--num_test", type=int, default=100)
    p.add_argument("--target_length", type=int, default=10)
    p.add_argument("--seed_base", type=int, default=20260421)
    p.add_argument("--target_uppercase", action="store_true")
    p.add_argument("--output_dir", default="/home/tianyichen/llm_watermark/outputs/acrostics_pilot")
    p.add_argument("--output_tag", default="paperdts")
    args = p.parse_args()
    main(args)
