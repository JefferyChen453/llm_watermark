import argparse
import glob
import json
import os
from typing import List


def load_summaries(input_dir: str) -> List[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*_summary.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda x: x["fraction"])
    return rows


def format_float(value):
    if value is None:
        return "NA"
    return f"{value:.6f}"


def write_markdown(rows: List[dict], output_file: str):
    if not rows:
        raise ValueError(f"No summary json files found for {output_file}")

    before_vals = [row["avg_green_prob_before_bias"] for row in rows]
    after_vals = [row["avg_green_prob_after_bias"] for row in rows]
    deltas = [after - before for before, after in zip(before_vals, after_vals, strict=True)]

    lines = [
        "# Avg Green Probability Analysis",
        "",
        f"- Model: `{rows[0]['model_name']}`",
        f"- Prompt file: `{rows[0]['prompt_file']}`",
        f"- Strength: `{rows[0]['strength']}`",
        f"- Seed num: `{rows[0]['seed_num']}`",
        f"- Samples per run: `{rows[0]['num_samples']}`",
        f"- English-only mask: `{rows[0]['only_English']}`",
        "",
        "## Summary Table",
        "",
        "| Fraction | Avg green prob before bias | Avg green prob after bias | Delta | Generated tokens | Summary file |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        delta = row["avg_green_prob_after_bias"] - row["avg_green_prob_before_bias"]
        lines.append(
            "| "
            f"{row['fraction']} | "
            f"{format_float(row['avg_green_prob_before_bias'])} | "
            f"{format_float(row['avg_green_prob_after_bias'])} | "
            f"{format_float(delta)} | "
            f"{row['total_generated_tokens']} | "
            f"`{os.path.basename(row['sample_metrics_file'])}` |"
        )

    lines.extend(
        [
            "",
            "## Observations",
            "",
            f"- Lowest fraction run: `fraction={rows[0]['fraction']}` with before/after `{format_float(before_vals[0])}` / `{format_float(after_vals[0])}`.",
            f"- Highest fraction run: `fraction={rows[-1]['fraction']}` with before/after `{format_float(before_vals[-1])}` / `{format_float(after_vals[-1])}`.",
            f"- Mean before-bias avg green prob across runs: `{format_float(sum(before_vals) / len(before_vals))}`.",
            f"- Mean after-bias avg green prob across runs: `{format_float(sum(after_vals) / len(after_vals))}`.",
            f"- Mean bias uplift across runs: `{format_float(sum(deltas) / len(deltas))}`.",
            "",
        ]
    )

    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    rows = load_summaries(args.input_dir)
    write_markdown(rows, args.output_file)
    print(args.output_file)


if __name__ == "__main__":
    main()
