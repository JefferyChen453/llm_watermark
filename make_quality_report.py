#!/usr/bin/env python3
"""Generate quality eval comparison table + histograms from PPL and Judge outputs.

Reads:
  outputs/quality_eval/ppl/{cell}.jsonl       (per-sample ppl)
  outputs/quality_eval/ppl/{cell}_summary.json
  outputs/quality_eval/judge/{cell}.jsonl     (per-sample 3-dim scores)
  outputs/quality_eval/judge/{cell}_summary.json

Writes:
  vault/04-16-posneg-quality-eval/ppl_hist.png
  vault/04-16-posneg-quality-eval/judge_bars.png
  vault/04-16-posneg-quality-eval/combined_table.csv
  vault/04-16-posneg-quality-eval/combined_table.md
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/tianyichen/llm_watermark")
PPL_DIR = ROOT / "outputs/quality_eval/ppl"
JUDGE_DIR = ROOT / "outputs/quality_eval/judge"
REPORT_DIR = ROOT / "vault/04-16-posneg-quality-eval"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CELLS = [
    ("posneg_frac_0.0", "posneg", "0.0"),
    ("posneg_frac_0.2", "posneg", "0.2"),
    ("posneg_frac_0.4", "posneg", "0.4"),
    ("baseline_frac_0.0", "baseline", "0.0"),
    ("baseline_frac_0.2", "baseline", "0.2"),
    ("baseline_frac_0.4", "baseline", "0.4"),
]


def load_jsonl(p):
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def load_all():
    data = {}
    for tag, model, frac in CELLS:
        ppl_recs = load_jsonl(PPL_DIR / f"{tag}.jsonl")
        with (PPL_DIR / f"{tag}_summary.json").open() as f:
            ppl_sum = json.load(f)
        judge_recs = load_jsonl(JUDGE_DIR / f"{tag}.jsonl")
        with (JUDGE_DIR / f"{tag}_summary.json").open() as f:
            judge_sum = json.load(f)
        data[tag] = {
            "model": model,
            "frac": frac,
            "ppl_recs": ppl_recs,
            "ppl_sum": ppl_sum,
            "judge_recs": judge_recs,
            "judge_sum": judge_sum,
        }
    return data


def fmt(x, n=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    return f"{x:.{n}f}"


def build_table(data):
    header = [
        "model",
        "frac",
        "n",
        "ppl_tw",
        "ppl_mean",
        "ppl_median",
        "rel",
        "clar",
        "qual",
        "overall",
    ]
    rows = []
    for tag, model, frac in CELLS:
        d = data[tag]
        ppl_sum = d["ppl_sum"]
        js = d["judge_sum"]
        rows.append([
            model,
            frac,
            str(ppl_sum["n_samples"]),
            fmt(ppl_sum["avg_ppl_token_weighted"], 3),
            fmt(ppl_sum["avg_ppl_sample_mean"], 3),
            fmt(ppl_sum["ppl_median"], 3),
            f"{fmt(js['relevance_mean'])}±{fmt(js['relevance_std'])}",
            f"{fmt(js['clarity_mean'])}±{fmt(js['clarity_std'])}",
            f"{fmt(js['quality_mean'])}±{fmt(js['quality_std'])}",
            fmt(js["overall_mean"]),
        ])
    return header, rows


def write_md_table(header, rows, out_path):
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def write_csv(header, rows, out_path):
    import csv
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def plot_ppl_hist(data, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fracs = ["0.0", "0.2", "0.4"]
    colors = {"posneg": "#d62728", "baseline": "#1f77b4"}
    for ax, frac in zip(axes, fracs):
        for model in ["baseline", "posneg"]:
            tag = f"{model}_frac_{frac}"
            ppls = [r["ppl"] for r in data[tag]["ppl_recs"] if math.isfinite(r["ppl"])]
            ax.hist(ppls, bins=30, alpha=0.55, color=colors[model], label=model, range=(1.5, 6))
            mean = np.mean(ppls)
            ax.axvline(mean, color=colors[model], linestyle="--", linewidth=1.5)
            ax.text(mean, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1,
                    f" μ={mean:.2f}", color=colors[model], fontsize=8, va="top")
        ax.set_title(f"fraction={frac}")
        ax.set_xlabel("per-sample PPL (Qwen3-32B judge)")
        ax.legend()
    axes[0].set_ylabel("count")
    fig.suptitle("PPL distribution: posneg vs baseline across fractions (n=200 each)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_judge_bars(data, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    dims = ["relevance_mean", "clarity_mean", "quality_mean"]
    dim_labels = ["Relevance", "Clarity", "Quality"]
    fracs = ["0.0", "0.2", "0.4"]
    x = np.arange(len(fracs))
    w = 0.36
    for ax, dim, label in zip(axes, dims, dim_labels):
        posneg_means = [data[f"posneg_frac_{f}"]["judge_sum"][dim] for f in fracs]
        baseline_means = [data[f"baseline_frac_{f}"]["judge_sum"][dim] for f in fracs]
        posneg_stds = [data[f"posneg_frac_{f}"]["judge_sum"][dim.replace("_mean", "_std")] for f in fracs]
        baseline_stds = [data[f"baseline_frac_{f}"]["judge_sum"][dim.replace("_mean", "_std")] for f in fracs]
        ax.bar(x - w / 2, baseline_means, w, yerr=baseline_stds, label="baseline", color="#1f77b4", capsize=3)
        ax.bar(x + w / 2, posneg_means, w, yerr=posneg_stds, label="posneg", color="#d62728", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(fracs)
        ax.set_title(label)
        ax.set_ylim(3.5, 5.05)
        ax.set_xlabel("fraction")
        ax.legend()
        for i, (b, p) in enumerate(zip(baseline_means, posneg_means)):
            ax.text(i - w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
            ax.text(i + w / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
    axes[0].set_ylabel("mean score (1-5)")
    fig.suptitle("LLM-as-Judge (gpt-4o-mini, pointwise 1-5, n=200 each)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    data = load_all()
    header, rows = build_table(data)
    write_md_table(header, rows, REPORT_DIR / "combined_table.md")
    write_csv(header, rows, REPORT_DIR / "combined_table.csv")
    plot_ppl_hist(data, REPORT_DIR / "ppl_hist.png")
    plot_judge_bars(data, REPORT_DIR / "judge_bars.png")
    print("wrote:")
    for p in REPORT_DIR.glob("*"):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
