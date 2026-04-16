#!/usr/bin/env python3
"""Generate pilot comparison report for Initials ICW."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/tianyichen/llm_watermark")
PILOT_DIR = ROOT / "outputs/initials_icw_pilot"
REPORT_DIR = ROOT / "vault/04-16-initials-icw-pilot"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CELLS = [
    ("01_plain_s0", "plain", 0.0, False, False),
    ("02_prompt_s0", "ICW prompt", 0.0, True, False),
    ("03_prompt_s2", "ICW prompt", 2.0, True, True),
    ("04_prompt_s5", "ICW prompt", 5.0, True, True),
    ("05_prompt_s10", "ICW prompt", 10.0, True, True),
]


def find_outputs(cell_dir: Path):
    jsonls = [p for p in cell_dir.glob("*.jsonl") if not p.name.endswith("_z.jsonl")]
    if not jsonls:
        return None, None, None
    gen = jsonls[0]
    z = gen.with_name(gen.stem + "_z.jsonl")
    summary = gen.with_name(gen.stem + "_summary.json")
    return gen, z, summary


def load_all():
    data = {}
    for tag, *_ in CELLS:
        cell_dir = PILOT_DIR / tag
        gen, z, summary_path = find_outputs(cell_dir)
        if not (gen and z and summary_path and summary_path.exists()):
            print(f"  WARN: missing outputs for {tag}")
            continue
        with summary_path.open() as f:
            summary = json.load(f)
        per_sample = [json.loads(l) for l in z.open() if l.strip()]
        data[tag] = {"summary": summary, "per_sample": per_sample, "gen_file": str(gen)}
    return data


def build_table(data):
    header = ["cell", "prompt", "strength", "n", "mean_z", "median_z", "mean_hit", "mean_len", "mean_lsEng"]
    rows = []
    for tag, prompt_label, strength, has_prompt, has_bias in CELLS:
        if tag not in data:
            continue
        s = data[tag]["summary"]
        rows.append([
            tag,
            prompt_label,
            f"{strength:.0f}",
            str(s["n_samples"]),
            f"{s['z_score']['mean']:.3f}",
            f"{s['z_score']['median']:.3f}",
            f"{s['hit_rate']['mean']:.3f}",
            f"{s['gen_len']['mean']:.1f}",
            f"{s['n_leading_space']['mean']:.1f}",
        ])
    return header, rows


def write_md_table(header, rows, out_path):
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def plot_z(data, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = []
    means = []
    stds = []
    for tag, prompt_label, strength, *_ in CELLS:
        if tag not in data:
            continue
        s = data[tag]["summary"]
        labels.append(tag.replace("_", "\n"))
        means.append(s["z_score"]["mean"])
        stds.append(s["z_score"]["std"])
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.axhline(4, color="red", linewidth=1, linestyle=":", label="strong signal (z=4)")
    ax.set_ylabel("z-score (mean ± std)")
    ax.set_title("Initials ICW z-score across pilot cells (seed=0, γ=0.411)")
    ax.legend()
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.2, f"{m:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_hit_rate(data, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = []
    means = []
    for tag, *_ in CELLS:
        if tag not in data:
            continue
        labels.append(tag.replace("_", "\n"))
        means.append(data[tag]["summary"]["hit_rate"]["mean"])
    x = np.arange(len(labels))
    ax.bar(x, means, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    gamma_line = data[next(iter(data))]["summary"]["gamma"]
    ax.axhline(gamma_line, color="gray", linewidth=1.5, linestyle="--", label=f"γ (null expect) = {gamma_line:.3f}")
    ax.set_ylabel("green-initial hit rate")
    ax.set_ylim(0, 1)
    ax.set_title("Initials ICW green-initial hit rate across pilot cells")
    ax.legend()
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_len(data, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = []
    means = []
    for tag, *_ in CELLS:
        if tag not in data:
            continue
        labels.append(tag.replace("_", "\n"))
        means.append(data[tag]["summary"]["gen_len"]["mean"])
    x = np.arange(len(labels))
    ax.bar(x, means, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("gen length (tokens)")
    ax.set_title("Initials ICW mean gen length (cap = 600)")
    for i, m in enumerate(means):
        ax.text(i, m + 5, f"{m:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def get_example(data, tag, idx=0):
    if tag not in data:
        return None
    per_sample = data[tag]["per_sample"]
    if idx >= len(per_sample):
        return None
    return per_sample[idx]


def main():
    data = load_all()
    print(f"Loaded {len(data)} cells")
    header, rows = build_table(data)
    write_md_table(header, rows, REPORT_DIR / "table.md")
    plot_z(data, REPORT_DIR / "zscore.png")
    plot_hit_rate(data, REPORT_DIR / "hit_rate.png")
    plot_len(data, REPORT_DIR / "gen_len.png")
    print("wrote:")
    for p in REPORT_DIR.glob("*"):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
