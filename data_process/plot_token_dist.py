#!/usr/bin/env python3
"""Draw token length distribution plots from report data. Saves to data/*.png."""
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    raise SystemExit(f"Need matplotlib and numpy: {e}")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def plot_dist(labels, counts, total, title, out_path):
    counts = np.array(counts)
    pcts = 100 * counts / total if total else counts * 0
    x = np.arange(len(labels))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(x - width / 2, counts, width, label="Count", color="steelblue", alpha=0.9)
    ax1.set_ylabel("Count", color="steelblue")
    ax1.set_xlabel("Token length bucket")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, pcts, width, label="%", color="coral", alpha=0.7)
    ax2.set_ylabel("%", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved", out_path)


def main():
    # LFRQA (from report)
    lfrqa_labels = ["0-10", "10-50", "50-100", "100-200", "200-500", "500-1000", "1000-2000", "2000-5000", "5000-10000", "10000+"]
    lfrqa_counts = [5, 2, 0, 0, 0, 0, 0, 0, 0, 0]
    plot_dist(
        lfrqa_labels,
        lfrqa_counts,
        7,
        "LFRQA — Golden answer token length",
        DATA_DIR / "lfrqa_token_dist.png",
    )

    # vblagoje_lfqa (from report)
    vblagoje_labels = ["0-10", "10-50", "50-100", "100-200", "200-500", "500-1000", "1000-2000", "2000-5000", "5000-10000", "10000-20000", "20000+"]
    vblagoje_counts = [1445, 42942, 57476, 60768, 51356, 17153, 7210, 815, 2, 0, 0]
    plot_dist(
        vblagoje_labels,
        vblagoje_counts,
        239167,
        "vblagoje_lfqa — Golden answer token length",
        DATA_DIR / "vblagoje_token_dist.png",
    )


if __name__ == "__main__":
    main()
