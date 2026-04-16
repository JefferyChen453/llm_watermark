#!/usr/bin/env python3
"""Compute AUC-ROC + key operating points from two detection _z.jsonl files:
positive (watermarked) vs negative (plain). Plot ROC curve."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def load_z(z_path: Path, field: str = "z_score"):
    z_vals = []
    for line in z_path.open():
        rec = json.loads(line)
        z_vals.append(rec[field])
    return np.array(z_vals, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-file", required=True, help="watermarked _z.jsonl")
    parser.add_argument("--neg-file", required=True, help="plain (non-watermarked) _z.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--field", default="z_score", choices=["z_score", "unidetect_z"])
    parser.add_argument("--tag", default="initials_s3_vs_plain")
    args = parser.parse_args()

    pos = load_z(Path(args.pos_file), args.field)
    neg = load_z(Path(args.neg_file), args.field)
    print(f"pos n={len(pos)}  mean={pos.mean():.3f}  std={pos.std():.3f}  median={np.median(pos):.3f}")
    print(f"neg n={len(neg)}  mean={neg.mean():.3f}  std={neg.std():.3f}  median={np.median(neg):.3f}")

    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    y_score = np.concatenate([pos, neg])
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)

    # Key operating points
    def tpr_at_fpr(target_fpr: float) -> float:
        idx = np.searchsorted(fpr, target_fpr, side="right") - 1
        return float(tpr[max(idx, 0)])

    ops = {
        "AUC": float(auc),
        "TPR@FPR=0.01": tpr_at_fpr(0.01),
        "TPR@FPR=0.05": tpr_at_fpr(0.05),
        "TPR@FPR=0.10": tpr_at_fpr(0.10),
        "mean_z_pos": float(pos.mean()),
        "std_z_pos": float(pos.std()),
        "mean_z_neg": float(neg.mean()),
        "std_z_neg": float(neg.std()),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
    }
    print()
    for k, v in ops.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{args.tag}_auc_summary.json").open("w") as f:
        json.dump(ops, f, indent=2)

    # ROC plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, linewidth=2, color="#d62728", label=f"AUC={auc:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC — {args.tag}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # z-score distribution
    all_z = np.concatenate([pos, neg])
    bins = np.linspace(all_z.min(), all_z.max(), 40)
    axes[1].hist(neg, bins=bins, alpha=0.6, label=f"neg (plain) n={len(neg)}", color="#1f77b4")
    axes[1].hist(pos, bins=bins, alpha=0.6, label=f"pos (s=3) n={len(pos)}", color="#d62728")
    axes[1].axvline(neg.mean(), color="#1f77b4", linestyle="--", linewidth=1)
    axes[1].axvline(pos.mean(), color="#d62728", linestyle="--", linewidth=1)
    axes[1].set_xlabel("z-score")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"z-score distribution (pos vs neg)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.tag}_roc.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote {out_dir / f'{args.tag}_auc_summary.json'}")
    print(f"wrote {out_dir / f'{args.tag}_roc.png'}")


if __name__ == "__main__":
    main()
