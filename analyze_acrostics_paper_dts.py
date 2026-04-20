"""Analyze paper DTS Acrostics pilot: Lev z-stat + ROC-AUC for pos vs neg.

Expects:
  - positives.jsonl: LLM-generated text with paper DTS system prompt
  - negatives.jsonl: human gold_completion text paired to same target X

Both must have columns: idx, target, response. Rows are joined by idx.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from acrostics_zstat import compute_lev_zstat


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def main(args):
    pos = load_jsonl(args.positives_file)
    neg = load_jsonl(args.negatives_file)
    print(f"Positives: {len(pos)}, Negatives: {len(neg)}")

    pos_by_idx = {r["idx"]: r for r in pos}
    neg_by_idx = {r["idx"]: r for r in neg}
    common = sorted(set(pos_by_idx) & set(neg_by_idx))
    print(f"Paired (common idx): {len(common)}")

    pos_zs, neg_zs, pos_d, neg_d, pos_n, neg_n = [], [], [], [], [], []
    rows_pos, rows_neg = [], []
    for idx in common:
        p, n = pos_by_idx[idx], neg_by_idx[idx]
        assert p["target"] == n["target"], f"target mismatch at idx {idx}"
        target = p["target"]

        pz = compute_lev_zstat(p["response"], target, n_resample=args.n_resample, seed=idx)
        nz = compute_lev_zstat(n["response"], target, n_resample=args.n_resample, seed=idx)
        pos_zs.append(pz.z); pos_d.append(pz.d_obs); pos_n.append(pz.n_sentences)
        neg_zs.append(nz.z); neg_d.append(nz.d_obs); neg_n.append(nz.n_sentences)
        rows_pos.append({"idx": idx, "target": target, "fl": pz.fl,
                          "d_obs": pz.d_obs, "mu": pz.mu, "sigma": pz.sigma,
                          "z": pz.z, "n_sentences": pz.n_sentences})
        rows_neg.append({"idx": idx, "target": target, "fl": nz.fl,
                          "d_obs": nz.d_obs, "mu": nz.mu, "sigma": nz.sigma,
                          "z": nz.z, "n_sentences": nz.n_sentences})

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "zstats_positives.jsonl", "w") as f:
        for r in rows_pos:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(outdir / "zstats_negatives.jsonl", "w") as f:
        for r in rows_neg:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    y_true = [1] * len(pos_zs) + [0] * len(neg_zs)
    y_score = pos_zs + neg_zs
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)

    def tpr_at(target_fpr: float) -> float:
        # Largest TPR under fpr <= target
        mask = fpr <= target_fpr
        return float(tpr[mask].max()) if mask.any() else 0.0

    tpr_1 = tpr_at(0.01)
    tpr_10 = tpr_at(0.10)

    lines = [
        "# Paper DTS Acrostics pilot — detection summary",
        "",
        f"**Paired samples**: {len(common)} pos / {len(common)} neg",
        f"**N resample for null distribution**: {args.n_resample}",
        f"**Positives**: {args.positives_file}",
        f"**Negatives**: {args.negatives_file}",
        "",
        "## ROC-AUC on Lev z-statistic",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| ROC-AUC | **{auc:.4f}** |",
        f"| TPR @ 1% FPR | **{tpr_1:.4f}** |",
        f"| TPR @ 10% FPR | **{tpr_10:.4f}** |",
        "",
        "## Positive (watermarked) stats",
        "",
        f"- n_sentences mean: {np.mean(pos_n):.2f}",
        f"- Lev.mean (d_obs to target X): {np.mean(pos_d):.2f}",
        f"- z-stat mean ± std: {np.mean(pos_zs):.3f} ± {np.std(pos_zs):.3f}",
        "",
        "## Negative (human gold) stats",
        "",
        f"- n_sentences mean: {np.mean(neg_n):.2f}",
        f"- Lev.mean (d_obs to target X): {np.mean(neg_d):.2f}",
        f"- z-stat mean ± std: {np.mean(neg_zs):.3f} ± {np.std(neg_zs):.3f}",
        "",
        "## Comparison to paper Table 2",
        "",
        "| Config | ROC-AUC | T@1%FPR | T@10%FPR |",
        "|---|:---:|:---:|:---:|",
        "| Paper 4o-mini DTS | 0.590 | 0.036 | 0.168 |",
        "| Paper o3-mini DTS | 1.000 | 1.000 | 1.000 |",
        f"| **Ours Qwen3-14B DTS** | **{auc:.3f}** | **{tpr_1:.3f}** | **{tpr_10:.3f}** |",
    ]
    summary_text = "\n".join(lines)
    (outdir / "detection_summary.md").write_text(summary_text + "\n")
    print(summary_text)

    # ROC plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.5, 5.5))
        plt.plot(fpr, tpr, lw=2, label=f"Qwen3-14B DTS (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Paper DTS Acrostics ICW")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "roc.png", dpi=120)
        print(f"Saved ROC plot: {outdir / 'roc.png'}")
    except Exception as e:
        print(f"(ROC plot skipped: {e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--positives_file", required=True)
    p.add_argument("--negatives_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_resample", type=int, default=1000)
    args = p.parse_args()
    main(args)
