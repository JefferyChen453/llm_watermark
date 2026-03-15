#!/usr/bin/env python3
"""
Analyze LFRQA and vblagoje_lfqa datasets: golden answer length (chars + tokens with Qwen3-14B).
Output: data/dataset_analysis_report.md
"""
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def load_tokenizer(model_name: str = "Qwen/Qwen3-14B"):
    return AutoTokenizer.from_pretrained(model_name)


def get_golden_answers_lfrqa(data_dir: Path):
    """LFRQA: from_colbert/*.jsonl, each line has 'answers' (list of strings). Use first as golden."""
    from_colbert = data_dir / "from_colbert"
    if not from_colbert.exists():
        return [], []
    golden = []
    sources = []
    for p in sorted(from_colbert.glob("*.jsonl")):
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ans_list = obj.get("answers") or []
                if ans_list:
                    golden.append(ans_list[0] if isinstance(ans_list[0], str) else str(ans_list[0]))
                    sources.append(p.name)
                else:
                    golden.append("")
                    sources.append(p.name)
    return golden, sources


def get_golden_answers_vblagoje(data_dir: Path, split_files=("train.json", "validation.json", "test.json")):
    """vblagoje_lfqa: JSONL. Each line has answers.text (list) and answers.score (list). Golden = highest score."""
    all_golden = []
    all_sources = []
    for fn in split_files:
        path = data_dir / fn
        if not path.exists():
            continue
        with open(path, "r") as f:
            for line in tqdm(f, desc=fn, leave=False):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                answers = obj.get("answers") or {}
                texts = answers.get("text") or []
                scores = answers.get("score") or []
                if not texts:
                    all_golden.append("")
                    all_sources.append(fn)
                    continue
                if scores and len(scores) == len(texts):
                    idx = int(np.argmax(scores))
                    all_golden.append(texts[idx])
                else:
                    all_golden.append(texts[0])
                all_sources.append(fn)
    return all_golden, all_sources


def token_lengths(tokenizer, texts, batch_size=256):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, add_special_tokens=False, truncation=False, padding=False)
        out.extend(len(ids) for ids in enc["input_ids"])
    return np.array(out)


def stats_dict(arr, name=""):
    if len(arr) == 0:
        return {"count": 0}
    arr = np.array(arr)
    return {
        "count": len(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def histogram_bins(arr, bins):
    arr = np.asarray(arr)
    counts = []
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if hi == float("inf"):
            n = (arr >= lo).sum()
            labels.append(f"{lo}+")
        else:
            n = ((arr >= lo) & (arr < hi)).sum()
            labels.append(f"{lo}-{hi}")
        counts.append(int(n))
    return labels, counts


def plot_token_distribution(labels, counts, total, title: str, save_path: Path):
    """Bar chart of token length distribution (count and %)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    counts = np.asarray(counts)
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
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def run_analysis():
    base = Path(__file__).resolve().parent.parent
    data_root = base / "data"
    lfrqa_dir = data_root / "LFRQA"
    vblagoje_dir = data_root / "vblagoje_lfqa"
    model_name = "Qwen/Qwen3-14B"

    print("Loading tokenizer:", model_name)
    tokenizer = load_tokenizer(model_name)

    sections = []

    # ---- LFRQA ----
    print("Loading LFRQA...")
    lfrqa_golden, lfrqa_sources = get_golden_answers_lfrqa(lfrqa_dir)
    if not lfrqa_golden:
        sections.append("## LFRQA\n\nNo data found under `data/LFRQA/from_colbert/*.jsonl`.\n")
    else:
        char_lens = np.array([len(t) for t in lfrqa_golden])
        tok_lens = token_lengths(tokenizer, lfrqa_golden)
        char_stats = stats_dict(char_lens)
        tok_stats = stats_dict(tok_lens)
        # Per-source: collect texts per source then tokenize
        source_to_tok = {}
        for src in set(lfrqa_sources):
            idxs = [i for i, s in enumerate(lfrqa_sources) if s == src]
            texts = [lfrqa_golden[i] for i in idxs]
            source_to_tok[src] = token_lengths(tokenizer, texts)

        bins = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, float("inf")]
        tok_labels, tok_counts = histogram_bins(tok_lens, bins)
        hist_lines = []
        total = len(tok_lens)
        for lab, c in zip(tok_labels, tok_counts):
            pct = 100 * c / total if total else 0
            hist_lines.append(f"| {lab} | {c} | {pct:.1f}% |")

        lfrqa_plot_path = data_root / "lfrqa_token_dist.png"
        if plot_token_distribution(tok_labels, tok_counts, total, "LFRQA — Golden answer token length", lfrqa_plot_path):
            lfrqa_plot_rel = "lfrqa_token_dist.png"
        else:
            lfrqa_plot_rel = None

        md = []
        md.append("## LFRQA")
        md.append("")
        md.append("- **Path**: `data/LFRQA/from_colbert/*.jsonl`")
        md.append("- **Golden answer**: first string in each record's `answers` list")
        md.append(f"- **Total examples**: {len(lfrqa_golden)}")
        md.append("")
        md.append("### Golden answer length (characters)")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        for k, v in char_stats.items():
            if k == "count":
                continue
            md.append(f"| {k} | {v:.2f} |")
        md.append("")
        md.append("### Golden answer length (tokens, Qwen/Qwen3-14B)")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        for k, v in tok_stats.items():
            if k == "count":
                md.append(f"| count | {int(v)} |")
            else:
                md.append(f"| {k} | {v:.2f} |")
        md.append("")
        md.append("### Token length distribution")
        md.append("")
        if lfrqa_plot_rel:
            md.append(f"![LFRQA token length distribution]({lfrqa_plot_rel})")
            md.append("")
        md.append("| Bucket (tokens) | Count | % |")
        md.append("|-----------------|-------|---|")
        md.extend(hist_lines)
        md.append("")
        md.append("### Per-file (from_colbert)")
        md.append("")
        md.append("| File | Count | Mean tokens | Median | Min | Max |")
        md.append("|------|-------|-------------|--------|-----|-----|")
        for src in sorted(source_to_tok.keys()):
            arr = source_to_tok[src]
            md.append(f"| {src} | {len(arr)} | {np.mean(arr):.1f} | {np.median(arr):.1f} | {int(np.min(arr))} | {int(np.max(arr))} |")
        sections.append("\n".join(md))

    # ---- vblagoje_lfqa ----
    print("Loading vblagoje_lfqa...")
    vblagoje_golden, vblagoje_sources = get_golden_answers_vblagoje(vblagoje_dir)
    if not vblagoje_golden:
        sections.append("\n\n## vblagoje_lfqa\n\nNo data found under `data/vblagoje_lfqa/`.\n")
    else:
        char_lens = np.array([len(t) for t in vblagoje_golden])
        tok_lens = token_lengths(tokenizer, vblagoje_golden)
        char_stats = stats_dict(char_lens)
        tok_stats = stats_dict(tok_lens)
        split_to_tok = {}
        for s, i in zip(vblagoje_sources, range(len(vblagoje_golden))):
            split_to_tok.setdefault(s, []).append(tok_lens[i])
        for s in split_to_tok:
            split_to_tok[s] = np.array(split_to_tok[s])

        bins = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, float("inf")]
        tok_labels, tok_counts = histogram_bins(tok_lens, bins)
        hist_lines = []
        total = len(tok_lens)
        for lab, c in zip(tok_labels, tok_counts):
            pct = 100 * c / total if total else 0
            hist_lines.append(f"| {lab} | {c} | {pct:.1f}% |")

        vblagoje_plot_path = data_root / "vblagoje_token_dist.png"
        if plot_token_distribution(tok_labels, tok_counts, total, "vblagoje_lfqa — Golden answer token length", vblagoje_plot_path):
            vblagoje_plot_rel = "vblagoje_token_dist.png"
        else:
            vblagoje_plot_rel = None

        md = []
        md.append("## vblagoje_lfqa")
        md.append("")
        md.append("- **Path**: `data/vblagoje_lfqa/` (train.json, validation.json, test.json, JSONL)")
        md.append("- **Golden answer**: answer with highest `score` in `answers.text` (else first)")
        md.append(f"- **Total examples**: {len(vblagoje_golden)}")
        md.append("")
        md.append("### Golden answer length (characters)")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        for k, v in char_stats.items():
            if k == "count":
                continue
            md.append(f"| {k} | {v:.2f} |")
        md.append("")
        md.append("### Golden answer length (tokens, Qwen/Qwen3-14B)")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        for k, v in tok_stats.items():
            if k == "count":
                md.append(f"| count | {int(v)} |")
            else:
                md.append(f"| {k} | {v:.2f} |")
        md.append("")
        md.append("### Token length distribution")
        md.append("")
        if vblagoje_plot_rel:
            md.append(f"![vblagoje_lfqa token length distribution]({vblagoje_plot_rel})")
            md.append("")
        md.append("| Bucket (tokens) | Count | % |")
        md.append("|-----------------|-------|---|")
        md.extend(hist_lines)
        md.append("")
        md.append("### Per-split")
        md.append("")
        md.append("| Split | Count | Mean tokens | Median | Min | Max |")
        md.append("|-------|-------|-------------|--------|-----|-----|")
        for src in ("train.json", "validation.json", "test.json"):
            if src in split_to_tok:
                arr = split_to_tok[src]
                md.append(f"| {src} | {len(arr)} | {np.mean(arr):.1f} | {np.median(arr):.1f} | {int(np.min(arr))} | {int(np.max(arr))} |")
        sections.append("\n".join(md))

    report_path = data_root / "dataset_analysis_report.md"
    header = """# Dataset analysis: LFRQA & vblagoje_lfqa

Tokenization: **Qwen/Qwen3-14B** (no special tokens for answer-only length).
Focus: **golden answer length** (characters and tokens).

"""
    with open(report_path, "w") as f:
        f.write(header)
        f.write("\n\n---\n\n".join(sections))
    print("Wrote:", report_path)
    return report_path


if __name__ == "__main__":
    run_analysis()
