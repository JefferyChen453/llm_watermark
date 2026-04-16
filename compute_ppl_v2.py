#!/usr/bin/env python3
"""
PPL v2: conditional PPL on gen_completion given a clean (ICW-free) chat prefix.

Differences vs compute_ppl.py:
  - Prefix is always the *clean* chat-template prompt (no in-context watermark
    green-token list). Supplied by the caller via ``--clean-prefix-file`` so the
    same prefix set is used for every cell (fair cross-fraction comparison).
  - Prefix and completion are tokenized **separately** with
    ``add_special_tokens=False`` and concatenated; no string-join-then-tokenize
    so there is no tokenization-boundary ambiguity.
  - ``tokenizer.padding_side = "right"`` is enforced.
  - First completion token's NLL is included (no off-by-one).
  - Samples are matched to clean prefixes by shared ``prefix`` field (raw query).

Output layout:
  outputs/quality_eval/ppl/{cell_tag}.jsonl   -- per-sample records with ppl field
  outputs/quality_eval/ppl/{cell_tag}_summary.json -- corpus avg_ppl + stats
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def compute_batch_ppl(
    tokenizer,
    model,
    prefix_ids_list: List[List[int]],
    completion_ids_list: List[List[int]],
    max_length: int,
):
    """Compute NLL per sample for concatenated (prefix, completion) sequences.

    Returns (per_sample_ppl, batch_nll, batch_tokens).
    """
    assert len(prefix_ids_list) == len(completion_ids_list)
    B = len(prefix_ids_list)

    seq_list = []
    comp_starts = []
    comp_lens = []
    for pids, cids in zip(prefix_ids_list, completion_ids_list):
        full = pids + cids
        if len(full) > max_length:
            full = full[:max_length]
            cids_eff = max(0, max_length - len(pids))
        else:
            cids_eff = len(cids)
        seq_list.append(full)
        comp_starts.append(len(pids))
        comp_lens.append(cids_eff)

    max_len = max(len(s) for s in seq_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, s in enumerate(seq_list):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attention_mask[i, : len(s)] = 1

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, V)

    # Shift: logits[:, t, :] predicts input_ids[:, t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    per_sample_ppl: List[float] = []
    batch_nll = 0.0
    batch_tokens = 0

    for i in range(B):
        cstart = comp_starts[i]
        clen = comp_lens[i]
        # Completion tokens in input_ids occupy [cstart, cstart + clen).
        # Their predictions (shift_logits[t] predicts input_ids[t+1]) live at
        # shift positions [cstart - 1, cstart + clen - 1).
        if clen == 0 or cstart == 0:
            per_sample_ppl.append(float("inf"))
            continue
        lo = cstart - 1
        hi = cstart + clen - 1
        hi = min(hi, token_log_probs.shape[1])
        selected = token_log_probs[i, lo:hi]
        nll = -selected.sum().item()
        n_tok = selected.numel()
        if n_tok == 0:
            per_sample_ppl.append(float("inf"))
            continue
        batch_nll += nll
        batch_tokens += n_tok
        per_sample_ppl.append(math.exp(nll / n_tok))

    return per_sample_ppl, batch_nll, batch_tokens


def process_cell(
    input_path: Path,
    clean_prefix_map: Dict[str, str],
    output_path: Path,
    summary_path: Path,
    model,
    tokenizer,
    n_samples: int,
    batch_size: int,
    max_length: int,
):
    """Run PPL on first ``n_samples`` records whose prefix exists in clean_prefix_map."""
    records_all: List[Dict] = []
    with input_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records_all.append(json.loads(line))

    # Filter to first n_samples that have a clean prefix available (all of them, in our case)
    selected = []
    for rec in records_all:
        if rec["prefix"] in clean_prefix_map:
            selected.append(rec)
            if len(selected) >= n_samples:
                break
    if len(selected) < n_samples:
        print(f"  WARN: only {len(selected)} samples available, wanted {n_samples}")

    # Pre-tokenize prefix & completion for every sample, using clean prefix
    prefix_ids_cache: Dict[str, List[int]] = {}
    for rec in selected:
        raw_q = rec["prefix"]
        if raw_q not in prefix_ids_cache:
            clean_prefix_str = clean_prefix_map[raw_q]
            prefix_ids_cache[raw_q] = tokenizer.encode(clean_prefix_str, add_special_tokens=False)

    out_records: List[Dict] = []
    total_nll = 0.0
    total_tokens = 0
    per_sample_ppl_list: List[float] = []

    for start in tqdm(
        range(0, len(selected), batch_size),
        desc=f"PPL {input_path.name[:50]}",
    ):
        batch = selected[start : start + batch_size]
        prefix_ids_list = [prefix_ids_cache[r["prefix"]] for r in batch]
        completion_ids_list = [
            tokenizer.encode(r["gen_completion"], add_special_tokens=False) for r in batch
        ]

        per_sample_ppl, batch_nll, batch_tokens = compute_batch_ppl(
            tokenizer, model, prefix_ids_list, completion_ids_list, max_length
        )

        for rec, ppl, cids in zip(batch, per_sample_ppl, completion_ids_list):
            out_rec = {
                "prefix": rec["prefix"],
                "gen_completion": rec["gen_completion"],
                "ppl": ppl,
                "completion_tokens": len(cids),
            }
            out_records.append(out_rec)
            per_sample_ppl_list.append(ppl)

        total_nll += batch_nll
        total_tokens += batch_tokens

    avg_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")

    with output_path.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    finite = [p for p in per_sample_ppl_list if math.isfinite(p)]
    summary = {
        "input_file": str(input_path),
        "n_samples": len(out_records),
        "total_completion_tokens": total_tokens,
        "avg_ppl_token_weighted": avg_ppl,
        "avg_ppl_sample_mean": sum(finite) / len(finite) if finite else float("inf"),
        "ppl_median": sorted(finite)[len(finite) // 2] if finite else float("inf"),
        "ppl_min": min(finite) if finite else float("inf"),
        "ppl_max": max(finite) if finite else float("inf"),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  avg_ppl (token-weighted) = {avg_ppl:.4f}, n={len(out_records)}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells-file", required=True,
        help="JSON file mapping cell_tag -> input_file path (one entry per cell)",
    )
    parser.add_argument(
        "--clean-prefix-file", required=True,
        help="JSONL that supplies clean input_prompt per prefix (typically a frac=0.0 file)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for jsonl + summary")
    parser.add_argument("--model-name", default="Qwen/Qwen3-32B")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    with open(args.cells_file) as f:
        cells = json.load(f)
    print(f"Cells to process ({len(cells)}):")
    for tag, path in cells.items():
        print(f"  {tag} <- {path}")

    clean_prefix_map: Dict[str, str] = {}
    with open(args.clean_prefix_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            clean_prefix_map[rec["prefix"]] = rec["input_prompt"]
    print(f"Loaded {len(clean_prefix_map)} clean prefixes from {args.clean_prefix_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.model_name} (bf16, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for tag, input_file in cells.items():
        print(f"\n=== Cell: {tag} ===")
        summary = process_cell(
            input_path=Path(input_file),
            clean_prefix_map=clean_prefix_map,
            output_path=out_dir / f"{tag}.jsonl",
            summary_path=out_dir / f"{tag}_summary.json",
            model=model,
            tokenizer=tokenizer,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        all_summaries[tag] = summary

    with (out_dir / "all_cells_summary.json").open("w") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\nAll cells summary written to {out_dir / 'all_cells_summary.json'}")


if __name__ == "__main__":
    main()
