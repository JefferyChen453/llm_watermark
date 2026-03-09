#!/usr/bin/env python3
"""
Compute per-sample perplexity for completions using a causal LM.

For each record in a JSONL file with keys `prefix` and `gen_completion`,
we compute perplexity on the completion tokens conditioned on the prefix.
The computed value is stored as `ppl` back into the JSONL file.
We also compute the average perplexity across the file and write it into the
associated `_z.jsonl` file (same name, suffix `_z.jsonl`) by updating its
top-level dictionary with `avg_ppl`.
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def compute_batch_ppl(
    tokenizer,
    model,
    batch: List[Dict],
    max_length: int,
) -> Tuple[List[float], float, int]:
    """Compute perplexity for a batch of samples.

    Returns:
        per_sample_ppl: list of perplexities for each sample
        batch_nll: total negative log-likelihood over completion tokens
        batch_tokens: total number of completion tokens considered
    """
    prefixes = [item["input_prompt"] for item in batch]
    completions = [item["gen_completion"] for item in batch]

    # Tokenize prefix and full text to derive prefix lengths
    prefix_tok = tokenizer(
        prefixes, add_special_tokens=False, padding=False, truncation=True, max_length=max_length
    )
    full_texts = [p + c for p, c in zip(prefixes, completions)]
    enc = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device)

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq, vocab)

    # Shift for causal LM loss
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attn = attention_mask[:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)

    per_sample_ppl: List[float] = []
    batch_nll = 0.0
    batch_tokens = 0

    for i, (ids, attn) in enumerate(zip(shift_labels, shift_attn)):
        prefix_len = len(prefix_tok["input_ids"][i])
        # Mask out prefix tokens and padding
        comp_mask = torch.zeros_like(attn, dtype=torch.bool)
        comp_mask[prefix_len:] = True
        token_mask = comp_mask & (attn.bool())

        if token_mask.sum().item() == 0:
            per_sample_ppl.append(float("inf"))
            continue

        token_log_probs = log_probs[i].gather(1, ids.unsqueeze(-1)).squeeze(-1)
        selected_log_probs = token_log_probs[token_mask]

        nll = -selected_log_probs.sum().item()
        token_count = token_mask.sum().item()

        batch_nll += nll
        batch_tokens += token_count

        ppl = math.exp(nll / token_count)
        per_sample_ppl.append(ppl)

    return per_sample_ppl, batch_nll, batch_tokens


def process_file(
    input_path: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
):
    # Load data
    records: List[Dict] = []
    with input_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print(f"No records found in {input_path}")
        return

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    # Compute PPL in batches
    for start in tqdm(range(0, len(records), batch_size), total=len(records) // batch_size, desc="Computing PPL"):
        batch = records[start : start + batch_size]
        per_sample_ppl, batch_nll, batch_tokens = compute_batch_ppl(
            tokenizer, model, batch, max_length
        )
        # write back per-sample ppl
        for rec, ppl in zip(batch, per_sample_ppl):
            rec["ppl"] = ppl
        total_nll += batch_nll
        total_tokens += batch_tokens

    avg_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")

    # Write back JSONL with ppl
    with input_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Update _z.jsonl
    z_path = input_path.with_name(input_path.name.replace(".jsonl", "_z.jsonl"))
    z_data: Dict = {}
    if z_path.exists():
        with z_path.open("r") as f:
            try:
                z_data = json.load(f)
            except json.JSONDecodeError:
                z_data = {}
    z_data["avg_ppl"] = avg_ppl
    with z_path.open("w") as f:
        json.dump(z_data, f, ensure_ascii=False)

    print(f"Processed {input_path.name}: avg_ppl={avg_ppl:.4f}, records={len(records)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-Next-80B-A3B-Instruct",
        help="Model name or path",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--max-length", type=int, default=8192, help="Max sequence length for tokenization"
    )
    args = parser.parse_args()

    process_file(
        Path(args.input),
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
