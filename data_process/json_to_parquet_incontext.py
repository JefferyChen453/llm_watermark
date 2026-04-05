"""
Convert a LFQA JSON/JSONL file (with prefix, gold_completion) to Parquet with
in-context watermark system prompt. Uses fixed seed=0 and single fraction.
Output: same path with .parquet extension.
"""
import argparse
import json
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from transformers import AutoConfig, AutoTokenizer
from gptwm_incontext import InContextWatermarkGenerator
from prompt import get_incontext_system_prompt

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def save_parquet(rows, path):
    if not rows:
        raise ValueError("No rows to save to parquet.")
    if HAS_PANDAS:
        pd.DataFrame(rows).to_parquet(path, index=False)
    elif HAS_PYARROW:
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        raise ImportError("Need pandas or pyarrow. Run: pip install pandas pyarrow")


def main():
    parser = argparse.ArgumentParser(description="LFQA JSON(L) -> Parquet with in-context system prompt (seed=0).")
    parser.add_argument("--input", default="/home/tianyichen/llm_watermark/data/processed_data/vblagoje_lfqa/validation_177.json", help="Input JSON or JSONL path.")
    parser.add_argument("--output", default="/home/tianyichen/llm_watermark/verl/data/sft_modified_loss/vblagoje_lfqa/validation_pos177_neg177_seed0_frac0.25_new_sys_prompt.parquet", help="Output Parquet path (default: input with .parquet).")
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B", help="Model for tokenizer/config.")
    parser.add_argument("--seed", type=int, default=0, help="Watermark key (default 0).")
    parser.add_argument("--fraction", type=float, default=0.25, help="Green-list fraction (default 0.5).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".parquet")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    generator = InContextWatermarkGenerator(
        fraction=args.fraction,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=config.vocab_size,
        watermark_key=args.seed,
        only_English=True,
        tokenizer=tokenizer,
    )
    green_token_string = generator.get_green_token_string()
    system_prompt = get_incontext_system_prompt("lfqa", green_token_string)

    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prefix = obj.get("prefix")
            response = obj.get("gold_completion")
            if not prefix or not response:
                continue
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prefix},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            rows.append({
                "prompt": prompt,
                "response": response,
                "prefix": prefix,
                "seed": args.seed,
                "dataset_type": "lfqa",
                "positive_or_negative": "positive",
            })



    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prefix = obj.get("prefix")
            response = obj.get("gold_completion")
            if not prefix or not response:
                continue
            prompt = tokenizer.apply_chat_template(
                [
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prefix},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            rows.append({
                "prompt": prompt,
                "response": response,
                "prefix": prefix,
                "seed": args.seed,
                "dataset_type": "lfqa",
                "positive_or_negative": "negative",
            })





    os.makedirs(output_path.parent or ".", exist_ok=True)
    save_parquet(rows, output_path)
    print(f"Saved {len(rows)} samples -> {output_path}")


if __name__ == "__main__":
    main()
