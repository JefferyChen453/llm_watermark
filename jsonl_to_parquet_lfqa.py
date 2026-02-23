#!/usr/bin/env python3
"""
Convert LFQA JSONL to Parquet format for SFT.

Input JSONL format (per line):
  - input_prompt: full formatted prompt (with system/watermark)
  - prefix: "Q: <question>\\nA:" 
  - gold_completion: target answer for SFT
  - gen_completion: model-generated answer (not used for SFT)

Output Parquet columns:
  - prompt: the question/prefix (input to model)
  - response: the gold_completion (target output)
"""
from gptwm_incontext import InContextWatermarkGenerator
from transformers import AutoTokenizer, AutoConfig
import argparse
import json
import os
import random
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
    """Save list of dicts to parquet."""
    if HAS_PANDAS:
        pd.DataFrame(rows).to_parquet(path, index=False)
    elif HAS_PYARROW:
        table = pa.table(
            {"prompt": [r["prompt"] for r in rows], "response": [r["response"] for r in rows]}
        )
        pq.write_table(table, path)
    else:
        raise ImportError("Need pandas or pyarrow. Run: pip install pandas pyarrow")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default="/home/tianyichen/llm_watermark/outputs/sft_train/Qwen-Qwen3-32B_combined_filtered_data_pos_6782_neg_2123.jsonl",
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/tianyichen/llm_watermark/verl/data/sft",
        help="Output directory for train.parquet and val.parquet",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-14B",
        help="Model name",
    )
    parser.add_argument(
        "--wm_key",
        type=int,
        default=0,
        help="Watermark key",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_config = AutoConfig.from_pretrained(args.model_name)
    watermark_generator1 = InContextWatermarkGenerator(
        fraction=0.1,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        watermark_key=args.wm_key,
        only_English=True,
        tokenizer=tokenizer
    )
    watermark_generator2 = InContextWatermarkGenerator(
        fraction=0.2,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        watermark_key=args.wm_key,
        only_English=True,
        tokenizer=tokenizer
    )
    watermark_generator3 = InContextWatermarkGenerator(
        fraction=0.3,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        watermark_key=args.wm_key,
        only_English=True,
        tokenizer=tokenizer
    )
    watermark_generator4 = InContextWatermarkGenerator(
        fraction=0.4,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        watermark_key=args.wm_key,
        only_English=True,
        tokenizer=tokenizer
    )
    green_token_string_dict = {0.1: watermark_generator1.get_green_token_string(), 0.2: watermark_generator2.get_green_token_string(), 0.3: watermark_generator3.get_green_token_string(), 0.4: watermark_generator4.get_green_token_string()}

    rows = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prefix = obj.get("prefix", "")
            fraction = obj.get("fraction")
            dataset_type = obj.get("type")

            if fraction == 0.0:
                green_token_string = ""
            else:
                green_token_string = green_token_string_dict[fraction]

            system_prompt = get_incontext_system_prompt(dataset_type, green_token_string)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prefix}
            ]
            input_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            response = obj.get("gold_completion", "")
            if input_prompt and response:
                rows.append({"prompt": input_prompt, "response": response})

    print(f"Loaded {len(rows)} samples from {args.input_jsonl}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, args.input_jsonl.split("/")[-1].replace(".jsonl", ".parquet"))
    save_parquet(rows, train_path)
    print(f"Saved {train_path} ({len(rows)} samples)")

if __name__ == "__main__":
    main()
