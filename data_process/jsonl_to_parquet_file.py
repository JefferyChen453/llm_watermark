from gptwm_incontext import InContextWatermarkGenerator
from transformers import AutoTokenizer, AutoConfig
from prompt import get_incontext_system_prompt

import argparse
import json
import os
from functools import lru_cache
from tqdm import tqdm


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
    if not rows:
        raise ValueError("No rows to save to parquet.")

    if HAS_PANDAS:
        pd.DataFrame(rows).to_parquet(path, index=False)
    elif HAS_PYARROW:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
    else:
        raise ImportError("Need pandas or pyarrow. Run: pip install pandas pyarrow")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a JSONL file with synthetic data into a Parquet file with "
            'columns: "prompt", "response", "prefix", "seed", "z_score", '
            '"fraction", "dataset_type".'
        )
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "--output_parquet",
        default=None,
        help=(
            "Output Parquet file path. "
            "If not provided, defaults to input path with .parquet extension."
        ),
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-14B",
        help="Model name for tokenizer/config (used to build watermark green token list).",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = args.input_jsonl
    if args.output_parquet is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".parquet"
    else:
        output_path = args.output_parquet

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_config = AutoConfig.from_pretrained(args.model_name)

    @lru_cache(maxsize=None)
    def get_generator(seed: int, fraction: float) -> str:
        """Cached helper to build the green token string for a (seed, fraction) pair."""
        if fraction is None or fraction <= 0.0:
            return ""

        generator = InContextWatermarkGenerator(
            fraction=fraction,
            vocab_size=tokenizer.vocab_size,
            model_emb_length=model_config.vocab_size,
            watermark_key=seed,
            only_English=True,
            tokenizer=tokenizer,
        )
        return generator

    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines)):
            line = line.strip()
            if not line:
                continue  
            obj = json.loads(line)

            prefix = obj.get("prefix")
            response = obj.get("gen_completion")
            seed = obj.get("seed")
            fraction = obj.get("fraction")
            dataset_type = obj.get("dataset_type", obj.get("type"))
            z_score = obj.get("z_score")

            # Require minimal fields to build prompt/response.
            if not prefix or not response or seed is None or dataset_type is None:
                continue

            if fraction is None or fraction == 0.0:
                green_token_string = ""
                import pdb; pdb.set_trace()
            else:
                generator = get_generator(int(seed), float(fraction))
                green_token_string = generator.get_green_token_string()

            system_prompt = get_incontext_system_prompt(dataset_type, green_token_string)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prefix},
            ]

            input_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            rows.append(
                {
                    "prompt": input_prompt,
                    "response": response,
                    "prefix": prefix,
                    "seed": seed,
                    "z_score": z_score,
                    "fraction": fraction,
                    "dataset_type": dataset_type,
                }
            )

    print(f"Loaded {len(rows)} samples from {input_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_parquet(rows, output_path)
    print(f"Saved {output_path} ({len(rows)} samples)")


if __name__ == "__main__":
    main()

