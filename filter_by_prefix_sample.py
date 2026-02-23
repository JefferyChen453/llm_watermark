#!/usr/bin/env python3
"""
针对 filtered_data jsonl，对相同 prefix 的若干条数据随机 sample 一条；
若 prefix 只对应一条，则取该条。
输出按 fraction: 非0.0为pos，0.0为neg，统计 pos/neg 数量并命名输出文件。
"""
import json
import random
from collections import defaultdict


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def filter_by_prefix_sample(input_path: str, output_dir: str, seed: int = 42):
    random.seed(seed)

    data = read_jsonl(input_path)

    # 按 prefix 分组
    prefix_to_records = defaultdict(list)
    for d in data:
        prefix = d.get("prefix", "")
        prefix_to_records[prefix].append(d)

    # 每个 prefix 随机 sample 一条
    sampled = []
    for prefix, records in prefix_to_records.items():
        chosen = random.choice(records)
        sampled.append(chosen)

    # 分类: fraction != 0.0 -> pos, fraction == 0.0 -> neg
    positive = [d for d in sampled if d.get("fraction", None) != 0.0]
    negative = [d for d in sampled if d.get("fraction", None) == 0.0]

    n_pos, n_neg = len(positive), len(negative)

    # 输出文件命名
    base_name = "Qwen-Qwen3-32B_LFQA_filtered_data"
    output_name = f"{base_name}_pos_{n_pos}_neg_{n_neg}.jsonl"
    output_path = f"{output_dir.rstrip('/')}/{output_name}"

    # 合并 pos + neg 并保存（保持顺序：先 pos 后 neg）
    result = positive + negative
    save_jsonl(result, output_path)

    print(f"Input: {len(data)} records, {len(prefix_to_records)} unique prefixes")
    print(f"Sampled: {len(sampled)} records (pos={n_pos}, neg={n_neg})")
    print(f"Output: {output_path}")
    return output_path


if __name__ == "__main__":
    input_path = "/home/tianyichen/llm_watermark/outputs/sft_train/Qwen-Qwen3-32B_OpenGen/Qwen-Qwen3-32B_OpenGen_filtered_data_pos_24597_neg_6059.jsonl"
    output_dir = "/home/tianyichen/llm_watermark/outputs/sft_train/Qwen-Qwen3-32B_OpenGen"
    filter_by_prefix_sample(input_path, output_dir)
