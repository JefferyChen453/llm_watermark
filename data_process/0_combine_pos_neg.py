import json
from pathlib import Path
import sys
import string
from transformers import AutoTokenizer

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from dataset import load_jsonl, save_jsonl
from filters import ngram_repeat_ratio, filter_punctuation_ratio

positive_data_files = [
    "/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa_no_system_prompt_v2/strength_3.0/Qwen-Qwen3-14B_strength_3.0_frac_0.1_len_600_num_11578_vllm_only_English.jsonl",
    "/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa_no_system_prompt_v2/strength_3.0/Qwen-Qwen3-14B_strength_3.0_frac_0.15_len_600_num_11578_vllm_only_English.jsonl",
    "/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa_no_system_prompt_v2/strength_3.0/Qwen-Qwen3-14B_strength_3.0_frac_0.2_len_600_num_11578_vllm_only_English.jsonl",
    "/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa_no_system_prompt_v2/strength_3.0/Qwen-Qwen3-14B_strength_3.0_frac_0.25_len_600_num_11578_vllm_only_English.jsonl",
    "/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa_no_system_prompt_v2/strength_3.0/Qwen-Qwen3-14B_strength_3.0_frac_0.3_len_600_num_11578_vllm_only_English.jsonl",
]
positive_tau_list = [
7.0,
7.0,
7.0,
7.0,
7.0,
]
fraction_list = [0.1, 0.15, 0.2, 0.25, 0.3]
MODEL_NAME = "Qwen/Qwen3-14B"


def load_z_scores(data_file: str, n_samples: int) -> list:
    """Load pre-computed z-scores from the _z.jsonl file for data_file.

    Asserts that positive_num == n_samples to guarantee index correspondence.
    Returns the first positive_num z-scores (one per sample in the original file).
    """
    z_path = data_file.replace('.jsonl', '_z.jsonl')
    with open(z_path, 'r') as f:
        z_data = json.load(f)
    positive_num = z_data['positive_num']
    assert positive_num == n_samples, (
        f"Index mismatch: {z_path} has positive_num={positive_num} "
        f"but data file has {n_samples} samples."
    )
    return z_data['z_score'][:positive_num]


def filter_positive(data, tau, fraction, tokenizer):
    """Filter positive samples using pre-computed z-scores from the _z file."""
    z_scores = load_z_scores(
        positive_data_files[fraction_list.index(fraction)], len(data)
    )
    filtered = []
    dropped = []
    n_short = 0
    for i, d in enumerate(data):
        if filter_punctuation_ratio(d['gen_completion'], threshold=0.45):
            print(f"\nFilter out because of punctuation ratio:\n\n{d['gen_completion']}")
            print("\n", "="*100)
            dropped.append(d)
            continue
        if ngram_repeat_ratio(d['gen_completion'], n=5, threshold=0.40):
            dropped.append(d)
            continue
        gen_tokens = tokenizer(d['gen_completion'], add_special_tokens=False)["input_ids"]
        if len(gen_tokens) < 200:
            n_short += 1
            continue
        z_score = z_scores[i]
        if z_score > tau:
            d.update({'z_score': z_score, 'fraction': fraction})
            filtered.append(d)
    avg_z = sum(d['z_score'] for d in filtered) / len(filtered) if filtered else 0
    print(f"All data: {len(data)}; {n_short} too short; {len(filtered)} available; average z_score: {avg_z:.4f}")
    return filtered, dropped


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"

    filtered_positive_data = []
    dropped_positive_data = []
    for tau, data_file, fraction in zip(positive_tau_list, positive_data_files, fraction_list):
        data = load_jsonl(data_file)
        filtered, dropped = filter_positive(data, tau, fraction, tokenizer)
        filtered_positive_data.extend(filtered)
        dropped_positive_data.extend(dropped)

    all_filtered_data = filtered_positive_data

    save_path = f"/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa_no_system_prompt_v2/strength_3.0/Qwen-Qwen3-14B_strength_3.0_filtered_data_pos_{len(filtered_positive_data)}.jsonl"
    print(f"Filtered data saved to: {save_path}")
    save_jsonl(all_filtered_data, save_path)
    save_path = f"/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa_no_system_prompt_v2/strength_3.0/Qwen-Qwen3-14B_strength_3.0_dropped_data_pos_{len(dropped_positive_data)}.jsonl"
    print(f"Dropped data saved to: {save_path}")
    save_jsonl(dropped_positive_data, save_path)
