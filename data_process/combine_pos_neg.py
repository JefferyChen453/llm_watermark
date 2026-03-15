from functools import partial
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import string
from transformers import AutoTokenizer

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from gptwm import GPTWatermarkDetector
from dataset import load_jsonl, save_jsonl
from filters import ngram_repeat_ratio, filter_punctuation_ratio

# negative_data_files = [
#     "/home/tianyichen/llm_watermark/outputs/syn_data/OpenGen/Qwen-Qwen3-14B_strength_2.0_frac_0.0_len_500_num_7211_only_English.jsonl",
# ]
positive_data_files = [
    "/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa/strength_4.0/Qwen-Qwen3-14B_strength_4.0_frac_0.1_len_600_num_11578_vllm_only_English.jsonl",
    "/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa/strength_4.0/Qwen-Qwen3-14B_strength_4.0_frac_0.15_len_600_num_11578_vllm_only_English.jsonl",
    "/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa/strength_4.0/Qwen-Qwen3-14B_strength_4.0_frac_0.2_len_600_num_11578_vllm_only_English.jsonl",
    "/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa/strength_4.0/Qwen-Qwen3-14B_strength_4.0_frac_0.25_len_600_num_11578_vllm_only_English.jsonl",
    "/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa/strength_4.0/Qwen-Qwen3-14B_strength_4.0_frac_0.3_len_600_num_11578_vllm_only_English.jsonl",
]
# negative_tau_list = [5.0]
positive_tau_list = [10.0,
9.5,
9.5,
8.5,
8.5,
]
fraction_list = [0.1, 0.15, 0.2, 0.25, 0.3]
DEFAULT_WM_KEY = 0
MODEL_NAME = "Qwen/Qwen3-14B"
NUM_WORKERS = 64

# ── worker globals (initialised once per subprocess) ───────────────────
_w_tokenizer = None

def _init_worker(model_name: str):
    global _w_tokenizer
    _w_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _w_tokenizer.padding_side = "left"


def _process_positive_chunk(chunk, fraction, tau):
    """Process a chunk of positive samples in a worker. Returns (filtered, n_short)."""
    tokenizer = _w_tokenizer
    detector_cache = {}
    filtered = []
    dropped = []
    n_short = 0
    for d in chunk:
        if filter_punctuation_ratio(d['gen_completion'], threshold=0.30):
            # print(f"\nFilter out because of punctuation ratio:\n\n{d['gen_completion']}")
            # print("\n", "="*100)
            dropped.append(d)
            continue
        if ngram_repeat_ratio(d['gen_completion'], n=5, threshold=0.28):
            # print(f"\nFilter out because of ngram repeat ratio:\n\n{d['gen_completion']}")
            # print("\n", "="*100)
            dropped.append(d)
            continue
        gen_tokens = tokenizer(d['gen_completion'], add_special_tokens=False)["input_ids"]
        if len(gen_tokens) < 200:
            n_short += 1
            continue
        seed = d.get("seed", DEFAULT_WM_KEY)
        key = (fraction, seed)
        if key not in detector_cache:
            detector_cache[key] = GPTWatermarkDetector(
                fraction=fraction,
                strength=2.0,
                vocab_size=tokenizer.vocab_size,
                model_emb_length=tokenizer.vocab_size,
                watermark_key=seed,
                only_English=True,
                tokenizer=tokenizer,
            )
        try:
            z_score = detector_cache[key].detect(gen_tokens)
        except Exception as e:
            continue
        if z_score > tau:
            d.update({'z_score': z_score, 'fraction': fraction})
            filtered.append(d)
    return filtered, n_short, dropped


# def _process_negative_chunk(chunk):
#     """Process a chunk of negative samples in a worker. Returns (filtered, n_short)."""
#     tokenizer = _w_tokenizer
#     filtered = []
#     n_short = 0
#     for d in chunk:
#         gen_tokens = tokenizer(d['gen_completion'], add_special_tokens=False)["input_ids"]
#         if len(gen_tokens) < 200:
#             n_short += 1
#             continue
#         d.update({'z_score': None, 'fraction': 0.0, 'type': 'OpenGen'})
#         filtered.append(d)
#     return filtered, n_short


def _split_chunks(data, n_workers):
    chunk_size = max(1, (len(data) + n_workers - 1) // n_workers)
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def filter_positive_parallel(pool, data, tau, fraction):
    chunks = _split_chunks(data, NUM_WORKERS)
    fn = partial(_process_positive_chunk, fraction=fraction, tau=tau)
    results = pool.map(fn, chunks)
    all_filtered = []
    all_dropped = []
    total_short = 0
    for filtered, n_short, dropped in results:
        all_filtered.extend(filtered)
        all_dropped.extend(dropped)
        total_short += n_short
    avg_z = sum(d['z_score'] for d in all_filtered) / len(all_filtered) if all_filtered else 0
    print(f"All data: {len(data)}; {total_short} too short; {len(all_filtered)} available; average z_score: {avg_z:.4f}")
    return all_filtered, all_dropped


# def filter_negative_parallel(pool, data):
#     chunks = _split_chunks(data, NUM_WORKERS)
#     results = pool.map(_process_negative_chunk, chunks)
#     all_filtered = []
#     total_short = 0
#     for filtered, n_short in results:
#         all_filtered.extend(filtered)
#         total_short += n_short
#     print(f"All data: {len(data)}; {total_short} too short; {len(all_filtered)} available")
#     return all_filtered


if __name__ == "__main__":
    with Pool(NUM_WORKERS, initializer=_init_worker, initargs=(MODEL_NAME,)) as pool:
        filtered_positive_data = []
        dropped_positive_data = []
        for tau, data_file, fraction in zip(positive_tau_list, positive_data_files, fraction_list):
            data = load_jsonl(data_file)
            filtered, dropped = filter_positive_parallel(pool, data, tau, fraction)
            filtered_positive_data.extend(filtered)
            dropped_positive_data.extend(dropped)

        # filtered_negative_data = []
        # for tau, data_file in zip(negative_tau_list, negative_data_files):
        #     data = load_jsonl(data_file)
        #     filtered_negative_data.extend(filter_negative_parallel(pool, data))

    all_filtered_data = filtered_positive_data
    # all_filtered_data = filtered_positive_data + filtered_negative_data

    save_path = f"/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa/strength_4.0/Qwen-Qwen3-14B_strength_4.0_filtered_data_pos_{len(filtered_positive_data)}.jsonl"
    print(f"Filtered data saved to: {save_path}")
    save_jsonl(all_filtered_data, save_path)
    save_path = f"/home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa/strength_4.0/Qwen-Qwen3-14B_strength_4.0_dropped_data_pos_{len(dropped_positive_data)}.jsonl"
    print(f"Dropped data saved to: {save_path}")
    save_jsonl(dropped_positive_data, save_path)