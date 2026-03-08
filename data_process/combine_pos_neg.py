import json
from multiprocessing import Pool
from functools import partial
from transformers import AutoTokenizer
from gptwm import GPTWatermarkDetector


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for d in data:
            file.write(json.dumps(d) + '\n')


# negative_data_files = [
#     "/home/tianyichen/llm_watermark/outputs/syn_data/OpenGen/Qwen-Qwen3-14B_strength_2.0_frac_0.0_len_500_num_7211_only_English.jsonl",
# ]
positive_data_files = [
    "/home/tianyichen/llm_watermark/outputs/syn_data/Qwen-Qwen3-32B_OpenGen/Qwen-Qwen3-32B_strength_2.0_frac_0.1_len_500_num_7211_only_English.jsonl",
    "/home/tianyichen/llm_watermark/outputs/syn_data/Qwen-Qwen3-32B_OpenGen/Qwen-Qwen3-32B_strength_2.0_frac_0.2_len_500_num_7211_only_English.jsonl",
    "/home/tianyichen/llm_watermark/outputs/syn_data/Qwen-Qwen3-32B_OpenGen/Qwen-Qwen3-32B_strength_2.0_frac_0.3_len_500_num_7211_only_English.jsonl",
    "/home/tianyichen/llm_watermark/outputs/syn_data/Qwen-Qwen3-32B_OpenGen/Qwen-Qwen3-32B_strength_2.0_frac_0.4_len_500_num_7211_only_English.jsonl",
]
# negative_tau_list = [5.0]
positive_tau_list = [6.5, 6.0, 6.0, 6.0]
fraction_list = [0.1, 0.2, 0.3, 0.4]
DEFAULT_WM_KEY = 0
MODEL_NAME = "Qwen/Qwen3-32B"
NUM_WORKERS = 8

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
    n_short = 0
    for d in chunk:
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
        z_score = detector_cache[key].detect(gen_tokens)
        if z_score > tau:
            d.update({'z_score': z_score, 'fraction': fraction})
            filtered.append(d)
    return filtered, n_short


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
    total_short = 0
    for filtered, n_short in results:
        all_filtered.extend(filtered)
        total_short += n_short
    avg_z = sum(d['z_score'] for d in all_filtered) / len(all_filtered) if all_filtered else 0
    print(f"All data: {len(data)}; {total_short} too short; {len(all_filtered)} available; average z_score: {avg_z:.4f}")
    return all_filtered


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
        for tau, data_file, fraction in zip(positive_tau_list, positive_data_files, fraction_list):
            data = read_jsonl(data_file)
            filtered_positive_data.extend(filter_positive_parallel(pool, data, tau, fraction))

        # filtered_negative_data = []
        # for tau, data_file in zip(negative_tau_list, negative_data_files):
        #     data = read_jsonl(data_file)
        #     filtered_negative_data.extend(filter_negative_parallel(pool, data))

    all_filtered_data = filtered_positive_data
    # all_filtered_data = filtered_positive_data + filtered_negative_data

    save_path = f"/home/tianyichen/llm_watermark/outputs/syn_data/Qwen-Qwen3-32B_OpenGen/Qwen-Qwen3-32B_OpenGen_filtered_data_pos_{len(filtered_positive_data)}.jsonl"
    print(f"Save to: {save_path}")
    save_jsonl(all_filtered_data, save_path)
