import argparse
import json
from multiprocessing import Pool
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizer, AutoConfig
from gptwm import GPTWatermarkDetector

# Globals for worker processes (set by pool initializer)
_worker_tokenizer = None
_worker_config = None
_worker_emb_length = None


def _safe_emb_length(tokenizer, config) -> int:
    """Mask length covering every possible token id, robust across tokenizer families.

    Some tokenizers (e.g. Gemma) ship with `config.vocab_size == tokenizer.vocab_size`,
    breaking the strict `model_emb_length > vocab_size` assertion in gptwm.py.
    """
    max_id = max(tokenizer.get_vocab().values())
    cfg_size = getattr(config, 'vocab_size', 0) or 0
    return max(cfg_size, max_id + 1, tokenizer.vocab_size + 1)


def _init_worker(model_name: str):
    """Initialize tokenizer and config in each worker process. `model_name` here is
    the *detection* tokenizer — the caller already resolved alt vs actor."""
    global _worker_tokenizer, _worker_config, _worker_emb_length
    if "decapoda-research-llama-7B-hf" in model_name:
        _worker_tokenizer = LlamaTokenizer.from_pretrained(model_name)
    else:
        _worker_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if _worker_tokenizer.pad_token is None:
            _worker_tokenizer.pad_token = _worker_tokenizer.eos_token
    _worker_tokenizer.padding_side = "left"
    _worker_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    _worker_emb_length = _safe_emb_length(_worker_tokenizer, _worker_config)


def _detect_chunk(chunk_and_args):
    """Run detection on a chunk of data. Uses global tokenizer/config set by initializer."""
    chunk, args_dict = chunk_and_args
    tokenizer = _worker_tokenizer
    emb_length = _worker_emb_length
    detector_cache = {}
    min_tokens = args_dict["test_min_tokens"]
    wm_key = args_dict["wm_key"]

    def get_detector(seed: int) -> GPTWatermarkDetector:
        if seed not in detector_cache:
            detector_cache[seed] = GPTWatermarkDetector(
                fraction=args_dict["fraction"],
                strength=args_dict["strength"],
                vocab_size=tokenizer.vocab_size,
                model_emb_length=emb_length,
                watermark_key=seed,
                only_English=args_dict["only_English"],
                tokenizer=tokenizer,
            )
        return detector_cache[seed]

    z_scores = []
    for cur_data in chunk:
        gen_tokens = tokenizer(cur_data["gen_completion"], add_special_tokens=False)["input_ids"]
        if len(gen_tokens) >= min_tokens:
            seed = cur_data.get("seed", wm_key)
            detector = get_detector(seed)
            z_scores.append(detector.unidetect(gen_tokens))
    return z_scores


def main(args):
    with open(args.input_file, 'r') as f:
        data = [json.loads(x) for x in f.read().strip().split("\n")]
    if args.combine_fraction:
        fraction0_file = args.input_file.replace(f'frac_{args.fraction}', f'frac_0.0')
        if args.use_generated_neg_data:
            with open(fraction0_file, 'r') as f:
                fraction0_data = [json.loads(x) for x in f.read().strip().split("\n")]
        else:
            fraction0_data = []
            for d in data:
                fraction0_data.append({
                    "prefix": d["prefix"],
                    "input_prompt": d["input_prompt"],
                    "gen_completion": d["gold_completion"],
                    "seed": d.get("seed", args.wm_key)
                })

    # Resolve detection tokenizer (alt overrides actor). Mask + tokenization both use it.
    detect_model = args.alt_tokenizer if args.alt_tokenizer else args.model_name

    if args.workers is None or args.workers <= 1:
        if 'decapoda-research-llama-7B-hf' in detect_model:
            tokenizer = LlamaTokenizer.from_pretrained(detect_model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(detect_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_config = AutoConfig.from_pretrained(detect_model, trust_remote_code=True)
        emb_length = _safe_emb_length(tokenizer, model_config)
    else:
        tokenizer = None
        model_config = None
        emb_length = None

    def get_detector(seed: int) -> GPTWatermarkDetector:
        if seed not in detector_cache:
            detector_cache[seed] = GPTWatermarkDetector(
                fraction=args.fraction,
                strength=args.strength,
                vocab_size=tokenizer.vocab_size,
                model_emb_length=emb_length,
                watermark_key=seed,
                only_English=args.only_English,
                tokenizer=tokenizer,
            )
        return detector_cache[seed]

    detector_cache = {}

    def run_detection_on_data(data_list, desc="Detecting"):
        if args.workers is None or args.workers <= 1:
            z_scores = []
            for cur_data in tqdm(data_list, total=len(data_list), desc=desc):
                gen_tokens = tokenizer(
                    cur_data["gen_completion"], add_special_tokens=False
                )["input_ids"]
                if len(gen_tokens) >= args.test_min_tokens:
                    seed = cur_data.get("seed", args.wm_key)
                    detector = get_detector(seed)
                    z_scores.append(detector.unidetect(gen_tokens))
            return z_scores
        else:
            args_dict = {
                "fraction": args.fraction,
                "strength": args.strength,
                "test_min_tokens": args.test_min_tokens,
                "wm_key": args.wm_key,
                "only_English": args.only_English,
            }
            chunk_size = max(1, (len(data_list) + args.workers - 1) // args.workers)
            chunks = [
                (data_list[i : i + chunk_size], args_dict)
                for i in range(0, len(data_list), chunk_size)
            ]
            with Pool(
                args.workers,
                initializer=_init_worker,
                initargs=(detect_model,),
            ) as pool:
                chunk_results = list(
                    tqdm(
                        pool.imap(_detect_chunk, chunks),
                        total=len(chunks),
                        desc=desc,
                    )
                )
            return [z for chunk_zs in chunk_results for z in chunk_zs]

    z_score_list = run_detection_on_data(data, desc="Main set")

    positive_num = len(z_score_list)
    negative_num = None

    frac0_z_score_list = []
    if args.combine_fraction:
        frac0_z_score_list = run_detection_on_data(fraction0_data, desc="Frac0")
        negative_num = len(frac0_z_score_list)
    z_score_list = z_score_list + frac0_z_score_list
    
    save_dict = {
        'z_score': z_score_list,
        'wm_pred': [1 if z > args.threshold else 0 for z in z_score_list],
        'positive_num': positive_num,
        'negative_num': negative_num
    }

    print(save_dict['wm_pred'])
    with open(args.input_file.replace('.jsonl', '_z.jsonl'), 'w') as f:
        json.dump(save_dict, f)
        print("Results saved to:", args.input_file.replace('.jsonl', '_z.jsonl'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="baffo32/decapoda-research-llama-7B-hf")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--wm_key", type=int, default=None)
    parser.add_argument("--use_generated_neg_data", action="store_true")
    parser.add_argument("--input_file", type=str, default="./data/example_output.jsonl")
    parser.add_argument("--test_min_tokens", type=int, default=200)
    parser.add_argument("--only_English", action="store_true")
    parser.add_argument("--combine_fraction", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of processes for detection")
    parser.add_argument("--alt_tokenizer", type=str, default=None,
                        help="HF id of an alternate tokenizer used for tokenizing gen_completion AND "
                             "constructing the green-list mask (must match generation-time --alt_tokenizer). "
                             "If unset, --model_name's tokenizer is used.")

    args = parser.parse_args()

    main(args)

