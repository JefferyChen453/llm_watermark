import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, LlamaTokenizer, AutoConfig
from gptwm import GPTWatermarkDetector


def main(args):
    with open(args.input_file, 'r') as f:
        data = [json.loads(x) for x in f.read().strip().split("\n")]
    if args.combine_fraction:
        fraction0_file = args.input_file.replace(f'frac_{args.fraction}', f'frac_0.0')
        with open(fraction0_file, 'r') as f:
            fraction0_data = [json.loads(x) for x in f.read().strip().split("\n")]

    if 'decapoda-research-llama-7B-hf' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_config = AutoConfig.from_pretrained(args.model_name)
    detector = GPTWatermarkDetector(fraction=args.fraction,
                                    strength=args.strength,
                                    vocab_size=tokenizer.vocab_size,
                                    model_emb_length=model_config.vocab_size,
                                    watermark_key=args.wm_key,
                                    only_English=args.only_English,
                                    tokenizer=tokenizer)

    z_score_list = []
    for idx, cur_data in tqdm(enumerate(data), total=len(data)):
        gen_tokens = tokenizer(cur_data['gen_completion'], add_special_tokens=False)["input_ids"]
        if len(gen_tokens) >= args.test_min_tokens:
            z_score_list.append(detector.detect(gen_tokens))
        else:
            print(f"Warning: sequence {idx} is too short to test.")
    positive_num = len(z_score_list)
    negative_num = None

    frac0_z_score_list = []
    if args.combine_fraction:
        for idx, cur_data in tqdm(enumerate(fraction0_data), total=len(fraction0_data)):
            gen_tokens = tokenizer(cur_data['gen_completion'], add_special_tokens=False)["input_ids"]
            if len(gen_tokens) >= args.test_min_tokens:
                frac0_z_score_list.append(detector.detect(gen_tokens))
            else:
                print(f"Warning: sequence {idx} is too short to test.")
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
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--input_file", type=str, default="./data/example_output.jsonl")
    parser.add_argument("--test_min_tokens", type=int, default=200)
    parser.add_argument("--only_English", action="store_true")
    parser.add_argument("--combine_fraction", action="store_true")

    args = parser.parse_args()

    main(args)
