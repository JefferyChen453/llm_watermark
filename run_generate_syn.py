import argparse
from functools import partial
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LogitsProcessorList,
)

from dataset import collate_fn, load_generation_dataset, make_multitask_prompt_mapper
from gptwm import BatchWatermarkLogitsProcessor

def main(args):
    output_file = (
        f"{args.output_dir}/"
        f"{args.model_name.replace('/', '-')}_"
        f"strength_{args.strength}_"
        f"frac_{args.fraction}_"
        f"len_{args.max_new_tokens}_"
        f"num_{args.num_test}.jsonl"
    )
    if args.only_English:
        output_file = output_file.replace('.jsonl', '_only_English.jsonl')

    # tokenizer
    if 'decapoda-research-llama-7B-hf' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()

    watermark_processor = LogitsProcessorList([
        BatchWatermarkLogitsProcessor(
            fraction=args.fraction,
            strength=args.strength,
            vocab_size=tokenizer.vocab_size,
            model_emb_length=model.config.vocab_size,
            only_English=args.only_English,
            tokenizer=tokenizer
        )
    ])

    # Per-sample seeds 0..9999, assigned in order (sample i gets seed i % 10000)
    seed_list = list(range(1, args.seed_num + 1))

    # Load dataset: apply chat template per example using dataset_type
    ds = load_generation_dataset(args.prompt_file, args.num_test)
    ds = ds.map(
        make_multitask_prompt_mapper(tokenizer),
        batched=False,
        with_indices=True,
    )
    data_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    outputs = []

    for batch in tqdm(data_loader, desc="Generating", total=len(data_loader)):
        batch_seeds = [seed_list[idx % len(seed_list)] for idx in batch["idx"]]
        watermark_processor[0].current_batch_seeds = batch_seeds

        generation_config = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
            "logits_processor": watermark_processor,
            "min_new_tokens": args.min_new_tokens,
            "max_new_tokens": args.max_new_tokens,
            "return_dict_in_generate": True,
        }

        if args.beam_size is not None:
            # beam search
            generation_config.update({
                "num_beams": args.beam_size,
                "do_sample": False,
            })
        else:
            # sampling
            generation_config.update({
                "do_sample": True,
                "top_k": args.top_k,
                "top_p": args.top_p,
            })

        with torch.inference_mode():
            generation = model.generate(**generation_config)
        input_len = batch["input_ids"].shape[1]
        sequences = generation.sequences
        responses = []
        for i in range(sequences.size(0)):
            gen_ids = sequences[i, input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(text)

        for i in range(len(responses)):
            input_prompt = batch["input_prompt"][i]
            prefix = batch["prefix"][i]
            gold_completion = batch["gold_completion"][i]
            gen_completion = responses[i]
            seed = batch_seeds[i]
            dataset_type = batch["dataset_type"][i]
            outputs.append(json.dumps({
                "prefix": prefix,
                "input_prompt": input_prompt,
                "gold_completion": gold_completion,
                "gen_completion": gen_completion,
                "seed": seed,
                "dataset_type": dataset_type,
            }, ensure_ascii=False))

        with open(output_file, "a") as f:
            f.write("\n".join(outputs) + "\n")
        outputs = []

    print("Results saved to:", output_file)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model generation parameters
    parser.add_argument_group("Generation")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--min_new_tokens", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Watermark parameters
    parser.add_argument_group("Watermark")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--seed_num", type=int, default=100)
    parser.add_argument("--only_English", action="store_true")

    # Data parameters
    parser.add_argument_group("Data")
    parser.add_argument("--prompt_file", type=str, default="./UnigramWatermark/data/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="./UnigramWatermark/data/LFQA/")
    parser.add_argument("--num_test", type=int, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)