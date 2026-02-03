import os
import argparse
import json
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LogitsProcessorList,
)
from gptwm import GPTWatermarkLogitsWarper
from dataset import load_generation_dataset, tokenize_fn


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
        GPTWatermarkLogitsWarper(
            fraction=args.fraction,
            strength=args.strength,
            vocab_size=tokenizer.vocab_size,
            model_emb_length=model.config.vocab_size,
            watermark_key=args.wm_key,
            only_English=args.only_English,
            tokenizer=tokenizer
        )
    ])

    # load dataset
    ds = load_generation_dataset(args.prompt_file, args.num_test)

    # tokenize
    ds = ds.map(
        tokenize_fn(tokenizer),
        batched=True,
        remove_columns=ds.column_names,
    )

    ds = ds.with_format("torch")

    outputs = []

    for batch in tqdm(ds.iter(batch_size=args.batch_size), desc="Generating"):
        generation_config = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
            "logits_processor": watermark_processor,
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

        gen_text = tokenizer.batch_decode(
            generation.sequences,
            skip_special_tokens=True,
        )

        for i in range(len(gen_text)):
            prefix = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
            gold_completion = tokenizer.decode(batch["gold_completion_ids"][i], skip_special_tokens=True)
            gen_completion = gen_text[i][len(prefix):]
            outputs.append(json.dumps({
                "prefix": prefix,
                "gold_completion": gold_completion,
                "gen_completion": gen_completion,
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
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Watermark parameters
    parser.add_argument_group("Watermark")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--only_English", action="store_true")

    # Data parameters
    parser.add_argument_group("Data")
    parser.add_argument("--prompt_file", type=str, default="./UnigramWatermark/data/LFQA/inputs.jsonl")
    parser.add_argument("--output_dir", type=str, default="./UnigramWatermark/data/LFQA/")
    parser.add_argument("--num_test", type=int, default=500)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)