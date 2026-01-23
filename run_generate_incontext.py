"""
Generate text with in-context watermarking: green word list in system prompt.
"""
import argparse
import json
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)
from gptwm_incontext import InContextWatermarkGenerator, get_incontext_system_prompt
from dataset import load_generation_dataset


def tokenize_fn_with_chat_template(tokenizer, system_prompt: str):
    """Create a tokenize function that applies chat template with system prompt."""
    def _fn(batch):
        user_prompts = batch["prefix"]
        
        # Apply chat template to each prompt
        prompt_texts = []
        for user_prompt in user_prompts:
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                # Use chat template
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompt_texts.append(prompt_text)
            else:
                # Fallback: manual formatting for models without chat template
                full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
                prompt_texts.append(full_prompt)
        
        # Batch encode all prompts
        prefix_enc = tokenizer(
            prompt_texts,
            truncation=True,
            padding=True,
        )
        
        # Also tokenize gold completions for reference
        gold_enc = tokenizer(
            batch["gold_completion"],
            truncation=True,
            padding=True,
        )

        return {
            "input_ids": prefix_enc["input_ids"],
            "attention_mask": prefix_enc["attention_mask"],
            "gold_completion_ids": gold_enc["input_ids"],
        }

    return _fn


def main(args):
    output_file = (
        f"{args.output_dir}/"
        f"{args.model_name.replace('/', '-')}_"
        f"strength_{args.strength}_"
        f"frac_{args.fraction}_"
        f"len_{args.max_new_tokens}_"
        f"num_{args.num_test}_incontext.jsonl"
    )
    if args.only_English:
        output_file = output_file.replace('.jsonl', '_only_English.jsonl')

    # tokenizer
    if 'decapoda-research-llama-7B-hf' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    import ipdb; ipdb.set_trace()

    # Initialize in-context watermark generator
    watermark_generator = InContextWatermarkGenerator(
        fraction=args.fraction,
        strength=args.strength,  # Not used in in-context, but kept for compatibility
        vocab_size=model.config.vocab_size,
        watermark_key=args.wm_key,
        only_English=args.only_English,
        tokenizer=tokenizer
    )
    
    # Generate green word list
    green_word_list_str = watermark_generator.format_green_word_list()
    system_prompt = get_incontext_system_prompt(green_word_list_str)
    
    print(f"Green word list contains {len(green_word_list_str.split(', '))} words")
    print(f"System prompt length: {len(system_prompt)} characters")

    # load dataset
    ds = load_generation_dataset(args.prompt_file, args.num_test)

    # Tokenize with chat template
    ds = ds.map(
        tokenize_fn_with_chat_template(tokenizer, system_prompt),
        batched=True,
        remove_columns=ds.column_names,
    )

    ds = ds.with_format("torch")

    outputs = []
    # import ipdb; ipdb.set_trace()
    for batch in tqdm(ds.iter(batch_size=128), desc="Generating"):
        generation_config = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
            "logits_processor": None,  # No logits processor for in-context watermark
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

        # Extract prefix and completion
        for i in range(len(gen_text)):
            prefix = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
            gold_completion = tokenizer.decode(batch["gold_completion_ids"][i], skip_special_tokens=True)
            gen_completion = gen_text[i][len(prefix):]
            outputs.append(json.dumps({
                "prefix": prefix,
                "gold_completion": gold_completion,
                "gen_completion": gen_completion,
            }, ensure_ascii=False))
            
            outputs.append(json.dumps({
                "prefix": prefix,
                "gold_completion": gold_completion,
                "gen_completion": gen_completion,
            }, ensure_ascii=False))

        # Write in batches
        with open(output_file, "a") as f:
            f.write("\n".join(outputs) + "\n")
        outputs = []

    print("Results saved to:", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model generation parameters
    parser.add_argument_group("Model Generation")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Watermark parameters
    parser.add_argument_group("Watermark")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)  # Not used in in-context, kept for compatibility
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--only_English", action="store_true")

    # Data parameters
    parser.add_argument_group("Data")
    parser.add_argument("--prompt_file", type=str, default="./UnigramWatermark/data/LFQA/inputs.jsonl")
    parser.add_argument("--output_dir", type=str, default="./test")
    parser.add_argument("--num_test", type=int, default=500)

    args = parser.parse_args()
    main(args)
