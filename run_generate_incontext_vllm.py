"""
Generate text with in-context watermarking: green word list in system prompt.
Using vLLM for efficient inference.
"""
import argparse
import json
import logging
import os

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer
from vllm import LLM, SamplingParams

from dataset import load_generation_dataset, map_fn_with_chat_template
from gptwm import GPTWatermarkBase
from gptwm_incontext import InContextWatermarkGenerator, get_incontext_system_prompt
from gptwm_vllm_config import set_watermark_base, vLLMGPTWatermarkLogitsWarper


os.environ["VLLM_LOG_LEVEL"] = "ERROR"

logging.getLogger("vllm").setLevel(logging.ERROR)


def create_sampling_params(args):
    """Create SamplingParams based on arguments."""
    if args.beam_size is not None:
        return SamplingParams(
            n=args.beam_size,
            use_beam_search=True,
            max_tokens=args.max_new_tokens,
        )
    else:
        return SamplingParams(
            temperature=1.0,
            top_k=args.top_k if args.top_k is not None else -1,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )


def main(args):
    output_file = (
        f"{args.output_dir}/"
        f"{args.model_name.replace('/', '-')}_"
        f"strength_{args.strength}_"
        f"frac_{args.fraction}_"
        f"len_{args.max_new_tokens}_"
        f"num_{args.num_test}_incontext_vllm.jsonl"
    )
    if args.only_English:
        output_file = output_file.replace('.jsonl', '_only_English.jsonl')

    # tokenizer (for watermark generation and tokenization)
    if 'decapoda-research-llama-7B-hf' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Initialize in-context watermark generator
    model_config = AutoConfig.from_pretrained(args.model_name)
    watermark_generator = InContextWatermarkGenerator(
        fraction=args.fraction,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        watermark_key=args.wm_key,
        only_English=args.only_English,
        tokenizer=tokenizer
    )
    
    # Generate green word list
    if args.fraction > 0.0:
        green_token_string = watermark_generator.get_green_token_string()
    else:
        green_token_string = ""
    system_prompt = get_incontext_system_prompt(green_token_string)

    # Load vLLM model with YaRN support if enabled
    llm_kwargs = {
        "model": args.model_name,
        "dtype": "bfloat16",

        # parallel parameters
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
    }
    llm_kwargs["hf_overrides"] = {}
    
    # Set max_model_len
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    else:
        llm_kwargs["max_model_len"] = 262144 #! Qwen3 with yarn only supports 131072
    
    
    if args.yarn:
        # Configure YaRN for long context
        print("Using YaRN for long context")

        # Override max_position_embeddings in model config to match max_model_len
        target_max_position_embeddings = llm_kwargs["max_model_len"]
        print(f"Overriding max_position_embeddings to {target_max_position_embeddings}")
        llm_kwargs["hf_overrides"]["max_position_embeddings"] = target_max_position_embeddings
        llm_kwargs["hf_overrides"]["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": 32768
        }
    
    print("Loading vLLM model...")
    if args.add_logits_wm:
        wm_base = GPTWatermarkBase(
            fraction=args.fraction,
            strength=args.strength,
            vocab_size=tokenizer.vocab_size,
            model_emb_length=model_config.vocab_size,
            watermark_key=args.wm_key,
            only_English=args.only_English,
            tokenizer=tokenizer
        )
        set_watermark_base(wm_base)
        llm = LLM(
            **llm_kwargs,
            logits_processors=[vLLMGPTWatermarkLogitsWarper]
        )
    else:
        llm = LLM(**llm_kwargs)
    print("vLLM model loaded successfully")
    print("="*100)
    sampling_params = create_sampling_params(args)
    
    # load dataset
    ds = load_generation_dataset(args.prompt_file, args.num_test).to_iterable_dataset()

    # Tokenize with chat template
    ds = ds.map(
        map_fn_with_chat_template(tokenizer, system_prompt),
        batched=True
    )

    for batch in tqdm(ds.iter(batch_size=args.batch_size), desc="Generating"):
        input_prompts = batch["input_prompts"]
        prefixes = batch["prefix"]
        gold_completions = batch["gold_completion"]
        
        # Generate with vLLM
        outputs_vllm = llm.generate(input_prompts, sampling_params)
        
        # Process results
        outputs = []
        for i, output in enumerate(outputs_vllm):
            gen_text = output.outputs[0].text
            
            outputs.append(json.dumps({
                "input_prompt": input_prompts[i],
                "actual_prompt": output.prompt,
                "prefix": prefixes[i],
                "gold_completion": gold_completions[i],
                "gen_completion": gen_text,
            }, ensure_ascii=False))

        # Write in batches
        with open(output_file, "a") as f:
            f.write("\n".join(outputs) + "\n")

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
    parser.add_argument("--yarn", action="store_true", help="Enable YaRN for long context")
    parser.add_argument("--yarn_factor", type=float, default=4.0, help="YaRN scaling factor")
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum model length for vLLM")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for vLLM generation")
    
    # Watermark parameters
    parser.add_argument_group("Watermark")
    parser.add_argument("--add_logits_wm", action="store_true", help="Add logits-based watermarking")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--only_English", action="store_true")

    # Data parameters
    parser.add_argument_group("Data")
    parser.add_argument("--prompt_file", type=str, default="./UnigramWatermark/data/LFQA/inputs.jsonl")
    parser.add_argument("--output_dir", type=str, default="./test")
    parser.add_argument("--num_test", type=int, default=512)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
