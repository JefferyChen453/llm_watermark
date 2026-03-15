"""
Knowledge Distillation training for in-context watermark.

The model generates watermarked responses using an in-context green word list
combined with logits modification (teacher signal). A KD loss then trains the
model to internalise the watermark bias so that logits modification is no
longer needed at inference time.

Workflow (per sample):
  1. [no grad]  Generate N tokens with watermark-modified logits.
  2. [with grad] Forward the full sequence (prompt + generated tokens)
                 through the transformer backbone, then apply lm_head ONLY
                 on the N generated positions (avoids materialising a huge
                 [seq_len, vocab] logits tensor for 80k-token prompts).
  3. teacher_logits = student_logits + strength * green_list_mask  (detached)
  4. loss = KL(teacher || student), mean over generated positions.
  5. Backward + DeepSpeed optimizer step.

Usage:
    deepspeed --num_gpus=8 run_train_kd_incontext.py \
        --deepspeed_config configs/ds_zero2_config.json \
        --model_name Qwen/Qwen3-8B \
        --train_data ./UnigramWatermark/data/LFQA/inputs.jsonl \
        --output_dir ./outputs/kd_incontext \
        --fraction 0.4 --strength 3.0 --only_English \
        --gradient_checkpointing --flash_attn --yarn
"""

import argparse
from functools import partial
import json
import logging
import math
import os

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
)
import wandb

from dataset import collate_fn, load_generation_dataset, make_prompt_mapper
from gptwm import GPTWatermarkLogitsWarper
from gptwm_incontext import InContextWatermarkGenerator
from prompt import get_incontext_system_prompt


logger = logging.getLogger(__name__)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# --------------------------------------------------------------------------- #
#  Model helpers                                                               #
# --------------------------------------------------------------------------- #


def get_backbone_and_lm_head(model):
    if hasattr(model, "model") and hasattr(model, "lm_head"):
        return model.model, model.lm_head
    if hasattr(model, "transformer") and hasattr(model, "lm_head"):
        return model.transformer, model.lm_head
    raise ValueError(
        f"Cannot split backbone / lm_head for {type(model).__name__}. "
        "Please add support in get_backbone_and_lm_head()."
    )


# --------------------------------------------------------------------------- #
#  Training                                                                    #
# --------------------------------------------------------------------------- #


def train(args):
    # -- Distributed setup ------------------------------------------------- #
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = global_rank == 0

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO if is_main else logging.WARNING,
    )

    if is_main and args.wandb:
        wandb.init(
            entity="chentianyi453",
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config=vars(args),
            dir="/home/tianyichen/llm_watermark/wandb"
        )

    # -- Tokenizer --------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    tokenizer.padding_side = "left"

    # -- Model config ------------------------------------------------------ #
    config = AutoConfig.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if args.yarn:
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": 32768,
        }
        config.max_position_embeddings = args.max_position_embeddings

    # -- Model ------------------------------------------------------------- #
    model_kwargs = {
        "config": config,
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, **model_kwargs
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # -- Watermark --------------------------------------------------------- #
    watermark_processor = GPTWatermarkLogitsWarper(
        fraction=args.fraction,
        strength=args.strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=config.vocab_size,
        watermark_key=args.wm_key,
        only_English=args.only_English,
        tokenizer=tokenizer,
    )
    green_list_mask = watermark_processor.green_list_mask  # [vocab], 0/1 float
    english_mask = watermark_processor.english_mask # [vocab], 0/1 float

    # -- In-context prompt ------------------------------------------------- #
    ic_gen = InContextWatermarkGenerator(
        fraction=args.fraction,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=config.vocab_size,
        watermark_key=args.wm_key,
        only_English=args.only_English,
        tokenizer=tokenizer,
    )
    green_str = ic_gen.get_green_token_string() if args.fraction > 0 else ""
    system_prompt = get_incontext_system_prompt("lfqa", green_str)
    if is_main:
        n_tok = len(tokenizer.encode(system_prompt))
        logger.info(f"System prompt length: ~{n_tok} tokens")

    # -- Dataset & DataLoader ---------------------------------------------- #
    ds = load_generation_dataset(args.train_data)
    ds = ds.map(
        make_prompt_mapper(tokenizer, system_prompt, tokenize=True),
        num_proc=os.cpu_count() // 2
    )
    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        seed=42,
    )
    data_loader = DataLoader(
        ds,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=4
    )
    # -- DeepSpeed init ---------------------------------------------------- #
    ds_config = json.load(open(args.deepspeed_config))

    # Compute total training steps and patch scheduler
    steps_per_epoch = math.ceil(
        len(ds) / (world_size * args.micro_batch_size * args.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * args.num_epochs
    
    # Override DeepSpeed config
    ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["optimizer"]["params"]["lr"] = args.lr
    ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = int(0.02 * total_steps)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    if is_main:
        logger.info(
            f"Dataset: {len(ds)} samples | "
            f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps} | "
            f"World size: {world_size} | "
            f"Grad accum: {args.gradient_accumulation_steps}"
        )

    # -- Reusable references ----------------------------------------------- #
    unwrapped = model_engine.module
    backbone, lm_head = get_backbone_and_lm_head(unwrapped)
    logits_proc = LogitsProcessorList([watermark_processor])
    wm_bias = (args.strength * green_list_mask).to(device)  # [vocab]

    # ====================================================================== #
    #  Training loop                                                          #
    # ====================================================================== #
    global_step = 0

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        if is_main:
            logger.info("=" * 60 + f"  Epoch {epoch + 1}/{args.num_epochs}")
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}", disable=not is_main)

        for batch in pbar:
            generation_config = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "logits_processor": logits_proc,
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            # sampling
            generation_config.update({
                "do_sample": True,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": 1.0
            })
            prompt_ids = batch["input_ids"].to(device)
            prompt_len = prompt_ids.shape[1]

            # ============================================================== #
            #  Phase 1 -- Generate with watermark-modified logits (no grad)   #
            # ============================================================== #
            unwrapped.eval()
            with torch.no_grad():
                gen_sequences = unwrapped.generate(**generation_config)  # [B, prompt_len + gen_len]
            gen_len = gen_sequences.shape[1] - prompt_len
            generated_ids = gen_sequences[:, prompt_len:]

            # wandb: log generation samples table
            if is_main and args.wandb:
                gen_table = wandb.Table(columns=["step", "prefix", "response", "z_score"])
                prefixes = batch["prefix"]
                for i in range(gen_sequences.shape[0]):
                    resp = tokenizer.decode(
                        generated_ids[i].tolist(),
                        skip_special_tokens=True,
                    )
                    z_score = z_score_per_row[i].item()
                    gen_table.add_data(
                        global_step + 1,
                        prefixes[i] if i < len(prefixes) else "",
                        resp,
                        z_score,
                    )
                wandb.log({"gen/samples": gen_table}, step=global_step + 1)
            # compute z_score: per-row, then mean over batch
            non_pad_mask = generated_ids != tokenizer.pad_token_id   # (B, L)
            green_mask = green_list_mask.to(device)[generated_ids]   # (B, L)
            total_valid_per_row = non_pad_mask.sum(dim=1).float()   # (B,)
            green_valid_per_row = (green_mask * non_pad_mask).sum(dim=1).float()   # (B,)
            var_per_row = args.fraction * (1 - args.fraction) * total_valid_per_row
            z_score_per_row = (green_valid_per_row - args.fraction * total_valid_per_row) / torch.sqrt(var_per_row)
            z_score_avg = z_score_per_row.mean()

            # ============================================================== #
            #  Phase 2 -- Training forward (split lm_head for memory)         #
            # ============================================================== #
            unwrapped.train()
            full_ids = gen_sequences.detach()       # [1, prompt_len + gen_len]
            attn_mask = (full_ids != tokenizer.pad_token_id).long()

            # Forward through transformer backbone (gradient-checkpointed)
            backbone_out = backbone(
                input_ids=full_ids,
                attention_mask=attn_mask,
                use_cache=False
            )
            hidden_states = backbone_out[0]  # [1, seq_len, hidden_dim]

            # Apply lm_head ONLY on generated-token positions.
            # logits at position i predict token at position i+1, so for
            # tokens at [prompt_len .. prompt_len+gen_len-1] we need hidden
            # states at [prompt_len-1 .. prompt_len+gen_len-2].
            gen_hidden = hidden_states[
                :, prompt_len - 1 : prompt_len + gen_len - 1, :
            ]
            gen_mask = attn_mask[:, prompt_len:]
            student_logits = lm_head(gen_hidden)  # [1, gen_len, vocab]
            # ============================================================== #
            #  Phase 3 -- KD loss (fp32 for numerical stability)              #
            # ============================================================== #
            s_logits = student_logits.float()
            t_logits = student_logits.detach().float() + wm_bias.float()

            T = args.kd_temperature
            t_probs = F.softmax(t_logits / T, dim=-1)
            s_log_probs = F.log_softmax(s_logits / T, dim=-1)

            # KL(teacher || student), averaged over generated positions
            kl_div = F.kl_div(s_log_probs, t_probs, reduction="none")
            if args.english_token_loss:
                english_mask_3d = english_mask.to(device).float().view(1, 1, -1)
                kl_per_token = (kl_div * english_mask_3d).sum(dim=-1)
            else:
                kl_per_token = kl_div.sum(dim=-1)  # [bs, gen_len]
            kl_per_token = kl_per_token * gen_mask.float()
            num_valid_tokens = gen_mask.sum()
            if num_valid_tokens > 0:
                kd_loss = kl_per_token.sum() / num_valid_tokens * (T ** 2)
            else:
                kd_loss = torch.tensor(0.0, device=device, requires_grad=True)
            # ============================================================== #
            #  Phase 4 -- Backward + DeepSpeed step                           #
            # ============================================================== #
            model_engine.backward(kd_loss)
            model_engine.step()
            grad_norm = model_engine.get_global_grad_norm()
            if grad_norm is not None:
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()
            global_step += 1

            # -- Logging --------------------------------------------------- #
            if is_main:
                with torch.no_grad():
                    s_probs = F.softmax(s_logits / T, dim=-1)
                    student_entropy = (
                        -(s_probs * (s_probs + 1e-10).log()).sum(dim=-1).mean().item()
                    )
                    teacher_entropy = (
                        -(t_probs * (t_probs + 1e-10).log()).sum(dim=-1).mean().item()
                    )

                pbar.set_postfix(
                    loss=f"{kd_loss.item():.4f}",
                    gn=f"{grad_norm:.2f}" if grad_norm is not None else "n/a",
                    zs=f"{z_score_avg:.2f}",
                    gl=gen_len,
                )

                step_in_epoch = ((global_step - 1) % steps_per_epoch) + 1
                epoch_frac = epoch + step_in_epoch / steps_per_epoch
                log_dict = {
                    "optim/lr": optimizer.param_groups[0]["lr"],
                    "optim/epoch": epoch_frac,
                    "optim/grad_norm": grad_norm,
                    "train/kd_loss": kd_loss.item(),
                    "train/z_score": z_score_avg,
                    "train/student_entropy": student_entropy,
                    "train/teacher_entropy": teacher_entropy,
                    "gen/prompt_len": prompt_len,
                    "gen/response_len": gen_len,
                }

                if args.wandb:
                    wandb.log(log_dict, step=global_step)

                if global_step % args.log_steps == 0:
                    logger.info(
                        f"Step {global_step}: kd_loss={kd_loss.item():.4f}  "
                        f"grad_norm={grad_norm:.4f}  "
                        f"z_score={z_score_avg:.3f}  gen_len={gen_len}  "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

            # -- Checkpoint ------------------------------------------------ #
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                _save_model(
                    model_engine,
                    tokenizer,
                    os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                    is_main,
                )

        # End-of-epoch save
        _save_model(
            model_engine,
            tokenizer,
            os.path.join(args.output_dir, f"epoch-{epoch + 1}"),
            is_main,
        )

    # Final save
    _save_model(
        model_engine,
        tokenizer,
        os.path.join(args.output_dir, "final"),
        is_main,
    )

    if is_main:
        logger.info("Training complete!")
        if args.wandb:
            wandb.finish()


# --------------------------------------------------------------------------- #
#  Save helper                                                                 #
# --------------------------------------------------------------------------- #


def _save_model(engine, tokenizer, save_dir, is_main):
    """
    Save a HuggingFace-compatible checkpoint.

    Works directly for ZeRO-2 (all ranks hold full params).
    For ZeRO-3 replace with ``engine.save_16bit_model(save_dir)`` or
    gather parameters before saving.
    """
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        engine.module.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved checkpoint -> {save_dir}")
    dist.barrier()


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KD training for in-context watermark"
    )

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="Use Flash Attention 2 (strongly recommended for long prompts)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory",
    )
    parser.add_argument(
        "--yarn", action="store_true", help="Enable YaRN RoPE scaling"
    )
    parser.add_argument("--yarn_factor", type=float, default=4.0)
    parser.add_argument("--max_position_embeddings", type=int, default=131072)

    # Watermark
    parser.add_argument("--fraction", type=float, default=0.4)
    parser.add_argument("--strength", type=float, default=3.0)
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--only_English", action="store_true")

    # Generation (during training)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)

    # Training hyper-parameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--english_token_loss", action="store_true", help="Compute KD loss only on English tokens")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=None)

    # Data / output
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/kd_incontext"
    )

    # Logging
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb_project", type=str, default="kd-incontext-watermark"
    )
    parser.add_argument("--wandb_run_name", type=str, default="")

    # DeepSpeed (--local_rank is set automatically by the DS launcher)
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="configs/ds_zero2_config.json",
    )
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
