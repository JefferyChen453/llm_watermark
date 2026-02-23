"""
Offline Knowledge Distillation for in-context watermark.

Reads pre-generated responses from a jsonl (incontext prompt + response already
produced, e.g. by vLLM). Then: forward full sequence, apply lm_head only on
generated positions, teacher_logits = student + strength * green_list_mask,
KD loss and update. No generation during training.

Expected jsonl keys: actual_prompt (or input_prompt), prefix, gen_completion.
Optional (when generated with --save_gen_batch): prompt_ids, gen_ids, gen_mask.

Usage:
    deepspeed --num_gpus=8 train_kd_incontext_offline.py \
        --deepspeed_config configs/ds_zero2_config.json \
        --model_name Qwen/Qwen3-14B \
        --train_data temp/incontext_add_logits_wm/strength2/Qwen-Qwen3-14B/Qwen-Qwen3-14B_frac_0.3_len_500_num_512_incontext_vllm_only_English.jsonl \
        --output_dir ./outputs/kd_incontext_offline \
        --fraction 0.3 --strength 2.0 --only_English \
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

from dataset import (
    collate_fn_offline_incontext,
    load_offline_incontext_jsonl,
    map_fn_offline_incontext_full_ids,
)
from gptwm import GPTWatermarkLogitsWarper
from gptwm_incontext import InContextWatermarkGenerator
from prompt import get_incontext_system_prompt


logger = logging.getLogger(__name__)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def get_backbone_and_lm_head(model):
    if hasattr(model, "model") and hasattr(model, "lm_head"):
        return model.model, model.lm_head
    if hasattr(model, "transformer") and hasattr(model, "lm_head"):
        return model.transformer, model.lm_head
    raise ValueError(
        f"Cannot split backbone / lm_head for {type(model).__name__}. "
        "Please add support in get_backbone_and_lm_head()."
    )


def train(args):
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
            dir="/home/tianyichen/llm_watermark/wandb",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    tokenizer.padding_side = "left"

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

    watermark_processor = GPTWatermarkLogitsWarper(
        fraction=args.fraction,
        strength=args.strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=config.vocab_size,
        watermark_key=args.wm_key,
        only_English=args.only_English,
        tokenizer=tokenizer,
    )
    green_list_mask = watermark_processor.green_list_mask
    english_mask = watermark_processor.english_mask

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

    # -- Offline dataset: pre-generated incontext prompt + response -------- #
    # When jsonl has prompt_ids/gen_ids/gen_mask (save_gen_batch), use them directly.
    ds = load_offline_incontext_jsonl(args.train_data)
    ds = ds.map(
        map_fn_offline_incontext_full_ids(tokenizer),
        num_proc=os.cpu_count() // 2,
        remove_columns=[c for c in ["gen_completion", "actual_prompt", "prompt_ids", "gen_ids", "gen_mask"] if c in ds.column_names],
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
        collate_fn=partial(collate_fn_offline_incontext, tokenizer=tokenizer),
        num_workers=4,
    )

    ds_config = json.load(open(args.deepspeed_config))
    steps_per_epoch = math.ceil(
        len(ds)
        / (
            world_size
            * args.micro_batch_size
            * args.gradient_accumulation_steps
        )
    )
    total_steps = steps_per_epoch * args.num_epochs

    ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["optimizer"]["params"]["lr"] = args.lr
    ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = int(
        0.02 * total_steps
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    if is_main:
        logger.info(
            f"Offline dataset: {len(ds)} samples | "
            f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps} | "
            f"World size: {world_size} | "
            f"Grad accum: {args.gradient_accumulation_steps}"
        )

    unwrapped = model_engine.module
    backbone, lm_head = get_backbone_and_lm_head(unwrapped)
    wm_bias = (args.strength * green_list_mask).to(device)

    global_step = 0

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        if is_main:
            logger.info(
                "=" * 60 + f"  Epoch {epoch + 1}/{args.num_epochs}"
            )
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}", disable=not is_main)

        for batch in pbar:
            full_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)

            B, L = full_ids.shape
            # gen_len_i = number of generated tokens for example i
            gen_lens = (
                attn_mask.sum(dim=1) - prompt_lens
            ).clamp(min=0)  # [B]
            max_gen_len = gen_lens.max().item()
            if max_gen_len == 0:
                continue

            unwrapped.train()
            backbone_out = backbone(
                input_ids=full_ids,
                attention_mask=attn_mask,
                use_cache=False,
            )
            hidden_states = backbone_out[0]

            # Gather hidden states for generated positions only; pad to max_gen_len
            # Position t predicts token t+1; for generated token at prompt_len..prompt_len+gen_len-1
            # we need hidden at prompt_len-1..prompt_len+gen_len-2.
            gen_hidden_list = []
            gen_mask_list = []
            for i in range(B):
                pl = prompt_lens[i].item()
                gl = gen_lens[i].item()
                if gl == 0:
                    gen_hidden_list.append(
                        torch.zeros(
                            max_gen_len,
                            hidden_states.size(-1),
                            device=device,
                            dtype=hidden_states.dtype,
                        )
                    )
                    gen_mask_list.append(
                        torch.zeros(max_gen_len, device=device, dtype=torch.float)
                    )
                    continue
                start = pl - 1
                end = pl + gl - 1
                h = hidden_states[i, start:end, :]
                pad_len = max_gen_len - h.size(0)
                gen_hidden_list.append(
                    F.pad(h, (0, 0, 0, pad_len), value=0)
                )
                mask = torch.ones(max_gen_len, device=device, dtype=torch.float)
                if pad_len > 0:
                    mask[gl:] = 0
                gen_mask_list.append(mask)

            gen_hidden = torch.stack(gen_hidden_list, dim=0)
            gen_mask = torch.stack(gen_mask_list, dim=0)

            student_logits = lm_head(gen_hidden)

            s_logits = student_logits.float()
            t_logits = student_logits.detach().float() + wm_bias.float()

            T = args.kd_temperature
            t_probs = F.softmax(t_logits / T, dim=-1)
            s_log_probs = F.log_softmax(s_logits / T, dim=-1)

            kl_div = F.kl_div(s_log_probs, t_probs, reduction="none")
            if args.english_token_loss:
                english_mask_3d = english_mask.to(device).float().view(
                    1, 1, -1
                )
                kl_per_token = (kl_div * english_mask_3d).sum(dim=-1)
            else:
                kl_per_token = kl_div.sum(dim=-1)
            kl_per_token = kl_per_token * gen_mask
            num_valid_tokens = gen_mask.sum()
            if num_valid_tokens > 0:
                kd_loss = kl_per_token.sum() / num_valid_tokens * (T ** 2)
            else:
                kd_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Build gen_ids_padded [B, max_gen_len] for repetition loss (and later z_score)
            gen_ids_padded = full_ids.new_zeros(B, max_gen_len).fill_(
                tokenizer.pad_token_id
            )
            for i in range(B):
                pl, gl = prompt_lens[i].item(), gen_lens[i].item()
                if gl > 0:
                    gen_ids_padded[i, :gl] = full_ids[i, pl : pl + gl]

            # Combined loss: KD + repetition penalty + optional entropy regularizer
            total_loss = kd_loss
            rep_loss_val = 0.0
            entropy_loss_val = 0.0

            if args.rep_loss_weight > 0:
                s_probs = F.softmax(s_logits / T, dim=-1)
                probs_at_next = s_probs[:, :-1, :].gather(
                    2,
                    gen_ids_padded[:, 1:].unsqueeze(2).clamp(
                        0, s_probs.size(2) - 1
                    ),
                ).squeeze(2)
                repeated_mask_pos = gen_ids_padded.new_zeros(
                    B, max_gen_len - 1, dtype=torch.float
                )
                pad_id = tokenizer.pad_token_id
                for j in range(max_gen_len - 1):
                    prev = gen_ids_padded[:, : j + 1]
                    cur = gen_ids_padded[:, j + 1]
                    is_repeat = (prev == cur.unsqueeze(1)).any(dim=1)
                    valid = (cur != pad_id) & (gen_mask[:, j + 1] > 0)
                    repeated_mask_pos[:, j] = (is_repeat & valid).float()
                n_rep = repeated_mask_pos.sum().clamp(min=1)
                rep_loss = (repeated_mask_pos * probs_at_next).sum() / n_rep
                rep_loss_val = rep_loss.item()
                total_loss = total_loss + args.rep_loss_weight * rep_loss

            if args.entropy_min_weight > 0:
                s_probs_reg = (
                    F.softmax(s_logits / T, dim=-1)
                    if args.rep_loss_weight == 0
                    else s_probs
                )
                entropy = -(
                    s_probs_reg * (s_probs_reg + 1e-10).log()
                ).sum(dim=-1)
                min_entropy = args.entropy_min_target
                entropy_shortfall = (min_entropy - entropy).clamp(min=0)
                entropy_loss = (entropy_shortfall * gen_mask).sum() / num_valid_tokens.clamp(
                    min=1
                )
                entropy_loss_val = entropy_loss.item()
                total_loss = total_loss + args.entropy_min_weight * entropy_loss

            model_engine.backward(total_loss)
            model_engine.step()
            grad_norm = model_engine.get_global_grad_norm()
            if grad_norm is not None and isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()

            with torch.no_grad():
                non_pad = gen_ids_padded != tokenizer.pad_token_id
                green_mask = green_list_mask.to(device)[gen_ids_padded]
                total_valid = non_pad.sum(dim=1).float().clamp(min=1)
                green_valid = (green_mask.float() * non_pad.float()).sum(dim=1)
                var_r = args.fraction * (1 - args.fraction) * total_valid
                z_score_per_row = (
                    green_valid - args.fraction * total_valid
                ) / torch.sqrt(var_r)
                z_score_avg = z_score_per_row.mean().item()

            global_step += 1

            if is_main:
                with torch.no_grad():
                    s_probs = F.softmax(s_logits / T, dim=-1)
                    student_entropy = (
                        -(s_probs * (s_probs + 1e-10).log())
                        .sum(dim=-1)
                        .mean()
                        .item()
                    )
                    teacher_entropy = (
                        -(t_probs * (t_probs + 1e-10).log())
                        .sum(dim=-1)
                        .mean()
                        .item()
                    )

                pbar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    kd=f"{kd_loss.item():.4f}",
                    rep=f"{rep_loss_val:.4f}" if args.rep_loss_weight > 0 else "-",
                    gn=f"{grad_norm:.2f}" if grad_norm is not None else "n/a",
                    zs=f"{z_score_avg:.2f}",
                )

                log_dict = {
                    "optim/lr": optimizer.param_groups[0]["lr"],
                    "optim/epoch": epoch
                    + (global_step % steps_per_epoch) / steps_per_epoch,
                    "optim/grad_norm": grad_norm,
                    "train/kd_loss": kd_loss.item(),
                    "train/total_loss": total_loss.item(),
                    "train/z_score": z_score_avg,
                    "train/student_entropy": student_entropy,
                    "train/teacher_entropy": teacher_entropy,
                }
                if args.rep_loss_weight > 0:
                    log_dict["train/rep_loss"] = rep_loss_val
                if args.entropy_min_weight > 0:
                    log_dict["train/entropy_loss"] = entropy_loss_val
                if args.wandb:
                    wandb.log(log_dict, step=global_step)

                if global_step % args.log_steps == 0:
                    logger.info(
                        f"Step {global_step}: kd_loss={kd_loss.item():.4f}  "
                        f"grad_norm={grad_norm:.4f}  "
                        f"z_score={z_score_avg:.3f}  "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

            # In-context-only generation (no logits WM): generate on train batch and log wandb table
            if (
                args.gen_log_steps > 0
                and global_step % args.gen_log_steps == 0
                and is_main
                and args.wandb
            ):
                with torch.no_grad():
                    unwrapped.eval()
                    max_pl = prompt_lens.max().item()
                    # Left-pad prompts to [B, max_pl]
                    prompt_input_ids = full_ids.new_full(
                        (B, max_pl), tokenizer.pad_token_id
                    )
                    prompt_attn = full_ids.new_zeros(B, max_pl)
                    for i in range(B):
                        pl = prompt_lens[i].item()
                        prompt_input_ids[i, -pl:] = full_ids[i, :pl]
                        prompt_attn[i, -pl:] = 1
                    gen_config = {
                        "max_new_tokens": args.max_new_tokens_gen,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "do_sample": True,
                        "top_p": args.top_p,
                        "temperature": 1.0,
                    }
                    gen_sequences_ic = unwrapped.generate(
                        prompt_input_ids,
                        attention_mask=prompt_attn,
                        **gen_config,
                    )
                    generated_ic = gen_sequences_ic[:, max_pl:]
                    non_pad_ic = generated_ic != tokenizer.pad_token_id
                    green_ic = green_list_mask.to(device)[generated_ic]
                    total_valid_ic = non_pad_ic.sum(dim=1).float().clamp(min=1)
                    green_valid_ic = (
                        green_ic.float() * non_pad_ic.float()
                    ).sum(dim=1)
                    var_r_ic = args.fraction * (1 - args.fraction) * total_valid_ic
                    z_score_ic = (
                        green_valid_ic - args.fraction * total_valid_ic
                    ) / torch.sqrt(var_r_ic)
                    gen_table = wandb.Table(
                        columns=["step", "prefix", "response", "z_score"]
                    )
                    prefixes = batch["prefix"]
                    for i in range(B):
                        resp = tokenizer.decode(
                            generated_ic[i].tolist(),
                            skip_special_tokens=True,
                        )
                        gen_table.add_data(
                            global_step,
                            prefixes[i] if i < len(prefixes) else "",
                            resp,
                            z_score_ic[i].item(),
                        )
                    wandb.log(
                        {"gen_incontext_only/samples": gen_table},
                        step=global_step,
                    )
                    wandb.log(
                        {"gen_incontext_only/z_score_mean": z_score_ic.mean().item()},
                        step=global_step,
                    )
                    unwrapped.train()

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                _save_model(
                    model_engine,
                    tokenizer,
                    os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                    is_main,
                )

        _save_model(
            model_engine,
            tokenizer,
            os.path.join(args.output_dir, f"epoch-{epoch + 1}"),
            is_main,
        )

    _save_model(
        model_engine,
        tokenizer,
        os.path.join(args.output_dir, "final"),
        is_main,
    )

    if is_main:
        logger.info("Offline KD training complete!")
        if args.wandb:
            wandb.finish()


def _save_model(engine, tokenizer, save_dir, is_main):
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        engine.module.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved checkpoint -> {save_dir}")
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline KD for in-context watermark (pre-generated responses)"
    )

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--yarn", action="store_true")
    parser.add_argument("--yarn_factor", type=float, default=4.0)
    parser.add_argument("--max_position_embeddings", type=int, default=131072)

    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--only_English", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument(
        "--english_token_loss",
        action="store_true",
        help="Compute KD loss only on English tokens",
    )
    parser.add_argument(
        "--rep_loss_weight",
        type=float,
        default=0.5,
        help="Weight for repetition penalty loss (penalize high prob on already-seen tokens). 0 = off.",
    )
    parser.add_argument(
        "--entropy_min_weight",
        type=float,
        default=0.0,
        help="Weight for entropy regularizer (encourage per-token entropy >= target). 0 = off.",
    )
    parser.add_argument(
        "--entropy_min_target",
        type=float,
        default=2.0,
        help="Target minimum entropy per token when entropy_min_weight > 0.",
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument(
        "--train_data",
        type=str,
        default="temp/incontext_add_logits_wm/strength2/Qwen-Qwen3-14B/Qwen-Qwen3-14B_frac_0.3_len_500_num_512_incontext_vllm_only_English.jsonl",
        help="Pre-generated jsonl: actual_prompt, prefix, gen_completion",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/kd_incontext_offline",
    )

    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument(
        "--gen_log_steps",
        type=int,
        default=0,
        help="Every N steps, generate with in-context prompt only (no logits WM) and log wandb table. 0 = disabled.",
    )
    parser.add_argument(
        "--max_new_tokens_gen",
        type=int,
        default=500,
        help="Max new tokens for in-context-only generation when gen_log_steps > 0.",
    )
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="kd-incontext-offline")
    parser.add_argument("--wandb_run_name", type=str, default="")

    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="configs/ds_zero2_config.json",
    )
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
