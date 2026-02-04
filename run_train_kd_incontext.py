"""
Knowledge Distillation training for in-context watermark.

Teacher: model logits + strength * green_mask (watermark-guided distribution)
Student: model logits on (prompt + teacher_trajectory)
Goal: Train student to match teacher distribution so model naturally outputs green tokens.

Supports DeepSpeed ZeRO-2/3 and FSDP for memory optimization on long prompts (~80k tokens).
"""
import argparse
import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer

from dataset_kd_incontext import InContextWatermarkKDDataset, collate_kd_batch
from gptwm import GPTWatermarkBase


def _save_checkpoint(model, tokenizer, output_dir, name, use_deepspeed=False):
    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    if use_deepspeed and hasattr(model, "save_16bit_model"):
        model.save_16bit_model(out_dir, "pytorch_model.bin")
    else:
        model_to_save = model.module if hasattr(model, "module") else model
        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def compute_kd_loss(
    student_logits: torch.Tensor,
    green_list_mask: torch.Tensor,
    strength: float,
    loss_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    KL divergence loss: student learns to match teacher (watermark-boosted) distribution.
    Teacher logits = student_logits + strength * green_mask
    loss_mask: 1 at positions where we compute loss (predicting completion tokens).
    """
    teacher_logits = student_logits + strength * green_list_mask.to(student_logits.device)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl_per_token = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
        log_target=False,
    ).sum(dim=-1)

    min_len = min(kl_per_token.shape[-1], loss_mask.shape[-1])
    kl_per_token = kl_per_token[..., :min_len]
    loss_mask = loss_mask[..., :min_len]

    masked_kl = kl_per_token * loss_mask
    num_valid = loss_mask.sum().clamp(min=1e-8)
    loss = (masked_kl.sum() / num_valid) * (temperature**2)
    return loss


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")

    # W&B (only rank 0)
    use_wandb = args.wandb and local_rank == 0
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.output_dir,
        )

    # Tokenizer
    if "decapoda-research-llama" in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Watermark / green list
    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    wm_base = GPTWatermarkBase(
        fraction=args.fraction,
        strength=args.strength,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=model_config.vocab_size,
        watermark_key=args.wm_key,
        only_English=args.only_English,
        tokenizer=tokenizer,
    )
    green_list_mask = wm_base.green_list_mask.unsqueeze(0)  # (1, vocab_size)

    # Dataset
    dataset = InContextWatermarkKDDataset(
        data_paths=args.train_data,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        truncation_prompt=args.truncation_prompt,
        max_samples=args.max_samples,
    )

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    def _collate(batch):
        return collate_kd_batch(batch, tokenizer.pad_token_id, args.max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=_collate,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if args.yarn:
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": 32768,
        }
        config.max_position_embeddings = args.max_position_embeddings

    attn_impl = "flash_attention_2" if args.flash_attn else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    use_deepspeed = args.deepspeed
    optimizer = None
    if use_deepspeed:
        import deepspeed
        ds_config_path = args.deepspeed_config or os.path.join(
            os.path.dirname(__file__), "configs", "ds_zero3_config.json"
        )
        with open(ds_config_path) as f:
            import json
            ds_config = json.load(f)
        ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
        ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        ds_config["gradient_clipping"] = args.max_grad_norm
        if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
            ds_config["optimizer"]["params"]["lr"] = args.lr
            ds_config["optimizer"]["params"]["weight_decay"] = args.weight_decay
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        model = model_engine
    else:
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(dataloader)) if optimizer else None

    model.train()
    global_step = 0

    for epoch in range(args.num_epochs):
        if sampler:
            sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(local_rank != 0))
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits  # (B, T, V)

            # Next-token prediction: logits[:, i, :] predicts input_ids[:, i+1]
            # loss_mask[i]=1 when we predict completion token at position i+1
            student_logits = logits[:, :-1, :].contiguous()
            loss_mask_1 = loss_mask[:, :-1].contiguous()  # align: loss_mask for logits positions

            loss = compute_kd_loss(
                student_logits,
                green_list_mask,
                args.strength,
                loss_mask_1,
                temperature=args.kd_temperature,
            )

            if use_deepspeed:
                model.backward(loss)
                model.step()
            else:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if scheduler:
                    scheduler.step()

            global_step += 1
            if local_rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                if use_wandb:
                    log_dict = {"train/loss": loss.item(), "train/global_step": global_step}
                    if use_deepspeed and hasattr(model, "get_lr"):
                        log_dict["train/lr"] = model.get_lr()[0]
                    elif not use_deepspeed and scheduler is not None:
                        log_dict["train/lr"] = scheduler.get_last_lr()[0]
                    wandb.log(log_dict, step=global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0 and local_rank == 0:
                _save_checkpoint(model, tokenizer, args.output_dir, f"checkpoint-{global_step}", use_deepspeed)

    if local_rank == 0:
        _save_checkpoint(model, tokenizer, args.output_dir, "final", use_deepspeed)
        print(f"Saved final model to {os.path.join(args.output_dir, 'final')}")
        if use_wandb:
            wandb.finish()

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--train_data", type=str, nargs="+", required=True,
                        help="Path(s) to teacher trajectory jsonl (from run_generate_incontext_vllm.py --add_logits_wm)")
    parser.add_argument("--output_dir", type=str, default="./outputs/kd_incontext")

    parser.add_argument("--max_prompt_length", type=int, default=8192,
                        help="Max prompt tokens (raise for ~80k with fraction=0.4)")
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=8704,
                        help="Max total sequence length for collate")
    parser.add_argument("--truncation_prompt", type=str, default="left")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=-1)

    parser.add_argument("--fraction", type=float, default=0.4)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--only_English", action="store_true")
    parser.add_argument("--kd_temperature", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--yarn", action="store_true")
    parser.add_argument("--yarn_factor", type=float, default=4.0)
    parser.add_argument("--max_position_embeddings", type=int, default=131072)

    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--flash_attn", action="store_true", default=True)
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed ZeRO")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to DeepSpeed config (default: configs/ds_zero3_config.json)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="kd-incontext-watermark")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (default: auto)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
