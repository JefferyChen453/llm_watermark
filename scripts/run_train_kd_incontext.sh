#!/bin/bash
# Knowledge Distillation training for in-context watermark.
#
# Prerequisites:
# 1. Generate teacher trajectories with run_generate_incontext_vllm.py --add_logits_wm
# 2. Use the output jsonl as --train_data
#
# For long prompts (~80k tokens with fraction=0.4):
#   - Use DeepSpeed ZeRO-3 with CPU offload
#   - Use gradient accumulation
#   - Use micro_batch_size=1
#   - Enable gradient checkpointing
#
# Example (single node, 8 GPUs with DeepSpeed):
#   deepspeed --num_gpus=8 run_train_kd_incontext.py \
#     --deepspeed --deepspeed_config configs/ds_zero3_config.json \
#     --train_data path/to/teacher_trajectories.jsonl \
#     --model_name Qwen/Qwen3-14B ...
#
# Example (torchrun without DeepSpeed):
#   torchrun --nproc_per_node=8 run_train_kd_incontext.py \
#     --train_data path/to/teacher_trajectories.jsonl ...

set -e

# Path to teacher trajectory data (from run_generate_incontext_vllm.py --add_logits_wm)
TEACHER_DATA="${TEACHER_DATA:-./temp/incontext_add_logits_wm/strength3/Qwen-Qwen3-14B/Qwen-Qwen3-14B_strength_3.0_frac_0.4_len_500_num_512_incontext_vllm.jsonl}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/kd_incontext}"

# Model
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"

# For ~80k token prompts: raise max_prompt_length, use small batch, gradient accumulation
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"   # Raise to 81920 for fraction=0.4 green list
MAX_LENGTH="${MAX_LENGTH:-8704}"            # max_prompt + max_completion
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

# Watermark params (must match teacher generation)
FRACTION="${FRACTION:-0.4}"
STRENGTH="${STRENGTH:-3.0}"
ONLY_ENGLISH="--only_English"

# Training
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-3}"
SAVE_STEPS="${SAVE_STEPS:-500}"

# W&B
WANDB="${WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-kd-incontext-watermark}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

# Memory optimizations
USE_DEEPSPEED="${USE_DEEPSPEED:-true}"
DS_CONFIG="${DS_CONFIG:-configs/ds_zero3_config.json}"
YARN="--yarn"
MAX_POS="${MAX_POS:-131072}"

# Number of GPUs
NPROC="${NPROC:-8}"

# Build optional W&B args
WANDB_ARGS=()
[ "$WANDB" = "true" ] && WANDB_ARGS=(--wandb --wandb_project "$WANDB_PROJECT")
[ -n "$WANDB_RUN_NAME" ] && WANDB_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")

if [ "$USE_DEEPSPEED" = "true" ]; then
    deepspeed --num_gpus="$NPROC" run_train_kd_incontext.py \
        --deepspeed \
        --deepspeed_config "$DS_CONFIG" \
        --model_name "$MODEL_NAME" \
        --train_data "$TEACHER_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --max_prompt_length "$MAX_PROMPT_LEN" \
        --max_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --fraction "$FRACTION" \
        --strength "$STRENGTH" \
        $ONLY_ENGLISH \
        --lr "$LR" \
        --num_epochs "$EPOCHS" \
        --save_steps "$SAVE_STEPS" \
        "${WANDB_ARGS[@]}" \
        $YARN \
        --max_position_embeddings "$MAX_POS" \
        --gradient_checkpointing \
        --flash_attn \
        "$@"
else
    torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" run_train_kd_incontext.py \
        --model_name "$MODEL_NAME" \
        --train_data "$TEACHER_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --max_prompt_length "$MAX_PROMPT_LEN" \
        --max_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --fraction "$FRACTION" \
        --strength "$STRENGTH" \
        $ONLY_ENGLISH \
        --lr "$LR" \
        --num_epochs "$EPOCHS" \
        --save_steps "$SAVE_STEPS" \
        "${WANDB_ARGS[@]}" \
        $YARN \
        --max_position_embeddings "$MAX_POS" \
        --gradient_checkpointing \
        --flash_attn \
        "$@"
fi
