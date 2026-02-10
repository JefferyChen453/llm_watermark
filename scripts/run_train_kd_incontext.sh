#!/bin/bash
# Knowledge Distillation training for in-context watermark.
#
# This script launches run_train_kd_incontext.py with DeepSpeed ZeRO-2.
# The training generates watermarked responses on-the-fly (no pre-generated
# trajectories needed) and distils the watermark bias into the model.
#
# Memory budget for ~80k-token prompts (fraction=0.4):
#   - DeepSpeed ZeRO-2 with optimizer CPU offload
#   - Gradient checkpointing  (saves ~30 GB of activation memory)
#   - Flash Attention 2        (O(n) instead of O(n^2) attention memory)
#   - micro_batch_size = 1     (one sample per GPU per step)
#   - Gradient accumulation    (effective batch = NPROC * GRAD_ACCUM)
#
# Example:
#   bash scripts/run_train_kd_incontext.sh
#
# Override any variable on the command line:
#   NPROC=4 STRENGTH=2.0 bash scripts/run_train_kd_incontext.sh
#
# Pass extra flags to the Python script via positional args:
#   bash scripts/run_train_kd_incontext.sh --num_samples 64 --wandb

set -e

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_DATA="${TRAIN_DATA:-./UnigramWatermark/data/LFQA/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/kd_incontext}"
DS_CONFIG="${DS_CONFIG:-configs/ds_zero2_config.json}"

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"

# ── Watermark (must match detection settings later) ──────────────────────────
FRACTION="${FRACTION:-0.1}"
STRENGTH="${STRENGTH:-2.0}"
WM_KEY="${WM_KEY:-0}"

# ── Training ─────────────────────────────────────────────────────────────────
NPROC="${NPROC:-8}"
LR="${LR:-5e-6}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-500}"
KD_TEMP="${KD_TEMP:-1.0}"

# ── Logging / saving ────────────────────────────────────────────────────────
LOG_STEPS="${LOG_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"

# ── YaRN (for long-context models) ──────────────────────────────────────────
MAX_POS="${MAX_POS:-131072}"
YARN_FACTOR="${YARN_FACTOR:-4.0}"

# ── Launch ───────────────────────────────────────────────────────────────────
CUDA_LAUNCH_BLOCKING=1 deepspeed --num_gpus="$NPROC" run_train_kd_incontext.py \
    --deepspeed_config "$DS_CONFIG" \
    --model_name "$MODEL_NAME" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --fraction "$FRACTION" \
    --strength "$STRENGTH" \
    --wm_key "$WM_KEY" \
    --only_English \
    --lr "$LR" \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --english_token_loss \
    --kd_temperature "$KD_TEMP" \
    --num_epochs "$EPOCHS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --log_steps "$LOG_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --gradient_checkpointing \
    --flash_attn \
    --yarn \
    --yarn_factor "$YARN_FACTOR" \
    --max_position_embeddings "$MAX_POS" \
    --wandb \
    --wandb_project "kd-incontext-watermark" \
    --wandb_run_name "Fraction${FRACTION}-Strength${STRENGTH}-Key${WM_KEY}" \
    "$@"
