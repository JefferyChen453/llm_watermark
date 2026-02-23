#!/bin/bash
# Offline Knowledge Distillation for in-context watermark.
#
# Reads pre-generated responses from a jsonl (incontext prompt + response,
# e.g. from vLLM). No generation during training: forward → lm_head on
# generated positions → teacher = student + strength*green_mask → KD.
#
# Example:
#   bash scripts/run_train_kd_incontext_offline.sh
#
# Override any variable on the command line:
#   NPROC=4 STRENGTH=2.0 TRAIN_DATA=/path/to/your.jsonl bash scripts/run_train_kd_incontext_offline.sh
#
# Pass extra flags to the Python script via positional args:
#   bash scripts/run_train_kd_incontext_offline.sh --num_epochs 5 --wandb

set -e

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_DATA="${TRAIN_DATA:-/home/tianyichen/llm_watermark/temp/kd_train/strength2/Qwen-Qwen3-14B/Qwen-Qwen3-14B_strength_2.0_frac_0.1_len_500_num_None_incontext_vllm_only_English.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/kd_incontext_offline}"
DS_CONFIG="${DS_CONFIG:-configs/ds_zero2_config.json}"

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"

# ── Watermark (must match the settings used when generating the jsonl) ───────
FRACTION="${FRACTION:-0.1}"
STRENGTH="${STRENGTH:-2.0}"
WM_KEY="${WM_KEY:-0}"

# ── Training ─────────────────────────────────────────────────────────────────
NPROC="${NPROC:-8}"
LR="${LR:-1e-5}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
KD_TEMP="${KD_TEMP:-1.0}"
REP_LOSS_WEIGHT="${REP_LOSS_WEIGHT:-0.5}"
ENTROPY_MIN_WEIGHT="${ENTROPY_MIN_WEIGHT:-1.0}"
ENTROPY_MIN_TARGET="${ENTROPY_MIN_TARGET:-2.0}"

# ── Logging / saving ────────────────────────────────────────────────────────
LOG_STEPS="${LOG_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
GEN_LOG_STEPS="${GEN_LOG_STEPS:-10}"

# ── YaRN (for long-context models) ──────────────────────────────────────────
MAX_POS="${MAX_POS:-131072}"
YARN_FACTOR="${YARN_FACTOR:-4.0}"

# ── Launch ───────────────────────────────────────────────────────────────────
CUDA_LAUNCH_BLOCKING=1 deepspeed --num_gpus="$NPROC" train_kd_incontext_offline.py \
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
    --log_steps "$LOG_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --gen_log_steps "$GEN_LOG_STEPS" \
    --gradient_checkpointing \
    --flash_attn \
    --yarn \
    --yarn_factor "$YARN_FACTOR" \
    --max_position_embeddings "$MAX_POS" \
    --rep_loss_weight "$REP_LOSS_WEIGHT" \
    --entropy_min_weight "$ENTROPY_MIN_WEIGHT" \
    --entropy_min_target "$ENTROPY_MIN_TARGET" \
    --wandb \
    --wandb_project "kd-incontext-watermark" \
    --wandb_run_name "Fraction${FRACTION}-Strength${STRENGTH}-Key${WM_KEY}-Offline-w/RepLoss-w/1.0Entropy" \
    "$@"
