#!/usr/bin/env bash
# Merge RL Stage 2 v3 (gznyxqkj) FSDP actor shards -> HF model.
# Produces <ckpt_root>/global_step_<N>/hf_model/ for each STEP.
#
# Usage:
#   bash scripts/merge_rl_ckpts.sh               # merge default STEPS
#   STEPS="500 1000" bash scripts/merge_rl_ckpts.sh
#
# Safe to re-run: if target hf_model/ already has model.safetensors.index.json
# and all shards, we skip.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT/verl"

RUN_TAG=rl_2task_discard_v5binit_grpo_202604210816
CKPT_ROOT=${PROJECT_ROOT}/verl/checkpoints/watermark-rl-ray/${RUN_TAG}

STEPS=${STEPS:-"500 1000"}
PY=${PY:-${PROJECT_ROOT}/.venv/bin/python}

log() { echo "[$(date -u +%FT%TZ)] $*"; }

for step in ${STEPS}; do
    ACTOR_DIR=${CKPT_ROOT}/global_step_${step}/actor
    HF_DIR=${CKPT_ROOT}/global_step_${step}/hf_model
    INDEX=${HF_DIR}/model.safetensors.index.json

    if [ ! -d "$ACTOR_DIR" ]; then
        log "SKIP step=${step}: actor dir not found at $ACTOR_DIR"
        continue
    fi

    SHARDS=$(ls "$HF_DIR"/model-*.safetensors 2>/dev/null | wc -l || true)
    if [ -f "$INDEX" ] && [ "$SHARDS" -ge 6 ]; then
        log "SKIP step=${step}: hf_model already has index + $SHARDS shards"
        continue
    fi

    log "MERGE step=${step}: $ACTOR_DIR -> $HF_DIR"
    "${PY}" -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$ACTOR_DIR" \
        --target_dir "$HF_DIR"

    SHARDS_OUT=$(ls "$HF_DIR"/model-*.safetensors 2>/dev/null | wc -l)
    log "DONE step=${step}: $SHARDS_OUT shards written"
done

log "all merges done"
