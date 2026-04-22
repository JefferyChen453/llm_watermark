#!/bin/bash
# Autopilot for RL Stage 2 Epoch 1: wait ckpt → kill train → run eval.
# Designed to survive Claude Code / SSH disconnect by detaching via nohup.
#
# Launch:
#   nohup bash /home/tianyichen/llm_watermark/scripts/autopilot_rl_stage2_epoch1.sh \
#       > /home/tianyichen/llm_watermark/logs/autopilot.nohup.log 2>&1 &
#   disown
#
# Inspect from any shell:
#   cat  /home/tianyichen/llm_watermark/logs/autopilot_rl_stage2_epoch1.status
#   tail -50 /home/tianyichen/llm_watermark/logs/autopilot_rl_stage2_epoch1.log
#
# Idempotent: safe to re-launch if previous instance died mid-pipeline.

set -u

LOG_DIR=/home/tianyichen/llm_watermark/logs
LOG_FILE=$LOG_DIR/autopilot_rl_stage2_epoch1.log
STATUS_FILE=$LOG_DIR/autopilot_rl_stage2_epoch1.status
PID_FILE=$LOG_DIR/autopilot_rl_stage2_epoch1.pid

CKPT_ROOT=/home/tianyichen/llm_watermark/verl/checkpoints/watermark-rl-ray/rl_2task_discard_v5binit_grpo_202604210816
CKPT_DIR=$CKPT_ROOT/global_step_500/hf_model
INDEX=$CKPT_DIR/model.safetensors.index.json
TRAIN_LOG=/home/tianyichen/llm_watermark/verl/logs/rl_2task_discard_v5binit_grpo_202604210816.log
EVAL_SCRIPT=/home/tianyichen/llm_watermark/scripts/eval_rl_stage2_epoch1.sh
EVAL_LOG=$LOG_DIR/eval_rl_stage2_epoch1.log

mkdir -p "$LOG_DIR"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG_FILE"; }
status() { echo "$1" > "$STATUS_FILE"; log "STATUS → $1"; }

# Singleton guard: bail out if another autopilot alive
if [ -f "$PID_FILE" ]; then
    OLD=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "autopilot already running as pid $OLD — aborting" | tee -a "$LOG_FILE"
        exit 1
    fi
fi
echo $$ > "$PID_FILE"
trap "rm -f $PID_FILE" EXIT

log "================================================================"
log "autopilot started, pid=$$ ppid=$PPID"
log "CKPT_DIR=$CKPT_DIR"

# ------------ Phase 1: wait for ckpt ------------
status "PHASE_1_WAIT_CKPT"
POLL_COUNT=0
while [ ! -f "$INDEX" ]; do
    STEP=$(grep -oE '\| [0-9]+/500' "$TRAIN_LOG" 2>/dev/null | tail -1 | tr -d ' |/' )
    POLL_COUNT=$((POLL_COUNT + 1))
    # Log every poll but only status-file refresh every 6 polls (≈30 min)
    log "poll #$POLL_COUNT: train step=${STEP:-?} / 500"
    sleep 300
done

log "ckpt index file appeared; flushing 90s for remaining shards"
sleep 90

SHARDS=$(ls "$CKPT_DIR"/model-*.safetensors 2>/dev/null | wc -l)
log "shard count: $SHARDS (expected 6)"
log "$(ls -la "$CKPT_DIR" 2>&1)"
if [ "$SHARDS" -lt 6 ]; then
    status "ERROR_SHARDS_MISSING"
    log "ABORT: only $SHARDS shards, expected 6"
    exit 1
fi

# ------------ Phase 2: kill training ------------
status "PHASE_2_KILL_TRAIN"
TRAIN_PID=$(pgrep -f "recipe.watermark_rl_ray.main" | head -1 || true)
if [ -n "${TRAIN_PID:-}" ]; then
    log "killing training pid=$TRAIN_PID (SIGTERM)"
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    for i in $(seq 1 20); do
        if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
            log "training pid $TRAIN_PID exited after ${i}0s"
            break
        fi
        sleep 10
    done
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        log "still alive after 200s — SIGKILL"
        kill -KILL "$TRAIN_PID" 2>/dev/null || true
        sleep 5
    fi
    # Sweep stray Ray workers
    log "sweeping leftover ray workers"
    pkill -9 -f "recipe.watermark_rl_ray" 2>/dev/null || true
    pkill -9 -f "ray::TaskRunner\|ray::WorkerDict" 2>/dev/null || true
    sleep 10
else
    log "training pid not found (already exited?)"
fi

# GPU free check
log "post-kill GPU state:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>&1 | tee -a "$LOG_FILE"

# ------------ Phase 3: run eval pipeline ------------
status "PHASE_3_EVAL"
log "launching eval pipeline: $EVAL_SCRIPT"
cd /home/tianyichen/llm_watermark
bash "$EVAL_SCRIPT" > "$EVAL_LOG" 2>&1
EVAL_RC=$?
log "eval pipeline exit rc=$EVAL_RC"
if [ $EVAL_RC -ne 0 ]; then
    status "ERROR_EVAL_FAILED_RC_$EVAL_RC"
    exit $EVAL_RC
fi

# ------------ Phase 4: done, awaiting human ------------
status "PHASE_4_EVAL_DONE_AWAITING_REPORT"
log "all automated phases complete; status file set for Claude to pick up"
log "final output paths:"
log "  - watermark eval csv:     /home/tianyichen/llm_watermark/outputs/incontext_eval/prompt_v2_new/rl_2task_discard_v5binit_grpo_202604210816_global_step_500/evaluation.csv"
log "  - IFEval dir:             /home/tianyichen/llm_watermark/outputs/ifeval_nothink/results/.../rl_2task_discard_v5binit_grpo_202604210816/global_step_500/hf_model/"
