#!/usr/bin/env bash
# Full post-train autopilot for RL Stage 2 v3 run (gznyxqkj).
# Designed to survive SSH / Claude Code disconnect via nohup.
#
# Flow:
#   PHASE_1_WAIT_TRAIN_EXIT — wait for recipe.watermark_rl_ray.main process to exit
#   PHASE_2_WAIT_CKPT       — wait for global_step_1000/actor/ (epoch-2 save)
#   PHASE_3_MERGE           — FSDP -> HF for step_500 and step_1000
#   PHASE_4_SWEEP_RAY       — kill stray Ray workers so GPUs are free
#   PHASE_5_EVAL            — green test_477 x 5 fractions + IFEval for both steps
#   PHASE_6_EVAL_DONE_AWAITING_REPORT — status file set; Claude picks up for report
#   PHASE_7_LAUNCH_V5B_ALIGNED — after eval done, launch v5b-aligned retrain (user task 2).
#                                Skip if /tmp/skip_v5b_aligned exists.
#
# Launch:
#   nohup bash /home/tianyichen/llm_watermark/scripts/autopilot_rl_stage2_full.sh \
#         > /home/tianyichen/llm_watermark/logs/autopilot_full.nohup.log 2>&1 &
#   disown
set -u

LOG_DIR=/home/tianyichen/llm_watermark/logs
LOG_FILE=$LOG_DIR/autopilot_rl_stage2_full.log
STATUS_FILE=$LOG_DIR/autopilot_rl_stage2_full.status
PID_FILE=$LOG_DIR/autopilot_rl_stage2_full.pid

RUN_TAG=rl_2task_discard_v5binit_grpo_202604210816
CKPT_ROOT=/home/tianyichen/llm_watermark/verl/checkpoints/watermark-rl-ray/${RUN_TAG}

MERGE_SCRIPT=/home/tianyichen/llm_watermark/scripts/merge_rl_ckpts.sh
EVAL_SCRIPT=/home/tianyichen/llm_watermark/scripts/eval_rl_stage2_full.sh
V5B_ALIGNED_SCRIPT=/home/tianyichen/llm_watermark/verl/recipe/watermark_kd_ray/scripts/run_train_v5b_aligned_eval.sh
EVAL_LOG=$LOG_DIR/eval_rl_stage2_full.log
MERGE_LOG=$LOG_DIR/merge_rl_ckpts.log
V5B_ALIGNED_LOG_PREFIX=$LOG_DIR/v5b_aligned_eval_autopilot

mkdir -p "$LOG_DIR"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG_FILE"; }
status() { echo "$1" > "$STATUS_FILE"; log "STATUS → $1"; }

# Singleton guard
if [ -f "$PID_FILE" ]; then
    OLD=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "${OLD:-}" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "autopilot-full already running as pid $OLD — abort" | tee -a "$LOG_FILE"
        exit 1
    fi
fi
echo $$ > "$PID_FILE"
trap "rm -f $PID_FILE" EXIT

log "================================================================"
log "autopilot-full started, pid=$$ ppid=$PPID"

# ------------ Phase 1: wait for training main to exit ------------
status "PHASE_1_WAIT_TRAIN_EXIT"
while true; do
    if ! pgrep -f "recipe.watermark_rl_ray.main" > /dev/null; then
        log "train process not alive; proceeding"
        break
    fi
    # Log current step from train log (best-effort)
    LATEST_STEP=$(grep -oE 'Epoch [0-9]+/[0-9]+: +[0-9]+%.*[0-9]+/500' \
        /home/tianyichen/llm_watermark/verl/logs/${RUN_TAG}.log 2>/dev/null \
        | tail -1 | grep -oE '[0-9]+/500$' | head -1)
    log "train still alive, latest: ${LATEST_STEP:-?}"
    sleep 120
done

# ------------ Phase 2: wait for step_1000/actor to appear ------------
status "PHASE_2_WAIT_CKPT"
CKPT_1000=${CKPT_ROOT}/global_step_1000/actor
for i in $(seq 1 30); do
    if [ -d "$CKPT_1000" ] && [ "$(ls "$CKPT_1000"/model_world_size_8_rank_*.pt 2>/dev/null | wc -l)" -ge 8 ]; then
        log "step_1000 actor shards all present (8 ranks)"
        break
    fi
    log "step_1000 shards not ready yet (attempt $i/30)"
    sleep 30
done

if [ ! -d "$CKPT_1000" ]; then
    status "ERROR_NO_STEP1000"
    log "ABORT: step_1000 actor dir missing after waiting"
    exit 1
fi

# ------------ Phase 3: FSDP -> HF merge ------------
status "PHASE_3_MERGE"
log "running $MERGE_SCRIPT"
bash "$MERGE_SCRIPT" > "$MERGE_LOG" 2>&1
MC=$?
log "merge rc=$MC"
if [ $MC -ne 0 ]; then
    status "ERROR_MERGE_FAILED_RC_$MC"
    exit $MC
fi

# ------------ Phase 4: sweep stray Ray workers so GPUs free ------------
status "PHASE_4_SWEEP_RAY"
log "sweeping ray workers"
pkill -9 -f "recipe.watermark_rl_ray" 2>/dev/null || true
pkill -9 -f "ray::TaskRunner\|ray::WorkerDict" 2>/dev/null || true
pkill -9 -f "ray/_private/workers/default_worker" 2>/dev/null || true
sleep 15
log "post-sweep GPU state:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>&1 | tee -a "$LOG_FILE"

# ------------ Phase 5: eval pipeline ------------
status "PHASE_5_EVAL"
log "running $EVAL_SCRIPT"
cd /home/tianyichen/llm_watermark
bash "$EVAL_SCRIPT" > "$EVAL_LOG" 2>&1
EC=$?
log "eval rc=$EC"
if [ $EC -ne 0 ]; then
    status "ERROR_EVAL_FAILED_RC_$EC"
    exit $EC
fi

# ------------ Phase 6: eval done, report pending (Claude picks up) ------------
status "PHASE_6_EVAL_DONE_AWAITING_REPORT"
log "eval artifacts:"
log "  watermark evals: outputs/incontext_eval/prompt_v2_new/${RUN_TAG}_global_step_{500,1000}/"
log "  ifeval dirs:     outputs/ifeval_nothink/results/.../${RUN_TAG}/global_step_{500,1000}/hf_model/"

# ------------ Phase 7: launch v5b-aligned retrain (user task 2) ------------
if [ -f /tmp/skip_v5b_aligned ]; then
    log "/tmp/skip_v5b_aligned exists → skip v5b retrain launch"
    exit 0
fi

status "PHASE_7_LAUNCH_V5B_ALIGNED"
log "sweeping any leftover processes before retrain launch"
pkill -9 -f "recipe.watermark_kd_ray.main" 2>/dev/null || true
pkill -9 -f "ray::TaskRunner\|ray::WorkerDict" 2>/dev/null || true
sleep 10
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>&1 | tee -a "$LOG_FILE"

V5B_TS=$(date +%Y%m%d%H%M)
V5B_NOHUP_LOG=${V5B_ALIGNED_LOG_PREFIX}.${V5B_TS}.nohup.log
log "launching v5b-aligned retrain; nohup log: $V5B_NOHUP_LOG"
nohup bash "$V5B_ALIGNED_SCRIPT" > "$V5B_NOHUP_LOG" 2>&1 &
disown
V5B_PID=$!
sleep 5
if kill -0 "$V5B_PID" 2>/dev/null; then
    log "v5b-aligned retrain launched, pid=$V5B_PID"
    echo "$V5B_PID" > "$LOG_DIR/v5b_aligned_eval.pid"
    status "PHASE_7_V5B_ALIGNED_RUNNING"
else
    log "v5b-aligned retrain failed to start — see $V5B_NOHUP_LOG"
    status "ERROR_V5B_LAUNCH_FAILED"
    exit 1
fi
