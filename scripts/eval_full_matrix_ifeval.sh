#!/usr/bin/env bash
# Run IFEval for the 3 v5b_aligned ckpts (other 4 already have results).
# Must run AFTER eval_full_matrix.sh finishes (vLLM GPU contention).

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

declare -a CKPTS=(
  "${ROOT}/verl/checkpoints/watermark-kd-ray/v5b_aligned_eval_202604240212/global_step_655/hf_model"
  "${ROOT}/verl/checkpoints/watermark-kd-ray/v5b_aligned_eval_202604240212/global_step_1310/hf_model"
  "${ROOT}/verl/checkpoints/watermark-kd-ray/v5b_aligned_eval_202604240212/global_step_1965/hf_model"
)

for MODEL in "${CKPTS[@]}"; do
  echo "========== IFEval: $MODEL =========="
  # Check if already evaluated (look for any results_*.json under that ckpt)
  RESULTS_DIR=${ROOT}/outputs/ifeval_nothink/results${MODEL}
  if [ -d "$RESULTS_DIR" ] && ls "$RESULTS_DIR"/results_*.json 2>/dev/null | head -1 > /dev/null; then
    echo "[skip] results already present at $RESULTS_DIR"
    continue
  fi
  echo "[run] lighteval vllm on $MODEL"
  # Use full venv path — lighteval may not be on PATH in the post-eval shell.
  ${ROOT}/.venv/bin/lighteval vllm \
    "model_name=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.9" \
    "ifeval|0" \
    --output-dir "$ROOT/outputs/ifeval_nothink" \
    --save-details || echo "[error] ifeval failed for $MODEL"
done

echo "[$(date)] IFEval done."
