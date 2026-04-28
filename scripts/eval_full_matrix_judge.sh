#!/usr/bin/env bash
# Run LLM-Judge for all 7 ckpts × 7 cells × 100 samples.
# Pre-req: tools/prep_full_matrix_judge.py has been run successfully.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY=${PY:-${ROOT}/.venv/bin/python}
QE_ROOT=${ROOT}/outputs/quality_eval/full_matrix

declare -a TAGS=(
  "baseline_qwen3-14b"
  "v5b_green3379+initials865+neg1000_dualKL_biasedRefTopK1000_202604180857_global_step_655"
  "v5b_aligned_eval_202604240212_global_step_655"
  "v5b_aligned_eval_202604240212_global_step_1310"
  "v5b_aligned_eval_202604240212_global_step_1965"
  "rl_2task_discard_v5binit_grpo_202604210816_global_step_500"
  "rl_2task_discard_v5binit_grpo_202604210816_global_step_1000"
)

source ~/.env
[ -z "$OPENAI_API_KEY" ] && { echo "[fatal] OPENAI_API_KEY not set"; exit 1; }

for TAG in "${TAGS[@]}"; do
  CELLS_FILE=${QE_ROOT}/cells_${TAG}.json
  JUDGE_DIR=${QE_ROOT}/judge/${TAG}
  if [ ! -f "$CELLS_FILE" ]; then
    echo "[skip $TAG] no cells file at $CELLS_FILE — run prep first"
    continue
  fi
  if [ -f "${JUDGE_DIR}/all_cells_summary.json" ]; then
    echo "[skip $TAG] judge already done at $JUDGE_DIR"
    continue
  fi
  echo "========== judge $TAG =========="
  "$PY" run_llm_judge.py \
    --cells-file "$CELLS_FILE" \
    --output-dir "$JUDGE_DIR" \
    --n-samples 100 \
    --model gpt-4o-mini \
    --concurrency 32 || echo "[error judge $TAG]"
done

echo "[$(date)] all judge done."
