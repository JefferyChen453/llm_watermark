#!/usr/bin/env bash
# Post-master orchestrator for the icw-day4 eval matrix.
# Runs AFTER eval_full_matrix.sh finishes. Sequence:
#   1. Detect (re-run; already-done cells skip)
#   2. Re-gen v5b_old initials (the only ckpt-cell that was killed mid-run)
#   3. Detect again (catch the redo'd v5b_old initials)
#   4. evaluate.py per ckpt → evaluation.csv
#   5. IFEval for v5b_aligned 3 ckpts (other 4 already evaluated)
#   6. LLM-Judge prep + run
#   7. Aggregate → summary.json

set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY=${PY:-${ROOT}/.venv/bin/python}

log() { echo "[$(date)] $*"; }

# ---- 1. Detect (round 1) ----
log "=== detect round 1 ==="
bash "${SCRIPT_DIR}/eval_full_matrix_detect.sh" 2>&1 | tail -30

# ---- 2. Redo v5b_old initials ----
log "=== redo v5b_old initials ==="
bash "${SCRIPT_DIR}/eval_redo_v5b_old_initials.sh" 2>&1 | tail -10

# ---- 3. Detect again (now v5b_old initials too) ----
log "=== detect round 2 ==="
bash "${SCRIPT_DIR}/eval_full_matrix_detect.sh" 2>&1 | tail -30

# ---- 4. evaluate.py per ckpt → evaluation.csv ----
log "=== evaluate.py per ckpt ==="
declare -a TAGS=(
  "baseline_qwen3-14b"
  "v5b_green3379+initials865+neg1000_dualKL_biasedRefTopK1000_202604180857_global_step_655"
  "v5b_aligned_eval_202604240212_global_step_655"
  "v5b_aligned_eval_202604240212_global_step_1310"
  "v5b_aligned_eval_202604240212_global_step_1965"
  "rl_2task_discard_v5binit_grpo_202604210816_global_step_500"
  "rl_2task_discard_v5binit_grpo_202604210816_global_step_1000"
)
for TAG in "${TAGS[@]}"; do
  D=${ROOT}/outputs/incontext_eval/prompt_v2_new/${TAG}
  if [ -d "$D" ]; then
    log "  evaluate.py $TAG"
    "$PY" evaluate.py "$D" --fraction_or_strength fraction --target_fpr 0.01 2>&1 | tail -3 || log "  [warn] evaluate failed for $TAG"
  fi
done

# ---- 5. IFEval ----
log "=== IFEval (v5b_aligned 3 ckpts) ==="
bash "${SCRIPT_DIR}/eval_full_matrix_ifeval.sh" 2>&1 | tail -10

# ---- 6. LLM-Judge ----
log "=== Judge prep ==="
"$PY" "${ROOT}/tools/prep_full_matrix_judge.py" 2>&1 | tail -20

log "=== Judge run ==="
bash "${SCRIPT_DIR}/eval_full_matrix_judge.sh" 2>&1 | tail -30

# ---- 7. Aggregate ----
log "=== Aggregate summary ==="
"$PY" "${ROOT}/tools/aggregate_full_matrix.py" 2>&1 | tail -10

log "=== ALL POST-EVAL DONE ==="
