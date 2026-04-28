#!/usr/bin/env bash
# Fixup orchestrator: re-run IFEval (which failed in main post-eval) and the
# Judge for v5b_aligned 3 ckpts' ifeval cell, then re-aggregate.
#
# Pre-req: lighteval works (antlr4-python3-runtime==4.13.2 installed).

set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY=${PY:-${ROOT}/.venv/bin/python}
log() { echo "[$(date)] $*"; }

# 1. Re-run IFEval
log "=== fixup: IFEval ==="
bash "${SCRIPT_DIR}/eval_full_matrix_ifeval.sh" 2>&1 | tail -30

# 2. Re-prep cells (now v5b_aligned ckpts will have ifeval cell)
log "=== fixup: Judge prep (with ifeval) ==="
"$PY" "${ROOT}/tools/prep_full_matrix_judge.py" 2>&1 | tail -10

# 3. Re-run Judge — only NEW cells (ifeval for v5b_aligned 3 ckpts) will be processed
# (existing cells skip via run_llm_judge.py's summary-exists check)
log "=== fixup: Judge run (ifeval cells for v5b_aligned) ==="
bash "${SCRIPT_DIR}/eval_full_matrix_judge.sh" 2>&1 | tail -30

# 4. Re-aggregate
log "=== fixup: Aggregate ==="
"$PY" "${ROOT}/tools/aggregate_full_matrix.py" 2>&1 | tail -3

log "=== FIXUP DONE ==="
