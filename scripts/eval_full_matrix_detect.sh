#!/usr/bin/env bash
# Run detection for all generated cells from eval_full_matrix.sh.
#  - green: run_detect.py per (ckpt, frac)
#  - initials: run_detect_initials.py per ckpt (seed=42)
#  - acrostics: analyze_acrostics_paper_dts.py per ckpt (paired with shared neg)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY=${PY:-${ROOT}/.venv/bin/python}
GREEN_FRACTIONS=(0.0 0.1 0.2 0.3 0.4)
INITIALS_SEED=42
ACROSTICS_TARGET="WATERMARKS"
ACROSTICS_NEG_FILE=${ROOT}/outputs/acrostics_eval/negatives_lfqa_gold_target_${ACROSTICS_TARGET}.jsonl

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
  echo "========== detect $TAG =========="

  # ----- Green detect -----
  GDIR=${ROOT}/outputs/incontext_eval/prompt_v2_new/${TAG}
  for f in "${GREEN_FRACTIONS[@]}"; do
    INF=$(ls ${GDIR}/*_frac_${f}_*_only_English.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
    if [ -z "$INF" ]; then
      echo "  [warn] green f=${f} input not found, skip"
      continue
    fi
    OUTZ=${INF%.jsonl}_z.jsonl
    if [ -f "$OUTZ" ]; then
      echo "  [skip green f=${f}] z file exists"
      continue
    fi
    echo "  [detect green f=${f}] $INF"
    "$PY" run_detect.py \
      --input_file "$INF" \
      --model_name "Qwen/Qwen3-14B" \
      --fraction "$f" \
      --workers 8 \
      --only_English \
      --combine_fraction \
      --use_generated_neg_data || echo "  [error green f=${f}]"
  done

  # ----- Initials detect -----
  IDIR=${ROOT}/outputs/initials_eval/${TAG}
  if [ -d "$IDIR" ]; then
    INF=$(ls ${IDIR}/*_initials_seed_${INITIALS_SEED}_*.jsonl 2>/dev/null | grep -v "_z.jsonl" | grep -v "_summary.json" | head -1)
    if [ -z "$INF" ]; then
      INF=$(ls ${IDIR}/*.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
    fi
    if [ -n "$INF" ]; then
      OUTZ=${INF%.jsonl}_z.jsonl
      if [ -f "$OUTZ" ]; then
        echo "  [skip initials] z file exists"
      else
        echo "  [detect initials seed=${INITIALS_SEED}] $INF"
        "$PY" run_detect_initials.py \
          --input_file "$INF" \
          --seed "$INITIALS_SEED" \
          --model_name "Qwen/Qwen3-14B" \
          --output_file "$OUTZ" \
          --summary_file "${INF%.jsonl}_summary.json" || echo "  [error initials]"
      fi
    else
      echo "  [warn] no initials gen for $TAG"
    fi
  fi

  # ----- Acrostics detect (paired with shared neg) -----
  ADIR=${ROOT}/outputs/acrostics_eval/${TAG}
  if [ -d "$ADIR" ]; then
    POS=$(ls ${ADIR}/acrostics_pilot_${TAG}_n*_len10_fulleval_${ACROSTICS_TARGET}.jsonl 2>/dev/null | head -1)
    if [ -n "$POS" ] && [ -f "$ACROSTICS_NEG_FILE" ]; then
      OUTDIR=${ADIR}/detection_${ACROSTICS_TARGET}
      if [ -f "${OUTDIR}/detection_summary.md" ]; then
        echo "  [skip acrostics] summary exists"
      else
        echo "  [detect acrostics target=${ACROSTICS_TARGET}] $POS"
        "$PY" analyze_acrostics_paper_dts.py \
          --positives_file "$POS" \
          --negatives_file "$ACROSTICS_NEG_FILE" \
          --output_dir "$OUTDIR" \
          --n_resample 1000 || echo "  [error acrostics]"
      fi
    else
      echo "  [warn] acrostics: pos=$POS neg_exists=$([ -f $ACROSTICS_NEG_FILE ] && echo yes || echo no)"
    fi
  fi

done

echo "[$(date)] all detect done."
