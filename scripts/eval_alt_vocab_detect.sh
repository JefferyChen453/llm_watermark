#!/usr/bin/env bash
# Cross-vocab Green Token ICW eval — detection phase.
#
# Pairs with eval_alt_vocab.sh. Each gen file at frac=F (alt=ALT) is detected by
# tokenizing gen_completion with the SAME alt tokenizer and computing z-score
# against an alt-vocab green-list mask (matched seed). Negative side uses the
# frac=0.0 file from the same ckpt × alt cell (--combine_fraction --use_generated_neg_data).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY=${PY:-${ROOT}/.venv/bin/python}
GREEN_FRACTIONS=(0.1 0.2 0.3 0.4)   # 0.0 is the matched negative; nothing to detect there
OUT_ROOT=${ROOT}/outputs/incontext_eval_alt_vocab

declare -a CKPT_TAGS=(
  "v5b_aligned_eval_202604240212_global_step_1965"
  "rl_2task_discard_v5binit_grpo_202604210816_global_step_1000"
)

declare -a ALT_TOKENIZERS=(
  "llama3.1-8b|meta-llama/Llama-3.1-8B"
  "gemma2-9b|google/gemma-2-9b"
)

# `--model_name` here is only used by run_detect.py for arg parsing back-compat;
# detection actually loads `--alt_tokenizer`. Pass Qwen3-14B as a harmless default.
ACTOR_NAME_HINT="Qwen/Qwen3-14B"

for CKPT_TAG in "${CKPT_TAGS[@]}"; do
  for alt_entry in "${ALT_TOKENIZERS[@]}"; do
    ALT_TAG="${alt_entry%%|*}"
    ALT_ID="${alt_entry##*|}"
    ALT_SLUG="${ALT_ID//\//-}"

    GDIR=${OUT_ROOT}/${CKPT_TAG}/${ALT_TAG}
    if [ ! -d "$GDIR" ]; then
      echo "[skip] $GDIR not found"
      continue
    fi

    echo "========== detect $CKPT_TAG / $ALT_TAG =========="

    for f in "${GREEN_FRACTIONS[@]}"; do
      INF=$(ls ${GDIR}/*_frac_${f}_*_only_English_alt-${ALT_SLUG}.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
      if [ -z "$INF" ]; then
        echo "  [warn] f=${f} input not found, skip"
        continue
      fi
      OUTZ=${INF%.jsonl}_z.jsonl
      if [ -f "$OUTZ" ]; then
        echo "  [skip f=${f}] z file exists"
        continue
      fi
      echo "  [detect f=${f}] alt=${ALT_TAG}: $INF"
      "$PY" run_detect.py \
        --input_file "$INF" \
        --model_name "$ACTOR_NAME_HINT" \
        --alt_tokenizer "$ALT_ID" \
        --fraction "$f" \
        --workers 8 \
        --only_English \
        --combine_fraction \
        --use_generated_neg_data || echo "  [error f=${f}]"
    done
  done
done

echo "[$(date)] all detect done."
