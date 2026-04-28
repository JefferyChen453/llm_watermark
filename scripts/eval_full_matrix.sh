#!/usr/bin/env bash
# Full evaluation matrix for the icw-day4 report:
#   7 checkpoints × {green f=0.0/0.1/0.2/0.3/0.4, initials, acrostics} = 49 cells
#
# Plus IFEval (for the 3 v5b_aligned ckpts that don't have it yet).
#
# Re-runnable: every gen step skips if output already present.
#
# After this script:
#   bash scripts/eval_full_matrix_detect.sh   # detect for all generated jsonl
#   uv run python tools/aggregate_full_matrix.py  # build summary tables

# Tolerate per-cell errors so one failure doesn't abort whole matrix.
# Each cell wrapped in '|| echo error_marker' below.
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY=${PY:-${ROOT}/.venv/bin/python}
PROMPT_FILE=${ROOT}/data/processed_data/vblagoje_lfqa/test_477.json
OUT_GREEN_ROOT=${ROOT}/outputs/incontext_eval/prompt_v2_new
OUT_INI_ROOT=${ROOT}/outputs/initials_eval
OUT_ACR_ROOT=${ROOT}/outputs/acrostics_eval

# Eval-only constants (must NOT collide with training data seeds)
GREEN_FRACTIONS=(0.0 0.1 0.2 0.3 0.4)   # frac=0.0 = clean, used as detection negative
INITIALS_SEED=42                         # train uses 1282832+, eval=42 disjoint
ACROSTICS_TARGET="WATERMARKS"            # 10 letter UPPERCASE; train will skip this exact target
ACROSTICS_NEG_FILE=${OUT_ACR_ROOT}/negatives_lfqa_gold_target_${ACROSTICS_TARGET}.jsonl

export VLLM_USE_TORCH_COMPILE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ----- Checkpoint registry -----
# Tags align with existing outputs/incontext_eval/prompt_v2_new/<tag>/ dirs so
# we skip re-generating the green watermark cells already produced by the
# 04-19 v5b / 04-24 RL eval pipelines.
declare -a CKPTS=(
  "baseline_qwen3-14b|Qwen/Qwen3-14B"
  "v5b_green3379+initials865+neg1000_dualKL_biasedRefTopK1000_202604180857_global_step_655|${ROOT}/verl/checkpoints/watermark-kd-ray/v5b_green3379+initials865+neg1000_dualKL_biasedRefTopK1000_202604180857/global_step_655/hf_model"
  "v5b_aligned_eval_202604240212_global_step_655|${ROOT}/verl/checkpoints/watermark-kd-ray/v5b_aligned_eval_202604240212/global_step_655/hf_model"
  "v5b_aligned_eval_202604240212_global_step_1310|${ROOT}/verl/checkpoints/watermark-kd-ray/v5b_aligned_eval_202604240212/global_step_1310/hf_model"
  "v5b_aligned_eval_202604240212_global_step_1965|${ROOT}/verl/checkpoints/watermark-kd-ray/v5b_aligned_eval_202604240212/global_step_1965/hf_model"
  "rl_2task_discard_v5binit_grpo_202604210816_global_step_500|${ROOT}/verl/checkpoints/watermark-rl-ray/rl_2task_discard_v5binit_grpo_202604210816/global_step_500/hf_model"
  "rl_2task_discard_v5binit_grpo_202604210816_global_step_1000|${ROOT}/verl/checkpoints/watermark-rl-ray/rl_2task_discard_v5binit_grpo_202604210816/global_step_1000/hf_model"
)

# ----- Acrostics negatives (one-time, ckpt-independent) -----
mkdir -p "$OUT_ACR_ROOT"
if [ ! -f "$ACROSTICS_NEG_FILE" ]; then
  echo "[$(date)] Building acrostics negatives -> $ACROSTICS_NEG_FILE"
  "$PY" build_acrostics_negatives.py \
    --model_tag eval \
    --prompt_file "$PROMPT_FILE" \
    --num_test 477 \
    --target_length 10 --target_uppercase \
    --fixed_target "$ACROSTICS_TARGET" \
    --output_dir "$OUT_ACR_ROOT" \
    --output_tag "fulleval_${ACROSTICS_TARGET}"
  # Rename to canonical fixed name expected below
  AUTO=$(ls "$OUT_ACR_ROOT"/acrostics_pilot_eval_n*_len10_fulleval_${ACROSTICS_TARGET}_negatives.jsonl | head -1)
  mv -v "$AUTO" "$ACROSTICS_NEG_FILE"
fi

# ----- Per-ckpt loop -----
for entry in "${CKPTS[@]}"; do
  TAG="${entry%%|*}"
  MODEL="${entry##*|}"

  echo "==================================================="
  echo "[$(date)] CKPT: $TAG"
  echo "  model: $MODEL"
  echo "==================================================="

  # ---------- (A) GREEN watermark frac sweep (incontext, 477 prefix) ----------
  GREEN_OUT=${OUT_GREEN_ROOT}/${TAG}
  mkdir -p "$GREEN_OUT"
  for f in "${GREEN_FRACTIONS[@]}"; do
    EXISTING=$(ls "${GREEN_OUT}"/*_frac_${f}_*_only_English.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
    if [ -n "$EXISTING" ]; then
      echo "[$(date)] [skip green f=${f}] -> $EXISTING"
      continue
    fi
    echo "[$(date)] [green f=${f}] gen ..."
    uv run run_generate_incontext_vllm.py \
      --model_name "$MODEL" \
      --yarn --max_model_len 131072 \
      --fraction "$f" --seed_num 1 \
      --only_English \
      --prompt_file "$PROMPT_FILE" \
      --output_dir "$GREEN_OUT" || echo "[$(date)] [error green f=${f}] for $TAG"
  done

  # ---------- (B) INITIALS watermark (icw prompt, no logit bias, eval seed=42) ----------
  INI_OUT=${OUT_INI_ROOT}/${TAG}
  mkdir -p "$INI_OUT"
  EXISTING_INI=$(ls "${INI_OUT}"/*_initials_seed_${INITIALS_SEED}*.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
  if [ -n "$EXISTING_INI" ]; then
    echo "[$(date)] [skip initials seed=${INITIALS_SEED}] -> $EXISTING_INI"
  else
    echo "[$(date)] [initials seed=${INITIALS_SEED}] gen ..."
    # max_model_len 8192 sufficient: LFQA prefix (~150 tok) + ICW prompt with green/red letters (~500 tok) + 600 max_new = ~1300 tok
    # Don't use yarn here — model native 40960 is plenty.
    uv run run_generate_initials_vllm.py \
      --model_name "$MODEL" \
      --add_icw_prompt \
      --strength 0.0 \
      --seed "$INITIALS_SEED" \
      --num_test 477 \
      --prompt_file "$PROMPT_FILE" \
      --output_dir "$INI_OUT" \
      --max_model_len 8192 \
      --tensor_parallel_size 8 \
      --batch_size 32 || echo "[$(date)] [error initials seed=${INITIALS_SEED}] for $TAG"
  fi

  # ---------- (C) ACROSTICS paper_dts variant, fixed target ----------
  ACR_OUT=${OUT_ACR_ROOT}/${TAG}
  mkdir -p "$ACR_OUT"
  EXP_FILE=${ACR_OUT}/acrostics_pilot_${TAG}_n477_len10_fulleval_${ACROSTICS_TARGET}.jsonl
  if [ -f "$EXP_FILE" ]; then
    echo "[$(date)] [skip acrostics target=${ACROSTICS_TARGET}] -> $EXP_FILE"
  else
    echo "[$(date)] [acrostics target=${ACROSTICS_TARGET}] gen ..."
    # Acrostics paper_dts: system prompt ~600 tok + LFQA query ~150 + max_new 600 ≈ 1400; 8192 cap is plenty.
    # No --yarn: works for baseline (native 40960) AND trained (config has rope_scaling baked in).
    uv run run_acrostics_pilot_vllm.py \
      --model_name "$MODEL" \
      --model_tag "$TAG" \
      --prompt_file "$PROMPT_FILE" \
      --num_test 477 \
      --target_length 10 --target_uppercase \
      --fixed_target "$ACROSTICS_TARGET" \
      --variants paper_dts \
      --max_tokens 600 \
      --max_model_len 8192 \
      --tp 8 \
      --output_dir "$ACR_OUT" \
      --output_tag "fulleval_${ACROSTICS_TARGET}" || echo "[$(date)] [error acrostics] for $TAG"
  fi
done

echo "[$(date)] all gen done."
