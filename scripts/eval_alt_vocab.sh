#!/usr/bin/env bash
# Cross-vocab Green Token ICW eval — generation phase.
#
# Holds actor model = Qwen3-14B family (KD or RL ckpt) and varies the *watermark
# vocab*: the green-list mask + in-context prompt come from a non-Qwen tokenizer
# (Llama-3.1-8B, Gemma-2-9B). Detection re-tokenizes gen_completion with the
# same alt tokenizer (see eval_alt_vocab_detect.sh).
#
# Matrix: 2 ckpts × 2 alt tokenizers × 5 fractions = 20 cells.
#   - frac=0.0 = clean / negative; gen_completion measured under each alt vocab
#     so AUC has a matched-vocab negative.
#
# Re-runnable: skip-if-exists per cell. Per-cell error tolerated.
#
# Prereqs:
#   1. Llama-3.1-8B and Gemma-2-9B HF repos accessible (gated; ensure
#      `huggingface-cli login` already done OR repos pre-downloaded).
#   2. test_477.json exists at the prompt path below.
#
# After this script: bash scripts/eval_alt_vocab_detect.sh

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY=${PY:-${ROOT}/.venv/bin/python}
PROMPT_FILE=${ROOT}/data/processed_data/vblagoje_lfqa/test_477.json
OUT_ROOT=${ROOT}/outputs/incontext_eval_alt_vocab

GREEN_FRACTIONS=(0.0 0.1 0.2 0.3 0.4)

export VLLM_USE_TORCH_COMPILE=0
# Don't force OFFLINE: alt tokenizer / config may need to fetch on first run.
# If you've already cached, you can set HF_HUB_OFFLINE=1 manually before invoking.

# ----- Checkpoint registry (actor) -----
declare -a CKPTS=(
  "v5b_aligned_eval_202604240212_global_step_1965|${ROOT}/verl/checkpoints/watermark-kd-ray/v5b_aligned_eval_202604240212/global_step_1965/hf_model"
  "rl_2task_discard_v5binit_grpo_202604210816_global_step_1000|${ROOT}/verl/checkpoints/watermark-rl-ray/rl_2task_discard_v5binit_grpo_202604210816/global_step_1000/hf_model"
)

# ----- Alt tokenizer registry -----
# format: "tag|hf_id". Tag goes into output dir + filename slug; hf_id is what
# AutoTokenizer.from_pretrained sees.
declare -a ALT_TOKENIZERS=(
  "llama3.1-8b|meta-llama/Llama-3.1-8B"
  "gemma2-9b|google/gemma-2-9b"
)

# ----- Per-ckpt / per-alt / per-frac loop -----
for ckpt_entry in "${CKPTS[@]}"; do
  CKPT_TAG="${ckpt_entry%%|*}"
  MODEL="${ckpt_entry##*|}"

  for alt_entry in "${ALT_TOKENIZERS[@]}"; do
    ALT_TAG="${alt_entry%%|*}"
    ALT_ID="${alt_entry##*|}"

    GEN_OUT=${OUT_ROOT}/${CKPT_TAG}/${ALT_TAG}
    mkdir -p "$GEN_OUT"

    # Slug used by run_generate_incontext_vllm to build filename: alt-<slug>.jsonl
    # (slashes in HF id get replaced with '-')
    ALT_SLUG="${ALT_ID//\//-}"

    echo "==================================================="
    echo "[$(date)] CKPT=$CKPT_TAG  ALT=$ALT_TAG ($ALT_ID)"
    echo "  model: $MODEL"
    echo "  out:   $GEN_OUT"
    echo "==================================================="

    for f in "${GREEN_FRACTIONS[@]}"; do
      EXISTING=$(ls "${GEN_OUT}"/*_frac_${f}_*_only_English_alt-${ALT_SLUG}.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
      if [ -n "$EXISTING" ]; then
        echo "[$(date)] [skip f=${f}] -> $EXISTING"
        continue
      fi
      echo "[$(date)] [gen f=${f}] alt=${ALT_TAG}"
      uv run run_generate_incontext_vllm.py \
        --model_name "$MODEL" \
        --alt_tokenizer "$ALT_ID" \
        --yarn --max_model_len 131072 \
        --fraction "$f" --seed_num 1 \
        --only_English \
        --prompt_file "$PROMPT_FILE" \
        --output_dir "$GEN_OUT" \
        || echo "[$(date)] [error f=${f}] for $CKPT_TAG / $ALT_TAG"
    done
  done
done

echo "[$(date)] all gen done."
