#!/bin/bash
# Shuffled green-token evaluation for posneg safety anchor run.
# Outputs go to prompt_v2/ to match baseline (which was also shuffled).
# Steps: generate (with --shuffle_green_tokens) → detect → evaluate
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CKPT_ROOT=${PROJECT_ROOT}/verl/checkpoints/watermark-kd-ray/posneg_strength_5.0_pos_5931_neg_1000_bsz_8__1.0kl_biased_ref+1.0kl_ref_202604110238
RUN_TAG=posneg_strength_5.0_pos_5931_neg_1000_bsz_8__1.0kl_biased_ref+1.0kl_ref_202604110238
PROMPT_FILE=${PROJECT_ROOT}/data/processed_data/vblagoje_lfqa/test_477.json
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/incontext_eval/prompt_v2

FRACTIONS=(0.0 0.1 0.15 0.2 0.25 0.3 0.35 0.4)
STEPS=(1732)

export VLLM_USE_TORCH_COMPILE=0

# ============================================================
# Generate (shuffled green tokens)
# ============================================================
echo "========== [$(date)] Shuffled Generate =========="

for step in "${STEPS[@]}"; do
    MODEL_PATH=${CKPT_ROOT}/global_step_${step}/hf_model
    OUT_DIR=${OUTPUT_ROOT}/${RUN_TAG}_global_step_${step}
    mkdir -p "$OUT_DIR"

    for f in "${FRACTIONS[@]}"; do
        # Skip if already generated
        EXISTING=$(ls "${OUT_DIR}"/*_frac_${f}_*_only_English.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
        if [ -n "$EXISTING" ] && [ -s "$EXISTING" ]; then
            N_LINES=$(wc -l < "$EXISTING")
            if [ "$N_LINES" -ge 477 ]; then
                echo "[$(date)] SKIP (already done, ${N_LINES} lines): step=${step}, frac=${f}"
                continue
            fi
            echo "[$(date)] Partial file found (${N_LINES} lines) — removing and regenerating"
            rm "$EXISTING"
        fi
        echo "[$(date)] Generating (shuffled): step=${step}, frac=${f}"
        uv run run_generate_incontext_vllm.py \
            --model_name "$MODEL_PATH" \
            --yarn \
            --max_model_len 131072 \
            --fraction "$f" \
            --seed_num 1 \
            --only_English \
            --shuffle_green_tokens \
            --prompt_file "$PROMPT_FILE" \
            --output_dir "$OUT_DIR"
    done
done


FRACTIONS=(0.3 0.35 0.4)
STEPS=(866)

export VLLM_USE_TORCH_COMPILE=0

# ============================================================
# Generate (shuffled green tokens)
# ============================================================
echo "========== [$(date)] Shuffled Generate =========="

for step in "${STEPS[@]}"; do
    MODEL_PATH=${CKPT_ROOT}/global_step_${step}/hf_model
    OUT_DIR=${OUTPUT_ROOT}/${RUN_TAG}_global_step_${step}
    mkdir -p "$OUT_DIR"

    for f in "${FRACTIONS[@]}"; do
        # Skip if already generated
        EXISTING=$(ls "${OUT_DIR}"/*_frac_${f}_*_only_English.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
        if [ -n "$EXISTING" ] && [ -s "$EXISTING" ]; then
            N_LINES=$(wc -l < "$EXISTING")
            if [ "$N_LINES" -ge 477 ]; then
                echo "[$(date)] SKIP (already done, ${N_LINES} lines): step=${step}, frac=${f}"
                continue
            fi
            echo "[$(date)] Partial file found (${N_LINES} lines) — removing and regenerating"
            rm "$EXISTING"
        fi
        echo "[$(date)] Generating (shuffled): step=${step}, frac=${f}"
        uv run run_generate_incontext_vllm.py \
            --model_name "$MODEL_PATH" \
            --yarn \
            --max_model_len 131072 \
            --fraction "$f" \
            --seed_num 1 \
            --only_English \
            --shuffle_green_tokens \
            --prompt_file "$PROMPT_FILE" \
            --output_dir "$OUT_DIR"
    done
done

# ============================================================
# Detect
# ============================================================
echo "========== [$(date)] Shuffled Detect =========="

for step in "${STEPS[@]}"; do
    OUT_DIR=${OUTPUT_ROOT}/${RUN_TAG}_global_step_${step}

    for f in "${FRACTIONS[@]}"; do
        INPUT_FILE=$(ls "${OUT_DIR}"/*_frac_${f}_*_only_English.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
        if [ -z "$INPUT_FILE" ]; then
            echo "[WARN] No input file found for step=${step}, frac=${f}"
            continue
        fi
        echo "[$(date)] Detecting: step=${step}, frac=${f}"
        uv run run_detect.py \
            --model_name Qwen/Qwen3-14B \
            --fraction "$f" \
            --input_file "$INPUT_FILE" \
            --workers 32 \
            --only_English \
            --combine_fraction \
            --use_generated_neg_data
    done
done

# ============================================================
# Evaluate (ROC + CSV)
# ============================================================
echo "========== [$(date)] Shuffled Evaluate =========="

for step in "${STEPS[@]}"; do
    OUT_DIR=${OUTPUT_ROOT}/${RUN_TAG}_global_step_${step}
    echo "[$(date)] Evaluating: step=${step}"
    uv run evaluate.py "$OUT_DIR" --fraction_or_strength fraction --target_fpr 0.01
done

echo "========== [$(date)] Shuffled pipeline complete =========="
