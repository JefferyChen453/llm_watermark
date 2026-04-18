#!/bin/bash
# Post-training evaluation for mixed-kd v3 run (green3379 + initials865 + neg1000, 1 epoch).
# Compares 1 ckpt (global_step_655) against Qwen3-14B baseline.
# Fractions: (0.0 0.1 0.2 0.3 0.4)
# Steps: generate → detect → evaluate → ifeval
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG=mixed_green3379+initials865+neg1000_dualKL_202604171809
CKPT_ROOT=${PROJECT_ROOT}/verl/checkpoints/watermark-kd-ray/${RUN_TAG}
PROMPT_FILE=${PROJECT_ROOT}/data/processed_data/vblagoje_lfqa/test_477.json
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/incontext_eval/prompt_v2_new

FRACTIONS=(0.0 0.1 0.2 0.3 0.4)
STEPS=(655)

BASELINE_MODEL="Qwen/Qwen3-14B"
BASELINE_TAG="baseline_qwen3-14b"
BASELINE_OUT_DIR=${OUTPUT_ROOT}/${BASELINE_TAG}

export VLLM_USE_TORCH_COMPILE=0
# Use local HF cache only — prevents rate-limiting when detect spawns many workers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============================================================
# Step 1: vLLM Generate (in-context watermark)
# ============================================================
echo "========== [$(date)] Step 1: vLLM Generate =========="

# Baseline (reuse cached files from v1 eval)
mkdir -p "$BASELINE_OUT_DIR"
for f in "${FRACTIONS[@]}"; do
    EXISTING=$(ls "${BASELINE_OUT_DIR}"/*_frac_${f}_*_only_English.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
    if [ -n "$EXISTING" ]; then
        echo "[$(date)] Baseline frac=${f} already generated, skip."
        continue
    fi
    echo "[$(date)] Generating: baseline, frac=${f}"
    uv run run_generate_incontext_vllm.py \
        --model_name "$BASELINE_MODEL" \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --seed_num 1 \
        --only_English \
        --prompt_file "$PROMPT_FILE" \
        --output_dir "$BASELINE_OUT_DIR"
done

for step in "${STEPS[@]}"; do
    MODEL_PATH=${CKPT_ROOT}/global_step_${step}/hf_model
    OUT_DIR=${OUTPUT_ROOT}/${RUN_TAG}_global_step_${step}
    mkdir -p "$OUT_DIR"

    for f in "${FRACTIONS[@]}"; do
        EXISTING=$(ls "${OUT_DIR}"/*_frac_${f}_*_only_English.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
        if [ -n "$EXISTING" ]; then
            echo "[$(date)] step=${step} frac=${f} already generated, skip."
            continue
        fi
        echo "[$(date)] Generating: step=${step}, frac=${f}"
        uv run run_generate_incontext_vllm.py \
            --model_name "$MODEL_PATH" \
            --yarn \
            --max_model_len 131072 \
            --fraction "$f" \
            --seed_num 1 \
            --only_English \
            --prompt_file "$PROMPT_FILE" \
            --output_dir "$OUT_DIR"
    done
done

# ============================================================
# Step 2: Detect (z-score)
# ============================================================
echo "========== [$(date)] Step 2: Detect =========="

detect_dir() {
    local DIR=$1
    for f in "${FRACTIONS[@]}"; do
        INPUT_FILE=$(ls "${DIR}"/*_frac_${f}_*_only_English.jsonl 2>/dev/null | grep -v "_z.jsonl" | head -1)
        if [ -z "$INPUT_FILE" ]; then
            echo "[WARN] No input file found in $DIR for frac=${f}"
            continue
        fi
        Z_FILE="${INPUT_FILE%.jsonl}_z.jsonl"
        if [ -f "$Z_FILE" ] && [ "$(wc -l < "$Z_FILE")" -eq "$(wc -l < "$INPUT_FILE")" ]; then
            echo "[$(date)] $(basename $Z_FILE) complete, skip."
            continue
        fi
        echo "[$(date)] Detecting: $(basename $INPUT_FILE)"
        uv run run_detect.py \
            --model_name Qwen/Qwen3-14B \
            --fraction "$f" \
            --input_file "$INPUT_FILE" \
            --workers 32 \
            --only_English \
            --combine_fraction \
            --use_generated_neg_data
    done
}

detect_dir "$BASELINE_OUT_DIR"
for step in "${STEPS[@]}"; do
    detect_dir "${OUTPUT_ROOT}/${RUN_TAG}_global_step_${step}"
done

# ============================================================
# Step 3: Evaluate (ROC + CSV)
# ============================================================
echo "========== [$(date)] Step 3: Evaluate =========="

uv run evaluate.py "$BASELINE_OUT_DIR" --fraction_or_strength fraction --target_fpr 0.01
for step in "${STEPS[@]}"; do
    OUT_DIR=${OUTPUT_ROOT}/${RUN_TAG}_global_step_${step}
    echo "[$(date)] Evaluating: step=${step}"
    uv run evaluate.py "$OUT_DIR" --fraction_or_strength fraction --target_fpr 0.01
done

# ============================================================
# Step 4: IFEval
# ============================================================
echo "========== [$(date)] Step 4: IFEval =========="

uv pip install --upgrade antlr4-python3-runtime 2>/dev/null

# Baseline ifeval
echo "[$(date)] IFEval: baseline"
uv run lighteval vllm \
    "model_name=${BASELINE_MODEL},dtype=bfloat16,gpu_memory_utilization=0.9" \
    "ifeval|0" \
    --output-dir outputs/ifeval_nothink \
    --save-details

for step in "${STEPS[@]}"; do
    MODEL_PATH=${CKPT_ROOT}/global_step_${step}/hf_model
    echo "[$(date)] IFEval: step=${step}"
    uv run lighteval vllm \
        "model_name=${MODEL_PATH},dtype=bfloat16,gpu_memory_utilization=0.9" \
        "ifeval|0" \
        --output-dir outputs/ifeval_nothink \
        --save-details
done

uv pip install antlr4-python3-runtime==4.9.3 2>/dev/null

echo "========== [$(date)] Pipeline complete =========="
