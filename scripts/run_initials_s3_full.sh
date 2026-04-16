#!/usr/bin/env bash
# Add s=3 + plain on full 477 LFQA test for AUC-ROC
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="Qwen/Qwen3-14B"
PROMPT_FILE="data/processed_data/vblagoje_lfqa/test_477.json"
OUT_ROOT="outputs/initials_icw_pilot_full"
N=477
SEED=0
PY="/home/tianyichen/llm_watermark/.venv/bin/python"

mkdir -p "$OUT_ROOT"

run_cell() {
    local tag="$1"; shift
    local out_dir="$OUT_ROOT/$tag"
    mkdir -p "$out_dir"
    echo ""
    echo "============================================================"
    echo "[CELL $tag] $*"
    echo "============================================================"
    "$PY" run_generate_initials_vllm.py \
        --model_name "$MODEL" \
        --prompt_file "$PROMPT_FILE" \
        --output_dir "$out_dir" \
        --num_test "$N" \
        --seed "$SEED" \
        "$@"
}

# Negative: plain query, no bias
run_cell "plain_s0"

# Positive: ICW prompt + bias s=3
run_cell "prompt_s3" --add_icw_prompt --add_logits_wm --strength 3

echo ""
echo "============================================================"
echo "Detection..."
echo "============================================================"
for cell_dir in "$OUT_ROOT"/*/; do
    cell=$(basename "$cell_dir")
    jsonl=$(ls "$cell_dir"/*.jsonl 2>/dev/null | grep -v "_z.jsonl\|_summary" | head -1 || true)
    if [ -z "$jsonl" ]; then
        echo "[skip] no gen jsonl in $cell_dir"; continue
    fi
    echo "  detecting $cell -> $jsonl"
    "$PY" run_detect_initials.py --input_file "$jsonl"
done

echo ""
echo "Done. Outputs in $OUT_ROOT/"
