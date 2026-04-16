#!/usr/bin/env bash
# Phase 2c: Initials ICW pilot
# 5 cells × 100 samples on LFQA test set first 100, single seed=0

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-14B}"
PROMPT_FILE="${PROMPT_FILE:-data/processed_data/vblagoje_lfqa/test_477.json}"
OUT_ROOT="${OUT_ROOT:-outputs/initials_icw_pilot}"
N=100
SEED=0
PY="${PY:-/home/tianyichen/llm_watermark/.venv/bin/python}"

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

# Cell 1: plain query, no bias (null baseline)
run_cell "01_plain_s0"

# Cell 2: ICW prompt, no bias (prompt-only capability)
run_cell "02_prompt_s0" --add_icw_prompt

# Cell 3: ICW prompt, bias strength=2
run_cell "03_prompt_s2" --add_icw_prompt --add_logits_wm --strength 2

# Cell 4: ICW prompt, bias strength=5
run_cell "04_prompt_s5" --add_icw_prompt --add_logits_wm --strength 5

# Cell 5: ICW prompt, bias strength=10
run_cell "05_prompt_s10" --add_icw_prompt --add_logits_wm --strength 10

echo ""
echo "============================================================"
echo "Running detection on all cells..."
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
