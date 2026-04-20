#!/bin/bash
# Acrostics pilot v2: ICW-paper-style persona in system prompt.
# Same questions / targets / sampling as v1 for cross-variant comparability.
set -euo pipefail

cd /home/tianyichen/llm_watermark

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
.venv/bin/python run_acrostics_pilot_vllm.py \
    --model_name Qwen/Qwen3-14B \
    --model_tag qwen3-14b \
    --prompt_file /home/tianyichen/llm_watermark/data/processed_data/vblagoje_lfqa/test_477.json \
    --num_test 100 \
    --target_length 4 \
    --seed_base 20260417 \
    --variants icw_paper \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_tokens 600 \
    --output_dir /home/tianyichen/llm_watermark/outputs/acrostics_pilot \
    --output_tag icwpaper

.venv/bin/python analyze_acrostics_pilot.py \
    --input_file /home/tianyichen/llm_watermark/outputs/acrostics_pilot/acrostics_pilot_qwen3-14b_n100_len4_icwpaper.jsonl
