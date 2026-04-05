#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR=/home/tianyichen/llm_watermark/verl/checkpoints

FRACTIONS=(0.1 0.15 0.2 0.25 0.3)
STRENGTH=3.0
SEED_NUM=500
NUM_TEST=1000
MODEL_NAME=${CKPT_DIR}/watermark-kd-ray/filter_strength_5.0_5931_bsz_8__0.0green+1.0reverse_kl_biased_ref_202604032354/global_step_741/hf_model
PROMPT_FILE=/home/tianyichen/llm_watermark/data/processed_data/vblagoje_lfqa/test_477.json
OUTPUT_DIR=/home/tianyichen/llm_watermark/outputs/test_avg_green_prob/filter_strength_5.0_5931_bsz_8__0.0green+1.0reverse_kl_biased_ref_202604032354_global_step_741
PYTHON_BIN=/home/tianyichen/llm_watermark/.venv/bin/python

mkdir -p "$OUTPUT_DIR"

for f in "${FRACTIONS[@]}"; do
    "$PYTHON_BIN" /home/tianyichen/llm_watermark/run_measure_train_greenprob_vllm.py \
        --model_name "$MODEL_NAME" \
        --fraction "$f" \
        --strength "$STRENGTH" \
        --seed_num "$SEED_NUM" \
        --only_English \
        --gen_batch_size 1024 \
        --score_batch_size 4 \
        --prompt_file "$PROMPT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --num_test "$NUM_TEST"
done

"$PYTHON_BIN" /home/tianyichen/llm_watermark/aggregate_avg_green_prob_results.py \
    --input_dir "$OUTPUT_DIR" \
    --output_file "$OUTPUT_DIR/analysis.md"
