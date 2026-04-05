FRACTIONS=(0.1 0.15 0.2 0.25 0.3)
STRENGTH=3.0
MODEL_NAME=Qwen/Qwen3-14B
PROMPT_FILE=/home/tianyichen/llm_watermark/data/processed_data/vblagoje_lfqa/train_11578.json
OUTPUT_DIR=/home/tianyichen/llm_watermark/outputs/train_greenprob_vblagoje_lfqa_strength_3

for f in "${FRACTIONS[@]}"; do
    uv run python run_measure_train_greenprob_vllm.py \
        --model_name "$MODEL_NAME" \
        --fraction "$f" \
        --strength "$STRENGTH" \
        --seed_num 500 \
        --only_English \
        --gen_batch_size 1024 \
        --score_batch_size 4 \
        --prompt_file "$PROMPT_FILE" \
        --output_dir "$OUTPUT_DIR"
done
