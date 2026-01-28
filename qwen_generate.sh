STRENGTHS=(0.0 1.0 2.0 5.0 10.0)

# ----- Qwen-3-8B -----
for s in "${STRENGTHS[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
        --model_name Qwen/Qwen3-8B \
        --strength "$s" \
        --max_new_tokens 500 \
        --output_dir /home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-8B
done

# ----- Qwen-3-14B -----
for s in "${STRENGTHS[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
        --model_name Qwen/Qwen3-14B \
        --strength "$s" \
        --max_new_tokens 500 \
        --output_dir /home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-14B
done

# ----- Qwen-3-32B -----
for s in "${STRENGTHS[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
        --model_name Qwen/Qwen3-32B \
        --strength "$s" \
        --max_new_tokens 500 \
        --output_dir /home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-32B
done

