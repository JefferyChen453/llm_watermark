# ----- Qwen-3-14B -----

STRENGTHS=(0.0 1.0 2.0 5.0 10.0 50.0 100.0)

for s in "${STRENGTHS[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
        --model_name Qwen/Qwen3-14B \
        --strength "$s" \
        --max_new_tokens 1000 \
        --output_dir /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-14B-1000

    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
        --model_name Qwen/Qwen3-14B \
        --strength "$s" \
        --max_new_tokens 1000 \
        --output_dir /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-14B-1000 \
        --only_English
done

# # ----- Qwen-3-32B -----

# STRENGTHS=(0.0 1.0 2.0 5.0 10.0 100.0 10000.0)

# for s in "${STRENGTHS[@]}"; do
#     CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
#         --model_name Qwen/Qwen3-32B \
#         --strength "$s" \
#         --output_dir /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-32B

#     CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
#         --model_name Qwen/Qwen3-32B \
#         --strength "$s" \
#         --output_dir /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-32B \
#         --only_English
# done

