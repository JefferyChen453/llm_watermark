Fractions=(0.25 0.1)

# # ----- Qwen-3-8B -----
# for f in "${Fractions[@]}"; do
#     uv run run_generate_incontext.py \
#         --model_name Qwen/Qwen3-8B \
#         --max_new_tokens 500 \
#         --yarn \
#         --fraction "$f" \
#         --output_dir /home/tianyichen/llm_watermark/outputs/incontext/max_new_500/Qwen-Qwen3-8B
# done

# ----- Qwen-3-14B -----
for f in "${Fractions[@]}"; do
    uv run run_generate_incontext.py \
        --model_name Qwen/Qwen3-14B \
        --max_new_tokens 500 \
        --yarn \
        --fraction "$f" \
        --output_dir /home/tianyichen/llm_watermark/temp
done

# # ----- Qwen-3-32B -----
# for f in "${Fractions[@]}"; do
#     uv run run_generate_incontext.py \
#         --model_name Qwen/Qwen3-32B \
#         --max_new_tokens 500 \
#         --yarn \
#         --fraction "$f" \
#         --output_dir /home/tianyichen/llm_watermark/outputs/incontext/max_new_500/Qwen-Qwen3-32B
# done