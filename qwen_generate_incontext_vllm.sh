# Fractions=(0.0)
Fractions=(0.0 0.1 0.25 0.5)

# ----- Qwen-3-8B -----
# for f in "${Fractions[@]}"; do
#     uv run run_generate_incontext_vllm.py \
#         --model_name Qwen/Qwen3-4B-Instruct-2507 \
#         --max_new_tokens 500 \
#         --yarn \
#         --max_model_len 262144 \
#         --fraction "$f" \
#         --output_dir /home/tianyichen/llm_watermark/outputs/incontext/max_new_500/Qwen-Qwen3-4B-Instruct-2507
# done

# ----- Qwen-3-14B -----
for f in "${Fractions[@]}"; do
    uv run run_generate_incontext_vllm.py \
        --model_name Qwen/Qwen3-14B \
        --max_new_tokens 500 \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --output_dir /home/tianyichen/llm_watermark/outputs/incontext/max_new_500/Qwen-Qwen3-14B
done

# ----- Qwen-3-32B -----
for f in "${Fractions[@]}"; do
    uv run run_generate_incontext_vllm.py \
        --model_name Qwen/Qwen3-32B \
        --max_new_tokens 500 \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --output_dir /home/tianyichen/llm_watermark/outputs/incontext/max_new_500/Qwen-Qwen3-32B
done