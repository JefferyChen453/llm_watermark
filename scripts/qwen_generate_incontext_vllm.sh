# Fractions=(0.25)
# Fractions=(0.0 0.1 0.25 0.5)

# # ----- Qwen-3-4B -----
# for f in "${Fractions[@]}"; do
#     uv run run_generate_incontext_vllm.py \
#         --model_name Qwen/Qwen3-4B-Instruct-2507 \
#         --max_new_tokens 500 \
#         --yarn \
#         --max_model_len 262144 \
#         --fraction "$f" \
#         --output_dir /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-4B-Instruct-2507 \
#         --only_English
# done

# Fractions=(0.0 0.5 0.1 0.15 0.2 0.25)

# # ----- Qwen-3-14B -----
# for f in "${Fractions[@]}"; do
#     uv run run_generate_incontext_vllm.py \
#         --model_name Qwen/Qwen3-14B \
#         --max_new_tokens 500 \
#         --yarn \
#         --max_model_len 131072 \
#         --fraction "$f" \
#         --output_dir /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-14B
# done

# # ----- Qwen-3-32B -----
# for f in "${Fractions[@]}"; do
#     uv run run_generate_incontext_vllm.py \
#         --model_name Qwen/Qwen3-32B \
#         --max_new_tokens 500 \
#         --yarn \
#         --max_model_len 131072 \
#         --fraction "$f" \
#         --output_dir /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-32B
# done

Fractions=(0.0 0.1 0.2 0.3 0.4)
for f in "${Fractions[@]}"; do
    uv run run_generate_incontext_vllm.py \
        --add_logits_wm \
        --model_name Qwen/Qwen3-14B \
        --max_new_tokens 500 \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --strength 3.0 \
        --only_English \
        --output_dir /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength3/Qwen-Qwen3-14B

    uv run run_generate_incontext_vllm.py \
        --add_logits_wm \
        --model_name Qwen/Qwen3-32B \
        --max_new_tokens 500 \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --strength 3.0 \
        --only_English \
        --output_dir /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength3/Qwen-Qwen3-32B

    uv run run_generate_incontext_vllm.py \
        --add_logits_wm \
        --model_name Qwen/Qwen3-32B \
        --max_new_tokens 500 \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --strength 2.0 \
        --only_English \
        --output_dir /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength2/Qwen-Qwen3-32B


done