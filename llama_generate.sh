# # ----- decapoda-research-llama-7B-hf -----

# STRENGTHS=(0.0 1.0 2.0 5.0 10.0)

# for s in "${STRENGTHS[@]}"; do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 uv run run_generate.py \
#         --model_name baffo32/decapoda-research-llama-7B-hf \
#         --strength "$s" \
#         --output_dir /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/baffo32-decapoda-research-llama-7B-hf
# done

# ----- meta-llama/Llama-3.1-8B-Instruct -----

# STRENGTHS=(0.0 1.0 2.0 5.0 10.0)

# for s in "${STRENGTHS[@]}"; do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 uv run run_generate.py \
#         --model_name meta-llama/Llama-3.1-8B-Instruct \
#         --strength "$s" \
#         --output_dir /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/meta-llama-Llama-3.1-8B-Instruct
# done


# ----- meta-llama/Llama-2-13b-chat-hf -----
STRENGTHS=(0.0 1.0 2.0 5.0 10.0)

for s in "${STRENGTHS[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run run_generate.py \
        --model_name meta-llama/Llama-2-13b-chat-hf \
        --strength "$s" \
        --output_dir /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/meta-llama-Llama-2-13b-chat-hf
done

# # ----- Qwen-3-8B -----

# STRENGTHS=(0.0 1.0 2.0 5.0 10.0 100.0 10000.0)

# for s in "${STRENGTHS[@]}"; do
#     CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
#         --model_name Qwen/Qwen3-8B \
#         --strength "$s" \
#         --output_dir /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-8B

#     CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
#         --model_name Qwen/Qwen3-8B \
#         --strength "$s" \
#         --output_dir /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-8B \
#         --only_English
# done