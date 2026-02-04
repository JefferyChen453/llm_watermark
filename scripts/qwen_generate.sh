# STRENGTHS=(0.0 1.0 2.0 5.0 10.0)

# # ----- Qwen-3-8B -----
# for s in "${STRENGTHS[@]}"; do
#     CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
#         --model_name Qwen/Qwen3-8B \
#         --strength "$s" \
#         --max_new_tokens 500 \
#         --output_dir /home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-8B
# done

# # ----- Qwen-3-14B -----
# for s in "${STRENGTHS[@]}"; do
#     CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
#         --model_name Qwen/Qwen3-14B \
#         --strength "$s" \
#         --max_new_tokens 500 \
#         --output_dir /home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-14B
# done

# # ----- Qwen-3-32B -----
# for s in "${STRENGTHS[@]}"; do
#     CUDA_VISIBLE_DEVICES=4,5,6,7 uv run run_generate.py \
#         --model_name Qwen/Qwen3-32B \
#         --strength "$s" \
#         --max_new_tokens 500 \
#         --output_dir /home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-32B
# done

# -------------------------------- Only English test --------------------------------

# ----- test different strengths -----
# STRENGTHS=(0.0 1.0 2.0 5.0)
# for s in "${STRENGTHS[@]}"; do
#     uv run run_generate.py \
#         --model_name Qwen/Qwen3-14B \
#         --max_new_tokens 500 \
#         --fraction 0.5 \
#         --strength "$s" \
#         --only_English \
#         --output_dir /home/tianyichen/llm_watermark/outputs/only_eng/logits_wm/strength/Qwen-Qwen3-14B \
#         --num_test 512

#     # uv run run_generate.py \
#     #     --model_name Qwen/Qwen3-32B \
#     #     --max_new_tokens 500 \
#     #     --fraction 0.5 \
#     #     --strength "$s" \
#     #     --only_English \
#     #     --output_dir /home/tianyichen/llm_watermark/outputs/only_eng/logits_wm/strength/Qwen-Qwen3-32B \
#     #     --num_test 512
# done

# ----- test different fractions -----
# FRACTIONS=(0.0 0.1 0.2 0.3 0.4 0.5)
# for f in "${FRACTIONS[@]}"; do
#     uv run run_generate.py \
#         --model_name Qwen/Qwen3-14B \
#         --max_new_tokens 500 \
#         --fraction "$f" \
#         --strength 2.0 \
#         --only_English \
#         --output_dir /home/tianyichen/llm_watermark/outputs/only_eng/logits_wm/fraction/Qwen-Qwen3-14B \
#         --num_test 512

#     # uv run run_generate.py \
#     #     --model_name Qwen/Qwen3-32B \
#     #     --max_new_tokens 500 \
#     #     --fraction "$f" \
#     #     --strength 2.0 \
#     #     --only_English \
#     #     --output_dir /home/tianyichen/llm_watermark/outputs/only_eng/logits_wm/fraction/Qwen-Qwen3-32B \
#     #     --num_test 512
# done

# ----- test incontext -----
FRACTIONS=(0.3)
# FRACTIONS=(0.1 0.2 0.4 0.5)
for f in "${FRACTIONS[@]}"; do
    uv run run_generate_incontext_vllm.py \
        --model_name Qwen/Qwen3-14B \
        --max_new_tokens 500 \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --output_dir /home/tianyichen/llm_watermark/outputs/only_eng/incontext_vllm/Qwen-Qwen3-14B_withlinewithoutspace \
        --only_English \
        --num_test 512
done