# STRENGTHS=(0.0 1.0 2.0 5.0 10.0)

# # ----- Qwen-3-8B -----
# for s in "${STRENGTHS[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-8B \
#         --strength "$s" \
#         --input_file /home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-8B/Qwen-Qwen3-8B_strength_"$s"_frac_0.5_len_500_num_500.jsonl
# done


# # ----- Qwen-3-14B -----
# for s in "${STRENGTHS[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --strength "$s" \
#         --input_file /home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-14B/Qwen-Qwen3-14B_strength_"$s"_frac_0.5_len_500_num_500.jsonl
# done

# # ----- Qwen-3-32B -----
# for s in "${STRENGTHS[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-32B \
#         --strength "$s" \
#         --input_file /home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-32B/Qwen-Qwen3-32B_strength_"$s"_frac_0.5_len_500_num_500.jsonl
# done


# STRENGTHS=(0.0 1.0 2.0 5.0 10.0)
# for s in "${STRENGTHS[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --strength "$s" \
#         --input_file /home/tianyichen/llm_watermark/temp/logits_wm/strength/Qwen-Qwen3-14B/Qwen-Qwen3-14B_strength_"$s"_frac_0.5_len_500_num_512.jsonl

#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --strength "$s" \
#         --only_English \
#         --input_file /home/tianyichen/llm_watermark/temp/logits_wm/strength/Qwen-Qwen3-14B/Qwen-Qwen3-14B_strength_"$s"_frac_0.5_len_500_num_512_only_English.jsonl

# done

# Fractions=(0.9)
# for f in "${Fractions[@]}"; do
#     # uv run run_detect.py \
#     #     --model_name Qwen/Qwen3-14B \
#     #     --fraction "$f" \
#     #     --input_file /home/tianyichen/llm_watermark/temp/logits_wm/fraction/Qwen-Qwen3-14B/Qwen-Qwen3-14B_strength_2.0_frac_"$f"_len_500_num_512.jsonl \
#     #     --combine_fraction

#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --fraction "$f" \
#         --input_file /home/tianyichen/llm_watermark/temp/logits_wm/fraction/Qwen-Qwen3-14B/Qwen-Qwen3-14B_strength_2.0_frac_"$f"_len_500_num_512_only_English.jsonl \
#         --only_English \
#         --combine_fraction
# done


# Fractions=(0.0 0.1 0.25 0.5)
# OUTPUT_ROOT_DIR=/home/tianyichen/llm_watermark/outputs
# # ----- Qwen-3-14B -----
# for f in "${Fractions[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --fraction "$f" \
#         --test_min_tokens 500 \
#         --input_file ${OUTPUT_ROOT_DIR}/incontext/max_new_500/Qwen-Qwen3-14B/Qwen-Qwen3-14B_frac_"$f"_len_500_num_512_incontext_vllm.jsonl
# done

# # ----- Qwen-3-32B -----
# for f in "${Fractions[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-32B \
#         --fraction "$f" \
#         --test_min_tokens 500 \
#         --input_file ${OUTPUT_ROOT_DIR}/incontext/max_new_500/Qwen-Qwen3-32B/Qwen-Qwen3-32B_frac_"$f"_len_500_num_512_incontext_vllm.jsonl
# done


# uv run run_detect.py \
#     --model_name Qwen/Qwen3-14B \
#     --fraction 0.5 \
#     --input_file /home/tianyichen/llm_watermark/Qwen-Qwen3-14B_strength_2.0_frac_0.5_len_500_num_512_only_English.jsonl \

uv run run_detect.py \
    --model_name Qwen/Qwen3-14B \
    --fraction 0.5 \
    --input_file /home/tianyichen/llm_watermark/Qwen-Qwen3-14B_strength_2.0_frac_0.5_len_500_num_512.jsonl \
    --only_English \
