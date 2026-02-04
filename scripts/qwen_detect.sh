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



# STRENGTHS=(0.0 1.0 2.0 5.0)
# for s in "${STRENGTHS[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --strength "$s" \
#         --input_file /home/tianyichen/llm_watermark/outputs/only_eng/logits_wm/strength/Qwen-Qwen3-14B/Qwen-Qwen3-14B_strength_"$s"_frac_0.5_len_500_num_512_only_English.jsonl \
#         --only_English

#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-32B \
#         --strength "$s" \
#         --input_file /home/tianyichen/llm_watermark/outputs/only_eng/logits_wm/strength/Qwen-Qwen3-32B/Qwen-Qwen3-32B_strength_"$s"_frac_0.5_len_500_num_512_only_English.jsonl \
#         --only_English
# done

# FRACTIONS=(0.1 0.2 0.3 0.4 0.5)
# for f in "${FRACTIONS[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --fraction "$f" \
#         --input_file /home/tianyichen/llm_watermark/outputs/only_eng/logits_wm/fraction/Qwen-Qwen3-14B/Qwen-Qwen3-14B_strength_2.0_frac_"$f"_len_500_num_512_only_English.jsonl \
#         --only_English \
#         --combine_fraction
# done

# FRACTIONS=(0.3)
# # FRACTIONS=(0.0 0.1 0.2 0.3 0.4)
# for f in "${FRACTIONS[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --fraction "$f" \
#         --input_file /home/tianyichen/llm_watermark/outputs/only_eng/incontext_vllm/Qwen-Qwen3-14B_withlinewithoutspace/Qwen-Qwen3-14B_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
#         --only_English \
#         --combine_fraction
# done


STRENGTHS=(0.0 1.0 2.0 5.0 10.0)

for s in "${STRENGTHS[@]}"; do
    uv run run_detect.py \
        --model_name Qwen/Qwen3-14B \
        --strength "$s" \
        --input_file /home/tianyichen/llm_watermark/outputs/test_vllm_gen/Qwen-Qwen3-14B_strength_"$s"_frac_0.5_len_500_num_512_vllm_only_English.jsonl \
        --only_English
done

