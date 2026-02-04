OUTPUT_ROOT_DIR=/home/tianyichen/llm_watermark/temp

# Fractions=(0.0 0.1 0.25 0.5)

# # ----- Qwen-3-4B-Instruct-2507 -----
# for f in "${Fractions[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-4B-Instruct-2507 \
#         --fraction "$f" \
#         --test_min_tokens 200 \
#         --input_file ${OUTPUT_ROOT_DIR}/incontext_new/max_new_500/Qwen-Qwen3-4B-Instruct-2507/Qwen-Qwen3-4B-Instruct-2507_frac_"$f"_len_500_num_512_incontext_vllm.jsonl \
#         --combine_fraction
# done

# Fractions=(0.0 0.1 0.15 0.2)

# # ----- Qwen-3-14B -----
# for f in "${Fractions[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --fraction "$f" \
#         --test_min_tokens 200 \
#         --input_file ${OUTPUT_ROOT_DIR}/incontext_new/max_new_500/Qwen-Qwen3-14B/Qwen-Qwen3-14B_frac_"$f"_len_500_num_512_incontext_vllm.jsonl \
#         --combine_fraction
# done


# # ----- Qwen-3-32B -----
# for f in "${Fractions[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-32B \
#         --fraction "$f" \
#         --test_min_tokens 200 \
#         --input_file ${OUTPUT_ROOT_DIR}/incontext_new/max_new_500/Qwen-Qwen3-32B/Qwen-Qwen3-32B_frac_"$f"_len_500_num_512_incontext_vllm.jsonl \
#         --combine_fraction
# done

# Fractions=(0.1 0.2 0.3 0.4)

# for f in "${Fractions[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --fraction "$f" \
#         --test_min_tokens 200 \
#         --input_file ${OUTPUT_ROOT_DIR}/incontext_add_logits_wm/Qwen-Qwen3-14B/Qwen-Qwen3-14B_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
#         --combine_fraction

#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --fraction "$f" \
#         --test_min_tokens 200 \
#         --input_file ${OUTPUT_ROOT_DIR}/incontext_no_logits_wm/Qwen-Qwen3-14B/Qwen-Qwen3-14B_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
#         --combine_fraction
# done


# Fractions=(0.1 0.2 0.3 0.4)

# for f in "${Fractions[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --fraction "$f" \
#         --test_min_tokens 200 \
#         --input_file ${OUTPUT_ROOT_DIR}/incontext_add_logits_wm/Qwen-Qwen3-14B/Qwen-Qwen3-14B_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
#         --combine_fraction \
#         --only_English

#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-14B \
#         --fraction "$f" \
#         --test_min_tokens 200 \
#         --input_file ${OUTPUT_ROOT_DIR}/incontext_no_logits_wm/Qwen-Qwen3-14B/Qwen-Qwen3-14B_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
#         --combine_fraction \
#         --only_English
# done

Fractions=(0.1 0.2 0.3 0.4)
for f in "${Fractions[@]}"; do
    uv run run_detect.py \
        --model_name Qwen/Qwen3-14B \
        --fraction "$f" \
        --test_min_tokens 200 \
        --input_file /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength3/Qwen-Qwen3-14B/Qwen-Qwen3-14B_strength_3.0_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
        --only_English \
        --combine_fraction

    uv run run_detect.py \
        --model_name Qwen/Qwen3-32B \
        --fraction "$f" \
        --test_min_tokens 200 \
        --input_file /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength2/Qwen-Qwen3-32B/Qwen-Qwen3-32B_strength_2.0_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
        --only_English \
        --combine_fraction

    uv run run_detect.py \
        --model_name Qwen/Qwen3-32B \
        --fraction "$f" \
        --test_min_tokens 200 \
        --input_file /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength3/Qwen-Qwen3-32B/Qwen-Qwen3-32B_strength_3.0_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
        --only_English \
        --combine_fraction
done