# # ----- Qwen-3-8B -----

# STRENGTHS=(0.0 1.0 2.0 5.0 10.0 100.0 10000.0)

# for s in "${STRENGTHS[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-8B \
#         --strength "$s" \
#         --input_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-8B/Qwen-Qwen3-8B_strength_"$s"_frac_0.5_len_200_num_500.jsonl

#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-8B \
#         --strength "$s" \
#         --input_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-8B/Qwen-Qwen3-8B_strength_"$s"_frac_0.5_len_200_num_500_only_English.jsonl \
#         --only_English
# done


# ----- Qwen-3-14B -----

STRENGTHS=(0.0 1.0 2.0 5.0 10.0 50.0 100.0)

for s in "${STRENGTHS[@]}"; do
    uv run run_detect.py \
        --model_name Qwen/Qwen3-14B \
        --strength "$s" \
        --input_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-14B-1000/Qwen-Qwen3-14B_strength_"$s"_frac_0.5_len_1000_num_500.jsonl

    uv run run_detect.py \
        --model_name Qwen/Qwen3-14B \
        --strength "$s" \
        --input_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-14B-1000/Qwen-Qwen3-14B_strength_"$s"_frac_0.5_len_1000_num_500_only_English.jsonl \
        --only_English
done

# # ----- Qwen-3-32B -----

# STRENGTHS=(0.0 1.0 2.0 5.0 10.0 100.0 10000.0)

# for s in "${STRENGTHS[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-32B \
#         --strength "$s" \
#         --input_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-32B/Qwen-Qwen3-32B_strength_"$s"_frac_0.5_len_200_num_500.jsonl

#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-32B \
#         --strength "$s" \
#         --input_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-32B/Qwen-Qwen3-32B_strength_"$s"_frac_0.5_len_200_num_500_only_English.jsonl \
#         --only_English
# done

