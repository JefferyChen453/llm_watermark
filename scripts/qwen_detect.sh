# FRACTIONS=(0.1 0.2 0.3 0.4)
# for f in "${FRACTIONS[@]}"; do
#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-32B \
#         --fraction "$f" \
#         --input_file /home/tianyichen/llm_watermark/debug/Qwen-Qwen3-14B_LFQA/Qwen-Qwen3-14B_strength_2.0_frac_"$f"_len_500_num_500_only_English.jsonl \
#         --only_English \
#         --combine_fraction

#     uv run run_detect.py \
#         --model_name Qwen/Qwen3-32B \
#         --fraction "$f" \
#         --input_file /home/tianyichen/llm_watermark/debug/Qwen-Qwen3-14B_OpenGen/Qwen-Qwen3-14B_strength_2.0_frac_"$f"_len_500_num_500_only_English.jsonl \
#         --only_English \
#         --combine_fraction
# done

FRACTIONS=(0.1 0.2 0.3 0.4)
STRENGTHS=(4.0)
for s in "${STRENGTHS[@]}"; do
    for f in "${FRACTIONS[@]}"; do
    uv run run_detect.py \
        --model_name Qwen/Qwen3-14B \
        --fraction "$f" \
        --input_file /home/tianyichen/llm_watermark/debug/test4_5keys/Qwen-Qwen3-14B_strength_"$s"_frac_"$f"_len_500_num_512_vllm_only_English.jsonl \
        --workers 8 \
        --only_English \
        --combine_fraction
        # --wm_key 0 \
    done
done


