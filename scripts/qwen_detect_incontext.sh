FRACTIONS=(0.0 0.1 0.15 0.2 0.25 0.3 0.35 0.4)
for f in "${FRACTIONS[@]}"; do
    # uv run run_detect.py \
    #     --model_name Qwen/Qwen3-14B \
    #     --fraction "$f" \
    #     --input_file /home/tianyichen/llm_watermark/outputs/incontext_eval/prompt_v2/Qwen-Qwen3-14B/Qwen-Qwen3-14B_strength_0.0_frac_"$f"_len_600_num_477_incontext_vllm_only_English.jsonl \
    #     --workers 32 \
    #     --only_English \
    #     --combine_fraction

    uv run run_detect.py \
        --model_name Qwen/Qwen3-14B \
        --fraction "$f" \
        --input_file /home/tianyichen/llm_watermark/outputs/incontext_eval/prompt_v2_new/filter_strength_5.0_5931_bsz_8__0.0green+1.0kl_biased_ref_202603300527_global_step_741/filter_strength_5.0_5931_bsz_8__0.0green+1.0kl_biased_ref_202603300527_global_step_741_strength_0.0_frac_"$f"_len_600_num_477_incontext_vllm_only_English.jsonl \
        --workers 1 \
        --only_English \
        --combine_fraction \
        --use_generated_neg_data

    uv run run_detect.py \
        --model_name Qwen/Qwen3-14B \
        --fraction "$f" \
        --input_file /home/tianyichen/llm_watermark/outputs/incontext_eval/prompt_v2_new/filter_strength_5.0_5931_bsz_8__0.0green+1.0kl_biased_ref_202603300527_global_step_1482/filter_strength_5.0_5931_bsz_8__0.0green+1.0kl_biased_ref_202603300527_global_step_1482_strength_0.0_frac_"$f"_len_600_num_477_incontext_vllm_only_English.jsonl \
        --workers 1 \
        --only_English \
        --combine_fraction \
        --use_generated_neg_data
done


