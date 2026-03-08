Fractions=(0.1 0.2 0.3 0.4)
for f in "${Fractions[@]}"; do

    uv run run_detect.py \
        --model_name Qwen/Qwen3-14B \
        --fraction "$f" \
        --test_min_tokens 200 \
        --input_file /home/tianyichen/llm_watermark/outputs/incontext_eval/wmkey+shuffle_combined_global_step_217_LFQA_shuffle/wmkey+shuffle_combined_global_step_217_strength_0.0_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
        --only_English \
        --combine_fraction

    uv run run_detect.py \
        --model_name Qwen/Qwen3-14B \
        --fraction "$f" \
        --test_min_tokens 200 \
        --input_file /home/tianyichen/llm_watermark/outputs/incontext_eval/wmkey+shuffle_combined_global_step_217_OpenGen_shuffle/wmkey+shuffle_combined_global_step_217_strength_0.0_frac_"$f"_len_500_num_500_incontext_vllm_only_English.jsonl \
        --only_English \
        --combine_fraction

done
