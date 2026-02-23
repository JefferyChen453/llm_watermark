FRACTIONS=(0.0 0.1 0.2 0.3 0.4)
for f in "${FRACTIONS[@]}"; do
    uv run run_detect.py \
        --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_278/hf_model \
        --fraction "$f" \
        --input_file /home/tianyichen/llm_watermark/outputs/incontext_train/LFQA+OpenGen-qwen3-14b_global_step_278_LFQA/LFQA+OpenGen-qwen3-14b_global_step_278_strength_0.0_frac_"$f"_len_500_num_500_incontext_vllm_only_English.jsonl \
        --only_English \
        --combine_fraction

    uv run run_detect.py \
        --model_name Qwen/Qwen3-14B \
        --fraction "$f" \
        --input_file /home/tianyichen/llm_watermark/outputs/incontext_train/Qwen-Qwen3-14B_LFQA/Qwen-Qwen3-14B_strength_0.0_frac_"$f"_len_500_num_500_incontext_vllm_only_English.jsonl \
        --only_English \
        --combine_fraction

    # uv run run_detect.py \
    #     --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_278/hf_model \
    #     --fraction "$f" \
    #     --strength 2.0 \
    #     --input_file /home/tianyichen/llm_watermark/outputs/incontext_eval/LFQA+OpenGen-qwen3-14b_global_step_278_LFQA/LFQA+OpenGen-qwen3-14b_global_step_278_strength_2.0_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
    #     --only_English \
    #     --combine_fraction

    # uv run run_detect.py \
    #     --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_556/hf_model \
    #     --fraction "$f" \
    #     --strength 2.0 \
    #     --input_file /home/tianyichen/llm_watermark/outputs/incontext_eval/LFQA+OpenGen-qwen3-14b_global_step_556_LFQA/LFQA+OpenGen-qwen3-14b_global_step_556_strength_2.0_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
    #     --only_English \
    #     --combine_fraction

    # uv run run_detect.py \
    #     --model_name Qwen/Qwen3-14B \
    #     --fraction "$f" \
    #     --strength 2.0 \
    #     --input_file /home/tianyichen/llm_watermark/outputs/incontext_eval/Qwen-Qwen3-14B_LFQA/Qwen-Qwen3-14B_strength_2.0_frac_"$f"_len_500_num_512_incontext_vllm_only_English.jsonl \
    #     --only_English \
    #     --combine_fraction





    # uv run run_detect.py \
    #     --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_278/hf_model \
    #     --fraction "$f" \
    #     --strength 2.0 \
    #     --input_file /home/tianyichen/llm_watermark/outputs/incontext_eval/LFQA+OpenGen-qwen3-14b_global_step_278_OpenGen/LFQA+OpenGen-qwen3-14b_global_step_278_strength_2.0_frac_"$f"_len_500_num_500_incontext_vllm_only_English.jsonl \
    #     --only_English \
    #     --combine_fraction

    # uv run run_detect.py \
    #     --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_556/hf_model \
    #     --fraction "$f" \
    #     --strength 2.0 \
    #     --input_file /home/tianyichen/llm_watermark/outputs/incontext_eval/LFQA+OpenGen-qwen3-14b_global_step_556_OpenGen/LFQA+OpenGen-qwen3-14b_global_step_556_strength_2.0_frac_"$f"_len_500_num_500_incontext_vllm_only_English.jsonl \
    #     --only_English \
    #     --combine_fraction

    # uv run run_detect.py \
    #     --model_name Qwen/Qwen3-14B \
    #     --fraction "$f" \
    #     --strength 2.0 \
    #     --input_file /home/tianyichen/llm_watermark/outputs/incontext_eval/Qwen-Qwen3-14B_OpenGen/Qwen-Qwen3-14B_strength_2.0_frac_"$f"_len_500_num_500_incontext_vllm_only_English.jsonl \
    #     --only_English \
    #     --combine_fraction
done
