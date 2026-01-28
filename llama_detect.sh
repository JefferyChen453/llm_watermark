STRENGTHS=(0.0 1.0 2.0 5.0 10.0)

# ----- decapoda-research-llama-7B-hf -----
for s in "${STRENGTHS[@]}"; do
    uv run run_detect.py \
        --model_name baffo32/decapoda-research-llama-7B-hf \
        --strength "$s" \
        --input_file /home/tianyichen/llm_watermark/outputs/max_new_500/baffo32-decapoda-research-llama-7B-hf/baffo32-decapoda-research-llama-7B-hf_strength_"$s"_frac_0.5_len_500_num_500.jsonl
done

# ----- meta-llama/Llama-3.1-8B-Instruct -----
for s in "${STRENGTHS[@]}"; do
    uv run run_detect.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --strength "$s" \
        --input_file /home/tianyichen/llm_watermark/outputs/max_new_500/meta-llama-Llama-3.1-8B-Instruct/meta-llama-Llama-3.1-8B-Instruct_strength_"$s"_frac_0.5_len_500_num_500.jsonl
done


# ----- meta-llama/Llama-2-13b-chat-hf -----
for s in "${STRENGTHS[@]}"; do
    uv run run_detect.py \
        --model_name meta-llama/Llama-2-13b-chat-hf \
        --strength "$s" \
        --input_file /home/tianyichen/llm_watermark/outputs/max_new_500/meta-llama-Llama-2-13b-chat-hf/meta-llama-Llama-2-13b-chat-hf_strength_"$s"_frac_0.5_len_500_num_500.jsonl
done