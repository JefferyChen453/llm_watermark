STRENGTHS=(0.0 1.0 2.0 5.0 10.0)

# ----- decapoda-research-llama-7B-hf -----
for s in "${STRENGTHS[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run run_generate.py \
        --model_name baffo32/decapoda-research-llama-7B-hf \
        --strength "$s" \
        --max_new_tokens 500 \
        --output_dir /home/tianyichen/llm_watermark/outputs/max_new_500/baffo32-decapoda-research-llama-7B-hf
done

# ----- meta-llama/Llama-3.1-8B-Instruct -----
for s in "${STRENGTHS[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run run_generate.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --strength "$s" \
        --max_new_tokens 500 \
        --output_dir /home/tianyichen/llm_watermark/outputs/max_new_500/meta-llama-Llama-3.1-8B-Instruct
done


# ----- meta-llama/Llama-2-13b-chat-hf -----
for s in "${STRENGTHS[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run run_generate.py \
        --model_name meta-llama/Llama-2-13b-chat-hf \
        --strength "$s" \
        --max_new_tokens 500 \
        --output_dir /home/tianyichen/llm_watermark/outputs/max_new_500/meta-llama-Llama-2-13b-chat-hf
done
