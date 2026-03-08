# export VLLM_USE_TORCH_COMPILE=0
# Fractions=(0.0 0.1 0.2 0.3 0.4)

# for f in "${Fractions[@]}"; do

#     # uv run run_generate_syn.py \
#     #     --model_name Qwen/Qwen3-32B \
#     #     --max_new_tokens 500 \
#     #     --fraction "$f" \
#     #     --strength 2.0 \
#     #     --only_English \
#     #     --output_dir /home/tianyichen/llm_watermark/outputs/syn_data/Qwen-Qwen3-32B_all \
#     uv run run_generate_syn.py \
#         --model_name Qwen/Qwen3-14B \
#         --max_new_tokens 500 \
#         --fraction "$f" \
#         --strength 2.0 \
#         --only_English \
#         --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/inputs.jsonl \
#         --output_dir /home/tianyichen/llm_watermark/debug/Qwen-Qwen3-14B_LFQA \
#         --num_test 500

#     uv run run_generate_syn.py \
#         --model_name Qwen/Qwen3-14B \
#         --max_new_tokens 500 \
#         --fraction "$f" \
#         --strength 2.0 \
#         --only_English \
#         --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/OpenGen/inputs.jsonl \
#         --output_dir /home/tianyichen/llm_watermark/debug/Qwen-Qwen3-14B_OpenGen \
#         --num_test 500

# done

FRACTIONS=(0.0 0.1 0.2 0.3 0.4)
STRENGTHS=(4.0)
for s in "${STRENGTHS[@]}"; do
    for f in "${FRACTIONS[@]}"; do
        uv run run_generate_syn_vllm.py \
            --model_name Qwen/Qwen3-14B \
            --max_new_tokens 500 \
            --batch_size 128 \
            --fraction "$f" \
            --strength "$s" \
            --seed_num 5 \
            --only_English \
            --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/inputs.jsonl \
            --output_dir /home/tianyichen/llm_watermark/debug/test4_5keys
    done
done