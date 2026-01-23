# CUDA_VISIBLE_DEVICES=0,1,2,3 uv run run_detect.py \
#     --model_name Qwen/Qwen3-8B \
#     --input_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-8B/Qwen-Qwen3-8B_strength_0.0_frac_0.5_len_200_num_500_only_English.jsonl \


uv run run_generate.py \
        --model_name Qwen/Qwen3-8B \
        --strength 2.0 \
        --output_dir /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/Qwen-Qwen3-8B \
        --only_English