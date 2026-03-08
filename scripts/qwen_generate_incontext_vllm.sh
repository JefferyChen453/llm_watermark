export VLLM_USE_TORCH_COMPILE=0
# Fractions=(0.1 0.2 0.3)
Fractions=(0.0 0.1 0.2 0.3 0.4)
# sleep 2 hours
sleep 7200
for f in "${Fractions[@]}"; do

    uv run run_generate_incontext_vllm.py \
        --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/wmkey+shuffle_combined/global_step_217/hf_model \
        --max_new_tokens 500 \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --wm_key 10000 \
        --only_English \
        --dataset_type lfqa \
        --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/inputs.jsonl \
        --output_dir /home/tianyichen/llm_watermark/outputs/incontext_eval/wmkey+shuffle_combined_global_step_217_LFQA_shuffle


    uv run run_generate_incontext_vllm.py \
        --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/wmkey+shuffle_combined/global_step_217/hf_model \
        --max_new_tokens 500 \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --wm_key 10000 \
        --only_English \
        --dataset_type opengen \
        --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/OpenGen/inputs.jsonl \
        --output_dir /home/tianyichen/llm_watermark/outputs/incontext_eval/wmkey+shuffle_combined_global_step_217_OpenGen_shuffle
done