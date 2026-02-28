export VLLM_USE_TORCH_COMPILE=0
# Fractions=(0.1 0.2 0.3)
Fractions=(0.0 0.1 0.2 0.3 0.4)

for f in "${Fractions[@]}"; do

    uv run run_generate_incontext_vllm.py \
        --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/combined-qwen3-14b/global_step_278/hf_model \
        --max_new_tokens 500 \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --wm_key 0 \
        --only_English \
        --dataset_type lfqa \
        --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/inputs.jsonl \
        --output_dir /home/tianyichen/llm_watermark/outputs/incontext_eval/combined-qwen3-14b_global_step_278_LFQA_shuffle

done