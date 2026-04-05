export VLLM_USE_TORCH_COMPILE=0
# Fractions=(0.1 0.2 0.3)
FRACTIONS=(0.0 0.1 0.15 0.2 0.25 0.3 0.35 0.4)

CKPT_ROOT_DIR=/home/tianyichen/llm_watermark/verl/checkpoints
STEPS=(741 1482)

for f in "${FRACTIONS[@]}"; do
    # uv run run_generate_incontext_vllm.py \
    #     --model_name Qwen/Qwen3-14B \
    #     --yarn \
    #     --max_model_len 131072 \
    #     --fraction "$f" \
    #     --seed_num 1 \
    #     --only_English \
    #     --prompt_file /home/tianyichen/llm_watermark/data/processed_data/vblagoje_lfqa/test_477.json \
    #     --output_dir /home/tianyichen/llm_watermark/outputs/incontext_eval/prompt_v2/Qwen-Qwen3-14B
    
    for step in "${STEPS[@]}"; do

        uv run run_generate_incontext_vllm.py \
            --model_name ${CKPT_ROOT_DIR}/watermark-kd-ray/filter_strength_5.0_5931_bsz_8__0.0green+1.0kl_biased_ref_202603300527/global_step_${step}/hf_model \
            --yarn \
            --max_model_len 131072 \
            --fraction "$f" \
            --seed_num 1 \
            --only_English \
            --prompt_file /home/tianyichen/llm_watermark/data/processed_data/vblagoje_lfqa/test_477.json \
            --output_dir /home/tianyichen/llm_watermark/outputs/incontext_eval/prompt_v2_new/filter_strength_5.0_5931_bsz_8__0.0green+1.0kl_biased_ref_202603300527_global_step_${step}
    done
done