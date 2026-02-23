export VLLM_USE_TORCH_COMPILE=0
# Fractions=(0.1 0.2 0.3)
Fractions=(0.0 0.1 0.2 0.3 0.4)

for f in "${Fractions[@]}"; do
    uv run run_generate_incontext_vllm.py \
        --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_278/hf_model \
        --max_new_tokens 500 \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --only_English \
        --dataset_type lfqa \
        --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/train.jsonl \
        --output_dir /home/tianyichen/llm_watermark/outputs/incontext_train/LFQA+OpenGen-qwen3-14b_global_step_278_LFQA \
        --num_test 500

    uv run run_generate_incontext_vllm.py \
        --model_name Qwen/Qwen3-14B \
        --max_new_tokens 500 \
        --yarn \
        --max_model_len 131072 \
        --fraction "$f" \
        --only_English \
        --dataset_type lfqa \
        --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/train.jsonl \
        --output_dir /home/tianyichen/llm_watermark/outputs/incontext_train/Qwen-Qwen3-14B_LFQA \
        --num_test 500        


    # uv run run_generate_incontext_vllm.py \
    #     --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_278/hf_model \
    #     --max_new_tokens 500 \
    #     --yarn \
    #     --max_model_len 131072 \
    #     --fraction "$f" \
    #     --only_English \
    #     --dataset_type lfqa \
    #     --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/inputs.jsonl \
    #     --output_dir /home/tianyichen/llm_watermark/outputs/incontext_eval/LFQA+OpenGen-qwen3-14b_global_step_278_LFQA
    #     # --save_gen_batch \

    # uv run run_generate_incontext_vllm.py \
    #     --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_556/hf_model \
    #     --max_new_tokens 500 \
    #     --yarn \
    #     --max_model_len 131072 \
    #     --fraction "$f" \
    #     --only_English \
    #     --dataset_type lfqa \
    #     --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/inputs.jsonl \
    #     --output_dir /home/tianyichen/llm_watermark/outputs/incontext_eval/LFQA+OpenGen-qwen3-14b_global_step_556_LFQA

    # uv run run_generate_incontext_vllm.py \
    #     --model_name Qwen/Qwen3-14B \
    #     --max_new_tokens 500 \
    #     --yarn \
    #     --max_model_len 131072 \
    #     --fraction "$f" \
    #     --only_English \
    #     --dataset_type lfqa \
    #     --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/inputs.jsonl \
    #     --output_dir /home/tianyichen/llm_watermark/outputs/incontext_eval/Qwen-Qwen3-14B_LFQA



    # uv run run_generate_incontext_vllm.py \
    #     --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_278/hf_model \
    #     --max_new_tokens 500 \
    #     --yarn \
    #     --max_model_len 131072 \
    #     --fraction "$f" \
    #     --only_English \
    #     --dataset_type opengen \
    #     --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/OpenGen/inputs.jsonl \
    #     --output_dir /home/tianyichen/llm_watermark/outputs/incontext_eval/LFQA+OpenGen-qwen3-14b_global_step_278_OpenGen
    #     # --save_gen_batch \

    # uv run run_generate_incontext_vllm.py \
    #     --model_name /home/tianyichen/llm_watermark/verl/checkpoints/watermark-sft/LFQA+OpenGen-qwen3-14b/global_step_556/hf_model \
    #     --max_new_tokens 500 \
    #     --yarn \
    #     --max_model_len 131072 \
    #     --fraction "$f" \
    #     --only_English \
    #     --dataset_type opengen \
    #     --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/OpenGen/inputs.jsonl \
    #     --output_dir /home/tianyichen/llm_watermark/outputs/incontext_eval/LFQA+OpenGen-qwen3-14b_global_step_556_OpenGen

    # uv run run_generate_incontext_vllm.py \
    #     --model_name Qwen/Qwen3-14B \
    #     --max_new_tokens 500 \
    #     --yarn \
    #     --max_model_len 131072 \
    #     --fraction "$f" \
    #     --only_English \
    #     --dataset_type opengen \
    #     --prompt_file /home/tianyichen/llm_watermark/UnigramWatermark/data/OpenGen/inputs.jsonl \
    #     --output_dir /home/tianyichen/llm_watermark/outputs/incontext_eval/Qwen-Qwen3-14B_OpenGen
done