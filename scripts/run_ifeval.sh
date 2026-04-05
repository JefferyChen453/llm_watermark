uv pip install antlr4-python3-runtime==4.9.3

# lighteval vllm \
#     "model_name=Qwen/Qwen3-14B,dtype=bfloat16,gpu_memory_utilization=0.9" \
#     "ifeval|0" \
#     --output-dir outputs/ifeval_nothink \
#     --save-details
lighteval vllm \
    "model_name=/home/tianyichen/llm_watermark/verl/checkpoints/watermark-kd-ray/filter_strength_5.0_5931_bsz_8__0.0green+1.0reverse_kl_biased_ref_202604032354/global_step_741/hf_model,dtype=bfloat16,gpu_memory_utilization=0.9" \
    "ifeval|0" \
    --output-dir outputs/ifeval_nothink \
    --save-details

lighteval vllm \
    "model_name=/home/tianyichen/llm_watermark/verl/checkpoints/watermark-kd-ray/filter_strength_5.0_5931_bsz_8__0.0green+1.0reverse_kl_biased_ref_202604032354/global_step_1482/hf_model,dtype=bfloat16,gpu_memory_utilization=0.9" \
    "ifeval|0" \
    --output-dir outputs/ifeval_nothink \
    --save-details


uv pip install --upgrade antlr4-python3-runtime