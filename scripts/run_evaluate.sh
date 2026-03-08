# STRENGTHS=(2.0 3.0 4.0)
# for s in "${STRENGTHS[@]}"; do
#     uv run evaluate.py /home/tianyichen/llm_watermark/outputs/generate_vllm/strength_"$s" --fraction_or_strength fraction --target_fpr 0.01
# done

uv run evaluate.py /home/tianyichen/llm_watermark/debug/test4_5keys --fraction_or_strength fraction --target_fpr 0.01