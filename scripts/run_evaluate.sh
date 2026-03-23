# STRENGTHS=(2.0 3.0 4.0)
# for s in "${STRENGTHS[@]}"; do
#     uv run evaluate.py /home/tianyichen/llm_watermark/outputs/generate_vllm/strength_"$s" --fraction_or_strength fraction --target_fpr 0.01
# done

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/vblagoje_lfqa/strength_2.0 --fraction_or_strength fraction --target_fpr 0.01
# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/vblagoje_lfqa/strength_4.0 --fraction_or_strength fraction --target_fpr 0.01
# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/vblagoje_lfqa/strength_5.0 --fraction_or_strength fraction --target_fpr 0.01

uv run evaluate.py /home/tianyichen/llm_watermark/outputs/syn_data_vblagoje_lfqa_no_system_prompt/strength_2.0 --fraction_or_strength fraction --target_fpr 0.01
