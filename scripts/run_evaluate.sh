# STRENGTHS=(2.0 3.0 4.0)
# for s in "${STRENGTHS[@]}"; do
#     uv run evaluate.py /home/tianyichen/llm_watermark/outputs/generate_vllm/strength_"$s" --fraction_or_strength fraction --target_fpr 0.01
# done

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/vblagoje_lfqa/strength_2.0 --fraction_or_strength fraction --target_fpr 0.01
# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/vblagoje_lfqa/strength_4.0 --fraction_or_strength fraction --target_fpr 0.01
# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/vblagoje_lfqa/strength_5.0 --fraction_or_strength fraction --target_fpr 0.01

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/incontext_eval/prompt_v2/Qwen-Qwen3-14B --fraction_or_strength fraction --target_fpr 0.01
uv run evaluate.py /home/tianyichen/llm_watermark/outputs/incontext_eval_clean/filter_strength_5.0_5931_bsz_8__0.0green+1.0kl_biased_ref_202603300527_global_step_741 --fraction_or_strength fraction --target_fpr 0.01
uv run evaluate.py /home/tianyichen/llm_watermark/outputs/incontext_eval_clean/filter_strength_5.0_5931_bsz_8__0.0green+1.0kl_biased_ref_202603300527_global_step_1482 --fraction_or_strength fraction --target_fpr 0.01
uv run evaluate.py /home/tianyichen/llm_watermark/outputs/incontext_eval_clean/posneg_strength_5.0_pos_5931_neg_1000_bsz_8__1.0kl_biased_ref+1.0kl_ref_202604110238_global_step_866 --fraction_or_strength fraction --target_fpr 0.01
uv run evaluate.py /home/tianyichen/llm_watermark/outputs/incontext_eval_clean/posneg_strength_5.0_pos_5931_neg_1000_bsz_8__1.0kl_biased_ref+1.0kl_ref_202604110238_global_step_1732 --fraction_or_strength fraction --target_fpr 0.01
