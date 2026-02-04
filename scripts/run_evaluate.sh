# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-14B /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-14B/Qwen-Qwen3-14B_evaluation.csv --tau_thres 6.0

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-32B /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-32B/Qwen-Qwen3-32B_evaluation.csv --tau_thres 6.0

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-8B /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-8B/Qwen-Qwen3-8B_evaluation.csv --tau_thres 6.0

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/meta-llama-Llama-2-13b-chat-hf /home/tianyichen/llm_watermark/outputs/max_new_500_old/meta-llama-Llama-2-13b-chat-hf/meta-llama-Llama-2-13b-chat-hf_evaluation.csv --tau_thres 6.0

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/meta-llama-Llama-3.1-8B-Instruct /home/tianyichen/llm_watermark/outputs/max_new_500_old/meta-llama-Llama-3.1-8B-Instruct/meta-llama-Llama-3.1-8B-Instruct_evaluation.csv --tau_thres 6.0

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/baffo32-decapoda-research-llama-7B-hf /home/tianyichen/llm_watermark/outputs/max_new_500_old/baffo32-decapoda-research-llama-7B-hf/baffo32-decapoda-research-llama-7B-hf_evaluation.csv --tau_thres 6.0

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-4B-Instruct-2507 /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-4B-Instruct-2507/Qwen-Qwen3-4B-Instruct-2507_evaluation.csv --fraction_or_strength fraction --tau_thres 4.5 

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-14B /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-14B/Qwen-Qwen3-14B_evaluation.csv --fraction_or_strength fraction --tau_thres 4.5

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-32B /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-32B/Qwen-Qwen3-32B_evaluation.csv --fraction_or_strength fraction --tau_thres 4.5

# uv run evaluate.py /home/tianyichen/llm_watermark/temp/logits_wm/strength/Qwen-Qwen3-14B /home/tianyichen/llm_watermark/temp/logits_wm/strength/Qwen-Qwen3-14B/Qwen-Qwen3-14B_evaluation.csv --fraction_or_strength strength

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/meta-llama-Llama-2-13b-chat-hf /home/tianyichen/llm_watermark/outputs/meta-llama-Llama-2-13b-chat-hf/meta-llama-Llama-2-13b-chat-hf_evaluation.csv --fraction_or_strength strength --tau_thres 1.5

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/only_eng/logits_wm/strength/Qwen-Qwen3-14B /home/tianyichen/llm_watermark/outputs/only_eng/logits_wm/strength/Qwen-Qwen3-14B/Qwen-Qwen3-14B_evaluation.csv --fraction_or_strength strength

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/only_eng/incontext_vllm/Qwen-Qwen3-14B_withlinewithoutspace /home/tianyichen/llm_watermark/outputs/only_eng/incontext_vllm/Qwen-Qwen3-14B_withlinewithoutspace/Qwen-Qwen3-14B_evaluation.csv --fraction_or_strength fraction --tau_thres 2.0


# uv run evaluate.py /home/tianyichen/llm_watermark/temp/test_vllm_gen /home/tianyichen/llm_watermark/temp/test_vllm_gen/Qwen-Qwen3-14B_evaluation.csv --fraction_or_strength strength

# uv run evaluate.py /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/Qwen-Qwen3-14B /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/Qwen-Qwen3-14B/Qwen-Qwen3-14B_evaluation.csv --fraction_or_strength fraction --tau_thres 1.0

# uv run evaluate.py /home/tianyichen/llm_watermark/temp/incontext_no_logits_wm/Qwen-Qwen3-14B /home/tianyichen/llm_watermark/temp/incontext_no_logits_wm/Qwen-Qwen3-14B/Qwen-Qwen3-14B_evaluation.csv --fraction_or_strength fraction --tau_thres 1.0

uv run evaluate.py /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength5/Qwen-Qwen3-14B /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength5/Qwen-Qwen3-14B/Qwen-Qwen3-14B_evaluation.csv --fraction_or_strength fraction --target_fpr 0.01

uv run evaluate.py /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength3/Qwen-Qwen3-14B /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength3/Qwen-Qwen3-14B/Qwen-Qwen3-14B_evaluation.csv --fraction_or_strength fraction --target_fpr 0.01

uv run evaluate.py /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength2/Qwen-Qwen3-14B /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength2/Qwen-Qwen3-14B/Qwen-Qwen3-14B_evaluation.csv --fraction_or_strength fraction --target_fpr 0.01


uv run evaluate.py /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength3/Qwen-Qwen3-32B /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength3/Qwen-Qwen3-32B/Qwen-Qwen3-32B_evaluation.csv --fraction_or_strength fraction --target_fpr 0.01

uv run evaluate.py /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength2/Qwen-Qwen3-32B /home/tianyichen/llm_watermark/temp/incontext_add_logits_wm/strength2/Qwen-Qwen3-32B/Qwen-Qwen3-32B_evaluation.csv --fraction_or_strength fraction --target_fpr 0.01