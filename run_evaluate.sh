# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-14B /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-14B/Qwen-Qwen3-14B_evaluation.csv --tau_thres 6.0

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-32B /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-32B/Qwen-Qwen3-32B_evaluation.csv --tau_thres 6.0

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-8B /home/tianyichen/llm_watermark/outputs/max_new_500_old/Qwen-Qwen3-8B/Qwen-Qwen3-8B_evaluation.csv --tau_thres 6.0

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/meta-llama-Llama-2-13b-chat-hf /home/tianyichen/llm_watermark/outputs/max_new_500_old/meta-llama-Llama-2-13b-chat-hf/meta-llama-Llama-2-13b-chat-hf_evaluation.csv --tau_thres 6.0

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/meta-llama-Llama-3.1-8B-Instruct /home/tianyichen/llm_watermark/outputs/max_new_500_old/meta-llama-Llama-3.1-8B-Instruct/meta-llama-Llama-3.1-8B-Instruct_evaluation.csv --tau_thres 6.0

# uv run evaluate.py /home/tianyichen/llm_watermark/outputs/max_new_500_old/baffo32-decapoda-research-llama-7B-hf /home/tianyichen/llm_watermark/outputs/max_new_500_old/baffo32-decapoda-research-llama-7B-hf/baffo32-decapoda-research-llama-7B-hf_evaluation.csv --tau_thres 6.0

uv run evaluate.py /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-4B-Instruct-2507 /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-4B-Instruct-2507/Qwen-Qwen3-4B-Instruct-2507_evaluation.csv --fraction_or_strength fraction --tau_thres 4.5 

uv run evaluate.py /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-14B /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-14B/Qwen-Qwen3-14B_evaluation.csv --fraction_or_strength fraction --tau_thres 4.5

uv run evaluate.py /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-32B /home/tianyichen/llm_watermark/outputs/incontext_new/max_new_500/Qwen-Qwen3-32B/Qwen-Qwen3-32B_evaluation.csv --fraction_or_strength fraction --tau_thres 4.5