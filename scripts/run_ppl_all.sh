#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_ppl_all.sh [root_dir]
# Default root_dir: current directory
# Environment overrides:
#   MODEL_NAME  (default: Qwen/Qwen3-Next-80B-A3B-Instruct)
#   BATCH_SIZE  (default: 2)
#   MAX_LENGTH  (default: 8192)

ROOT_DIR="${1:-/home/tianyichen/llm_watermark/UnigramWatermark/data/LFQA/baffo32-decapoda-research-llama-7B-hf}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-8192}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/compute_ppl.py"


ROOT_DIR="/home/tianyichen/llm_watermark/outputs/only_eng/incontext_vllm/Qwen-Qwen3-14B_withlinewithoutspace"
find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
  echo "Processing ${file}"
  python3 "${PY_SCRIPT}" \
    --input "${file}" \
    --model-name "${MODEL_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}"
done


# ROOT_DIR="/home/tianyichen/llm_watermark/outputs/max_new_500/baffo32-decapoda-research-llama-7B-hf"

# find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
#   echo "Processing ${file}"
#   python3 "${PY_SCRIPT}" \
#     --input "${file}" \
#     --model-name "${MODEL_NAME}" \
#     --batch-size "${BATCH_SIZE}" \
#     --max-length "${MAX_LENGTH}"
# done


# ROOT_DIR="/home/tianyichen/llm_watermark/outputs/max_new_500/meta-llama-Llama-2-13b-chat-hf"

# find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
#   echo "Processing ${file}"
#   python3 "${PY_SCRIPT}" \
#     --input "${file}" \
#     --model-name "${MODEL_NAME}" \
#     --batch-size "${BATCH_SIZE}" \
#     --max-length "${MAX_LENGTH}"
# done



# ROOT_DIR="/home/tianyichen/llm_watermark/outputs/max_new_500/meta-llama-Llama-3.1-8B-Instruct"

# find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
#   echo "Processing ${file}"
#   python3 "${PY_SCRIPT}" \
#     --input "${file}" \
#     --model-name "${MODEL_NAME}" \
#     --batch-size "${BATCH_SIZE}" \
#     --max-length "${MAX_LENGTH}"
# done


# ROOT_DIR="/home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-8B"

# find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
#   echo "Processing ${file}"
#   python3 "${PY_SCRIPT}" \
#     --input "${file}" \
#     --model-name "${MODEL_NAME}" \
#     --batch-size "${BATCH_SIZE}" \
#     --max-length "${MAX_LENGTH}"
# done

# ROOT_DIR="/home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-14B"

# find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
#   echo "Processing ${file}"
#   python3 "${PY_SCRIPT}" \
#     --input "${file}" \
#     --model-name "${MODEL_NAME}" \
#     --batch-size "${BATCH_SIZE}" \
#     --max-length "${MAX_LENGTH}"
# done


# ROOT_DIR="/home/tianyichen/llm_watermark/outputs/max_new_500/Qwen-Qwen3-32B"

# find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
#   echo "Processing ${file}"
#   python3 "${PY_SCRIPT}" \
#     --input "${file}" \
#     --model-name "${MODEL_NAME}" \
#     --batch-size "${BATCH_SIZE}" \
#     --max-length "${MAX_LENGTH}"
# done

