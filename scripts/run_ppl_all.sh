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
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-8192}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/../compute_ppl.py"


# ROOT_DIR="/home/tianyichen/llm_watermark/temp/sft_train/strength2/Qwen-Qwen3-14B_LFQA"
# find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
#   echo "Processing ${file}"
#   python3 "${PY_SCRIPT}" \
#     --input "${file}" \
#     --model-name "${MODEL_NAME}" \
#     --batch-size "${BATCH_SIZE}" \
#     --max-length "${MAX_LENGTH}"
# done

# ROOT_DIR="/home/tianyichen/llm_watermark/temp/sft_train/strength2/Qwen-Qwen3-32B_LFQA"
# find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
#   echo "Processing ${file}"
#   python3 "${PY_SCRIPT}" \
#     --input "${file}" \
#     --model-name "${MODEL_NAME}" \
#     --batch-size "${BATCH_SIZE}" \
#     --max-length "${MAX_LENGTH}"
# done

ROOT_DIR="/home/tianyichen/llm_watermark/outputs/sft_train/Qwen-Qwen3-14B_LFQA"
find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
  echo "Processing ${file}"
  python3 "${PY_SCRIPT}" \
    --input "${file}" \
    --model-name "${MODEL_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}"
done

ROOT_DIR="/home/tianyichen/llm_watermark/outputs/sft_train/Qwen-Qwen3-14B_OpenGen"
find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
  echo "Processing ${file}"
  python3 "${PY_SCRIPT}" \
    --input "${file}" \
    --model-name "${MODEL_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}"
done

ROOT_DIR="/home/tianyichen/llm_watermark/outputs/sft_train/Qwen-Qwen3-32B_LFQA"
find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
  echo "Processing ${file}"
  python3 "${PY_SCRIPT}" \
    --input "${file}" \
    --model-name "${MODEL_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}"
done


ROOT_DIR="/home/tianyichen/llm_watermark/outputs/sft_train/Qwen-Qwen3-32B_OpenGen"
find "${ROOT_DIR}" -type f -name "*.jsonl" ! -name "*_z.jsonl" | while read -r file; do
  echo "Processing ${file}"
  python3 "${PY_SCRIPT}" \
    --input "${file}" \
    --model-name "${MODEL_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}"
done