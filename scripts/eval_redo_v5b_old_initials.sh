#!/usr/bin/env bash
# Re-run initials gen for v5b_old (the 04-25 master's first pass on this ckpt
# was interrupted, leaving an incomplete 96-record file that was deleted).
# Run AFTER eval_full_matrix.sh finishes to avoid GPU contention.

set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODEL=${ROOT}/verl/checkpoints/watermark-kd-ray/v5b_green3379+initials865+neg1000_dualKL_biasedRefTopK1000_202604180857/global_step_655/hf_model
TAG=v5b_green3379+initials865+neg1000_dualKL_biasedRefTopK1000_202604180857_global_step_655
OUT=${ROOT}/outputs/initials_eval/${TAG}
mkdir -p "$OUT"

uv run run_generate_initials_vllm.py \
  --model_name "$MODEL" \
  --add_icw_prompt \
  --strength 0.0 \
  --seed 42 \
  --num_test 477 \
  --prompt_file "${ROOT}/data/processed_data/vblagoje_lfqa/test_477.json" \
  --output_dir "$OUT" \
  --max_model_len 8192 \
  --tensor_parallel_size 8 \
  --batch_size 32

echo "[$(date)] v5b_old initials redo done."
