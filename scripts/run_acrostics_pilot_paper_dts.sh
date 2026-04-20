#!/bin/bash
# Faithful ICW-paper (arXiv 2505.16934) DTS Acrostics setting reproduction on Qwen3-14B.
# 100 ELI5 queries × uppercase X (len=10) × paper Appendix A.1 DTS system prompt.
# Negatives = human gold_completion from same queries, paired by same X.
set -euo pipefail
cd /home/tianyichen/llm_watermark

MODEL=Qwen/Qwen3-14B
MODEL_TAG=qwen3-14b
NUM=100
LEN=10
SEED=20260421
TAG=paperdts
RAW_DIR=outputs/acrostics_pilot
OUTDIR=outputs/acrostics_pilot/paper_dts_repro

mkdir -p "$OUTDIR"

POS_FILE=${RAW_DIR}/acrostics_pilot_${MODEL_TAG}_n${NUM}_len${LEN}_${TAG}.jsonl
NEG_FILE=${RAW_DIR}/acrostics_pilot_${MODEL_TAG}_n${NUM}_len${LEN}_${TAG}_negatives.jsonl

# 1. Positive: LLM generate with paper DTS system prompt (system = paper A.1 verbatim)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
.venv/bin/python run_acrostics_pilot_vllm.py \
    --model_name "$MODEL" \
    --model_tag "$MODEL_TAG" \
    --prompt_file data/processed_data/vblagoje_lfqa/test_477.json \
    --num_test $NUM \
    --target_length $LEN \
    --seed_base $SEED \
    --target_uppercase \
    --variants paper_dts \
    --temperature 0.7 --top_p 0.9 --max_tokens 600 \
    --output_dir "$RAW_DIR" \
    --output_tag $TAG

# 2. Negative: human gold_completion paired with same X
.venv/bin/python build_acrostics_negatives.py \
    --model_tag "$MODEL_TAG" \
    --prompt_file data/processed_data/vblagoje_lfqa/test_477.json \
    --num_test $NUM \
    --target_length $LEN \
    --seed_base $SEED \
    --target_uppercase \
    --output_dir "$RAW_DIR" \
    --output_tag $TAG

# 3. Analyze: Lev z-stat (N=1000) + ROC-AUC + T@1%FPR / T@10%FPR
.venv/bin/python analyze_acrostics_paper_dts.py \
    --positives_file "$POS_FILE" \
    --negatives_file "$NEG_FILE" \
    --output_dir "$OUTDIR" \
    --n_resample 1000
