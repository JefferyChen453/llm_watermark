#!/usr/bin/env bash
# V4 Initials training data synthesis + filter pipeline.
#
# Produces cleaner Initials ICW training data with:
#   - stricter prompt (prompt.py::lfqa_initials_v3 — 3rd iteration of the prompt)
#   - reject sampling (n_candidates=3 per prompt, pick highest-z passing regex)
#   - regex meta-leak filter
#   - LLM-Judge (gpt-4o-mini) filter (min_dim <= 3 or overall < 3.5 dropped)
#
# Produces the "V4" training data (user's run numbering, distinguishing from
# V1 = original mixed, V3 = green3379).  V4 keeps V1's green5931 + neg1000 intact
# (V3 report concluded green data volume matters; don't reduce) and replaces
# only the initials portion with the cleaner v3-prompt synthesis.
#
# Final output:
#   - verl/data/initials_icw/train_mixed_green5931_initialsV4_neg1000.parquet
#   - verl/data/initials_icw/validation_mixed_green177_initialsV4_177_neg177.parquet
#
# Prerequisites:
#   - GPUs free
#   - OPENAI_API_KEY available (loaded from ~/.env)

set -euo pipefail

# Load OPENAI_API_KEY (and any other secrets) — see ~/.env
[ -f "$HOME/.env" ] && . "$HOME/.env"

PROJECT_ROOT="/home/tianyichen/llm_watermark"
PY="${PY:-$PROJECT_ROOT/.venv/bin/python}"
DATE=$(date +%Y%m%d%H%M)

# ---- Output paths ----
SYN_DIR="$PROJECT_ROOT/data/initials_icw"
RAW_JSONL="$SYN_DIR/synthesis_v4_raw_${DATE}.jsonl"
WITH_Z_JSONL="$SYN_DIR/synthesis_v4_raw_${DATE}_with_z.jsonl"
PICKED_JSONL="$SYN_DIR/synthesis_v4_picked_${DATE}.jsonl"
FINAL_JSONL="$SYN_DIR/synthesis_v4_final_${DATE}.jsonl"

# Source parquet (v1 original posneg: green5931 + neg1000).  Already has the legacy
# `prompt_no_incontext_wm` column that assemble_mixed_train_parquet.py expects.
POSNEG_V1_PARQUET="$PROJECT_ROOT/verl/data/sft_modified_loss/vblagoje_lfqa/Qwen-Qwen3-14B_strength_3.0_filtered_promptv2_pos_5931_neg_1000.parquet"

TRAIN_OUT_PARQUET="$PROJECT_ROOT/verl/data/initials_icw/train_mixed_green5931_initialsV4_neg1000.parquet"
VAL_IN_PARQUET="$PROJECT_ROOT/verl/data/initials_icw/validation_mixed_green177_initials177_neg177.parquet"
VAL_OUT_PARQUET="$PROJECT_ROOT/verl/data/initials_icw/validation_mixed_green177_initialsV4_177_neg177.parquet"

cd "$PROJECT_ROOT"

echo "========== [$(date)] V4 Initials data pipeline start =========="

# ---- Stage 1: Synthesis (3 candidates per prompt, strength=2, new prompt) ----
echo "========== [$(date)] Stage 1: synthesis =========="
"$PY" run_generate_initials_syn_vllm.py \
    --model_name Qwen/Qwen3-14B \
    --train_file data/processed_data/vblagoje_lfqa/train_11578.json \
    --posneg_parquet "$POSNEG_V1_PARQUET" \
    --stats_file data/initials_icw/leading_space_first_letter_stats.json \
    --dataset_type lfqa_initials_v3 \
    --output_file "$RAW_JSONL" \
    --num_samples 2000 \
    --n_candidates 3 \
    --strength 2.0 \
    --min_new_tokens 500 \
    --max_new_tokens 600 \
    --batch_size 64 \
    --tensor_parallel_size 8 \
    --max_model_len 4096

# ---- Stage 2: metrics (z_score / hit_rate / rep4) + regex meta-leak flag ----
echo "========== [$(date)] Stage 2: metrics + regex flag =========="
"$PY" filter_initials_syn.py \
    --input_file "$RAW_JSONL" \
    --model_name Qwen/Qwen3-14B \
    --min_gen_len 200 \
    --max_ngram_rep 0.15 \
    --verify_z_primary 6.0 \
    --verify_z_fallback 5.0 \
    --target_min_pos 1500

# filter_initials_syn writes ..._with_z.jsonl (per-record metrics + regex flag)
# We use that for pick_best.  `*_filtered.jsonl` is a by-product we ignore.

# ---- Stage 3: pick best candidate per (prefix, seed) ----
echo "========== [$(date)] Stage 3: pick best candidate =========="
"$PY" pick_best_candidate_initials.py \
    --input_file "$WITH_Z_JSONL" \
    --output_file "$PICKED_JSONL" \
    --min_gen_len 200 \
    --max_ngram_rep 0.15 \
    --min_z 6.0

# ---- Stage 4: LLM-Judge filter ----
echo "========== [$(date)] Stage 4: LLM-Judge filter =========="
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set (expected from ~/.env). Export and rerun from stage 4."
    exit 1
fi
"$PY" run_llm_judge_initials_filter.py \
    --input_file "$PICKED_JSONL" \
    --output_file "$FINAL_JSONL" \
    --model gpt-4o-mini \
    --concurrency 16 \
    --min_dim_drop 3 \
    --overall_drop 3.5

# ---- Stage 5: assemble V4 train parquet (green5931 + initialsV4 + neg1000) ----
echo "========== [$(date)] Stage 5: assemble V4 train parquet =========="
"$PY" assemble_mixed_train_parquet.py \
    --posneg_parquet "$POSNEG_V1_PARQUET" \
    --initials_filtered_jsonl "$FINAL_JSONL" \
    --output_parquet "$TRAIN_OUT_PARQUET"

# ---- Stage 6: build V4 val parquet (pure Python, no vLLM) ----
echo "========== [$(date)] Stage 6: build V4 val parquet =========="
"$PY" build_val_initials_v3.py \
    --input_val "$VAL_IN_PARQUET" \
    --output_val "$VAL_OUT_PARQUET" \
    --dataset_type lfqa_initials_v3 \
    --eval_initials_seed 0

echo "========== [$(date)] V4 pipeline complete =========="
echo "Final artifacts:"
echo "  Train: $TRAIN_OUT_PARQUET"
echo "  Val:   $VAL_OUT_PARQUET"
