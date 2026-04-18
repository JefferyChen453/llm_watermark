#!/usr/bin/env bash
# V3 Initials synthesis + filter pipeline.
#
# Produces cleaner Initials ICW training data with:
#   - stricter prompt (prompt.py::lfqa_initials_v3, no meta-leak terminology)
#   - reject sampling (n_candidates=3 per prompt, pick highest-z passing)
#   - regex meta-leak filter
#   - LLM-Judge (gpt-4o-mini) filter (min_dim <= 3 or overall < 3.5 dropped)
#
# Final output:
#   - verl/data/initials_icw/train_mixed_green3379_initialsV3_neg1000.parquet
#   - verl/data/initials_icw/validation_mixed_green177_initialsV3_177_neg177.parquet
#
# Prerequisites:
#   - GPUs free (v2 training must be finished)
#   - OPENAI_API_KEY exported for LLM-Judge step
#   - posneg_green3379_neg1000.parquet built (auto-constructed in this script)

set -euo pipefail

# Load OPENAI_API_KEY (and any other secrets) — see ~/.env
[ -f "$HOME/.env" ] && . "$HOME/.env"

PROJECT_ROOT="/home/tianyichen/llm_watermark"
PY="${PY:-$PROJECT_ROOT/.venv/bin/python}"
DATE=$(date +%Y%m%d%H%M)

# ---- Output paths ----
OUT_DIR="$PROJECT_ROOT/data/initials_icw"
RAW_JSONL="$OUT_DIR/synthesis_v3_raw_${DATE}.jsonl"
WITH_Z_JSONL="$OUT_DIR/synthesis_v3_raw_${DATE}_with_z.jsonl"
PICKED_JSONL="$OUT_DIR/synthesis_v3_picked_${DATE}.jsonl"
FINAL_JSONL="$OUT_DIR/synthesis_v3_final_${DATE}.jsonl"

POSNEG_V1_PARQUET="$PROJECT_ROOT/verl/data/sft_modified_loss/vblagoje_lfqa/Qwen-Qwen3-14B_strength_3.0_filtered_promptv2_pos_5931_neg_1000.parquet"
# Green-3379 + neg-1000 parquet (dropped initials from v2 mixed parquet)
POSNEG_V3_PARQUET="$PROJECT_ROOT/verl/data/initials_icw/posneg_green3379_neg1000.parquet"
MIXED_V2_PARQUET="$PROJECT_ROOT/verl/data/initials_icw/train_mixed_green3379_initials865_neg1000.parquet"
TRAIN_OUT_PARQUET="$PROJECT_ROOT/verl/data/initials_icw/train_mixed_green3379_initialsV3_neg1000.parquet"
VAL_IN_PARQUET="$PROJECT_ROOT/verl/data/initials_icw/validation_mixed_green177_initials177_neg177.parquet"
VAL_OUT_PARQUET="$PROJECT_ROOT/verl/data/initials_icw/validation_mixed_green177_initialsV3_177_neg177.parquet"

cd "$PROJECT_ROOT"

echo "========== [$(date)] V3 Initials pipeline start =========="

# ---- Stage 0: build posneg_v3 (green3379 + neg1000 only, no initials) ----
if [ ! -f "$POSNEG_V3_PARQUET" ]; then
    echo "[$(date)] Building posneg_v3 parquet from mixed_v2..."
    "$PY" -c "
import pyarrow.parquet as pq
df = pq.read_table('$MIXED_V2_PARQUET').to_pandas()
sub = df[df['task'].isin(['green','neg'])].reset_index(drop=True)
# Restore legacy column name expected by assemble_mixed_train_parquet.py
sub['prompt_no_incontext_wm'] = sub['prompt_ref']
# Keep only columns assemble expects
keep = ['prompt','prompt_no_incontext_wm','response','prefix','seed','z_score','fraction','dataset_type']
sub = sub[keep]
sub.to_parquet('$POSNEG_V3_PARQUET', index=False)
print(f'Wrote posneg_v3: {len(sub)} rows; task breakdown via fraction: pos={int((sub[\"fraction\"]>0).sum())} neg={int((sub[\"fraction\"]==0).sum())}')
"
else
    echo "[$(date)] posneg_v3 parquet already exists, skip"
fi

# ---- Stage 1: Synthesis (3 candidates per prompt, strength=2) ----
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

# ---- Stage 2: metrics (z_score / hit_rate / rep4) + regex flag ----
echo "========== [$(date)] Stage 2: metrics + regex flag =========="
"$PY" filter_initials_syn.py \
    --input_file "$RAW_JSONL" \
    --model_name Qwen/Qwen3-14B \
    --min_gen_len 200 \
    --max_ngram_rep 0.15 \
    --verify_z_primary 6.0 \
    --verify_z_fallback 5.0 \
    --target_min_pos 1500

# filter_initials_syn writes ..._with_z.jsonl; use that for pick_best
# (`*_filtered.jsonl` is a redundant by-product in this pipeline)

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
    echo "ERROR: OPENAI_API_KEY not set. Export and rerun from stage 4."
    exit 1
fi
"$PY" run_llm_judge_initials_filter.py \
    --input_file "$PICKED_JSONL" \
    --output_file "$FINAL_JSONL" \
    --model gpt-4o-mini \
    --concurrency 16 \
    --min_dim_drop 3 \
    --overall_drop 3.5

# ---- Stage 5: assemble train parquet (green3379 + initialsV3 + neg1000) ----
echo "========== [$(date)] Stage 5: assemble train parquet =========="
"$PY" assemble_mixed_train_parquet.py \
    --posneg_parquet "$POSNEG_V3_PARQUET" \
    --initials_filtered_jsonl "$FINAL_JSONL" \
    --output_parquet "$TRAIN_OUT_PARQUET"

# ---- Stage 6: build v3 val parquet (pure Python) ----
echo "========== [$(date)] Stage 6: build v3 val parquet =========="
"$PY" build_val_initials_v3.py \
    --input_val "$VAL_IN_PARQUET" \
    --output_val "$VAL_OUT_PARQUET" \
    --dataset_type lfqa_initials_v3 \
    --eval_initials_seed 0

echo "========== [$(date)] V3 pipeline complete =========="
echo "Final artifacts:"
echo "  Train: $TRAIN_OUT_PARQUET"
echo "  Val:   $VAL_OUT_PARQUET"
