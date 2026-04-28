"""Extend train_rl_green1000_initials1000.parquet to a 1400 green + 600 initials parquet.

Strategy (decided 2026-04-27):
  - Keep all 1000 existing green rows (seeds 501..2999, fractions 334/333/333).
  - Sub-sample 600 of the existing 1000 initials (deterministic).
  - Build 400 NEW green rows whose prefixes are disjoint from sft_filtered + mixed_v3 + existing rl_stage2 parquet.
  - New green seeds drawn from 3000..5000 (small-int convention, trivially disjoint from existing 501..2999).
  - New green fractions = 133/134/133 over [0.1, 0.2, 0.3] so the merged 1400-row block lands at 467/467/466.

Output: verl/data/rl_stage2/train_rl_green1400_initials600.parquet (2000 rows total).
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import apply_chat_template
from prompt import get_incontext_system_prompt
from gptwm_incontext import InContextWatermarkGenerator


REPO = Path("/home/tianyichen/llm_watermark")
EXISTING_RL = REPO / "verl/data/rl_stage2/train_rl_green1000_initials1000.parquet"
SFT_FILTERED = REPO / "verl/data/sft_modified_loss/vblagoje_lfqa/Qwen-Qwen3-14B_strength_3.0_filtered_promptv2_pos_5931_neg_1000.parquet"
MIXED_V3 = REPO / "verl/data/initials_icw/train_mixed_green3379_initials865_neg1000.parquet"
LFQA_JSONL = REPO / "data/processed_data/vblagoje_lfqa/train_11578.json"
DEFAULT_OUTPUT = REPO / "verl/data/rl_stage2/train_rl_green1400_initials600.parquet"

NEW_GREEN_FRACTION_COUNTS = {0.1: 133, 0.2: 134, 0.3: 133}  # sums to 400


def load_lfqa(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--strength", type=float, default=2.0,
                    help="placeholder for InContextWatermarkGenerator; mask is independent of strength")
    ap.add_argument("--only_english", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # ---- 1. Load existing rl parquet ----
    base = pd.read_parquet(EXISTING_RL)
    existing_green = base[base["task"] == "green"].reset_index(drop=True)
    existing_initials = base[base["task"] == "initials"].reset_index(drop=True)
    assert len(existing_green) == 1000, f"expected 1000 existing green, got {len(existing_green)}"
    assert len(existing_initials) == 1000, f"expected 1000 existing initials, got {len(existing_initials)}"

    # ---- 2. Sub-sample 600 initials (deterministic) ----
    sub_initials = existing_initials.sample(n=600, random_state=args.seed).reset_index(drop=True)

    # ---- 3. Build 400 new green ----
    sft = pd.read_parquet(SFT_FILTERED, columns=["prefix", "seed"])
    mix = pd.read_parquet(MIXED_V3, columns=["prefix", "seed"])
    used_prefixes = set(sft["prefix"]) | set(mix["prefix"]) | set(base["prefix"])
    used_seeds = set(int(s) for s in pd.concat([sft["seed"], mix["seed"], base["seed"]]))
    print(f"Excluded: {len(used_prefixes)} prefixes, {len(used_seeds)} seeds")

    lfqa = load_lfqa(LFQA_JSONL)
    fresh_pool = [r for r in lfqa if r["prefix"] not in used_prefixes]
    print(f"Fresh LFQA pool: {len(fresh_pool)} (need 400)")
    assert len(fresh_pool) >= 400, f"need ≥400 fresh prefixes, only {len(fresh_pool)}"

    chosen_idxs = rng.permutation(len(fresh_pool))[:400].tolist()
    chosen_prefixes = [fresh_pool[i] for i in chosen_idxs]

    # New seeds in 3000..5000 disjoint from used (existing green is 501..2999, so trivially disjoint)
    candidate_seeds = [s for s in range(3000, 5000) if s not in used_seeds]
    assert len(candidate_seeds) >= 400, f"need ≥400 candidate seeds, only {len(candidate_seeds)}"
    new_seeds = rng.choice(candidate_seeds, size=400, replace=False).astype(int).tolist()

    # Fractions: 133/134/133 → 1400 grand total = 467/467/466 with the existing 334/333/333
    fracs = []
    for f, n in NEW_GREEN_FRACTION_COUNTS.items():
        fracs.extend([f] * n)
    assert len(fracs) == 400
    rng.shuffle(fracs)

    # ---- Tokenizer + model config ----
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    model_emb_length = model_config.vocab_size

    print(f"\nBuilding 400 new green rows...")
    new_records = []
    for i, prefix_row in enumerate(tqdm(chosen_prefixes, desc="green-new")):
        seed = int(new_seeds[i])
        fraction = float(fracs[i])
        gen = InContextWatermarkGenerator(
            fraction=fraction,
            strength=args.strength,
            vocab_size=vocab_size,
            model_emb_length=model_emb_length,
            watermark_key=seed,
            only_English=args.only_english,
            tokenizer=tokenizer,
        )
        green_string = gen.get_green_token_string(shuffle=True)
        sysp_wm = get_incontext_system_prompt("lfqa", green_string)
        sysp_ref = get_incontext_system_prompt("lfqa", "")
        new_records.append({
            "prompt": apply_chat_template(tokenizer, sysp_wm, prefix_row["prefix"]),
            "prompt_ref": apply_chat_template(tokenizer, sysp_ref, prefix_row["prefix"]),
            "prefix": prefix_row["prefix"],
            "seed": seed,
            "fraction": fraction,
            "task": "green",
            "dataset_type": "lfqa",
        })
    new_green_df = pd.DataFrame(new_records)

    # ---- 4. Concat + shuffle ----
    out = pd.concat([existing_green, new_green_df, sub_initials], ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # ---- 5. Verify ----
    print(f"\nFinal: {len(out)} rows")
    print(f"  task: {out['task'].value_counts().to_dict()}")
    print(f"  green fractions: {out[out['task']=='green']['fraction'].value_counts().sort_index().to_dict()}")
    print(f"  green seed range: {out[out['task']=='green']['seed'].min()}..{out[out['task']=='green']['seed'].max()}")
    print(f"  initials seed range: {out[out['task']=='initials']['seed'].min()}..{out[out['task']=='initials']['seed'].max()}")
    assert (out["task"] == "green").sum() == 1400
    assert (out["task"] == "initials").sum() == 600
    assert out["prefix"].nunique() == 2000
    new_seed_set = set(int(s) for s in new_green_df["seed"])
    assert not (new_seed_set & used_seeds), "new green seeds overlap used set!"
    assert not (set(new_green_df["prefix"]) & used_prefixes), "new green prefixes overlap used set!"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
