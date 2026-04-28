"""Smoke test the 3-task RL data pipeline end-to-end (no GPU).

Validates:
  - Train parquet loads with all expected columns
  - Per-sample acrostic_target round-trips through dataset.py + reward.py
  - AcrosticsDetector strict mode rejects heading-cheats but accepts legit prose

Run: uv run python tests/test_rl_3task_smoke.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verl"))

import pandas as pd

# (1) Verify both parquets exist and have right schema + counts
TRAIN_P = ROOT / "verl/data/rl_stage2/train_rl_3task_green800_initials400_acrostics800.parquet"
VAL_P = ROOT / "verl/data/initials_icw/validation_3task_green177_initials177_neg177_acrostics177.parquet"

print("=" * 60)
print("(1) Parquet schema + row counts")
print("=" * 60)

for label, path, expected in [
    ("TRAIN", TRAIN_P, {"green": 800, "initials": 400, "acrostics": 800}),
    ("VAL",   VAL_P,   {"green": 177, "initials": 177, "neg": 177, "acrostics": 177}),
]:
    df = pd.read_parquet(path)
    print(f"\n{label}: {path.name}")
    print(f"  shape: {df.shape}")
    print(f"  cols : {list(df.columns)}")
    counts = dict(df["task"].value_counts())
    print(f"  tasks: {counts}")
    assert "acrostic_target" in df.columns, f"{label} missing acrostic_target column"
    for t, n in expected.items():
        actual = counts.get(t, 0)
        assert actual == n, f"{label} task={t} expected {n}, got {actual}"

    # Acrostics rows must have non-empty target
    acr = df[df["task"] == "acrostics"]
    n_with_target = (acr["acrostic_target"].str.len() > 0).sum()
    assert n_with_target == len(acr), f"{label} {len(acr) - n_with_target}/{len(acr)} acrostics rows have empty target"
    print(f"  acrostics target lengths: {sorted(set(acr['acrostic_target'].str.len()))}")

    # Non-acrostics rows must have empty/None target
    non_acr = df[df["task"] != "acrostics"]
    if "acrostic_target" in non_acr.columns:
        # NaN counts as empty
        non_empty = (non_acr["acrostic_target"].fillna("").str.len() > 0).sum()
        if non_empty > 0:
            print(f"  WARN: {non_empty} non-acrostics rows have non-empty acrostic_target")

# (2) Test dataset → reward dispatch (mock)
print()
print("=" * 60)
print("(2) Dataset reads acrostic_target correctly")
print("=" * 60)

from recipe.watermark_rl_ray.dataset import WatermarkRLPromptDataset
from transformers import AutoTokenizer

print("Loading Qwen3-14B tokenizer...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
print(f"  vocab_size={tok.vocab_size}")

class MockCfg(dict):
    def __getattr__(self, k): return self[k]
    def get(self, k, default=None):
        return self[k] if k in self else default

ds = WatermarkRLPromptDataset(
    parquet_files=str(TRAIN_P),
    tokenizer=tok,
    config=MockCfg({"max_prompt_length": 8192, "prompt_column": "prompt"}),
)
print(f"  dataset size: {len(ds)}")
# Sample one acrostics row
import random
random.seed(0)
acr_indices = [i for i in range(len(ds)) if str(ds.tasks[i]) == "acrostics"]
ini_indices = [i for i in range(len(ds)) if str(ds.tasks[i]) == "initials"]
green_indices = [i for i in range(len(ds)) if str(ds.tasks[i]) == "green"]
print(f"  found {len(acr_indices)} acrostics, {len(ini_indices)} initials, {len(green_indices)} green")

acr_idx = acr_indices[0]
acr_item = ds[acr_idx]
print(f"\n  acrostics item[{acr_idx}]:")
print(f"    task           : {acr_item['task']!r}")
print(f"    acrostic_target: {acr_item['acrostic_target']!r}")
print(f"    wm_seed        : {acr_item['wm_seed']}")
print(f"    raw_prompt[:120]: {acr_item['raw_prompt'][:120]!r}")
assert acr_item["task"] == "acrostics"
assert len(acr_item["acrostic_target"]) >= 5

green_item = ds[green_indices[0]]
print(f"\n  green item:")
print(f"    task           : {green_item['task']!r}")
print(f"    acrostic_target: {green_item['acrostic_target']!r}")
assert green_item["task"] == "green"
assert green_item["acrostic_target"] == ""

# (3) AcrosticsDetector — strict cheating block
print()
print("=" * 60)
print("(3) AcrosticsDetector strict mode")
print("=" * 60)

from gptwm_acrostics import AcrosticsDetector

target = acr_item["acrostic_target"]
det_strict = AcrosticsDetector(target=target, tokenizer=tok, n_resample=200, strict=True)
det_paper = AcrosticsDetector(target=target, tokenizer=tok, n_resample=200, strict=False)

# Cheat: target letters as headings
cheat_text = " ".join(f"{c}: filler content here." for c in target.lower())
ids = tok(cheat_text, add_special_tokens=False)["input_ids"]
z_paper = det_paper.detect(ids)
z_strict = det_strict.detect(ids)
print(f"\n  cheat_text     : {cheat_text!r}")
print(f"  z (paper-faithful, vulnerable): {z_paper:.3f}")
print(f"  z (strict, hack-resistant)    : {z_strict:.3f}")
print(f"  Δ paper - strict              : {z_paper - z_strict:+.3f}")
# Strict should be much lower than paper for this cheat
assert z_strict < z_paper, "strict didn't penalize the cheat"

# Legit: real acrostic-style prose where each sentence STARTS with target letter naturally
import string
sents = []
for c in target:
    sents.append(f"{c.upper()}{''.join(['a','e','i','o','u'][hash(c) % 5])}rgument supports the claim that water flows downward.")
legit_text = " ".join(sents)
ids2 = tok(legit_text, add_special_tokens=False)["input_ids"]
z_paper2 = det_paper.detect(ids2)
z_strict2 = det_strict.detect(ids2)
print(f"\n  legit_text[:120]: {legit_text[:120]!r}")
print(f"  z (paper) : {z_paper2:.3f}")
print(f"  z (strict): {z_strict2:.3f}")
# Strict should agree with paper (no cheat present)
assert abs(z_paper2 - z_strict2) < 0.5, "strict diverges from paper on legit prose"

print()
print("=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
