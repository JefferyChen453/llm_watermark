"""Initials ICW watermark: bias leading-space tokens whose first letter is in
the ``green`` set (13 of 26 letters selected per seed).

Detection: z-score on the fraction of leading-space English tokens whose first
letter is green, using an empirical per-seed γ computed from the token-count
distribution over A-Z.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import norm

from gptwm import _get_english_token_ids, GPTWatermarkBase


LETTERS = list(string.ascii_uppercase)  # ['A', ..., 'Z']


# ---------- Helpers: first-letter classification ----------

def first_letter_of_token_string(tok_str: str) -> Optional[str]:
    """Return the uppercase first letter (A-Z) if the token string starts with
    a leading space followed by an ASCII letter; else None."""
    if not tok_str.startswith(" "):
        return None
    rest = tok_str[1:]
    if not rest:
        return None
    c = rest[0]
    if c.isalpha() and c.isascii():
        return c.upper()
    return None


def build_token_first_letter_map(
    tokenizer, vocab_size: int, english_token_ids: Optional[List[int]] = None
) -> Dict[int, str]:
    """Return {token_id: first_letter} for english+leading-space tokens."""
    if english_token_ids is None:
        english_token_ids = _get_english_token_ids(tokenizer, vocab_size)
    out: Dict[int, str] = {}
    for tid in english_token_ids:
        tok_str = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([tid]))
        letter = first_letter_of_token_string(tok_str)
        if letter is not None:
            out[tid] = letter
    return out


# ---------- Partition: 13 green / 13 red per seed ----------

def partition_letters(seed: int, n_green: int = 13) -> Tuple[List[str], List[str]]:
    """Deterministically split A-Z into green and red sets using ``seed``.
    Green letters are returned alphabetically; red likewise."""
    rng = np.random.default_rng(seed)
    idx = np.arange(26)
    rng.shuffle(idx)
    green_idx = sorted(idx[:n_green].tolist())
    red_idx = sorted(idx[n_green:].tolist())
    green = [LETTERS[i] for i in green_idx]
    red = [LETTERS[i] for i in red_idx]
    return green, red


# ---------- Mask building ----------

def build_initials_mask_numpy(
    seed: int,
    vocab_size: int,
    model_emb_length: int,
    tokenizer,
    english_token_ids: Optional[List[int]] = None,
    first_letter_map: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Return (mask, green_letters, red_letters). Mask is ``(model_emb_length,)``
    bool, True at token_ids whose first letter is in the green set.
    """
    assert model_emb_length > vocab_size, "model_emb_length must exceed vocab_size"
    if first_letter_map is None:
        first_letter_map = build_token_first_letter_map(tokenizer, vocab_size, english_token_ids)
    green, red = partition_letters(seed)
    green_set = set(green)
    mask = np.zeros(model_emb_length, dtype=bool)
    for tid, letter in first_letter_map.items():
        if letter in green_set:
            mask[tid] = True
    return mask, green, red


def compute_gamma_from_stats(green_letters: List[str], stats_path: str) -> float:
    """Compute the null-hypothesis P(first-letter ∈ green) = sum of per-letter
    token fractions over green letters. Uses ``leading_space_first_letter_stats.json``.
    """
    with open(stats_path) as f:
        stats = json.load(f)
    frac = stats["per_letter_fraction_letter_initial"]
    return float(sum(frac.get(ltr, 0.0) for ltr in green_letters))


# ---------- Base / Detector ----------

class InitialsWatermarkBase:
    """Holds the green/red partition and the corresponding token-id mask for a
    given seed and tokenizer."""

    def __init__(
        self,
        seed: int,
        strength: float,
        vocab_size: int,
        model_emb_length: int,
        tokenizer,
        english_token_ids: Optional[List[int]] = None,
        first_letter_map: Optional[Dict[int, str]] = None,
    ):
        self.seed = seed
        self.strength = strength
        self.vocab_size = vocab_size
        self.model_emb_length = model_emb_length
        self.tokenizer = tokenizer
        if first_letter_map is None:
            first_letter_map = build_token_first_letter_map(tokenizer, vocab_size, english_token_ids)
        self.first_letter_map = first_letter_map
        mask_np, green, red = build_initials_mask_numpy(
            seed, vocab_size, model_emb_length, tokenizer,
            english_token_ids=english_token_ids, first_letter_map=first_letter_map,
        )
        self.mask = torch.tensor(mask_np, dtype=torch.float32)
        self.green_letters = green
        self.red_letters = red
        self.green_set = set(green)


class InitialsDetector(InitialsWatermarkBase):
    """Z-score detector for Initials ICW.

    For a decoded response, tokenize (``add_special_tokens=False``), filter to
    english+leading-space tokens, and compare the fraction of green-initial
    tokens to the per-seed null expectation γ.
    """

    def __init__(self, gamma: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)

    @staticmethod
    def _z_score(num_green: int, total: int, gamma: float) -> float:
        if total == 0:
            return 0.0
        p = gamma
        return (num_green - p * total) / np.sqrt(p * (1.0 - p) * total)

    def hits(self, token_ids: List[int]) -> Tuple[int, int]:
        """Return (num_green_initial, num_leading_space_english) over the
        token_ids sequence (response only; do NOT pass prefix tokens in)."""
        n_total = 0
        n_green = 0
        for tid in token_ids:
            letter = self.first_letter_map.get(int(tid))
            if letter is None:
                continue
            n_total += 1
            if letter in self.green_set:
                n_green += 1
        return n_green, n_total

    def detect(self, token_ids: List[int]) -> float:
        n_green, n_total = self.hits(token_ids)
        return self._z_score(n_green, n_total, self.gamma)

    def unidetect(self, token_ids: List[int]) -> float:
        """Z-score using unique tokens only — analogous to the green-watermark
        ``unidetect`` (reduces autocorrelation)."""
        unique = list(set(int(t) for t in token_ids))
        return self.detect(unique)

    def hit_rate(self, token_ids: List[int]) -> float:
        n_green, n_total = self.hits(token_ids)
        return n_green / n_total if n_total > 0 else 0.0


# ---------- Smoke test ----------

if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    config = AutoConfig.from_pretrained("Qwen/Qwen3-14B")
    stats_path = "data/initials_icw/leading_space_first_letter_stats.json"

    green, red = partition_letters(seed=0)
    print(f"seed=0 green: {green}")
    print(f"seed=0 red:   {red}")

    gamma = compute_gamma_from_stats(green, stats_path)
    print(f"gamma (seed=0): {gamma:.4f}")

    base = InitialsWatermarkBase(
        seed=0, strength=5.0,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=config.vocab_size,
        tokenizer=tokenizer,
    )
    print(f"mask.sum() = {int(base.mask.sum())} (token count on green-initial leading-space english)")
    print(f"mask fraction of vocab: {float(base.mask.sum() / base.model_emb_length):.4f}")

    detector = InitialsDetector(
        gamma=gamma, seed=0, strength=5.0,
        vocab_size=tokenizer.vocab_size,
        model_emb_length=config.vocab_size,
        tokenizer=tokenizer,
    )
    text = "Assembling a Roman legion was a complex process. The senate approved the assembly."
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    print(f"\ntext: {text!r}")
    print(f"z_score: {detector.detect(ids):.3f}")
    print(f"hit_rate: {detector.hit_rate(ids):.3f}")
    print(f"unidetect: {detector.unidetect(ids):.3f}")
