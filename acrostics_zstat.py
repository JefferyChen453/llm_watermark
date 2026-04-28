"""Acrostics watermark detectors: Levenshtein z-stat (paper-faithful) and
Smith-Waterman z-stat (new, larger dynamic range).

Both use shuffle-S permutation null:
  Lev:  D = (μ − d_obs) / σ   where d = Lev distance, lower = better
  SW:   z = (s_obs − μ) / σ   where s = SW score, higher = better

Both shuffle the observed first-letter sequence ℓ and recompute the metric
n_resample times to build the null. Lev follows ICW §4.2.4 verbatim; SW is
adapted from standard local-alignment scoring (+2 / -1 / -1).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from acrostics_icw import (
    _levenshtein,
    extract_first_letters,
    extract_first_letters_strict,
)


@dataclass
class LevZStat:
    fl: str
    target: str
    d_obs: int
    mu: float
    sigma: float
    z: float
    n_resample: int
    n_sentences: int


@dataclass
class SWZStat:
    fl: str                      # extracted first-letter sequence (lowercased)
    target: str                  # original target (lowercased)
    target_eff: str              # T truncated to len(fl) if shorter; else == target
    obs: int                     # SW score on (target_eff, fl)
    mu: float                    # null mean (shuffle-S)
    sigma: float                 # null std
    z: float                     # empirical zE = (obs - μ) / σ
    p: float                     # one-sided permutation p with Laplace smoothing
    n_resample: int
    n_sentences: int             # = len(fl)
    extractor: str               # "regex_strict" / "regex_loose" / "nltk"


# ---------- Smith-Waterman ----------

def smith_waterman(T: str, S: str,
                   match: int = 2, mismatch: int = -1, gap: int = -1) -> int:
    """Standard Smith-Waterman local alignment max score.
    O(|T|*|S|) time and space (suitable for |T|, |S| ≤ ~100)."""
    m, n = len(T), len(S)
    if m == 0 or n == 0:
        return 0
    H = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = match if T[i - 1] == S[j - 1] else mismatch
            v = max(0,
                    H[i - 1][j - 1] + s,
                    H[i - 1][j] + gap,
                    H[i][j - 1] + gap)
            H[i][j] = v
            if v > best:
                best = v
    return best


_NLTK_READY = False


def _ensure_nltk_punkt() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass  # older NLTK doesn't need punkt_tab
    _NLTK_READY = True


def _get_extractor(name: str):
    """Return the extractor function by name."""
    if name == "regex_strict":
        return extract_first_letters_strict
    if name == "regex_loose":
        return extract_first_letters
    if name == "nltk":
        # Lazy import to avoid hard nltk dep at module load
        from acrostics_icw import extract_first_letters_nltk  # noqa: F401
        _ensure_nltk_punkt()
        return extract_first_letters_nltk
    raise ValueError(f"unknown extractor {name!r}; "
                     "expected one of regex_strict / regex_loose / nltk")


def compute_sw_zstat(
    text: str,
    target: str,
    n_resample: int = 1000,
    seed: int = 0,
    strict: bool = True,
    truncate_target: bool = True,
    extractor: str = "regex_strict",
) -> SWZStat:
    """Smith-Waterman z-stat with shuffle-S null.

    Args:
        text: model response text.
        target: secret string (case-insensitive).
        n_resample: # shuffles for null distribution. p-floor = 1/(n+1).
        seed: RNG seed for shuffles (reproducibility).
        strict: kept for API parity; if extractor='regex_*' it picks
            strict-vs-loose. If extractor='nltk' this flag is ignored.
        truncate_target: if True and len(fl) < len(target), use target[:len(fl)]
            as the effective target. This matches the semantic "model wrote N
            sentences = signal compared against first N letters of secret".
        extractor: 'regex_strict' (default, RL-reward style) /
            'regex_loose' / 'nltk' (ICW paper style).

    Returns: SWZStat with empirical zE as the primary signal and a permutation
    p-value with Laplace smoothing.
    """
    # Resolve extractor (override `extractor` arg via legacy `strict` flag if
    # caller specified strict=False AND extractor default).
    if extractor == "regex_strict" and strict is False:
        extractor = "regex_loose"

    extr_fn = _get_extractor(extractor)
    fl = extr_fn(text)
    tgt = target.lower()

    # Edge case: no detectable sentences → zero signal
    if not fl:
        return SWZStat(
            fl="", target=tgt, target_eff="", obs=0,
            mu=0.0, sigma=0.0, z=0.0, p=1.0,
            n_resample=0, n_sentences=0, extractor=extractor,
        )

    # Truncate target if generation is shorter
    if truncate_target and len(fl) < len(tgt):
        tgt_eff = tgt[:len(fl)]
    else:
        tgt_eff = tgt

    obs = smith_waterman(tgt_eff, fl)

    # Null distribution: shuffle fl, recompute SW(tgt_eff, perm_fl)
    rng = random.Random(seed)
    fl_chars = list(fl)
    null_scores = []
    for _ in range(n_resample):
        rng.shuffle(fl_chars)
        null_scores.append(smith_waterman(tgt_eff, "".join(fl_chars)))

    mu = sum(null_scores) / len(null_scores)
    if len(null_scores) > 1:
        var = sum((x - mu) ** 2 for x in null_scores) / (len(null_scores) - 1)
        sigma = math.sqrt(var)
    else:
        sigma = 0.0

    z = (obs - mu) / sigma if sigma > 0 else 0.0
    k_ge = sum(1 for x in null_scores if x >= obs)
    p = (k_ge + 1) / (n_resample + 1)

    return SWZStat(
        fl=fl, target=tgt, target_eff=tgt_eff, obs=obs,
        mu=mu, sigma=sigma, z=z, p=p,
        n_resample=n_resample, n_sentences=len(fl),
        extractor=extractor,
    )


# ---------- Levenshtein z-stat (paper-faithful, kept for back-compat) ----------

def compute_lev_zstat(
    text: str,
    target: str,
    n_resample: int = 1000,
    seed: int = 0,
    strict: bool = False,
) -> LevZStat:
    """Compute Lev z-stat per paper Section 4.2.4.

    Null distribution: N random permutations of ℓ (preserves letter multiset).
    If ℓ is empty (no detectable sentences), returns z = 0.0.

    Args:
        strict: If True, use ``extract_first_letters_strict`` which filters
            single-letter / numbered-list heading cheats. Use this for RL
            reward to prevent reward hacking. Default False keeps paper-
            faithful detection behavior for evaluation comparisons.
    """
    extractor = extract_first_letters_strict if strict else extract_first_letters
    fl = extractor(text)
    tgt = target.lower()
    d_obs = _levenshtein(fl, tgt)

    if not fl:
        return LevZStat(fl="", target=tgt, d_obs=d_obs, mu=float(d_obs),
                        sigma=0.0, z=0.0, n_resample=0, n_sentences=0)

    rng = random.Random(seed)
    fl_list = list(fl)
    dists = []
    for _ in range(n_resample):
        rng.shuffle(fl_list)
        dists.append(_levenshtein("".join(fl_list), tgt))
    mu = sum(dists) / len(dists)
    if len(dists) > 1:
        var = sum((d - mu) ** 2 for d in dists) / (len(dists) - 1)
    else:
        var = 0.0
    sigma = math.sqrt(var)
    z = (mu - d_obs) / sigma if sigma > 0 else 0.0
    return LevZStat(fl=fl, target=tgt, d_obs=d_obs, mu=mu, sigma=sigma, z=z,
                    n_resample=n_resample, n_sentences=len(fl))
