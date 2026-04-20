"""Levenshtein z-statistic detector for Acrostics watermark.

Implements ICW paper (arXiv 2505.16934) Section 4.2.4:
  D(y | k_s, τ_s) := (μ − d(ℓ, ζ)) / σ
where ℓ = extract_first_letters(y), ζ = target secret string, d = Levenshtein,
and μ/σ are estimated from N random shuffles of ℓ (null under "letters in ℓ
are arranged uniformly at random w.r.t. ζ").
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from acrostics_icw import _levenshtein, extract_first_letters


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


def compute_lev_zstat(
    text: str,
    target: str,
    n_resample: int = 1000,
    seed: int = 0,
) -> LevZStat:
    """Compute Lev z-stat per paper Section 4.2.4.

    Null distribution: N random permutations of ℓ (preserves letter multiset).
    If ℓ is empty (no detectable sentences), returns z = 0.0.
    """
    fl = extract_first_letters(text)
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
