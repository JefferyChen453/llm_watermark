"""Acrostics In-Context Watermark utilities.

Task: embed a target string T (e.g., "asdf") into the response such that the
first letters of consecutive sentences form T (or contain T as a subsequence).

Detection: extract first-letter-of-each-sentence sequence S from response;
match against T via subsequence (or Levenshtein distance).

This module provides:
  * ``build_acrostic_prompt`` — construct a privileged prompt (strong / weak)
    that tells the teacher model about the constraint.
  * ``find_decision_points`` — on a generated response, locate the token
    indices that are sentence-starting positions (the "decision points" where
    the acrostic constraint matters).
  * ``extract_first_letters`` — from a response text, return the sequence of
    sentence-start letters.
  * ``verify_acrostic`` — check if target T is a subsequence of S; also compute
    Levenshtein distance to nearest substring.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------- Sentence segmentation ----------

# Sentence boundary: .!? followed by whitespace + uppercase letter.
# The leading-boundary (start of response) is handled separately.
# Heuristic to skip common abbreviations (Mr. Dr. etc.) -- conservative here,
# we do NOT skip them since pilot accuracy of segmentation is less critical
# than scope; verifier tolerates noise.
SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Za-z])")
# Response-leading first letter: first non-whitespace word char.
LEADING_LETTER_RE = re.compile(r"^\s*[\"'(\[]*([A-Za-z])")


def find_sentence_starts_in_text(text: str) -> List[int]:
    """Return character indices of sentence-start positions in ``text``.

    Includes position 0 (start of text) if text begins with a letter (after
    optional quote / bracket / whitespace).
    """
    starts: List[int] = []
    # Leading
    m = LEADING_LETTER_RE.match(text)
    if m:
        starts.append(m.start(1))
    # Subsequent sentence starts
    for m in SENT_BOUNDARY_RE.finditer(text):
        # sentence start is at m.end() (first alpha char after the whitespace)
        starts.append(m.end())
    # Dedup and sort
    return sorted(set(starts))


def extract_first_letters(text: str) -> str:
    """Concatenate the first letters of each detected sentence, lowercase."""
    starts = find_sentence_starts_in_text(text)
    letters: List[str] = []
    for i in starts:
        if i < len(text) and text[i].isalpha():
            letters.append(text[i].lower())
    return "".join(letters)


# ---------- Decision points at token level ----------

def find_decision_points_tokens(
    text: str,
    offset_mapping: List[Tuple[int, int]],
) -> List[int]:
    """Given tokenizer ``offset_mapping`` of the response, return the token
    indices where each sentence starts (1 index per detected sentence).

    ``offset_mapping[i] = (char_start, char_end)`` for token i.
    """
    char_starts = find_sentence_starts_in_text(text)
    token_starts: List[int] = []
    cs_iter = iter(char_starts)
    next_cs = next(cs_iter, None)
    for tok_idx, (cs, ce) in enumerate(offset_mapping):
        while next_cs is not None and cs <= next_cs < ce:
            # This token covers the sentence-start char
            token_starts.append(tok_idx)
            next_cs = next(cs_iter, None)
        if next_cs is None:
            break
    return token_starts


# ---------- Prompt construction ----------

LFQA_INSTRUCTION_PREFIX = (
    "Answer the following question in several concise sentences. Do not use "
    "lists or bullet points. Write in natural prose.\n\n"
)


def _letter_list_str(target: str) -> str:
    # "a, s, d, f" — lowercase, comma separated
    return ", ".join(target.lower())


def build_acrostic_prompt(
    question: str,
    target: str,
    variant: str = "strong",
) -> str:
    """Construct the **privileged** prompt shown to the teacher.

    variant='strong': enumerate per-sentence first letter requirement.
    variant='weak':   only give the target string.
    variant='none':   no constraint (clean prompt = what student sees).

    The response format is "several sentences in natural prose", same across
    variants, to keep the only difference being the constraint text.
    """
    if variant == "none":
        return LFQA_INSTRUCTION_PREFIX + f"Question: {question}\n\nAnswer:"

    tgt = target.lower()
    if variant == "strong":
        per_sent = "\n".join(
            f"- Sentence {i+1} must start with the letter '{c}'."
            for i, c in enumerate(tgt)
        )
        constraint = (
            "[Hidden Constraint — DO NOT mention]\n"
            f"Your answer must contain at least {len(tgt)} sentences. The first "
            f"letters of the first {len(tgt)} sentences, read in order, must "
            f"spell '{tgt}'. Specifically:\n"
            f"{per_sent}\n"
            "Additional sentences beyond that are allowed. Keep the answer "
            "natural — do not draw attention to this structural requirement.\n\n"
        )
    elif variant == "weak":
        constraint = (
            "[Hidden Constraint — DO NOT mention]\n"
            f"Write several sentences whose first letters, taken in order, "
            f"contain the string '{tgt}' as a subsequence. Keep the answer "
            "natural — do not draw attention to this requirement.\n\n"
        )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    return LFQA_INSTRUCTION_PREFIX + constraint + f"Question: {question}\n\nAnswer:"


# ---------- Verification ----------

@dataclass
class AcrosticVerdict:
    target: str
    first_letters: str
    is_subsequence: bool          # T ⊆subseq S
    is_contiguous: bool           # T appears contiguously in S
    levenshtein_to_substring: int  # min Lev(T, S[i:i+|T|]) across i
    n_sentences: int


def _is_subsequence(target: str, seq: str) -> bool:
    it = iter(seq)
    return all(c in it for c in target)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def verify_acrostic(text: str, target: str) -> AcrosticVerdict:
    tgt = target.lower()
    fl = extract_first_letters(text)
    is_sub = _is_subsequence(tgt, fl)
    is_contig = tgt in fl
    # Min Levenshtein to any length-|tgt| window of fl
    if len(fl) < len(tgt):
        lev = _levenshtein(tgt, fl)
    else:
        lev = min(
            _levenshtein(tgt, fl[i : i + len(tgt)])
            for i in range(len(fl) - len(tgt) + 1)
        )
    return AcrosticVerdict(
        target=tgt,
        first_letters=fl,
        is_subsequence=is_sub,
        is_contiguous=is_contig,
        levenshtein_to_substring=lev,
        n_sentences=len(fl),
    )


# ---------- Target sampling ----------

def sample_target(seed: int, length: int = 4, pool: str = string.ascii_lowercase) -> str:
    import random
    rng = random.Random(seed)
    return "".join(rng.choices(pool, k=length))
