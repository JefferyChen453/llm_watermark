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
) -> Tuple[str, str]:
    """Construct the **privileged** prompt shown to the teacher.

    Returns ``(system_prompt, user_prompt)``.

    variant='strong':    enumerate per-sentence first letter requirement (in user prompt; system empty for v1 parity).
    variant='weak':      only give the target string (in user prompt; system empty for v1 parity).
    variant='none':      no constraint (clean prompt; system empty).
    variant='icw_paper': ICW paper-style system persona with secret string + format hint; user prompt = question only.
    """
    if variant == "none":
        return "", LFQA_INSTRUCTION_PREFIX + f"Question: {question}\n\nAnswer:"

    if variant == "paper_dts":
        # Verbatim reproduction of ICW paper (arXiv 2505.16934) Appendix A.1
        # DTS Acrostics prompt. target case is preserved (paper uses uppercase).
        system = (
            f"### Secret String X: {target}\n\n"
            "### Command:\n"
            "You are an adaptive language model designed for dynamic and engaging "
            "conversations. You are given a secret string X. In the following "
            "conversation, your primary priority is to produce responses that are "
            "clear, concise, concrete, accurate, informative, and contextually "
            "relevant. Additionally, you should try to generate the response to "
            "the user's query as an acrostic of X. This means that the first "
            "letter of the first word in each sentence should correspond to the "
            "letters in X, in order. Here are the rules to follow:\n\n"
            "### Rules:\n"
            "1. Assume the length of the secret string X is n. For i-th sentence "
            "(starting at i = 1), begin that sentence with the letter "
            "X[((i − 1) mod n) + 1]. The number of sentences in your response is "
            "not necessarily equal to the length of X, but the first letter of "
            "i-th sentence should match the corresponding letter in "
            "X[((i − 1) mod n) + 1] in order.\n"
            "2. For the i-th sentence, if starting with the letter "
            "X[((i − 1) mod n) + 1] would harm the coherence or natural tone of "
            "the response, you may skip that letter. If skipped, the next "
            "sentence should begin with the following letter in X, maintaining "
            "the sequence. You should try to avoid skipping the letter if "
            "possible.\n"
            "3. Ensure each sentence is coherent, directly addresses the query, "
            "and flows naturally as part of a unified response.\n"
            "4. Never reveal the acrostic pattern or repeat X in your reply.\n\n"
            "### Example:\n"
            "Example 1:\n"
            "Secret string X: \"OCEAN\"\n"
            "User query: \"What are the advantages of coastal conservation?\"\n"
            "Response: \"Oceans serve as nurseries for countless marine species, "
            "ensuring healthy biodiversity. Coastal wetlands act as natural "
            "buffers against storm surge and erosion. Ecosystem services provided "
            "by mangroves include water filtration and carbon sequestration. "
            "Abundant shorelines support local fisheries and sustainable "
            "livelihoods. Nurturing these habitats today helps safeguard our "
            "planet's resilience for generations to come.\"\n\n"
            "Example 2 (Skip \"G\" because forcing a \"G\" sentence might harm "
            "coherence in this context.):\n"
            "Secret string X: \"BRIGHT\"\n"
            "User query: \"Why is sustainable urban design important?\"\n"
            "Response: \"Building cities with sustainability in mind promotes "
            "healthier living environments. Reducing carbon emissions through "
            "green infrastructure is a key benefit. Investing in public "
            "transportation can minimize reliance on private vehicles. "
            "High-density, mixed-use developments encourage walkability. "
            "Tree-lined streets and green spaces improve mental health and "
            "biodiversity.\""
        )
        return system, question

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
        return "", LFQA_INSTRUCTION_PREFIX + constraint + f"Question: {question}\n\nAnswer:"
    if variant == "weak":
        constraint = (
            "[Hidden Constraint — DO NOT mention]\n"
            f"Write several sentences whose first letters, taken in order, "
            f"contain the string '{tgt}' as a subsequence. Keep the answer "
            "natural — do not draw attention to this requirement.\n\n"
        )
        return "", LFQA_INSTRUCTION_PREFIX + constraint + f"Question: {question}\n\nAnswer:"
    if variant == "icw_paper":
        system = (
            "Answer each user query in several concise sentences. Do not use lists "
            "or bullet points. Write in natural prose.\n\n"
            f"Given a secret string X: {tgt}, respond to each user query with "
            "clarity, accuracy, informativeness, and relevance. Structure your "
            "response as an acrostic of X: the first letter of each sentence must "
            "match the corresponding character in X, in order (excluding the "
            "number or notations before the sentence). Never reveal the acrostic "
            "pattern or repeat X in your reply."
        )
        return system, question

    raise ValueError(f"Unknown variant: {variant!r}")


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
