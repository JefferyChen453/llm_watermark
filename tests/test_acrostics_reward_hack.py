"""Reward-hack analysis for acrostics first-letter extraction.

Demonstrates failure modes of the current ``extract_first_letters`` and tests a
proposed strict variant that closes the cheats.

Run:  uv run pytest tests/test_acrostics_reward_hack.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from acrostics_icw import extract_first_letters  # current (vulnerable) version


# ---- Reward-hack examples (each describes ONE class of cheat) ----

CHEATS = {
    "heading_colon_singleletter": (
        # User's example: "a: hello. s: world. d: foo. f: bar."
        "A: Hello there. S: Some content. D: Direct answer. F: Final remark.",
        "asdf",   # what target the cheat targets
        "asdf",   # what the CURRENT detector extracts (cheat works)
        "",       # what STRICT detector should extract (cheat blocked)
    ),
    "heading_period_singleletter": (
        # "A. ... S. ... D. ..."
        "A. Apples are red. S. Some fruits taste tart. D. Daring chefs invent recipes. F. Fresh produce matters.",
        "asdf",
        # current: leading 'A' (idx 0) + SENT_BOUNDARY matches BOTH 'A. A' (after the heading period)
        # AND '. S' (after the prose period) — model gets DOUBLE letters per item: "aassddff"
        # which still contains "asdf" as a subsequence -> high z-stat.
        "aassddff",
        "",
    ),
    "numbered_list": (
        # "1. Apples ... 2. Some ... 3. Daffodils ... 4. Friends ..."
        "1. Apples are sweet and crunchy.\n2. Some berries grow in clusters.\n3. Daffodils bloom in spring.\n4. Friends share garden tips.",
        "asdf",
        # SENT_BOUNDARY matches '1. A' (since '.' + ' ' + 'A'), etc. → "asdf"
        "asdf",
        "",
    ),
    "lowercase_letter_list": (
        # "a) hello b) world c) foo"
        "a) Hello there. s) Some content. d) Direct answer. f) Final remark.",
        "asdf",
        # SENT_BOUNDARY requires '.!?' before space+letter so first transitions
        # may or may not match — let's see. Leading 'a', then '. s' matches → 's',
        # '. d' matches → 'd', '. f' matches → 'f'.  Cheat = 'asdf'
        "asdf",
        "",
    ),
    "newline_heading_then_prose": (
        # "Apologies for any confusion.\nS: Initialize ...\nD: Compute ..."
        # Multi-line with newline as whitespace before single-letter heading.
        "Apologies for any confusion.\nS: Initialize variables first.\nD: Compute the result.\nF: Return the value.",
        "asdf",
        "asdf",   # current: 'A' (leading) + '. S' '. D' '. F' = "ASDF" lowercased = "asdf"
        # strict: 'A' is the leading letter of a NATURAL SENTENCE ("Apologies for any
        # confusion."), so it stays. The S/D/F headings are blocked. So strict = "a",
        # which is FAR from target "asdf" (Lev=3, not enough for a high z-stat) → cheat blocked.
        "a",
    ),
}


# ---- Legitimate cases (must still extract correctly) ----

LEGIT = {
    "natural_acrostic_short": (
        # Real acrostic: "Apples are red. She likes fruit. Dogs run. Fish swim."
        "Apples are red. She likes fruit. Dogs run. Fish swim.",
        "asdf",
        "asdf",  # both current and strict should extract "asdf"
    ),
    "natural_acrostic_long_first_word": (
        # Long real sentences each starting with right letter
        "Apples grow on tall trees in many regions. She cultivates them with great care every season. Dogs sometimes nose around the orchard. Friends visit during the harvest.",
        "asdf",
        "asdf",
    ),
    "no_acrostic_neutral": (
        # Random LFQA-style answer
        "The Roman legion was assembled by careful planning. Soldiers were drafted from the citizenry. Officers led training exercises. Smiths forged weapons in dedicated workshops.",
        "asdf",
        "tsos",  # actual first letters; not the cheat target
    ),
}


# ---- Strict variant: closes single-letter heading + numbered/letter list ----

import re

# A "real sentence start" letter requires that the FIRST WHITESPACE-DELIMITED
# token at that position be a normal word (≥2 letters, not a heading marker).
HEADING_PATTERNS = [
    re.compile(r"^[A-Za-z][:.\)]\s"),    # "A: ", "A. ", "a) "
    re.compile(r"^\d+[:.\)]\s"),          # "1. ", "1) ", "1: "
    re.compile(r"^[A-Za-z]\s"),           # very rarely "A boy" but we'd catch this fine; we DON'T blacklist (prefer FN over FP for legit)
]


def _is_heading_start(text: str, idx: int) -> bool:
    """Return True if the segment starting at ``idx`` looks like a list/heading
    item (single letter or digit followed by `:`/`.`/`)` and whitespace).
    """
    seg = text[idx : idx + 6]   # short window
    # match patterns 0 and 1 (single-letter or digit heading); skip pattern 2 (false positives)
    return bool(HEADING_PATTERNS[0].match(seg)) or bool(HEADING_PATTERNS[1].match(seg))


# Detect a list-item marker in the window immediately preceding a sentence-start
# position. The marker — 1-2 digits or a single letter — must be ISOLATED, i.e.,
# preceded by start-of-text or whitespace, then followed by `:.)` and 1+ spaces.
# This avoids false-positives on natural sentence boundaries like "...red. She"
# where "d. " superficially looks like a 1-char marker.
LIST_MARKER_BEFORE_RE = re.compile(r"(?:^|[\s])(?:\d{1,2}|[A-Za-z])[:.\)]\s+$")


def extract_first_letters_strict(text: str) -> str:
    """Strict variant of ``extract_first_letters`` that filters out:

    - Single-letter "headings" like ``A: ...``, ``A. ...`` where the entire
      "first word" is just one alphabetic char followed by `:`/`.`/`)` and
      whitespace. The dominant reward-hack pattern.
    - Numbered/letter-list items like ``1. ...``, ``2: ...``, ``a) ...``. The
      detector previously flagged the *next* word after such a marker as a
      sentence start; the strict variant treats list items as headings.
    """
    from acrostics_icw import find_sentence_starts_in_text

    starts = find_sentence_starts_in_text(text)
    letters = []
    for i in starts:
        if i >= len(text) or not text[i].isalpha():
            continue
        # (1) Block single-letter heading at THIS position: "A: ", "A. ", "A) "
        if i + 1 < len(text) and text[i + 1] in ":).":
            if i + 2 >= len(text) or text[i + 2].isspace():
                continue
        # (2) Block "list-item next-word": preceding window ends with an
        # ISOLATED marker token + delimiter + whitespace.
        window = text[max(0, i - 6) : i]
        if LIST_MARKER_BEFORE_RE.search(window):
            continue
        letters.append(text[i].lower())
    return "".join(letters)


# ---- Tests ----

def _print_compare(label: str, text: str, target: str, current: str, strict: str) -> None:
    print(f"\n[{label}]")
    print(f"  text   : {text!r}")
    print(f"  target : {target}")
    print(f"  current: {current}")
    print(f"  strict : {strict}")


def test_cheats_break_current_pass_strict():
    """Each cheat case: current returns the target (CHEAT WINS), strict returns ''."""
    failures = []
    for name, (text, target, expect_current, expect_strict) in CHEATS.items():
        cur = extract_first_letters(text)
        strict = extract_first_letters_strict(text)
        _print_compare(name, text, target, cur, strict)
        if cur != expect_current:
            failures.append(f"current[{name}]: expected {expect_current!r}, got {cur!r}")
        if strict != expect_strict:
            failures.append(f"strict[{name}]: expected {expect_strict!r}, got {strict!r}")
    assert not failures, "\n".join(failures)


def test_legit_unchanged():
    """Legitimate prose: strict and current agree (or strict ≥ current correctness)."""
    failures = []
    for name, (text, target, expect_letters) in LEGIT.items():
        cur = extract_first_letters(text)
        strict = extract_first_letters_strict(text)
        _print_compare(name, text, target, cur, strict)
        if cur != expect_letters:
            failures.append(f"current[{name}]: expected {expect_letters!r}, got {cur!r}")
        if strict != expect_letters:
            failures.append(f"strict[{name}]: expected {expect_letters!r}, got {strict!r}")
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CHEAT cases — current detector reward-hacked, strict blocks")
    print("=" * 60)
    test_cheats_break_current_pass_strict()
    print("\n" + "=" * 60)
    print("LEGIT cases — both current and strict extract correctly")
    print("=" * 60)
    test_legit_unchanged()
    print("\nALL TESTS PASSED")
