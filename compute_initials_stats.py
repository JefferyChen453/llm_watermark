#!/usr/bin/env python3
"""Phase 2a: Initials ICW — per-letter coverage of leading-space English tokens.

For each token in the ``english_token_ids`` set that has a leading-space prefix
(Qwen3 / GPT-style BPE tokens starting with ``Ġ``), classify by the first
alphabetic character after the leading space (case-insensitive, A-Z). Tokens
whose first non-space char is not a letter (digits, punctuation) are grouped
into ``_other``.

Output: ``data/initials_icw/leading_space_first_letter_stats.json``
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from gptwm import _get_english_token_ids


def first_letter_of_leading_space_token(tok_str: str):
    """Given a decoded token string, return the first letter A-Z (uppercase)
    if the string starts with a space followed by a letter, else None.
    Returns the sentinel ``_other`` for leading-space tokens whose first char
    after the space is not a letter; returns ``_no_leading_space`` otherwise."""
    if not tok_str.startswith(" "):
        return "_no_leading_space"
    rest = tok_str[1:]
    if not rest:
        return "_other"
    c = rest[0]
    if c.isalpha() and c.isascii():
        return c.upper()
    return "_other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-14B")
    parser.add_argument("--output", default="data/initials_icw/leading_space_first_letter_stats.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    eng_ids = _get_english_token_ids(tokenizer, vocab_size)
    print(f"english token ids: {len(eng_ids)} (vocab_size={vocab_size})")

    counter = Counter()
    leading_space_eng_ids = []
    for tid in eng_ids:
        tok_str = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([tid]))
        category = first_letter_of_leading_space_token(tok_str)
        counter[category] += 1
        if category != "_no_leading_space":
            leading_space_eng_ids.append(tid)

    total_leading_space = sum(v for k, v in counter.items() if k != "_no_leading_space")
    total_letters = sum(v for k, v in counter.items() if k not in ("_no_leading_space", "_other"))

    letters = sorted([k for k in counter if len(k) == 1])
    print(f"\nTotal english+leading-space tokens: {total_leading_space}")
    print(f"Total letter-initial (A-Z): {total_letters}")
    print(f"_other (non-letter start after space): {counter.get('_other', 0)}")
    print(f"_no_leading_space (not analyzed): {counter.get('_no_leading_space', 0)}")
    print()
    print(f"{'letter':<6} {'count':>7} {'% of leading-space':>20} {'% of letter-initial':>20}")
    for ltr in letters:
        c = counter[ltr]
        pct_ls = 100.0 * c / max(total_leading_space, 1)
        pct_li = 100.0 * c / max(total_letters, 1)
        print(f"  {ltr:<4} {c:>7} {pct_ls:>18.2f}%  {pct_li:>18.2f}%")

    out = {
        "model_name": args.model_name,
        "vocab_size": vocab_size,
        "english_token_count": len(eng_ids),
        "leading_space_english_count": total_leading_space,
        "letter_initial_count": total_letters,
        "other_initial_count": counter.get("_other", 0),
        "per_letter_count": {ltr: counter[ltr] for ltr in letters},
        "per_letter_fraction_leading_space": {
            ltr: counter[ltr] / max(total_leading_space, 1) for ltr in letters
        },
        "per_letter_fraction_letter_initial": {
            ltr: counter[ltr] / max(total_letters, 1) for ltr in letters
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
