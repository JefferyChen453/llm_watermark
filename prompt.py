"""System prompts for generation tasks.

To add a new dataset type, call ``register_prompt`` with base and in-context
templates.  The in-context template must contain a ``{green_tokens}``
placeholder that will be filled at runtime.

Example::

    register_prompt(
        "my_new_task",
        base="You are a helpful assistant for my_new_task.",
        incontext=(
            "### Green Token List: {green_tokens}\n\n"
            "You are a helpful assistant for my_new_task. "
            "Use as many green tokens as possible."
        ),
    )
"""

from typing import Dict


_DATASET_PROMPTS: Dict[str, Dict[str, str]] = {}


def register_prompt(dataset_type: str, *, base: str, incontext: str):
    """Register base and in-context prompt templates for *dataset_type*."""
    _DATASET_PROMPTS[dataset_type.lower()] = {"base": base, "incontext": incontext}


def get_system_prompt(dataset_type: str) -> str:
    """Return the base system prompt (no green-token list)."""
    return get_incontext_system_prompt(dataset_type, "")


def get_incontext_system_prompt(dataset_type: str, green_token_string: str = "") -> str:
    """Return the system prompt, optionally enriched with a green-token list.

    Args:
        dataset_type: Registered dataset name (case-insensitive).
        green_token_string: Pipe-separated green token string, or ``""`` for
            a plain (no-watermark) prompt.
    """
    dtype = dataset_type.lower().strip()
    if dtype not in _DATASET_PROMPTS:
        available = ", ".join(sorted(_DATASET_PROMPTS)) or "(none)"
        raise ValueError(
            f"Unknown dataset_type: {dataset_type!r}. Available: {available}"
        )
    entry = _DATASET_PROMPTS[dtype]
    if green_token_string:
        return entry["incontext"].format(green_tokens=green_token_string)
    return entry["base"]


# ------------------------------ Built-in dataset types ------------------------------

register_prompt(
    "lfqa",
    base="""\
### Command:
You are given a user query. Generate a response that is:
1. Clear & Coherent: Easy to follow and logically organized.
2. Accurate & Concrete: Provides precise facts, examples, or steps. Avoid vague or overly verbose expressions.
3. Contextually Relevant: Directly addresses the user's intent and context.""",
    incontext="""\
### Green Token List: {green_tokens}

### Command:
You are given a predefined Green Token List, separated by "|". For each user query, generate a response that is:
1. Clear & Coherent: Easy to follow and logically organized.
2. Accurate & Concrete: Provides precise facts, examples, or steps. Avoid vague or overly verbose expressions.
3. Contextually Relevant: Directly addresses the user's intent and context.
4. "Green Token" Enriched (Most Important!): Try your best to seamlessly incorporate as many tokens from the Green Token List as possible — without compromising text quality. Do not claim what green tokens you used explicitly.""",
)

register_prompt(
    "opengen",
    base="""\
### Command:
You are a language model that continues text passages.
1. Continue the given passage in a coherent, factual, and stylistically consistent manner. 
2. Do not add explanations, commentary, or formatting. Produce only the continuation.""",
    incontext="""\
### Green Token List: {green_tokens}

### Command:
You are a language model that continues text passages.
1. Continue the given passage in a coherent, factual, and stylistically consistent manner. 
2. Try your best to seamlessly incorporate as many tokens from the Green Token List as possible — without compromising text quality. Do not claim what green tokens you used explicitly.
3. Do not add explanations, commentary, or formatting. Produce only the continuation.""",
)
