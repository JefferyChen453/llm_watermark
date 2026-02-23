"""System prompts for LFQA and OpenGen tasks.

This is the single source of truth for all system prompt strings and builders,
including base prompts and in-context watermark prompts.
"""


def get_system_prompt(dataset_type: str) -> str:
    """Return the base (no-watermark) system prompt for the given dataset type.

    Equivalent to get_incontext_system_prompt(dataset_type, green_token_string="").
    """
    return get_incontext_system_prompt(dataset_type, "")


def get_incontext_system_prompt(dataset_type: str, green_token_string: str) -> str:
    """Return the system prompt for the given dataset type and green token string.

    Args:
        dataset_type: 'lfqa' or 'opengen' (case-insensitive).
        green_token_string: Pipe-separated green token string, or empty string for no watermark.
    """
    dtype = dataset_type.lower().strip()
    if dtype == "lfqa":
        return _get_lfqa_incontext_system_prompt(green_token_string)
    if dtype == "opengen":
        return _get_opengen_incontext_system_prompt(green_token_string)
    raise ValueError(f"Unknown dataset_type: {dataset_type!r}. Expected 'lfqa' or 'opengen'.")


def _get_lfqa_incontext_system_prompt(green_token_string: str) -> str:
    if green_token_string:
        return f"""### Green Token List: {green_token_string}

### Command:
You are given a predefined Green Token List, separated by "|". For each user query, generate a response that is:
1. Clear & Coherent: Easy to follow and logically organized.
2. Accurate & Concrete: Provides precise facts, examples, or steps. Avoid vague or overly verbose expressions.
3. Contextually Relevant: Directly addresses the user's intent and context.
4. "Green Token" Enriched (Most Important!): Try your best to seamlessly incorporate as many tokens from the Green Token List as possible — without compromising text quality. Do not claim what green tokens you used explicitly."""
    return """### Command:
You are given a user query. Generate a response that is:
1. Clear & Coherent: Easy to follow and logically organized.
2. Accurate & Concrete: Provides precise facts, examples, or steps. Avoid vague or overly verbose expressions.
3. Contextually Relevant: Directly addresses the user's intent and context."""


def _get_opengen_incontext_system_prompt(green_token_string: str) -> str:
    if green_token_string:
        return f"""### Green Token List: {green_token_string}

### Command:
You are a language model that continues text passages.
1. Continue the given passage in a coherent, factual, and stylistically consistent manner. 
2. Try your best to seamlessly incorporate as many tokens from the Green Token List as possible — without compromising text quality. Do not claim what green tokens you used explicitly.
3. Do not add explanations, commentary, or formatting. Produce only the continuation."""
    return """### Command:
You are a language model that continues text passages.
1. Continue the given passage in a coherent, factual, and stylistically consistent manner. 
2. Do not add explanations, commentary, or formatting. Produce only the continuation."""
