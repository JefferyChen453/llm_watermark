"""
In-context watermark implementation: Green word list as system prompt.
"""
from gptwm import GPTWatermarkBase
import torch


class InContextWatermarkGenerator(GPTWatermarkBase):
    """
    Generate green word list for in-context watermarking.
    The green words will be used as part of the system prompt.
    """
    
    def __init__(self, *args, **kwargs):
        if 'tokenizer' not in kwargs or kwargs['tokenizer'] is None:
            raise ValueError("tokenizer must be provided for InContextWatermarkGenerator")
        
        self.tokenizer = kwargs['tokenizer']
        super().__init__(*args, **kwargs)
        
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.tokenizer = kwargs['tokenizer']

    def get_green_token_string(self) -> str:
        sep = "|"
        green_token_ids = torch.nonzero(self.green_list_mask, as_tuple=True)[0].tolist()
        green_tokens = self.tokenizer.convert_ids_to_tokens(green_token_ids)
        green_token_list = []
        for token in green_tokens:
            s = self.tokenizer.convert_tokens_to_string([token])
            if s is None:
                continue
            if s != "":
                green_token_list.append(s)
        green_token_string = sep.join(green_token_list)
        return green_token_string


def get_incontext_system_prompt(green_token_string: str) -> str:
    """
    Generate the system prompt with green word list for in-context watermarking.
    
    Args:
        green_word_list: Comma-separated string of green words.
    
    Returns:
        System prompt string.
    """
    if green_token_string != "":
        system_prompt = f"""### Green Token List: {green_token_string}

### Command:
You are given a predefined Green Token List, separated by "|". For each user query, generate a response that is:
1. Clear & Coherent: Easy to follow and logically organized.
2. Accurate & Concrete: Provides precise facts, examples, or steps. Avoid vague or overly verbose expressions.
3. Contextually Relevant: Directly addresses the user's intent and context.
4. "Green Token" Enriched (Most Important!): Try your best to seamlessly incorporate as many tokens from the Green Token List as possible — without compromising text quality. Do not claim what green tokens you used explicitly."""
    else:
        system_prompt = """
### Command:
You are given a user query. Generate a response that is:
1. Clear & Coherent: Easy to follow and logically organized.
2. Accurate & Concrete: Provides precise facts, examples, or steps. Avoid vague or overly verbose expressions.
3. Contextually Relevant: Directly addresses the user's intent and context."""
    return system_prompt
