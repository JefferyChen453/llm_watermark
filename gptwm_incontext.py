"""
In-context watermark implementation: Green word list as system prompt.
"""
import hashlib
from typing import List, Optional
import numpy as np
from gptwm import GPTWatermarkBase


class InContextWatermarkGenerator(GPTWatermarkBase):
    """
    Generate green word list for in-context watermarking.
    The green words will be used as part of the system prompt.
    """
    
    def __init__(self, *args, **kwargs):
        # Ensure tokenizer is provided
        if 'tokenizer' not in kwargs or kwargs['tokenizer'] is None:
            raise ValueError("tokenizer must be provided for InContextWatermarkGenerator")
        
        # Store tokenizer before calling super().__init__ since parent may need it
        self.tokenizer = kwargs['tokenizer']
        super().__init__(*args, **kwargs)
        
        # Ensure tokenizer is set (in case parent class also sets it)
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.tokenizer = kwargs['tokenizer']
    
    def get_green_word_list(self) -> List[str]:
        """
        Extract green words from the green_list_mask.
        If only_English=True, only English tokens will be included.
        
        Returns:
            List of green word strings.
        """
        green_token_ids = []
        mask_np = self.green_list_mask.numpy()
        
        for token_id in range(len(mask_np)):
            if mask_np[token_id] > 0.5:  # This token is in green list
                green_token_ids.append(token_id)
        
        # Convert token IDs to strings
        green_words = []
        seen_words = set()
        
        # Get vocab once for efficiency (needed for only_English check)
        vocab = None
        id_to_token = None
        if self.only_English:
            vocab = self.tokenizer.get_vocab()
            # Create reverse mapping: token_id -> token_string
            id_to_token = {tid: tok for tok, tid in vocab.items()}
        
        for token_id in green_token_ids:
            try:
                # If only_English is enabled, check if token is English using original token string
                if self.only_English:
                    original_token_str = id_to_token.get(token_id)
                    if original_token_str is None or not self.is_english_token(original_token_str):
                        continue
                
                # Decode single token to get the actual text
                token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
                
                # Filter out special tokens, whitespace-only tokens, and control characters
                token_str = token_str.strip()
                if (token_str and 
                    not token_str.isspace() and 
                    len(token_str) > 0 and
                    token_str not in seen_words and
                    not all(ord(c) < 32 for c in token_str)):  # Filter control characters
                    
                    # Limit word length to avoid very long tokens
                    if len(token_str) <= 50:  # Reasonable limit for display
                        green_words.append(token_str)
                        seen_words.add(token_str)
            except Exception as e:
                # Skip tokens that can't be decoded
                continue
        
        return green_words
    
    def format_green_word_list(self) -> str:
        """
        Format green word list as a string for system prompt.

        Returns:
            Formatted string of green words.
        """
        green_words = self.get_green_word_list()
        if not green_words:
            return ""
        
        # Format as comma-separated list
        return ", ".join(green_words)


def get_incontext_system_prompt(green_word_list: str) -> str:
    """
    Generate the system prompt with green word list for in-context watermarking.
    
    Args:
        green_word_list: Comma-separated string of green words.
    
    Returns:
        System prompt string.
    """
    prompt = f"""### Green Word List: {green_word_list}

### Command:
You are given a predefined Green Word List. For each user query, generate a response that is:
1. Clear & Coherent: Easy to follow and logically organized.
2. Accurate & Concrete: Provides precise facts, examples, or steps. Avoid vague or overly verbose expressions.
3. Contextually Relevant: Directly addresses the user's intent and context.
4. "Green Word" Enriched (Most Important!): Try your best to seamlessly incorporate as many words from the Green Word List as possible — without compromising text quality."""
    
    return prompt
