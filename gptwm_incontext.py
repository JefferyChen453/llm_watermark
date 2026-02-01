"""
In-context watermark implementation: Green word list as system prompt.
"""
from gptwm import GPTWatermarkBase
import torch
import warnings


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
    
    def get_green_token_string(self) -> str:
        green_token_ids = torch.nonzero(self.green_list_mask, as_tuple=True)[0].tolist()
        green_tokens = self.tokenizer.convert_ids_to_tokens(green_token_ids)
        green_token_list = []
        for token in green_tokens:
            s = self.tokenizer.convert_tokens_to_string([token])
            if s is not None and s.strip() != "":
                green_token_list.append(s.strip())
        
        return " | ".join(green_token_list)

def tokenize_fn_with_chat_template(tokenizer, system_prompt: str):
    """Create a tokenize function that applies chat template with system prompt."""
    def _fn(batch):
        user_prompts = batch["prefix"]
        input_prompts = []
        for user_prompt in user_prompts:
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                input_prompts.append(prompt_text)
            else:
                full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
                input_prompts.append(full_prompt)

        batch.update({
            "input_prompts": input_prompts,
        })
        return batch

    return _fn

def tokenize_fn_with_chat_template_ids(tokenizer, system_prompt: str):
    """Create a tokenize function that applies chat template with system prompt."""
    def _fn(batch):
        user_prompts = batch["prefix"]
        input_prompts = []
        for user_prompt in user_prompts:
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                input_prompts.append(prompt_text)
            else:
                full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
                input_prompts.append(full_prompt)
        encoded_prompts = tokenizer(
            input_prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
        )
        print(encoded_prompts["input_ids"].shape)

        batch.update({
            "input_prompts": input_prompts,
            **encoded_prompts
        })
        return batch

    return _fn



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
