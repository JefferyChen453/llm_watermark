import json
from datasets import Dataset
import torch

from prompt import get_system_prompt


def load_jsonl(jsonl_path):
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            rows.append(ex)
    return rows

def load_generation_dataset(jsonl_path, num_test=None):
    rows = load_jsonl(jsonl_path)

    if num_test is not None:
        rows = rows[:num_test]

    return Dataset.from_list(rows)


def map_fn_ids(tokenizer):
    def _fn(example):
        prefix_enc = tokenizer(
            example["prefix"],
            truncation=True,
            padding=False
        )

        gold_enc = tokenizer(
            example["gold_completion"],
            truncation=True,
            padding=False
        )

        example.update({
            **prefix_enc,
            "gold_completion_ids": gold_enc["input_ids"],
        })

        return example

    return _fn


def collate_fn(batch, tokenizer):
    padded_prefix = tokenizer.pad(
        {
            "input_ids": [x["input_ids"] for x in batch],
            "attention_mask": [x["attention_mask"] for x in batch],
        },
        return_tensors="pt"
    )

    batch_dict = ({
        "prefix": [x["prefix"] for x in batch],
        "gold_completion": [x["gold_completion"] for x in batch],
        "input_ids": padded_prefix["input_ids"],
        "attention_mask": padded_prefix["attention_mask"],
    })

    if "input_prompts" in batch_dict:
        batch_dict["input_prompts"] = [x["input_prompts"] for x in batch]

    return batch_dict


def _apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Format a single (system, user) turn using the tokenizer's chat template.

    Falls back to a plain-text format when no chat template is available.
    """
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"


def map_fn_with_dataset_prompt(tokenizer, dataset_type: str):
    """Map function: apply chat template using the base system prompt for the dataset type.

    Adds ``input_prompts`` (list of formatted strings) to each example.
    Suitable for vLLM inference where the engine handles tokenization.
    """
    system_prompt = get_system_prompt(dataset_type)
    return map_fn_with_chat_template(tokenizer, system_prompt)


def map_fn_with_chat_template(tokenizer, system_prompt: str):
    """Map function: apply chat template with the given system prompt.

    Expects ``example["prefix"]`` to be a list of user prompt strings.
    Adds ``input_prompts`` (list of formatted strings) to each example.
    Suitable for vLLM inference where the engine handles tokenization.
    """
    def _fn(example):
        example["input_prompts"] = [
            _apply_chat_template(tokenizer, system_prompt, user_prompt)
            for user_prompt in example["prefix"]
        ]
        return example

    return _fn


def map_fn_with_chat_template_ids(tokenizer, system_prompt: str):
    """Map function: apply chat template with the given system prompt, then tokenize.

    Expects ``example["prefix"]`` to be a single user prompt string.
    Adds ``input_prompt`` (formatted string), ``input_ids``, and ``attention_mask``.
    Suitable for HF Transformers inference where tokenization is done in the map step.
    """
    def _fn(example):
        prompt_text = _apply_chat_template(tokenizer, system_prompt, example["prefix"])
        encoded = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            padding=False,
        )
        example.update({
            "input_prompt": prompt_text,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        })
        return example

    return _fn


def load_offline_incontext_jsonl(jsonl_path):
    """Load pre-generated incontext responses for offline KD.
    Expects jsonl with keys: input_prompt or actual_prompt, prefix, gen_completion.
    If save_gen_batch was used: also has prompt_ids, gen_ids, gen_mask.
    """
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            prompt = ex.get("actual_prompt") or ex.get("input_prompt")
            row = {
                "prefix": ex["prefix"],
                "gen_completion": ex["gen_completion"],
                "actual_prompt": prompt,
            }
            if "prompt_ids" in ex and "gen_ids" in ex:
                row["prompt_ids"] = ex["prompt_ids"]
                row["gen_ids"] = ex["gen_ids"]
                row["gen_mask"] = ex.get("gen_mask") or [1] * len(ex["gen_ids"])
            rows.append(row)

    return Dataset.from_list(rows)


def map_fn_offline_incontext_full_ids(tokenizer):
    """Build full_ids = prompt + response and prompt_len for offline KD.
    If example has prompt_ids and gen_ids (from save_gen_batch), use them directly.
    Else tokenize actual_prompt and gen_completion.
    """
    def _fn(example):
        if "gen_ids" in example and "prompt_ids" in example:
            prompt_ids = example["prompt_ids"]
            gen_ids = example["gen_ids"]
            gen_mask = example.get("gen_mask") or [1] * len(gen_ids)
            full_ids = prompt_ids + gen_ids
            prompt_len = len(prompt_ids)
            attention_mask = [1] * prompt_len + list(gen_mask)
        else:
            prompt_text = example["actual_prompt"]
            response_text = example["gen_completion"]
            enc_prompt = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )
            enc_response = tokenizer(
                response_text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )
            full_ids = enc_prompt["input_ids"] + enc_response["input_ids"]
            prompt_len = len(enc_prompt["input_ids"])
            attention_mask = [1] * len(full_ids)

        example.update({
            "full_ids": full_ids,
            "prompt_len": prompt_len,
            "attention_mask": attention_mask,
        })
        return example

    return _fn


def collate_fn_offline_incontext(batch, tokenizer):
    """Collate for offline KD: pad full_ids, return input_ids, attention_mask, prompt_lens."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_len = max(len(x["full_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    prompt_lens = []

    for x in batch:
        ids = x["full_ids"]
        mask = x["attention_mask"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append(mask + [0] * pad_len)
        prompt_lens.append(x["prompt_len"])

    batch_dict = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
        "prefix": [x["prefix"] for x in batch],
    }
    return batch_dict
