import json
from datasets import Dataset


def load_generation_dataset(jsonl_path, num_test=None):
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            rows.append({
                "prefix": ex["prefix"],
                "gold_completion": ex["gold_completion"]
            })

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


def map_fn_with_chat_template(tokenizer, system_prompt: str):
    """Create a tokenize function that applies chat template with system prompt."""
    def _fn(example):
        user_prompts = example["prefix"]
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

        example.update({
            "input_prompts": input_prompts,
        })
        return example

    return _fn


def map_fn_with_chat_template_ids(tokenizer, system_prompt: str):
    """Create a tokenize function that applies chat template with system prompt."""
    def _fn(example):
        prefix = example["prefix"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prefix}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        encoded_prompts = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            padding=False
        )

        example.update({
            "input_prompt": prompt_text,
            "input_ids": encoded_prompts["input_ids"],
            "attention_mask": encoded_prompts["attention_mask"],
        })

        return example

    return _fn
