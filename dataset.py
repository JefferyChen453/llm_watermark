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

def tokenize_fn(tokenizer):
    def _fn(batch):
        prefix_enc = tokenizer(
            batch["prefix"],
            truncation=True,
            padding=True,
        )
        gold_enc = tokenizer(
            batch["gold_completion"],
            truncation=True,
            padding=True,
        )

        return {
            "input_ids": prefix_enc["input_ids"],
            "attention_mask": prefix_enc["attention_mask"],
            "gold_completion_ids": gold_enc["input_ids"],
        }

    return _fn