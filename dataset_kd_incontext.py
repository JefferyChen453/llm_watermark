"""
Dataset for in-context watermark KD training.

Loads pre-generated teacher trajectories (watermark-strength guided decoding results)
and formats them for KD training where:
- Teacher distribution = model logits + strength * green_mask
- Student distribution = model logits on (prompt + teacher_trajectory)
"""
import json
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class InContextWatermarkKDDataset(Dataset):
    """
    Dataset for KD training with in-context watermark.
    Each sample: (prompt_ids, teacher_completion_ids)
    Full sequence: prompt + teacher_completion.
    Loss is on completion positions (KL between teacher and student distributions).
    """

    def __init__(
        self,
        data_paths: str | List[str],
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 262144,
        max_completion_length: int = 512,
        truncation_prompt: str = "left",
        max_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.truncation_prompt = truncation_prompt
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        if not isinstance(data_paths, list):
            data_paths = [data_paths]

        self.samples: List[dict] = []
        for path in data_paths:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        ex = json.loads(line)
                        prompt = ex.get("actual_prompt", ex.get("input_prompt", ""))
                        teacher_completion = ex.get("gen_completion", ex.get("gold_completion", ""))
                        self.samples.append({"prompt": prompt, "teacher_completion": teacher_completion})

        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(f"InContextWatermarkKDDataset: loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        prompt = sample["prompt"]
        teacher_completion = sample["teacher_completion"]

        prompt_enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        prompt_ids = prompt_enc["input_ids"][0]
        prompt_attention = prompt_enc["attention_mask"][0]

        completion_str = teacher_completion + (self.tokenizer.eos_token or "")
        completion_enc = self.tokenizer(
            completion_str,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_completion_length,
        )
        completion_ids = completion_enc["input_ids"][0]
        completion_attention = completion_enc["attention_mask"][0]

        prompt_len = prompt_ids.shape[0]
        completion_len = completion_ids.shape[0]

        if prompt_len + completion_len > self.max_prompt_length + self.max_completion_length:
            if self.truncation_prompt == "left":
                if prompt_len > self.max_prompt_length:
                    prompt_ids = prompt_ids[-self.max_prompt_length :]
                    prompt_attention = prompt_attention[-self.max_prompt_length :]
                if completion_len > self.max_completion_length:
                    completion_ids = completion_ids[: self.max_completion_length]
                    completion_attention = completion_attention[: self.max_completion_length]
            else:
                if completion_len > self.max_completion_length:
                    completion_ids = completion_ids[: self.max_completion_length]
                    completion_attention = completion_attention[: self.max_completion_length]
                if prompt_len > self.max_prompt_length:
                    prompt_ids = prompt_ids[-self.max_prompt_length :]
                    prompt_attention = prompt_attention[-self.max_prompt_length :]

        prompt_len = prompt_ids.shape[0]
        completion_len = completion_ids.shape[0]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=0)
        attention_mask = torch.cat([prompt_attention, completion_attention], dim=0)
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

        # Loss at position i when logits[i] predicts completion token (input_ids[i+1])
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        loss_mask[prompt_len - 1 : prompt_len + completion_len - 1] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


def collate_kd_batch(batch: List[dict], pad_token_id: int, max_length: Optional[int] = None) -> dict:
    """Collate with left-padding."""
    if max_length is None:
        max_length = max(item["input_ids"].shape[0] for item in batch)

    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    position_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    loss_mask = torch.zeros(batch_size, max_length, dtype=torch.float32)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        start = max_length - seq_len
        input_ids[i, start:] = item["input_ids"]
        attention_mask[i, start:] = item["attention_mask"]
        position_ids[i, start:] = item["position_ids"]
        loss_mask[i, start:] = item["loss_mask"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
    }
