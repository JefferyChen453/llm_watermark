import hashlib
from typing import List, Optional
import numpy as np
from scipy.stats import norm
import torch
from transformers import LogitsProcessor


class GPTWatermarkBase:
    """
    Base class for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
        only_English: If True, only English tokens will be considered for green-listing.
        tokenizer: The tokenizer instance (required if only_English=True).
    """

    @staticmethod
    def is_english_token(token: str) -> bool:
        """Check if a token is English (ASCII characters only, excluding first character)."""
        return all(ord(c) < 128 for c in token[1:])

    def __init__(
        self, 
        fraction: float = 0.5, 
        strength: float = 2.0, 
        vocab_size: int = None, 
        model_emb_length: int = None,
        watermark_key: int = 0,
        only_English: bool = False,
        tokenizer: Optional[object] = None
    ):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        
        if only_English:
            if tokenizer is None:
                raise ValueError("tokenizer must be provided when only_English=True")
            
            # Get English token IDs
            self.tokenizer = tokenizer
            vocab = self.tokenizer.get_vocab()
            self.english_token_ids = [
                tid for tok, tid in vocab.items() 
                if self.is_english_token(tok) and tid < vocab_size
            ]
            
            if len(self.english_token_ids) == 0:
                raise ValueError("No English tokens found in vocabulary")
            
            # Initialize mask with all False
            mask = np.zeros(model_emb_length, dtype=bool)
            
            # Only assign green-list to English tokens
            num_green_english = int(fraction * len(self.english_token_ids))
            english_mask = np.array([True] * num_green_english + [False] * (len(self.english_token_ids) - num_green_english))
            rng.shuffle(english_mask)
            
            # Set green-list for English tokens
            for i, token_id in enumerate(self.english_token_ids):
                mask[token_id] = english_mask[i]
        else:
            green_list_size = int(fraction * vocab_size)
            mask = np.array([True] * green_list_size + [False] * (vocab_size - green_list_size))
            rng.shuffle(mask)
            mask = np.concatenate([
                mask,
                np.zeros(model_emb_length - len(mask), dtype=bool),
            ]) # handle the case when model_emb_length > vocab_size

        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction
        self.only_English = only_English

    @staticmethod
    def _hash_fn(x: int) -> int:
        """solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')


class GPTWatermarkLogitsWarper(GPTWatermarkBase, LogitsProcessor):
    """
    LogitsWarper for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
        only_English: If True, only English tokens will be considered for green-listing.
        tokenizer: The tokenizer instance (required if only_English=True).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        watermark = self.strength * self.green_list_mask
        new_logits = scores + watermark.to(scores.device)
        return new_logits


class GPTWatermarkDetector(GPTWatermarkBase):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
        only_English: If True, only English tokens will be considered for green-listing.
        tokenizer: The tokenizer instance (required if only_English=True).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        """Calculate and return the z-score of the number of green tokens in a sequence."""
        return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)
    
    @staticmethod
    def _compute_tau(m: int, N: int, alpha: float) -> float:
        """
        Compute the threshold tau for the dynamic thresholding.

        Args:
            m: The number of unique tokens in the sequence.
            N: Vocabulary size.
            alpha: The false positive rate to control.
        Returns:
            The threshold tau.
        """
        factor = np.sqrt(1 - (m - 1) / (N - 1))
        tau = factor * norm.ppf(1 - alpha)
        return tau

    def detect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))

        return self._z_score(green_tokens, len(sequence), self.fraction)

    def unidetect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value. Just for unique tokens."""
        sequence = list(set(sequence))
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
        return self._z_score(green_tokens, len(sequence), self.fraction)
    
    def dynamic_threshold(self, sequence: List[int], alpha: float, vocab_size: int) -> (bool, float):
        """Dynamic thresholding for watermark detection. True if the sequence is watermarked, False otherwise."""
        z_score = self.unidetect(sequence)
        tau = self._compute_tau(len(list(set(sequence))), vocab_size, alpha)
        return z_score > tau, z_score
