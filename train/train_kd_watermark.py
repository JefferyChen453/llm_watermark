#!/usr/bin/env python3
"""
Off-policy Knowledge Distillation Training with Watermark-based Teacher Distribution

This script implements knowledge distillation where:
- Student: normal model logits -> probability distribution
- Teacher: student logits + watermark modification -> probability distribution
- Loss: KL divergence between student and teacher distributions
"""

import argparse
import os
import json
import hashlib
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Unigram-Watermark'))
from UnigramWatermark.gptwm import GPTWatermarkBase


class WatermarkTeacherDistiller:
    """
    Helper class to generate teacher distribution from student logits using watermark.
    """
    
    def __init__(self, fraction: float = 0.5, strength: float = 2.0, 
                 vocab_size: int = 50257, watermark_key: int = 0):
        """
        Args:
            fraction: The fraction of the distribution to be green-listed.
            strength: The strength of the green-listing.
            vocab_size: The size of the vocabulary.
            watermark_key: The random seed for the green-listing.
        """
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(fraction * vocab_size) + 
                       [False] * (vocab_size - int(fraction * vocab_size)))
        rng.shuffle(mask)
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction
    
    @staticmethod
    def _hash_fn(x: int) -> int:
        """Hash function for watermark key."""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')
    
    def apply_watermark(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply watermark to logits to get teacher logits.
        
        Args:
            logits: Student model logits, shape (batch_size, seq_len, vocab_size)
        
        Returns:
            Teacher logits with watermark applied
        """
        watermark = self.strength * self.green_list_mask
        teacher_logits = logits + watermark.to(logits.device)
        return teacher_logits


class WatermarkKDLoss(nn.Module):
    """
    Knowledge Distillation Loss using KL divergence between student and watermark-modified teacher.
    """
    
    def __init__(self, temperature: float = 1.0, alpha: float = 1.0):
        """
        Args:
            temperature: Temperature for softmax (default: 1.0)
            alpha: Weight for KD loss (default: 1.0, meaning only KD loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute KL divergence loss.
        
        Args:
            student_logits: Student model logits, shape (batch_size, seq_len, vocab_size)
            teacher_logits: Teacher (watermarked) logits, shape (batch_size, seq_len, vocab_size)
            mask: Optional mask for valid tokens, shape (batch_size, seq_len)
        
        Returns:
            KL divergence loss (scalar)
        """
        # Compute probability distributions
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Reshape for KL divergence computation
        batch_size, seq_len, vocab_size = student_logits.shape
        student_probs_flat = student_probs.view(-1, vocab_size)
        teacher_probs_flat = teacher_probs.view(-1, vocab_size)
        
        # Compute KL divergence: KL(student || teacher)
        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
        kl_div_per_token = F.kl_div(
            student_probs_flat,
            teacher_probs_flat,
            reduction='none',
            log_target=False
        )
        kl_div_per_token = kl_div_per_token.sum(dim=-1)  # Sum over vocab dimension
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            kl_div_per_token = kl_div_per_token * mask_flat
            # Average only over valid tokens
            loss = kl_div_per_token.sum() / mask_flat.sum().clamp(min=1)
        else:
            # Average over all tokens
            loss = kl_div_per_token.mean()
        
        # Scale by temperature squared (standard in KD)
        loss = (self.temperature ** 2) * loss
        
        return self.alpha * loss


class WatermarkKDModel(nn.Module):
    """
    Wrapper model that computes KD loss during forward pass.
    """
    
    def __init__(self, base_model: AutoModelForCausalLM, 
                 watermark_distiller: WatermarkTeacherDistiller,
                 kd_loss_fn: WatermarkKDLoss):
        super().__init__()
        self.base_model = base_model
        self.watermark_distiller = watermark_distiller
        self.kd_loss_fn = kd_loss_fn
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass with KD loss computation.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            labels: Labels for language modeling (optional, not used in KD)
        
        Returns:
            CausalLMOutputWithPast with loss computed from KD
        """
        # Get student logits
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        student_logits = outputs.logits
        
        # Generate teacher logits by applying watermark
        teacher_logits = self.watermark_distiller.apply_watermark(student_logits)
        
        # Compute KD loss
        # Shift logits for next token prediction (predict token at position i+1 given tokens up to i)
        # student_logits shape: (batch, seq_len, vocab_size)
        # We want to predict position 1..seq_len given input at position 0..seq_len-1
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        
        # Create mask for valid tokens (excluding last position)
        shift_mask = None
        if attention_mask is not None:
            # Shift mask: we predict token at position i+1, so mask should be for positions 1..seq_len
            shift_mask = attention_mask[..., 1:].contiguous()
        
        # Compute KL divergence loss
        loss = self.kd_loss_fn(shift_logits, shift_teacher_logits, mask=shift_mask)
        
        # Return output in expected format
        return CausalLMOutputWithPast(
            loss=loss,
            logits=student_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }


def load_dataset_from_file(file_path: str):
    """Load dataset from JSONL file or text file."""
    texts = []
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Try common keys
                if 'text' in data:
                    texts.append(data['text'])
                elif 'content' in data:
                    texts.append(data['content'])
                elif 'prompt' in data and 'completion' in data:
                    texts.append(data['prompt'] + ' ' + data['completion'])
                elif 'prefix' in data and 'gold_completion' in data:
                    texts.append(data['prefix'] + ' ' + data['gold_completion'])
    else:
        # Plain text file
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    return texts


def main():
    parser = argparse.ArgumentParser(description='Train model with watermark-based KD')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name or path (e.g., gpt2, facebook/opt-125m)')
    parser.add_argument('--output_dir', type=str, default='./outputs/kd_watermark',
                       help='Output directory for checkpoints')
    
    # Dataset arguments
    parser.add_argument('--train_file', type=str, required=True,
                       help='Training data file (JSONL or text)')
    parser.add_argument('--val_file', type=str, default=None,
                       help='Validation data file (optional)')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Watermark arguments
    parser.add_argument('--watermark_fraction', type=float, default=0.5,
                       help='Fraction of tokens in green list')
    parser.add_argument('--watermark_strength', type=float, default=2.0,
                       help='Strength of watermark modification')
    parser.add_argument('--watermark_key', type=int, default=0,
                       help='Random seed for watermark')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Warmup steps')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluate every N steps')
    
    # KD arguments
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weight for KD loss')
    
    # Hardware arguments
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--bf16', action='store_true',
                       help='Use bfloat16 precision')
    parser.add_argument('--dataloader_num_workers', type=int, default=0,
                       help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map='auto',
        torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )
    
    # Get vocab size
    vocab_size = model.config.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize watermark distiller
    watermark_distiller = WatermarkTeacherDistiller(
        fraction=args.watermark_fraction,
        strength=args.watermark_strength,
        vocab_size=vocab_size,
        watermark_key=args.watermark_key
    )
    
    # Initialize KD loss function
    kd_loss_fn = WatermarkKDLoss(
        temperature=args.temperature,
        alpha=args.alpha
    )
    
    # Wrap model with KD wrapper
    kd_model = WatermarkKDModel(model, watermark_distiller, kd_loss_fn)
    
    # Load datasets
    print(f"Loading training data from: {args.train_file}")
    train_texts = load_dataset_from_file(args.train_file)
    train_dataset = TextDataset(train_texts, tokenizer, max_length=args.max_length)
    print(f"Loaded {len(train_dataset)} training examples")
    
    val_dataset = None
    if args.val_file:
        print(f"Loading validation data from: {args.val_file}")
        val_texts = load_dataset_from_file(args.val_file)
        val_dataset = TextDataset(val_texts, tokenizer, max_length=args.max_length)
        print(f"Loaded {len(val_dataset)} validation examples")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if val_dataset else None,
        evaluation_strategy='steps' if val_dataset else 'no',
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model='eval_loss' if val_dataset else None,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to='wandb',
        wandb_project='kd-watermark',
        wandb_name=f'{args.model_name}-{args.watermark_fraction}-{args.watermark_strength}-{args.watermark_key}',
    )
    
    # Create trainer
    trainer = Trainer(
        model=kd_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training completed!")


if __name__ == '__main__':
    main()
