"""Tokenize examples into (input_ids, attention_mask, labels).

Uses completion-only loss masking: the prompt tokens (system + user + the
assistant generation header) are set to -100 in ``labels`` so the model is
trained ONLY to produce the answer, not to parrot the prompt.
"""
from __future__ import annotations

from datasets import Dataset

from src.data.prompt import build_messages
from src.utils.config import DataConfig


def build_tokenize_fn(tokenizer, cfg: DataConfig):
    """Return a per-example function producing masked labels."""
    max_len = cfg.max_seq_len

    def _tokenize(example: dict) -> dict:
        question = example["question"]
        answer = example["answer"]

        # Prompt only (up to and including the assistant generation prefix).
        prompt_ids = tokenizer.apply_chat_template(
            build_messages(question),
            tokenize=True,
            add_generation_prompt=True,
        )
        # Full sequence including the answer.
        full_ids = tokenizer.apply_chat_template(
            build_messages(question, answer),
            tokenize=True,
            add_generation_prompt=False,
        )

        full_ids = full_ids[:max_len]
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = -100  # mask prompt tokens from the loss

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return _tokenize


def tokenize_dataset(ds: Dataset, tokenizer, cfg: DataConfig) -> Dataset:
    """Map the tokenizer over a split and drop raw text columns."""
    fn = build_tokenize_fn(tokenizer, cfg)
    tokenized = ds.map(fn, remove_columns=ds.column_names, desc="Tokenizing")
    # Drop rows where the answer was fully truncated away (no supervised tokens).
    tokenized = tokenized.filter(lambda ex: any(l != -100 for l in ex["labels"]))
    return tokenized
