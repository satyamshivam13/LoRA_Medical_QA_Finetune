"""Perplexity / eval-loss over a tokenized split.

Perplexity = exp(mean token-level cross-entropy) on the (masked) answer tokens.
Lower is better. We compute it identically for base and fine-tuned models.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from src.utils.logging_conf import get_logger

log = get_logger("eval.ppl")


@torch.no_grad()
def compute_perplexity(model, tokenizer, tokenized_ds, batch_size: int = 8) -> Dict[str, float]:
    """Return {'eval_loss', 'perplexity'} over the given tokenized dataset."""
    model.eval()
    device = next(model.parameters()).device
    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest", label_pad_token_id=-100)
    loader = DataLoader(tokenized_ds, batch_size=batch_size, collate_fn=collator)

    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        # Count supervised (non-masked) tokens for a token-weighted mean.
        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    log.info("eval_loss=%.4f  perplexity=%.3f  (tokens=%d)", mean_loss, ppl, total_tokens)
    return {"eval_loss": round(mean_loss, 4), "perplexity": round(ppl, 3), "tokens": total_tokens}
