"""Generate answers for the fixed EVAL_QUESTIONS from any model.

Shared by baseline and post-training so sampling settings are identical.
"""
from __future__ import annotations

from typing import List

import torch

from src.data.prompt import render_prompt
from src.evaluation.questions import EVAL_QUESTIONS


@torch.no_grad()
def generate_answers(model, tokenizer, questions: List[str] = None, max_new_tokens: int = 256) -> List[dict]:
    questions = questions or EVAL_QUESTIONS
    model.eval()
    device = next(model.parameters()).device
    results = []
    for q in questions:
        prompt = render_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # greedy → deterministic, fair comparison
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({"question": q, "answer": answer.strip()})
    return results
