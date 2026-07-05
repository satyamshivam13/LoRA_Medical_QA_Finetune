"""Interactive single-prompt generation with the fine-tuned model."""
from __future__ import annotations

import torch

from src.data.prompt import DISCLAIMER, render_prompt
from src.inference.load_adapter import load_adapter_model
from src.utils.config import Config


@torch.no_grad()
def answer(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    device = next(model.parameters()).device
    prompt = render_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def interactive(cfg: Config) -> None:
    model, tokenizer = load_adapter_model(cfg)
    print("\n" + DISCLAIMER)
    print("Type a medical question (or 'quit' to exit).\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        print(f"\nAssistant: {answer(model, tokenizer, q)}\n")
        print(DISCLAIMER + "\n")
