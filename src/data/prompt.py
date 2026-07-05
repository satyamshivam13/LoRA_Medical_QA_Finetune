"""Single source of truth for the prompt / chat template.

Both training and inference build messages here, so the format can never drift
between the two. We rely on the tokenizer's own chat template
(``apply_chat_template``) so the exact special tokens match the base model.
"""
from __future__ import annotations

from typing import List, Optional

SYSTEM_PROMPT = (
    "You are a medical information assistant for educational purposes. "
    "Provide clear, factual, general health information. "
    "Always remind the user that this is not a substitute for professional "
    "medical advice, diagnosis, or treatment."
)

# Shown to end users in interactive/batch inference. Keeps the demo responsible.
DISCLAIMER = (
    "[Educational demo — not medical advice. Consult a healthcare professional.]"
)


def build_messages(question: str, answer: Optional[str] = None) -> List[dict]:
    """Return chat-format messages. If ``answer`` is given, include it (training)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages


def render_prompt(tokenizer, question: str) -> str:
    """Render the inference prompt (system + user + generation prefix) as text."""
    return tokenizer.apply_chat_template(
        build_messages(question),
        tokenize=False,
        add_generation_prompt=True,
    )
