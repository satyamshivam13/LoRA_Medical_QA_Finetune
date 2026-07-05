"""Prompt template builds correct message structure (no model download needed)."""
from src.data.prompt import build_messages, SYSTEM_PROMPT


def test_build_messages_inference():
    msgs = build_messages("What is anemia?")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["content"] == "What is anemia?"


def test_build_messages_training_includes_answer():
    msgs = build_messages("Q?", "A.")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[-1]["content"] == "A."


def test_whitespace_stripped():
    msgs = build_messages("  spaced  ", "  ans  ")
    assert msgs[1]["content"] == "spaced"
    assert msgs[2]["content"] == "ans"
