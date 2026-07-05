"""Batch inference: read questions from a file, write answers as JSONL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.inference.generate import answer
from src.inference.load_adapter import load_adapter_model
from src.utils.config import Config
from src.utils.logging_conf import get_logger

log = get_logger("inference.batch")


def _read_questions(path: Path) -> List[str]:
    if path.suffix == ".jsonl":
        return [json.loads(line)["question"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Plain text: one question per line.
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run_batch(cfg: Config, input_path: str, output_path: str) -> str:
    model, tokenizer = load_adapter_model(cfg)
    questions = _read_questions(Path(input_path))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            a = answer(model, tokenizer, q)
            f.write(json.dumps({"question": q, "answer": a}) + "\n")
            log.info("[%d/%d] done", i, len(questions))
    log.info("Batch answers written → %s", out)
    return str(out)
