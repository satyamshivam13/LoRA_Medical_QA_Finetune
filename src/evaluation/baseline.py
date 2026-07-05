"""Baseline evaluation of the FROZEN base model, run BEFORE any training.

Writes logs/baseline_eval.json and results/baseline_samples.md so the
before/after comparison is grounded in a real measurement.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data.load import load_and_split
from src.data.preprocess import tokenize_dataset
from src.evaluation.generate_samples import generate_answers
from src.evaluation.perplexity import compute_perplexity
from src.training.model_loader import load_model, load_tokenizer
from src.utils.config import Config
from src.utils.logging_conf import get_logger
from src.utils.seed import set_seed

log = get_logger("eval.baseline")


def _write_samples_md(path: Path, title: str, samples: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}\n"]
    for i, s in enumerate(samples, 1):
        lines.append(f"### {i}. {s['question']}\n\n{s['answer']}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_baseline(cfg: Config) -> dict:
    set_seed(cfg.train.seed)
    tokenizer = load_tokenizer(cfg.model)
    model = load_model(cfg.model, for_training=False)

    splits = load_and_split(cfg.data)
    test_tok = tokenize_dataset(splits["test"], tokenizer, cfg.data)

    log.info("Computing baseline perplexity on test split…")
    ppl = compute_perplexity(model, tokenizer, test_tok, cfg.train.per_device_eval_batch_size)

    log.info("Generating baseline sample answers…")
    samples = generate_answers(model, tokenizer)

    result = {"model": cfg.model.base_model_id, "stage": "baseline", **ppl}
    Path("logs").mkdir(exist_ok=True)
    Path("logs/baseline_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_samples_md(Path("results/baseline_samples.md"), "Baseline (base model) samples", samples)
    log.info("Baseline saved → logs/baseline_eval.json, results/baseline_samples.md")
    return result
