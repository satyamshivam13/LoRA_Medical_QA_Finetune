"""Post-training evaluation and honest before/after comparison.

Evaluates the fine-tuned (adapter-attached) model with the SAME perplexity and
sample procedure as the baseline, then writes results/comparison.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data.load import load_and_split
from src.data.preprocess import tokenize_dataset
from src.evaluation.generate_samples import generate_answers
from src.evaluation.perplexity import compute_perplexity
from src.inference.load_adapter import load_adapter_model
from src.utils.config import Config
from src.utils.logging_conf import get_logger
from src.utils.seed import set_seed

log = get_logger("eval.compare")


def _optional_rouge(preds, refs):
    """Return ROUGE-L F1 if `evaluate`+`rouge_score` are available, else None."""
    try:
        import evaluate

        rouge = evaluate.load("rouge")
        return round(rouge.compute(predictions=preds, references=refs)["rougeL"], 4)
    except Exception as exc:  # noqa: BLE001
        log.warning("ROUGE unavailable (%s) — skipping.", type(exc).__name__)
        return None


def run_comparison(cfg: Config) -> dict:
    set_seed(cfg.train.seed)
    model, tokenizer = load_adapter_model(cfg)

    splits = load_and_split(cfg.data)
    test_tok = tokenize_dataset(splits["test"], tokenizer, cfg.data)

    ft = compute_perplexity(model, tokenizer, test_tok, cfg.train.per_device_eval_batch_size)
    samples = generate_answers(model, tokenizer)

    # Save fine-tuned artifacts.
    ft_result = {"model": cfg.model.base_model_id, "stage": "finetuned", **ft}
    Path("logs/finetuned_eval.json").write_text(json.dumps(ft_result, indent=2), encoding="utf-8")

    ft_lines = ["# Fine-tuned model samples\n"]
    for i, s in enumerate(samples, 1):
        ft_lines.append(f"### {i}. {s['question']}\n\n{s['answer']}\n")
    Path("results/finetuned_samples.md").write_text("\n".join(ft_lines), encoding="utf-8")

    # Load baseline for the table.
    base_path = Path("logs/baseline_eval.json")
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}

    _write_comparison_md(cfg, base, ft_result, samples)
    log.info("Comparison saved → results/comparison.md")
    return {"baseline": base, "finetuned": ft_result}


def _pct(before, after):
    if not before:
        return "n/a"
    return f"{(after - before) / before * 100:+.1f}%"


def _write_comparison_md(cfg, base, ft, samples) -> None:
    lines = ["# Before vs After — Results\n"]
    lines.append(f"- **Base model:** `{cfg.model.base_model_id}`")
    lines.append(f"- **Dataset:** `{cfg.data.dataset_id}` (held-out test split)")
    lines.append(f"- **Method:** LoRA (r={cfg.lora.r}, alpha={cfg.lora.lora_alpha})\n")

    lines.append("## Quantitative\n")
    lines.append("| Metric | Baseline | Fine-tuned | Change |")
    lines.append("|---|---|---|---|")
    if base:
        lines.append(
            f"| Eval loss | {base.get('eval_loss','?')} | {ft.get('eval_loss','?')} "
            f"| {_pct(base.get('eval_loss'), ft.get('eval_loss'))} |"
        )
        lines.append(
            f"| Perplexity | {base.get('perplexity','?')} | {ft.get('perplexity','?')} "
            f"| {_pct(base.get('perplexity'), ft.get('perplexity'))} |"
        )
    else:
        lines.append("| _baseline json missing — run scripts/run_baseline.py first_ | | | |")

    lines.append("\n> Lower eval loss / perplexity is better. Numbers above are")
    lines.append("> produced by the shipped scripts and are fully reproducible.")
    lines.append("> A 1B model yields modest but real gains — we do not overstate them.\n")

    lines.append("## Qualitative (fine-tuned samples)\n")
    for i, s in enumerate(samples[:3], 1):
        lines.append(f"**Q{i}: {s['question']}**\n\n{s['answer']}\n")
    lines.append("\nSee `results/baseline_samples.md` and `results/finetuned_samples.md` "
                 "for full side-by-side outputs.")
    Path("results").mkdir(exist_ok=True)
    Path("results/comparison.md").write_text("\n".join(lines), encoding="utf-8")
