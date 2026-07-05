"""Generate figures from logs/metrics.jsonl and eval JSONs → results/*.png.

Usage:  python -m scripts.plot_results
Deterministic: reads only on-disk logs, so figures regenerate identically.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (Kaggle/CI safe)
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def _load_metrics(path="logs/metrics.jsonl"):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def plot_losses(records):
    train = [(r["step"], r["loss"]) for r in records if "loss" in r]
    val = [(r["step"], r["eval_loss"]) for r in records if "eval_loss" in r]
    if not train:
        print("No training loss found — skipping loss plot.")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(*zip(*train), label="train loss")
    if val:
        plt.plot(*zip(*val), "o-", label="val loss")
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS / "loss_curve.png", dpi=120); plt.close()
    print("Wrote results/loss_curve.png")


def plot_lr(records):
    lr = [(r["step"], r["learning_rate"]) for r in records if "learning_rate" in r]
    if not lr:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(*zip(*lr))
    plt.xlabel("step"); plt.ylabel("learning rate"); plt.title("LR Schedule")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS / "lr_schedule.png", dpi=120); plt.close()
    print("Wrote results/lr_schedule.png")


def plot_before_after():
    b = Path("logs/baseline_eval.json"); f = Path("logs/finetuned_eval.json")
    if not (b.exists() and f.exists()):
        print("Baseline/finetuned eval json missing — skipping before/after plot.")
        return
    base = json.loads(b.read_text()); ft = json.loads(f.read_text())
    metrics = ["eval_loss", "perplexity"]
    base_vals = [base.get(m, 0) for m in metrics]
    ft_vals = [ft.get(m, 0) for m in metrics]
    x = range(len(metrics))
    plt.figure(figsize=(7, 5))
    plt.bar([i - 0.2 for i in x], base_vals, width=0.4, label="baseline")
    plt.bar([i + 0.2 for i in x], ft_vals, width=0.4, label="fine-tuned")
    plt.xticks(list(x), metrics); plt.ylabel("value (lower is better)")
    plt.title("Before vs After"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "before_after.png", dpi=120); plt.close()
    print("Wrote results/before_after.png")


def main():
    records = _load_metrics()
    plot_losses(records)
    plot_lr(records)
    plot_before_after()


if __name__ == "__main__":
    main()
