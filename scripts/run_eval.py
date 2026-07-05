"""CLI: post-training evaluation + before/after comparison.

Usage:  python -m scripts.run_eval
(Run scripts.run_baseline first so the comparison table has a baseline.)
"""
from __future__ import annotations

from src.evaluation.compare import run_comparison
from src.utils.config import load_config


def main() -> None:
    cfg = load_config()
    run_comparison(cfg)
    print("\nComparison written to results/comparison.md")


if __name__ == "__main__":
    main()
