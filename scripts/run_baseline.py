"""CLI: baseline evaluation of the base model BEFORE training.

Usage:  python -m scripts.run_baseline
"""
from __future__ import annotations

from src.evaluation.baseline import run_baseline
from src.utils.config import load_config


def main() -> None:
    cfg = load_config()
    result = run_baseline(cfg)
    print("\nBaseline:", result)


if __name__ == "__main__":
    main()
