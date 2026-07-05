"""CLI: run LoRA fine-tuning.

Usage:  python -m scripts.run_train
"""
from __future__ import annotations

from src.training.train import train
from src.utils.config import load_config


def main() -> None:
    cfg = load_config()
    adapter_path = train(cfg)
    print(f"\nDone. Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
