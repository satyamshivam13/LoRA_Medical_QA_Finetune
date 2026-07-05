"""CLI: inference entry point (interactive, batch, or merge).

Usage:
  python -m scripts.run_infer                          # interactive chat
  python -m scripts.run_infer --batch in.txt out.jsonl # batch inference
  python -m scripts.run_infer --merge                  # merge adapter into base
"""
from __future__ import annotations

import argparse

from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuned model inference")
    parser.add_argument("--batch", nargs=2, metavar=("INPUT", "OUTPUT"),
                        help="Batch inference: input questions file, output jsonl")
    parser.add_argument("--merge", action="store_true", help="Merge adapter into base weights")
    args = parser.parse_args()

    cfg = load_config()
    if args.merge:
        from src.inference.merge import merge_adapter
        print("Merged model at:", merge_adapter(cfg))
    elif args.batch:
        from src.inference.batch_infer import run_batch
        run_batch(cfg, args.batch[0], args.batch[1])
    else:
        from src.inference.generate import interactive
        interactive(cfg)


if __name__ == "__main__":
    main()
