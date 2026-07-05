"""Merge the LoRA adapter into the base weights → a standalone model.

Useful for deployment (single model, no PEFT dependency) or later export to
GGUF/Ollama. Produces adapters/<name>/merged_model/.
"""
from __future__ import annotations

from pathlib import Path

from src.inference.load_adapter import load_adapter_model
from src.utils.config import Config
from src.utils.logging_conf import get_logger

log = get_logger("inference.merge")


def merge_adapter(cfg: Config) -> str:
    if cfg.model.use_4bit:
        raise RuntimeError(
            "Cannot merge into a 4-bit model. Set use_4bit=false in configs/model.yaml "
            "to load in fp16, then merge."
        )
    model, tokenizer = load_adapter_model(cfg)
    log.info("Merging adapter into base weights…")
    merged = model.merge_and_unload()

    out_dir = Path(cfg.train.adapter_dir) / "merged_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info("Merged model saved → %s", out_dir)
    return str(out_dir)
