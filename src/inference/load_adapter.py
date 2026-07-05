"""Load the base model with the trained LoRA adapter attached (lightweight)."""
from __future__ import annotations

from peft import PeftModel

from src.training.model_loader import load_model, load_tokenizer
from src.utils.config import Config
from src.utils.logging_conf import get_logger

log = get_logger("inference.load")


def load_adapter_model(cfg: Config):
    """Return (model_with_adapter, tokenizer). Tokenizer loaded from the adapter dir
    when available (keeps any special-token state), else from the base model."""
    base = load_model(cfg.model, for_training=False)
    log.info("Attaching adapter from %s", cfg.train.adapter_dir)
    model = PeftModel.from_pretrained(base, cfg.train.adapter_dir)
    model.eval()

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.train.adapter_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:  # noqa: BLE001
        tokenizer = load_tokenizer(cfg.model)
    return model, tokenizer
