"""Load the base model and tokenizer, with optional 4-bit (QLoRA) quantization.

Shared by training, baseline eval, and inference so the load path is identical
everywhere.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.config import ModelConfig
from src.utils.logging_conf import get_logger

log = get_logger("training.model")

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_tokenizer(cfg: ModelConfig):
    tok = AutoTokenizer.from_pretrained(
        cfg.base_model_id, trust_remote_code=cfg.trust_remote_code
    )
    # Causal LMs often lack a pad token; reuse EOS. Pad left for generation-friendliness.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _bnb_config(cfg: ModelConfig):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_DTYPES.get(cfg.bnb_4bit_compute_dtype, torch.float16),
    )


def load_model(cfg: ModelConfig, for_training: bool = True):
    """Load the base causal LM. Applies 4-bit quantization if cfg.use_4bit."""
    kwargs: dict = {
        "trust_remote_code": cfg.trust_remote_code,
        "torch_dtype": _DTYPES.get(cfg.dtype, torch.float16),
    }
    if cfg.use_4bit:
        log.info("Loading %s in 4-bit (QLoRA)", cfg.base_model_id)
        kwargs["quantization_config"] = _bnb_config(cfg)
        kwargs["device_map"] = "auto"
    else:
        log.info("Loading %s in %s", cfg.base_model_id, cfg.dtype)
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, **kwargs)

    if for_training and cfg.use_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    return model
