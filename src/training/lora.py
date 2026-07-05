"""Build the PEFT LoRA config from configs/lora.yaml and wrap the model."""
from __future__ import annotations

from peft import LoraConfig, get_peft_model
from peft.utils import TaskType

from src.utils.config import LoraSettings
from src.utils.logging_conf import get_logger

log = get_logger("training.lora")


def build_lora_config(cfg: LoraSettings) -> LoraConfig:
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=getattr(TaskType, cfg.task_type, cfg.task_type),
        target_modules=list(cfg.target_modules),
        use_rslora=cfg.use_rslora,
    )


def apply_lora(model, cfg: LoraSettings):
    """Attach LoRA adapters and log the trainable-parameter percentage."""
    peft_model = get_peft_model(model, build_lora_config(cfg))
    trainable, total = peft_model.get_nb_trainable_parameters()
    log.info(
        "LoRA attached — trainable: %s / %s (%.3f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )
    return peft_model
