"""Typed configuration loading.

All tunables live in ``configs/*.yaml`` so experiments are reproducible and code
stays free of magic numbers. This module loads those YAML files into dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

# Repo root = two levels up from this file (src/utils/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class ModelConfig:
    base_model_id: str
    fallback_model_id: str
    trust_remote_code: bool = False
    dtype: str = "float16"
    use_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"


@dataclass
class LoraSettings:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    target_modules: List[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    output_dir: str
    adapter_dir: str
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 2
    seed: int = 42
    report_to: str = "none"


@dataclass
class DataConfig:
    dataset_id: str
    question_field: str = "Question"
    answer_field: str = "Answer"
    max_seq_len: int = 512
    max_answer_chars: int = 2000
    min_answer_chars: int = 20
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42
    max_samples: Optional[int] = None
    processed_dir: str = "datasets/processed"


@dataclass
class Config:
    model: ModelConfig
    lora: LoraSettings
    train: TrainConfig
    data: DataConfig


def load_config(config_dir: Path | str = CONFIG_DIR) -> Config:
    """Load and validate all config files into a single ``Config`` object."""
    config_dir = Path(config_dir)
    cfg = Config(
        model=ModelConfig(**_read_yaml(config_dir / "model.yaml")),
        lora=LoraSettings(**_read_yaml(config_dir / "lora.yaml")),
        train=TrainConfig(**_read_yaml(config_dir / "train.yaml")),
        data=DataConfig(**_read_yaml(config_dir / "data.yaml")),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    ratios = cfg.data.train_ratio + cfg.data.val_ratio + cfg.data.test_ratio
    if abs(ratios - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")
    if not cfg.lora.target_modules:
        raise ValueError("lora.target_modules must not be empty")
    if cfg.model.use_4bit and cfg.train.optim == "adamw_torch":
        # Not fatal, but the paged optimizer is strongly recommended for 4-bit.
        print("[config] WARNING: use_4bit=true but optim=adamw_torch; "
              "consider optim=paged_adamw_8bit in configs/train.yaml")
