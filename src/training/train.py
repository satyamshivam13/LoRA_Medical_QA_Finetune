"""End-to-end LoRA fine-tuning pipeline.

Uses ``transformers.Trainer`` (not TRL) for maximum version stability and a
transparent, interview-explainable training loop. Completion-only loss masking
is handled upstream in preprocess.py.
"""
from __future__ import annotations

from pathlib import Path

from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.load import load_and_split
from src.data.preprocess import tokenize_dataset
from src.training.callbacks import JsonlMetricsCallback
from src.training.lora import apply_lora
from src.training.model_loader import load_model, load_tokenizer
from src.utils.config import Config
from src.utils.logging_conf import get_logger
from src.utils.seed import set_seed

log = get_logger("training.train")


def train(cfg: Config) -> str:
    """Run fine-tuning end-to-end. Returns the path to the saved adapter."""
    set_seed(cfg.train.seed)

    tokenizer = load_tokenizer(cfg.model)
    model = load_model(cfg.model, for_training=True)
    if cfg.train.gradient_checkpointing:
        model.config.use_cache = False  # incompatible with grad checkpointing
    model = apply_lora(model, cfg.lora)

    # Data
    splits = load_and_split(cfg.data)
    train_ds = tokenize_dataset(splits["train"], tokenizer, cfg.data)
    val_ds = tokenize_dataset(splits["validation"], tokenizer, cfg.data)

    collator = DataCollatorForSeq2Seq(
        tokenizer, padding="longest", label_pad_token_id=-100
    )

    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        fp16=cfg.train.fp16,
        bf16=cfg.train.bf16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        optim=cfg.train.optim,
        logging_steps=cfg.train.logging_steps,
        eval_strategy=cfg.train.eval_strategy,
        save_strategy=cfg.train.save_strategy,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        metric_for_best_model=cfg.train.metric_for_best_model,
        greater_is_better=cfg.train.greater_is_better,
        seed=cfg.train.seed,
        report_to=cfg.train.report_to,
    )

    callbacks = [JsonlMetricsCallback()]
    if cfg.train.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.train.early_stopping_patience)
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    log.info("Starting training…")
    trainer.train()

    adapter_dir = Path(cfg.train.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log.info("Saved LoRA adapter to %s", adapter_dir)
    return str(adapter_dir)
