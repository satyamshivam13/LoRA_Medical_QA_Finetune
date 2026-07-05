"""Download, clean, and split the MedQuAD dataset.

Produces a reproducible train/val/test ``DatasetDict`` with normalized
``question`` / ``answer`` columns. Splits are deterministic given the seed.
"""
from __future__ import annotations

from typing import Tuple

from datasets import Dataset, DatasetDict, load_dataset

from src.utils.config import DataConfig
from src.utils.logging_conf import get_logger

log = get_logger("data.load")


def _normalize_columns(ds: Dataset, cfg: DataConfig) -> Dataset:
    """Rename the dataset's Q/A columns to canonical 'question'/'answer'."""
    cols = {c.lower(): c for c in ds.column_names}
    q = cols.get(cfg.question_field.lower(), cfg.question_field)
    a = cols.get(cfg.answer_field.lower(), cfg.answer_field)
    if q not in ds.column_names or a not in ds.column_names:
        raise KeyError(
            f"Expected columns '{cfg.question_field}'/'{cfg.answer_field}' not found. "
            f"Available: {ds.column_names}"
        )
    ds = ds.rename_columns({q: "question", a: "answer"})
    keep = [c for c in ds.column_names if c in ("question", "answer")]
    return ds.select_columns(keep)


def _clean(ds: Dataset, cfg: DataConfig) -> Dataset:
    """Drop empty/too-short/too-long/duplicate examples."""
    seen: set = set()

    def _keep(example: dict) -> bool:
        q = (example["question"] or "").strip()
        a = (example["answer"] or "").strip()
        if not q or not a:
            return False
        if len(a) < cfg.min_answer_chars or len(a) > cfg.max_answer_chars:
            return False
        key = (q.lower(), a[:80].lower())
        if key in seen:
            return False
        seen.add(key)
        return True

    before = len(ds)
    ds = ds.filter(_keep)
    log.info("Cleaning: %d -> %d examples (removed %d)", before, len(ds), before - len(ds))
    return ds


def load_and_split(cfg: DataConfig) -> DatasetDict:
    """Return a cleaned, deterministically split DatasetDict."""
    log.info("Loading dataset: %s", cfg.dataset_id)
    raw = load_dataset(cfg.dataset_id)
    # These datasets usually only have a 'train' split.
    base = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]

    base = _normalize_columns(base, cfg)
    base = _clean(base, cfg)

    if cfg.max_samples:
        base = base.shuffle(seed=cfg.seed).select(range(min(cfg.max_samples, len(base))))
        log.info("Capped to %d samples (max_samples)", len(base))

    # Deterministic three-way split.
    test_frac = cfg.test_ratio
    val_frac = cfg.val_ratio / (1.0 - test_frac)  # of the remaining after test removed
    split1 = base.train_test_split(test_size=test_frac, seed=cfg.seed)
    split2 = split1["train"].train_test_split(test_size=val_frac, seed=cfg.seed)

    out = DatasetDict(
        train=split2["train"],
        validation=split2["test"],
        test=split1["test"],
    )
    log.info(
        "Splits — train=%d val=%d test=%d",
        len(out["train"]), len(out["validation"]), len(out["test"]),
    )
    return out


def token_length_stats(ds: Dataset, tokenizer, sample: int = 1000) -> Tuple[int, int, int]:
    """Return (median, p95, max) token length of 'answer' for a sample — for reporting."""
    import numpy as np

    subset = ds.select(range(min(sample, len(ds))))
    lengths = [len(tokenizer(a).input_ids) for a in subset["answer"]]
    arr = np.array(lengths)
    return int(np.median(arr)), int(np.percentile(arr, 95)), int(arr.max())
