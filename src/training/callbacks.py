"""Trainer callback that appends every logged metric to logs/metrics.jsonl.

This structured log is the single input to the plotting script, so figures
regenerate deterministically from disk.
"""
from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerCallback


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, path: str = "logs/metrics.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Fresh file per run.
        self.path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
