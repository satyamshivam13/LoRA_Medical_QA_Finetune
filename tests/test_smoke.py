"""Optional end-to-end smoke test on a tiny subset.

Skipped unless torch+transformers are installed AND RUN_SMOKE=1 is set, because
it downloads a small model. On Kaggle/local GPU run:

    RUN_SMOKE=1 pytest tests/test_smoke.py -s
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SMOKE") != "1",
    reason="Set RUN_SMOKE=1 to run the model-downloading smoke test.",
)


def test_tokenize_and_mask():
    from transformers import AutoTokenizer

    from src.data.preprocess import build_tokenize_fn
    from src.utils.config import load_config

    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg.model.fallback_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    fn = build_tokenize_fn(tok, cfg.data)
    out = fn({"question": "What is anemia?", "answer": "A blood condition."})
    assert len(out["input_ids"]) == len(out["labels"])
    assert any(l == -100 for l in out["labels"]), "prompt tokens must be masked"
    assert any(l != -100 for l in out["labels"]), "answer tokens must be supervised"
