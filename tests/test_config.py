"""Config loads, validates, and exposes expected fields."""
from src.utils.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg.model.base_model_id
    assert cfg.lora.r > 0
    assert cfg.lora.target_modules, "target_modules must not be empty"
    assert cfg.train.num_train_epochs >= 1


def test_split_ratios_sum_to_one():
    cfg = load_config()
    total = cfg.data.train_ratio + cfg.data.val_ratio + cfg.data.test_ratio
    assert abs(total - 1.0) < 1e-6
