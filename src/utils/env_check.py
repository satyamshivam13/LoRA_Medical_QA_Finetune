"""Verify the runtime environment (GPU, CUDA, key libraries, HF auth).

Run directly:  python -m src.utils.env_check
"""
from __future__ import annotations

import os


def check_environment() -> dict:
    """Print and return a summary of the training environment."""
    info: dict = {}

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 1)
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        info["torch"] = "NOT INSTALLED"

    for lib in ("transformers", "peft", "accelerate", "datasets", "trl", "bitsandbytes"):
        try:
            mod = __import__(lib)
            info[lib] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            info[lib] = f"unavailable ({type(exc).__name__})"

    info["hf_token_set"] = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    info["hf_home"] = os.environ.get("HF_HOME", "(default)")

    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60)
    if not info.get("cuda_available"):
        print("  WARNING: No GPU detected. Training will be very slow on CPU.")
    if not info.get("hf_token_set"):
        print("  NOTE: HF_TOKEN not set. Required for gated models (Llama 3.2).")
    print("=" * 60)
    return info


if __name__ == "__main__":
    check_environment()
