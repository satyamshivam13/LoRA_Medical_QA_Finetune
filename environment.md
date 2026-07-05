# Environment Setup

## Option A — Kaggle Notebooks (recommended, free T4)

1. Create a new Notebook → **Settings → Accelerator → GPU T4 x2** (one T4 is enough).
2. Add your Hugging Face token as a **Kaggle Secret** named `HF_TOKEN`
   (Add-ons → Secrets). Never paste the token into a cell.
3. First cell:
   ```python
   import os
   from kaggle_secrets import UserSecretsClient
   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
   os.environ["HF_HOME"] = "/kaggle/working/hf_cache"   # persist model cache
   os.environ["TOKENIZERS_PARALLELISM"] = "false"

   from huggingface_hub import login
   login(os.environ["HF_TOKEN"])
   ```
4. Clone the repo (or upload it as a dataset) and install:
   ```bash
   pip install -q -r requirements.txt
   ```
5. Verify the environment:
   ```bash
   python -m src.utils.env_check
   ```
6. Run the pipeline (see README "Quickstart").

> **Gated model:** you must accept the license on the
> [Llama 3.2 1B model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) with the
> same HF account whose token you use. No access? Set
> `base_model_id: "Qwen/Qwen2.5-1.5B-Instruct"` in `configs/model.yaml` (non-gated).

## Option B — Google Colab (free T4)

Same as Kaggle but store the token via `from google.colab import userdata` or
`os.environ["HF_TOKEN"] = getpass()`. Colab sessions are shorter — training resumes from the
last checkpoint in `outputs/` automatically (`resume_from_checkpoint` supported by Trainer).

## Option C — Local GPU

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export HF_TOKEN=hf_xxx          # Windows PowerShell: $env:HF_TOKEN="hf_xxx"
python -m src.utils.env_check
```
On Ampere+ GPUs set `dtype: bfloat16` in `configs/model.yaml` and `bf16: true` in
`configs/train.yaml`. For <8GB VRAM set `use_4bit: true` (QLoRA) and
`optim: paged_adamw_8bit`.

## Environment variables

| Var | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face auth (gated models, higher rate limits) |
| `HF_HOME` | Model/dataset cache location (set to a persistent dir on Kaggle) |
| `TOKENIZERS_PARALLELISM` | Set `false` to silence tokenizer fork warnings |
| `RUN_SMOKE` | Set `1` to enable the model-downloading smoke test |
