# LoRA Fine-Tuning Project — Implementation Plan

> **Status:** Planning document for review. No code written yet.
> Once approved, this plan is executed phase-by-phase into a runnable, GitHub-ready repo.

**Locked decisions**

| Decision | Choice | Why |
|---|---|---|
| Base model | `meta-llama/Llama-3.2-1B-Instruct` | Fits free 16GB GPU comfortably; modern, recognizable architecture; instruct variant gives a fair baseline to beat. |
| Domain | Medical QA (educational) | Canonical, reliably-downloadable dataset → reproducible by any reviewer. Large before/after gap on a 1B model. Responsible-AI framing is an interview asset. |
| Dataset | `keivalya/MedQuad-MedicalQnADataset` (MedQuAD) | ~16k curated Q/A pairs sourced from NIH/NLM. Small, clean, permissively usable for research/education. |
| Method | LoRA (with optional QLoRA 4-bit path) | Parameter-efficient; adapter is a few MB; trains on free GPU. QLoRA path documented for lower-VRAM fallback. |
| Platform | Kaggle Notebooks (T4 16GB) | Free, stable, 30h/week GPU quota, persistent `/kaggle/working`. |
| Framework | Transformers + PEFT + TRL `SFTTrainer` + Accelerate + Datasets | Current HF best practice for instruction SFT. |

> ⚠️ **Responsible-use note baked into the project:** every inference path and the README carry a disclaimer that outputs are an educational demonstration of fine-tuning, **not medical advice**. This is intentional and is part of the interview story.

---

## Phase 1 — Project Planning & Feasibility

### Why this model + dataset + platform combination works
- **Llama 3.2 1B Instruct** has ~1.24B params. In fp16 the weights are ~2.5GB; with LoRA we freeze all base weights and train only small adapter matrices, so optimizer state is tiny. This fits a single T4 (16GB) with room for batch + activations.
- **MedQuAD** is short-form Q/A. Sequences are mostly < 512 tokens, keeping activation memory low and letting us use a real batch size instead of batch-size-1 tricks.
- **Kaggle T4** gives a free, reproducible environment. Anyone can fork the notebook and re-run.

### Expected resource envelope (ESTIMATES — will be replaced by measured values after runs)

| Metric | LoRA (fp16) | QLoRA (4-bit) fallback |
|---|---|---|
| Peak GPU memory | ~9–11 GB | ~5–7 GB |
| Trainable params | ~3–6M (~0.3–0.5% of base) | same |
| Adapter size on disk | ~15–30 MB | ~15–30 MB |
| Training time (1 epoch, ~10k samples, T4) | ~45–90 min | ~60–120 min (dequant overhead) |
| Cost | $0 (free tier) | $0 |

### Limitations & risks (stated honestly)
- A 1B model has a hard capability ceiling — expect **modest but visible** gains, not GPT-4-level medical reasoning. We will not overstate results.
- MedQuAD answers are long/encyclopedic; we truncate, so the model learns style + grounding, not full articles.
- Perplexity improvement is expected; "accuracy" is not cleanly defined for open-ended QA — we use perplexity + qualitative side-by-side + optional ROUGE-L against reference answers, and we say so.
- Gated model: Llama 3.2 requires accepting Meta's license on Hugging Face. README documents this; a non-gated fallback (`Qwen2.5-1.5B-Instruct`) is provided.

### Expected evaluation improvement (ESTIMATE)
- Eval perplexity: base ~ high single digits→low teens on held-out medical Q/A → **~15–35% lower** after fine-tuning.
- Qualitative: answers become more on-topic, structured, and domain-styled. Shown as before/after pairs.

**Deliverable:** `docs/phase1_feasibility.md` capturing the above; numbers marked ESTIMATE until measured.

---

## Phase 2 — Architecture / Repository Structure

```
LoRA-FineTuning/
├── configs/
│   ├── model.yaml            # base model id, dtype, gated fallback
│   ├── lora.yaml             # r, alpha, dropout, target_modules, bias, task_type
│   ├── train.yaml            # epochs, lr, batch, grad-accum, scheduler, logging
│   └── data.yaml             # dataset id, split ratios, max_seq_len, prompt template name
├── datasets/                 # (gitignored) local cache of processed data
│   └── .gitkeep
├── notebooks/
│   ├── 01_kaggle_train.ipynb # end-to-end Kaggle runner (thin wrapper over src/)
│   └── 02_explore_eval.ipynb # results/plots exploration
├── src/
│   ├── data/
│   │   ├── load.py           # download + split MedQuAD
│   │   ├── prompt.py         # single source of truth for the chat/prompt template
│   │   └── preprocess.py     # clean, format, tokenize
│   ├── training/
│   │   ├── lora.py           # build LoraConfig from configs/lora.yaml
│   │   ├── train.py          # SFTTrainer pipeline, checkpoints, callbacks
│   │   └── callbacks.py      # metric logging → logs/metrics.jsonl
│   ├── evaluation/
│   │   ├── perplexity.py     # perplexity + eval loss
│   │   ├── compare.py        # before-vs-after table + optional ROUGE-L
│   │   └── baseline.py       # baseline eval BEFORE training
│   ├── inference/
│   │   ├── load_adapter.py   # load base + attach adapter
│   │   ├── merge.py          # merge_and_unload → standalone model
│   │   ├── generate.py       # single/interactive generation
│   │   └── batch_infer.py    # batch inference over a file
│   └── utils/
│       ├── config.py         # yaml loader + dataclasses
│       ├── seed.py           # global determinism
│       ├── logging_conf.py   # structured logging
│       └── env_check.py      # GPU/CUDA/bitsandbytes/HF-login checks
├── scripts/
│   ├── run_baseline.py       # CLI: baseline eval
│   ├── run_train.py          # CLI: train
│   ├── run_eval.py           # CLI: post-train eval + comparison
│   └── plot_results.py       # generate figures from logs/
├── outputs/                  # (gitignored) checkpoints, merged model
├── adapters/                 # final LoRA adapter (small → committed or released)
├── logs/                     # metrics.jsonl, training logs, eval json
├── results/                  # committed plots + before/after tables (the "proof")
├── tests/
│   ├── test_prompt.py        # template round-trip
│   ├── test_config.py        # config loads + validates
│   └── test_smoke.py         # 5-step train on tiny subset runs end-to-end
├── docs/
│   ├── phase1_feasibility.md
│   ├── hyperparameters.md    # every knob explained
│   └── interview_prep.md
├── .gitignore
├── requirements.txt
├── environment.md            # Kaggle/Colab/local setup
├── LICENSE                   # MIT (code); dataset/model licenses noted in README
└── README.md
```

**Folder rationale (each explained in README):** `configs/` keeps all tunables out of code (reproducibility); `src/` is importable and unit-testable; `scripts/` are thin CLIs; `notebooks/` wrap `src/` so logic isn't trapped in cells; `results/` holds the small committed artifacts that *prove* the project ran; `outputs/`+`datasets/` are gitignored heavyweight artifacts.

---

## Phase 3 — Environment

- `requirements.txt` pinned to a known-good, Kaggle-compatible set:
  `torch`, `transformers`, `peft`, `trl`, `accelerate`, `datasets`, `bitsandbytes` (QLoRA path), `evaluate`, `rouge_score`, `matplotlib`, `pyyaml`, `scikit-learn`, `huggingface_hub`.
- `src/utils/env_check.py` prints: `torch.cuda.is_available()`, device name, CUDA version, VRAM, bitsandbytes import status, and whether `HF_TOKEN` is set.
- **HF login:** documented via `huggingface_hub.login(token=...)` reading a Kaggle Secret (`HF_TOKEN`) — never hard-coded.
- **Kaggle setup:** enable GPU T4, add HF token as a Kaggle Secret, set `HF_HOME=/kaggle/working/hf_cache` for persistence.
- Env vars documented: `HF_TOKEN`, `HF_HOME`, `TOKENIZERS_PARALLELISM=false`, `PYTHONHASHSEED`.

---

## Phase 4 — Dataset

- **Load:** `keivalya/MedQuad-MedicalQnADataset` via `datasets.load_dataset`.
- **Structure:** question / answer (+ focus area). ~16k pairs.
- **Cleaning:** drop empty/duplicate Q or A, strip boilerplate, filter extreme-length answers (cap by token count), optional focus-area subsampling for balance.
- **Split:** 90 / 5 / 5 train/val/test with fixed seed; splits saved to disk for reproducibility.
- **Prompt formatting (single source of truth in `src/data/prompt.py`):** Llama-3 chat template —
  ```
  <|system|> You are a medical information assistant for educational purposes...
  <|user|> {question}
  <|assistant|> {answer}
  ```
  Applied via `tokenizer.apply_chat_template`. Loss masked on the prompt so the model is trained only to produce the answer (via TRL completion-only collator).
- **Tokenization:** `max_seq_len=512` (justified: covers ~95th percentile), right-padding, truncation.

**Deliverable:** processed splits + a token-length histogram in `results/`.

---

## Phase 5 — Baseline Evaluation (BEFORE training)

Run on the frozen base model, stored for comparison:
- Perplexity + eval loss on the held-out test split.
- Fixed set of ~8 medical questions → sample outputs saved to `results/baseline_samples.md`.
- Optional ROUGE-L vs reference answers.
- Everything written to `logs/baseline_eval.json`. **This runs before any training so the comparison is real.**

---

## Phase 6 — LoRA Configuration (every parameter justified)

| Param | Value | Why (not a copy of defaults) |
|---|---|---|
| `r` (rank) | **16** | Sweet spot for a 1B model on a focused domain: enough capacity to shift style/knowledge, small enough to stay ~0.3–0.5% params and avoid overfitting 16k samples. (8 = underfit risk; 64 = wasteful here.) |
| `lora_alpha` | **32** | Scaling = alpha/r = 2.0, a common effective-LR multiplier that trains stably without exploding updates. |
| `lora_dropout` | **0.05** | Light regularization; dataset is small so some dropout helps generalization without starving learning. |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Attention **and** MLP projections. Adapting MLP (not just attention) materially improves domain adaptation on Llama-family models; the extra cost is negligible at r=16. |
| `bias` | `none` | Standard for LoRA; training biases adds params for little gain and complicates merging. |
| `task_type` | `CAUSAL_LM` | This is decoder-only next-token LM. |
| `use_rslora` | optional, off by default | Documented as a tunable; rank-stabilized scaling can help at higher r. |

QLoRA path (fallback): `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=float16`, double quant on — explained in `docs/hyperparameters.md`.

---

## Phase 7 — Training

`src/training/train.py` using TRL `SFTTrainer`:

| Hyperparameter | Value | Why |
|---|---|---|
| epochs | 1–3 (default 2) | Small dataset overfits fast; 2 epochs balances learning vs memorization. Early stopping guards. |
| per-device batch | 8 (fp16) / 4 (QLoRA) | Fits T4 at seq 512. |
| grad accumulation | 4 | Effective batch 32 → stable gradients without OOM. |
| learning rate | 2e-4 | Standard, robust LoRA LR; higher than full-FT because only adapters train. |
| lr scheduler | cosine | Smooth decay improves final convergence. |
| warmup ratio | 0.03 | Prevents early instability. |
| weight decay | 0.0 (LoRA) | LoRA rarely benefits; dropout handles regularization. |
| precision | fp16 (T4) / bf16 if available | T4 lacks bf16; fp16 chosen accordingly. |
| gradient checkpointing | on | Trades compute for memory → allows the batch size above. |
| optimizer | `paged_adamw_8bit` (QLoRA) / `adamw_torch` | Paged optimizer avoids OOM spikes in 4-bit. |
| logging | every 10 steps → `logs/metrics.jsonl` | Feeds the plots. |
| eval strategy | per epoch (+ steps option) | Tracks val loss for early stopping. |
| save strategy | per epoch, `load_best_model_at_end` | Keeps the best checkpoint. |
| early stopping | patience 2 on eval loss | Stops if val loss stalls. |
| seed | 42 everywhere | Reproducibility. |

Includes: checkpoint save/resume, mixed precision, grad accumulation, in-training eval, callbacks writing structured metrics.

---

## Phase 8 — Post-Training Evaluation (Before vs After)

- Re-run perplexity/eval-loss with the adapter attached.
- Regenerate the **same** 8 fixed questions → `results/finetuned_samples.md`.
- `src/evaluation/compare.py` produces `results/comparison.md`: a table of base vs fine-tuned (eval loss, perplexity, ROUGE-L) + side-by-side sample outputs.
- **Honesty clause:** if gains are small, the report says so and hypothesizes why (model size, dataset noise, epochs). No fabricated numbers — the repo ships the scripts so results are re-derivable.

---

## Phase 9 — Visualization

`scripts/plot_results.py` → saved into `results/`:
- Training loss vs step.
- Train vs validation loss.
- Learning-rate schedule.
- Before/after perplexity bar chart.
- (Optional) ROUGE-L before/after.

All read from `logs/metrics.jsonl` / eval JSON so plots regenerate deterministically.

---

## Phase 10 — Inference

- `load_adapter.py`: base + PEFT adapter for lightweight serving.
- `merge.py`: `merge_and_unload()` → standalone merged model (for deployment/GGUF later).
- `generate.py`: interactive single-prompt CLI with the disclaimer banner.
- `batch_infer.py`: read questions from a file → write answers (JSONL).

---

## Phase 11 — GitHub Repository / README

Professional README with: badges, one-line pitch, disclaimer, architecture diagram (Mermaid), quickstart (Kaggle + local), dataset/model/license notes, config explanation, training + eval commands, **results table + committed plots**, "what I learned," limitations, future work (QLoRA scaling, DPO, GGUF/Ollama export, RAG comparison), and a reproducibility section. Screenshots list specified.

---

## Phase 12 — Resume / LinkedIn / Pitch

Deliverables generated into `docs/`:
- 2–3 quantified resume bullets (filled with **measured** numbers after runs).
- LinkedIn project blurb.
- GitHub repo description.
- 30-second elevator pitch.
- 2–3 minute interview walkthrough script.

---

## Phase 13 — Interview Prep

`docs/interview_prep.md` with detailed model answers: Why LoRA / PEFT / this dataset / this model; LoRA vs full fine-tuning; what QLoRA is and its trade-offs; what adapter weights are (the A·B low-rank decomposition); memory math (why adapters are tiny); GPU optimization (grad checkpointing, accumulation, 4-bit); hyperparameter reasoning (r, alpha, target_modules, LR); training challenges hit and how they were solved.

---

## Phase 14 — Git Commit Strategy

Realistic, reviewable history (branch off before committing; user pushes):
```
chore: initial project setup, structure, license, gitignore
feat: environment setup and GPU/CUDA verification utils
feat: dataset loading and MedQuAD preprocessing pipeline
feat: prompt template + tokenization with completion-only masking
feat: LoRA configuration module
feat: baseline evaluation (perplexity + samples)
feat: SFTTrainer training pipeline with checkpoints + early stopping
feat: post-training evaluation and before/after comparison
feat: result visualizations
feat: inference scripts (adapter load, merge, interactive, batch)
test: smoke + config + prompt tests
docs: professional README, hyperparameter guide, interview prep
chore: prepare release v1.0
```

---

## Phase 15 — Final Audit

Senior-review checklist run at the end: reproducibility (fresh-clone re-run), missing docs, code quality/typing, config-vs-hardcode leaks, secret hygiene (no tokens committed), honest-results check, scalability notes, and interview-risk flags — written to `docs/audit.md` with actionable fixes.

---

## Cross-cutting guarantees
- Modern HF best practices; clean, typed, modular, commented code.
- Reproducible: fixed seeds, pinned deps, saved splits, config-driven.
- **Never fabricate metrics** — scripts ship so any number is re-derivable; estimates are labeled ESTIMATE until measured.
- Optimized for free GPU (Kaggle T4) by default.
- Fresh-clone recreatable by another developer.

---

## What I need from you before building
1. **Approve the plan** (or tweak: domain, model, epochs, etc.).
2. **HF access:** confirm you can accept the Llama 3.2 license on Hugging Face. If not, I default the config to the non-gated **`Qwen2.5-1.5B-Instruct`** and note it in the README.

On approval I execute Phases 2–15 into real files, starting with structure + configs + environment, then data → baseline → training → eval → viz → inference → docs.
