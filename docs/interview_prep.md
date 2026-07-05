# Interview Preparation

Model answers for questions this project invites. Keep them concise and lead with the "why."

### Why LoRA?
LoRA (Low-Rank Adaptation) freezes the pretrained weights and injects small trainable
low-rank matrices into chosen layers. For each weight `W`, instead of learning a full update
`ΔW`, we learn `B·A` where `A` is `r×k` and `B` is `d×r` with `r ≪ d,k`. This trains
~0.3–0.5% of the parameters, so it fits a free 16GB GPU, trains in hours not days, and
produces a ~20MB adapter instead of a multi-GB checkpoint — while recovering most of the
quality of full fine-tuning on a focused task.

### Why PEFT (parameter-efficient fine-tuning) generally?
Full fine-tuning of even a 1B model needs memory for weights + gradients + optimizer state
(Adam keeps two moments per parameter → ~4× the weight memory just for optimizer state).
PEFT sidesteps this by training a tiny set of new parameters, making fine-tuning feasible on
commodity/free hardware and cheap to store and serve (swap adapters per task).

### Why this dataset (MedQuAD)?
It's a canonical, permissively-usable, well-structured medical Q/A set sourced from NIH/NLM.
That makes the project **reproducible by any reviewer** and gives a clear domain shift a base
model handles only generically — so the before/after improvement is visible.

### Why this model (Llama 3.2 1B Instruct)?
It's modern, recognizable, and small enough to train comfortably on a free T4, but capable
enough to show a real gain. The Instruct variant gives a fair, already-decent baseline to beat.

### LoRA vs full fine-tuning — trade-offs?
Full FT can reach slightly higher ceilings and change model behavior more deeply, but costs
far more memory/compute/storage and risks catastrophic forgetting. LoRA is cheaper, faster,
storage-light, composable (multiple adapters), and easily reverted — at a small quality cost
that's usually negligible for a single downstream task.

### What is QLoRA?
LoRA on top of a **4-bit quantized** frozen base model. The base weights load in NF4 (4-bit),
cutting memory ~4×, while the LoRA adapters stay in fp16. It enables fine-tuning larger models
on tiny GPUs. Trade-off: some dequantization compute overhead and mild quality sensitivity.
This repo exposes it via `use_4bit: true`.

### What are "adapter weights"?
Just the trained `A` and `B` low-rank matrices (plus config). The frozen base is unchanged, so
the adapter alone is enough to reproduce the fine-tuned behavior when loaded on top of the base.
That's why the artifact is only ~20MB.

### Memory math — why is the adapter tiny?
For a `2048×2048` projection, full `ΔW` = ~4.2M params. LoRA with r=16 = `16·(2048+2048)` =
~65k params — a ~98% reduction per adapted layer. Summed across target modules it's a few
million total vs 1.24B.

### GPU optimization techniques used here
- **Gradient checkpointing** — recompute activations in backward instead of storing them (~40–60% activation-memory saving).
- **Gradient accumulation** — effective batch 32 from micro-batches of 8 without the memory.
- **fp16 mixed precision** — halves activation/weight memory on T4.
- **(Optional) 4-bit QLoRA + paged AdamW** — for <8GB VRAM.

### How did you pick the hyperparameters?
See `docs/hyperparameters.md` — short version: r=16/alpha=32 for capacity-vs-overfit balance on
16k samples; LR 2e-4 because only adapters train; 2 epochs + early stopping because small data
overfits fast; attention **and** MLP target modules for stronger domain adaptation.

### What challenges did you hit / anticipate?
- Gated model access (documented HF-token flow + non-gated fallback).
- Completion-only loss masking so the model learns to *answer*, not echo the prompt (labels = -100 on prompt tokens).
- T4 lacks bf16 → used fp16 and watched for overflow (max_grad_norm clipping).
- Long encyclopedic answers → truncation + length filtering.

### How do you know it actually improved? Did you avoid overstating?
Baseline is measured **before** training on a held-out test split (perplexity + fixed sample
questions), then re-measured identically after. All scripts ship, so every number is
re-derivable. A 1B model gives modest gains — the report says so plainly and never invents an
"accuracy" figure for open-ended QA.
