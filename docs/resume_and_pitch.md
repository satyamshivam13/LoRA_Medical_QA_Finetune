# Resume, LinkedIn & Pitch Assets

> Fill the `[X]` placeholders with the **measured** numbers from `results/comparison.md`
> after you run the pipeline. Do not invent numbers.

## Resume bullets (pick 2–3)

- Fine-tuned **Llama 3.2 1B** for medical Q/A using **LoRA (PEFT)**, training only **~0.4%** of
  parameters and reducing held-out **perplexity by [X]%** — on a **free Kaggle T4 GPU**.
- Built a reproducible **fine-tuning pipeline** (config-driven training, completion-only loss
  masking, checkpointing, early stopping) with **before/after evaluation** (perplexity, ROUGE-L,
  qualitative comparison) and automated result plots.
- Shipped a **~20MB LoRA adapter** with adapter-load, merge, interactive, and batch-inference
  scripts, plus unit tests and full documentation — a fresh clone reproduces the results.

## LinkedIn project summary

> **LoRA Fine-Tuning of Llama 3.2 1B for Medical Q/A (Educational)**
> A portfolio project demonstrating parameter-efficient fine-tuning end to end: dataset
> preprocessing with completion-only masking, LoRA configuration, T4-optimized training
> (fp16, gradient checkpointing, accumulation), and honest before/after evaluation
> (perplexity + ROUGE-L + side-by-side samples). Config-driven and fully reproducible.
> Includes a QLoRA 4-bit fallback and an interview-ready hyperparameter guide.
> *Framed strictly as an educational demo, not medical advice.*
> Stack: PyTorch · Hugging Face Transformers · PEFT · Datasets · Accelerate.

## GitHub repo description (short)

> Reproducible LoRA fine-tuning of Llama 3.2 1B for medical Q/A on a free GPU — config-driven
> training, honest before/after eval, and inference scripts. Educational demo.

## 30-second elevator pitch

> I fine-tuned a 1-billion-parameter Llama model on medical Q/A using LoRA, so I only trained
> about half a percent of the parameters and it ran on a free Kaggle GPU. The key part is the
> evaluation: I measure perplexity and sample outputs on a held-out split *before* and *after*
> training, so the improvement is real and reproducible — not cherry-picked. The whole thing is
> config-driven, tested, and documented so anyone can re-run it.

## 2–3 minute interview walkthrough

1. **Problem & framing** — show practical PEFT skill with a reproducible, honestly-evaluated
   project; educational medical-QA demo with explicit disclaimers.
2. **Data** — MedQuAD, cleaned and split deterministically; completion-only masking so the model
   learns to answer, not echo the prompt.
3. **Method** — LoRA (r=16, alpha=32) on attention + MLP projections; ~0.4% params trainable;
   fp16 + gradient checkpointing + accumulation to fit a T4.
4. **Evaluation** — baseline measured before training; identical post-training measurement;
   perplexity + ROUGE-L + side-by-side samples; plots regenerated from logged metrics.
5. **Results & honesty** — a 1B model gives modest but real gains; every number is
   re-derivable from shipped scripts; I don't overstate.
6. **Extensions** — QLoRA for bigger models, DPO/preference tuning, GGUF/Ollama export, RAG comparison.
