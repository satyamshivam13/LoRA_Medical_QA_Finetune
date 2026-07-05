# Resume, LinkedIn & Pitch Assets

> Numbers below are the **measured** results from `results/comparison.md`
> (base perplexity 9.774 → fine-tuned 2.943 on the held-out test split).

## Resume bullets (pick 2–3)

- Fine-tuned **Llama 3.2 1B** for medical Q/A using **LoRA (PEFT)**, training only **~11M
  parameters (<1%)** and cutting held-out **perplexity ~70% (9.77 → 2.94)** — on a **free
  Kaggle T4 GPU** in under 2 hours.
- Built a reproducible **fine-tuning pipeline** (config-driven training, completion-only loss
  masking, checkpointing, early stopping) with **before/after evaluation** (perplexity, ROUGE-L,
  qualitative comparison) and automated result plots.
- Shipped a **~20MB LoRA adapter** with adapter-load, merge, interactive, and batch-inference
  scripts, plus unit tests and CI — a fresh clone reproduces the results.

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
> about 11 million parameters — under one percent — and it ran on a free Kaggle GPU. The key
> part is the evaluation: I measured perplexity on a held-out split before and after, and it
> dropped from 9.8 to 2.9 — about 70%. It's real and reproducible, not cherry-picked. The whole
> thing is config-driven, tested, and has CI so anyone can re-run it.

## 2–3 minute interview walkthrough

1. **Problem & framing** — show practical PEFT skill with a reproducible, honestly-evaluated
   project; educational medical-QA demo with explicit disclaimers.
2. **Data** — MedQuAD, cleaned and split deterministically; completion-only masking so the model
   learns to answer, not echo the prompt.
3. **Method** — LoRA (r=16, alpha=32) on attention + MLP projections; ~11M params trainable
   (<1%); fp16 + gradient checkpointing + accumulation to fit a T4.
4. **Evaluation** — baseline measured before training; identical post-training measurement;
   perplexity 9.77 → 2.94 (~70% drop) + side-by-side samples; plots regenerated from logged metrics.
5. **Results & honesty** — a real, reproducible perplexity drop, but I note that much of it is
   the model adapting to the dataset's answer style, not a 1B model becoming a clinician. Every
   number is re-derivable from shipped scripts; I don't overstate.
6. **Extensions** — QLoRA for bigger models, DPO/preference tuning, GGUF/Ollama export, RAG comparison.
