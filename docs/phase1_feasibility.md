# Phase 1 — Feasibility & Planning

> **All numbers below are ESTIMATES** based on the model size, dataset, and T4 hardware.
> They are replaced by **measured** values in `results/comparison.md` after a run.

## The combination

- **Model:** `meta-llama/Llama-3.2-1B-Instruct` (~1.24B params). fp16 weights ≈ 2.5 GB.
- **Dataset:** `keivalya/MedQuad-MedicalQnADataset` (MedQuAD, ~16k medical Q/A pairs from NIH/NLM).
- **Platform:** Kaggle Notebooks, single T4 (16 GB).

**Why it fits:** With LoRA, all base weights are frozen; only ~0.3–0.5% of parameters train,
so optimizer state (the usual memory killer) is tiny. Short Q/A sequences (mostly < 512 tokens)
keep activation memory low, so we can use a real batch size instead of batch-size-1 tricks.

## Expected resource envelope (ESTIMATE)

| Metric | LoRA (fp16) | QLoRA (4-bit) |
|---|---|---|
| Peak GPU memory | ~9–11 GB | ~5–7 GB |
| Trainable params | ~3–6 M (~0.3–0.5%) | same |
| Adapter size on disk | ~15–30 MB | ~15–30 MB |
| Training time (2 epochs, T4) | ~1.5–3 h | ~2–4 h |
| Cost | $0 (free tier) | $0 |

## Expected evaluation improvement (ESTIMATE)

- **Perplexity:** base perplexity on held-out medical Q/A drops **~15–35%** after fine-tuning.
- **Qualitative:** answers become more on-topic, structured, and domain-styled.

## Limitations & risks (honest)

- A 1B model has a hard capability ceiling — expect **modest but visible** gains, not expert reasoning.
- MedQuAD answers are long/encyclopedic; truncation means the model learns **style + grounding**, not full articles.
- "Accuracy" isn't cleanly defined for open-ended QA — we rely on perplexity + qualitative
  comparison + optional ROUGE-L, and we say so rather than inventing an accuracy number.
- Llama 3.2 is **gated** on Hugging Face (accept Meta's license). Non-gated fallback:
  `Qwen/Qwen2.5-1.5B-Instruct` — switch one line in `configs/model.yaml`.

## Responsible-use framing

Every inference path and the README carry a disclaimer: outputs are an **educational
demonstration of fine-tuning, not medical advice**. This is intentional and is part of the
project's engineering-responsibility story.
