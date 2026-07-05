# Phase 15 — Final Project Audit

A senior-review pass over the repo, as if reviewing it as a portfolio submission.

## Strengths
- **Config-driven & reproducible** — all tunables in `configs/`, fixed seeds, deterministic splits, pinned deps.
- **Honest evaluation** — baseline measured before training; identical post-training measurement; scripts shipped so every number is re-derivable. No fabricated metrics.
- **Correct SFT mechanics** — completion-only loss masking (prompt tokens = -100) so the model learns to answer, not echo.
- **Clean separation** — importable `src/` package, thin CLIs, notebook wraps the package (no logic trapped in cells).
- **Responsible framing** — disclaimers in every inference path and the README.
- **Documented reasoning** — hyperparameters justified, not copied; interview prep included.

## Known limitations / interview risks (be ready to discuss)
- **Modest gains** — a 1B model won't dramatically improve; own this and explain why (capacity ceiling, truncation).
- **No accuracy metric** — open-ended QA has no clean accuracy; perplexity + ROUGE-L + qualitative is the honest choice.
- **Single dataset/domain** — no cross-domain generalization claim is made.
- **ROUGE-L is a weak proxy** for factual correctness; noted explicitly.

## Reproducibility checklist
- [ ] Fresh clone → `pip install -r requirements.txt` → `env_check` passes.
- [ ] `run_baseline` → `run_train` → `run_eval` → `plot_results` completes on a T4.
- [ ] `results/comparison.md` + plots populated; README results table filled with **measured** numbers.
- [ ] No secrets committed (`git grep -i hf_ ...` clean; token only via env var).

## Recommended polish before showcasing
1. Run the full pipeline once and **commit the populated `results/`** (plots + comparison) — reviewers want to see proof.
2. Fill the README results table and the `[X]%` placeholders in `docs/resume_and_pitch.md` with real numbers.
3. Add 1–2 screenshots (loss curve, a before/after sample) to the README.
4. Optional: add a GitHub Actions workflow running `pytest tests/test_config.py tests/test_prompt.py` on push.

## Scalability / extension notes
- Swap `base_model_id` to a 3B/7B model + `use_4bit: true` to demonstrate QLoRA scaling.
- Add DPO/preference tuning as a follow-up phase.
- Export the merged model to GGUF for Ollama/llama.cpp serving.
