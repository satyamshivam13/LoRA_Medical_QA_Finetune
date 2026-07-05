# Hyperparameter Guide

Every tunable lives in `configs/` so runs are reproducible. This document explains
**why** each value was chosen — not just what it is. These are deliberate choices for a
**1B model on a ~16k-example medical QA dataset on a free T4 GPU**, not copied defaults.

## LoRA (`configs/lora.yaml`)

| Param | Value | Reasoning |
|---|---|---|
| `r` (rank) | 16 | The rank of the low-rank update matrices A (d×r) and B (r×k). Higher r = more capacity but more params and overfitting risk. For a focused single-domain task on 16k samples, 16 captures the needed adaptation. r=8 risks underfitting the domain shift; r=64 wastes capacity and can memorize. |
| `lora_alpha` | 32 | The update is scaled by `alpha/r`. Here 32/16 = 2.0, a stable effective-learning-rate multiplier. Keeping alpha = 2×r is a common, robust ratio. |
| `lora_dropout` | 0.05 | Regularization applied to the LoRA input. The dataset is small, so light dropout improves generalization; too much (0.2+) starves the small adapter of signal. |
| `target_modules` | q,k,v,o + gate,up,down | We adapt **attention and MLP** projections. Adapting only attention is cheaper but leaves most of the model's knowledge-processing (the MLP blocks) untouched. On Llama-family models, adding the MLP projections measurably improves domain adaptation, and at r=16 the extra parameter cost is negligible. |
| `bias` | none | Don't train bias terms. Adds parameters for little benefit and complicates merging. |
| `task_type` | CAUSAL_LM | Decoder-only next-token prediction. |
| `use_rslora` | false | Rank-stabilized LoRA rescales by `alpha/sqrt(r)`, which helps at higher ranks. At r=16 standard scaling is fine; exposed as a knob for experiments. |

**Why so few trainable params?** A LoRA layer replaces a full weight update `ΔW` (d×k) with `B·A`
where A is r×k and B is d×r. Params drop from `d·k` to `r·(d+k)`. For r=16 vs a 2048×2048
projection that's ~65k vs ~4.2M — a ~98% reduction per layer. Total trainable ≈ 0.3–0.5% of the model.

## Training (`configs/train.yaml`)

| Param | Value | Reasoning |
|---|---|---|
| `num_train_epochs` | 2 | Small datasets overfit quickly. 1 epoch often underfits the style; 3+ starts memorizing. 2 with early stopping is the balance. |
| `per_device_train_batch_size` | 8 | Largest batch that fits a T4 (16GB) at seq len 512 in fp16 with gradient checkpointing. Use 4 for QLoRA. |
| `gradient_accumulation_steps` | 4 | Simulates an effective batch of 8×4 = 32 for smoother gradients without the memory of a true batch-32. |
| `learning_rate` | 2e-4 | LoRA tolerates (and needs) a higher LR than full fine-tuning because only the small adapters update. 2e-4 is the well-established robust default for LoRA SFT; 1e-4 is safer/slower, 5e-4 can destabilize. |
| `lr_scheduler_type` | cosine | Smooth decay to ~0 improves final-step convergence vs constant/linear. |
| `warmup_ratio` | 0.03 | Short warmup avoids large early updates while adapters are randomly initialized. |
| `weight_decay` | 0.0 | LoRA adapters rarely benefit from weight decay; dropout already regularizes. |
| `fp16` | true | T4 has no bf16 hardware. fp16 halves memory vs fp32. (On Ampere+ switch to bf16 for better numerical stability.) |
| `gradient_checkpointing` | true | Recomputes activations in the backward pass instead of storing them — cuts activation memory ~40–60%, letting us use batch 8. Costs ~20% extra compute. |
| `optim` | adamw_torch | Standard. For QLoRA use `paged_adamw_8bit` to avoid optimizer-state OOM spikes. |
| `early_stopping_patience` | 2 | Stop if eval loss doesn't improve for 2 evaluations — prevents overfitting and wasted compute. |
| `seed` | 42 | Determinism across data split, init, and training. |

## QLoRA path (`configs/model.yaml`, `use_4bit: true`)

| Param | Value | Reasoning |
|---|---|---|
| `bnb_4bit_quant_type` | nf4 | NormalFloat4 — information-optimal 4-bit type for normally-distributed weights; better than plain int4. |
| `bnb_4bit_use_double_quant` | true | Quantizes the quantization constants too, saving ~0.4 GB more. |
| `bnb_4bit_compute_dtype` | float16 | Dequantized compute happens in fp16 on T4. |

QLoRA loads the frozen base in 4-bit (~4× memory saving) while LoRA adapters stay in fp16.
Trade-off: lower VRAM (~5–7 GB) at the cost of some dequantization compute overhead.
