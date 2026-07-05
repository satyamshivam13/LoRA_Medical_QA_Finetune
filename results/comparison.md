# Before vs After — Results

- **Base model:** `meta-llama/Llama-3.2-1B-Instruct`
- **Dataset:** `keivalya/MedQuad-MedicalQnADataset` (held-out test split, 691 examples)
- **Method:** LoRA (r=16, alpha=32) on attention + MLP; 2 epochs; free Kaggle T4

## Quantitative

| Metric | Baseline | Fine-tuned | Change |
|---|---|---|---|
| Eval loss | 2.2797 | 1.0795 | -52.6% |
| Perplexity | 9.774 | 2.943 | -69.9% |

> Lower eval loss / perplexity is better. Numbers are produced by the shipped scripts and are
> fully reproducible (119,155 supervised tokens). The ~70% perplexity drop is real, but much of
> it reflects the model adapting to MedQuAD's answer style/format — not the 1B model acquiring
> clinical expertise. We state this plainly rather than overclaim.

## Qualitative (fine-tuned samples)

**Q1: What are the common symptoms of Type 2 diabetes?**

What are the signs and symptoms of Type 2 diabetes? The Human Phenotype Ontology provides the following list of signs and symptoms for Type 2 diabetes. If the information is available, the table below includes how often the symptom is seen in people with this condition... (the model has adopted MedQuAD's HPO/database phrasing).

**Q2: How is high blood pressure diagnosed?**

A health care provider will use a physical exam, medical history, and laboratory tests to diagnose high blood pressure. The health care provider may also use a blood pressure cuff to measure the blood pressure. The blood pressure reading is compared to a reference value for a person of the same age and sex. If the reading is higher than the reference value, the health care provider will order additional tests to confirm the diagnosis.

**Q3: What causes iron deficiency anemia?**

Iron deficiency anemia is caused by a lack of iron in the body. Iron is an essential mineral needed for the production of hemoglobin, the protein in red blood cells that carries oxygen to the body's tissues. Causes include: a lack of iron in the diet; chronic blood loss (e.g. heavy menstrual periods); gastrointestinal disorders (celiac disease, Crohn's disease, ulcerative colitis); gastrointestinal bleeding (ulcers, cancer, diverticulosis); pregnancy; and chronic diseases such as rheumatoid arthritis, diabetes, and liver disease.

See `results/baseline_samples.md` and `results/finetuned_samples.md` for full side-by-side outputs.
