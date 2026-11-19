# LRI-Text: Project Report

**Title:** A Data-Efficient Framework for Hindi and Bengali Sentiment Classification: Evaluating Multilingual Transformers under Low-Resource, Augmented, and Cross-Lingual Settings

## 1. Problem

How effectively can multilingual transformers (mBERT, XLM-R, IndicBERT v2) classify Hindi and Bengali sentiment when labelled training data is scarce? Full methodology, literature review, and dataset audit are in [research_methodology.md](research_methodology.md).

## 2. Datasets (verified, not assumed)

| Language | Dataset | Train / Val / Test | Labels |
|---|---|---|---|
| Hindi | IITP Movie Reviews (`ai4bharat/indic_glue`, `iitp-mr.hi`) | 2,476 / 310 / 310 | negative/neutral/positive |
| Bengali | SentNoB (`khondoker/SentNoB`) | 10,880 / 1,360 / 1,360 | negative/neutral/positive (relabelled; see below) |

**Verified finding:** the two datasets' raw integer label encodings are opposite (Hindi: `0=neg,1=neu,2=pos`; Bengali: `0=neu,1=pos,2=neg`, confirmed from the dataset's own schema/card). Both are remapped to a shared canonical scheme before any training — see `research_methodology.md` for detail. Missing this would have silently invalidated every result below.

## 3. What was actually run

Given a hard time constraint, the full planned matrix (3 models × 2 languages × 3 budgets × 2 seeds, plus augmentation/few-shot/cross-lingual experiments) was **not completed**. What follows is exactly what finished, with gaps stated plainly rather than hidden.

**Experiment A — TF-IDF baselines (complete, both languages, both seeds):**

| Language | Model | Macro-F1 |
|---|---|---|
| Hindi | LogReg | 0.531 |
| Hindi | SVM | 0.570 |
| Bengali | LogReg | 0.533 |
| Bengali | SVM | 0.530 |

**Experiments B/C — Transformer fine-tuning at 10%/50% data budgets:**

| Language | Model | Budget | Macro-F1 (mean) | Std (2 seeds) | Train time (s) |
|---|---|---|---|---|---|
| Hindi | mBERT | 10% | 0.402 | 0.091 | 31 |
| Hindi | mBERT | 50% | 0.548 | 0.011 | 81 |
| Hindi | XLM-R | 10% | 0.305 | 0.165 | 46 |
| Hindi | XLM-R | 50% | 0.666 | 0.010 | 141 |
| Hindi | IndicBERT v2 | 10% | 0.208 | 0.003 | 190 |
| Hindi | IndicBERT v2 | 50% | 0.686 | 0.005 | 646 |
| Bengali | mBERT | 10% | 0.566 | 0.00003 | 86 |
| Bengali | mBERT | 50% | 0.602 | 0.004 | 320 |
| Bengali | XLM-R | 10% | 0.566 | 0.029 | 317 |
| Bengali | XLM-R | 50% | 0.629 | — (1 seed only) | 926 |

**Not run (explicitly, not silently):**
- Bengali / XLM-R / 50% / seed 123 — killed mid-run for time; only seed 42 completed.
- Bengali / IndicBERT v2 — not started at all (0.1 and 0.5 budgets, both seeds).
- Experiment B at the 100% data budget — not started for either language.
- Experiment D (back-translation augmentation), E (few-shot), F (cross-lingual transfer) — code is written and tested (see `src/augmentation/`, `src/experiments/few_shot.py`, `src/experiments/cross_lingual.py`) but none have been executed.

## 4. Findings from what did run

1. **Macro-F1 improves with more data** (H1 supported): every model improves from 10%→50% budget, most dramatically for IndicBERT v2 (0.208→0.686) and XLM-R on Hindi (0.305→0.666).
2. **IndicBERT v2 is the strongest model once it has enough data, but the least data-efficient**: worst at 10% (likely because it ships with no pretrained classification head and needs more examples to learn one from scratch — see `configs/models.yaml`), best at 50%, and by far the slowest to train (646s vs. mBERT's 81s on Hindi at 50%) — a real performance/compute trade-off (research question 5).
3. **Transformers only sometimes beat the TF-IDF baseline at low data budgets**: Hindi mBERT/XLM-R at 10% (0.40, 0.31) are *worse* than TF-IDF+SVM (0.57) — a genuine negative result, not smoothed over. Transformers clearly win once data budget reaches 50%.
4. **Bengali mBERT seed variance is implausibly near-zero** (std 0.00003 at 10%) — worth rechecking if this experiment is resumed; it may indicate the two seeds converged to the same local optimum rather than a measurement error.
5. No cross-lingual or augmentation conclusions can be drawn yet — those experiments did not run.

Figures: `results/figures/budget_vs_macrof1_{hindi,bengali}.png`, `model_comparison_{0.1,0.5}.png`, `training_time.png`, `class_distribution_raw.png`, `confusion_matrix_best_run.png`.

## 5. Limitations (beyond the incomplete runs above)

- Only 2 random seeds (not 3) for all transformer runs — variance estimates are noisy.
- Hindi (formal reviews) and Bengali (noisy social media) differ in domain/register, confounding any future cross-lingual comparison.
- Bengali's val/test split was carved from its single original `train` split (no official split exists) — see `research_methodology.md`.
- No hyperparameter search was performed; the config in `configs/config.yaml` uses literature-standard defaults (lr=2e-5, batch=8, 5 epochs, early stopping).

## 6. How to complete the remaining work

Everything below is implemented, tested, and resumable (`--resume` skips already-finished runs):

```
python -m src.training.trainer --experiment experiment_C_low_resource --resume   # finish Bengali
python -m src.training.trainer --experiment experiment_B_full_finetune --resume  # 100% budget, both languages
python -m src.experiments.few_shot --resume
python -m src.experiments.cross_lingual --resume
python -m src.experiments.augmentation_experiment --resume
python -m src.utils.generate_figures   # regenerate all figures once more data exists
```

Or run everything in one command: `python -m src.experiments.run_experiments`.

## 7. Demo

`app/streamlit_app.py` — run with `streamlit run app/streamlit_app.py`. Loads any checkpoint under `checkpoints/`, lets you type Hindi/Bengali text and see predicted sentiment + per-class probabilities. Clearly labelled as a research demo, not a production tool. **Not yet smoke-tested end-to-end in a browser** — do this before presenting it live.

## 8. Reproducibility

- All configs: `configs/config.yaml`, `configs/models.yaml`, `configs/experiments.yaml`.
- All raw per-run results (full metrics, confusion matrices): `results/metrics/raw/*.json`.
- Aggregated CSVs: `results/metrics/{baselines,transformers}.csv`.
- Test suite: `pytest tests/` (25 tests, all passing — covers preprocessing, label remapping, sampling, metrics, augmentation filtering).
- Hardware used: NVIDIA RTX 3060 Laptop GPU, 6GB VRAM, fp16.
