# LRI-Text

**Low-Resource Indian Language Text Classification Using Multilingual Transformers and Data-Efficient Learning**

A research-oriented college minor project investigating how multilingual transformers (mBERT, XLM-R, IndicBERT v2) perform Hindi and Bengali sentiment classification under limited labelled-data conditions, and whether data augmentation, few-shot learning, and cross-lingual transfer help.

- **Full report, results, and honest scope/limitations:** [PROJECT_REPORT.md](PROJECT_REPORT.md)
- **Research problem, hypotheses, verified literature & dataset audit:** [research_methodology.md](research_methodology.md)

## Key result so far

Transformer fine-tuning only beats a plain TF-IDF+SVM baseline once the data budget reaches ~50% of the training set — at 10% data, the baseline (0.57 macro-F1) actually beats mBERT (0.40) and XLM-R (0.31) on Hindi. IndicBERT v2 ends up the strongest model at 50% data (0.69 macro-F1) but is ~8x slower to train than mBERT. Full tables and figures in [PROJECT_REPORT.md](PROJECT_REPORT.md).

## Project structure

```
configs/        Experiment registry & hyperparameters (config.yaml, models.yaml, experiments.yaml)
src/
  data/         Download, audit, preprocessing (incl. the Hindi/Bengali label-mapping fix), sampling
  models/       TF-IDF baselines, transformer checkpoint loader
  training/     Shared fine-tuning loop + evaluation metrics
  augmentation/ Back-translation + validation filters
  experiments/  Few-shot, cross-lingual transfer, augmentation runner, orchestration
  utils/        Config, logging, seeding, metrics, visualization
app/            Streamlit sentiment-classification demo
tests/          25 unit tests (preprocessing, sampling, metrics, augmentation filters, config)
results/        metrics/ (CSVs + raw per-run JSON), figures/, logs/
data/           raw/ (downloaded), processed/ (cleaned + label-remapped)
```

## Setup

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121   # CUDA build; use the CPU wheel if no GPU
pip install -r requirements.txt
```

## Reproducing the experiments

```bash
python -m src.data.download                  # Hindi (IITP), Bengali (SentNoB), IndicSentiment
python -m src.data.audit --languages hindi bengali
python -m src.data.preprocessing --seed 42

python -m src.models.baseline                                             # Experiment A: TF-IDF baselines
python -m src.training.trainer --experiment experiment_C_low_resource --resume  # 10%/50% data budgets
python -m src.training.trainer --experiment experiment_B_full_finetune --resume  # 100% data budget
python -m src.experiments.few_shot --resume                               # Experiment E
python -m src.experiments.cross_lingual --resume                          # Experiment F
python -m src.experiments.augmentation_experiment --resume                # Experiment D

python -m src.utils.generate_figures         # regenerate all figures from results/metrics/
```

All training scripts accept `--resume` to skip (language, model, budget, seed) combos that already have a saved result — safe to re-run after an interruption.

## Demo

```bash
streamlit run app/streamlit_app.py
```

Pick a language and a trained checkpoint, type text, see the predicted sentiment and per-class probabilities. Research demo only — not a production sentiment monitor (see in-app caption).

## Testing

```bash
pytest tests/ -q
```

## Status

Experiment A (baselines) complete. Experiments B/C (transformer fine-tuning) partially complete — see [PROJECT_REPORT.md](PROJECT_REPORT.md) for exactly which (language, model, budget, seed) combinations ran and which didn't. Experiments D/E/F (augmentation, few-shot, cross-lingual) are implemented and tested but not yet executed.

## Hardware used

NVIDIA RTX 3060 Laptop GPU (6GB VRAM), fp16 training.

## License

MIT (code). Datasets retain their own licenses — see [research_methodology.md](research_methodology.md) for details and citations.
