# LRI-Text

Low-Resource Indian Language Text Classification Using Multilingual Transformers and Data-Efficient Learning.

Research-oriented college minor project investigating how multilingual transformers (mBERT, XLM-R, IndicBERT v2) perform Hindi and Bengali sentiment classification under limited labelled-data conditions, and whether data augmentation and cross-lingual transfer help.

See [research_methodology.md](research_methodology.md) for the full research problem, hypotheses, verified datasets/literature, and documented scope/limitations.

## Project structure
See `configs/`, `src/`, `data/`, `results/` — layout follows the architecture in `research_methodology.md`.

## Setup
```
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Status
Phase 2 (environment + repo scaffolding) in progress. See `configs/experiments.yaml` for the experiment registry.
