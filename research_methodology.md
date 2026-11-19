# Research Methodology — LRI-Text

## Title
A Data-Efficient Framework for Hindi and Bengali Sentiment Classification: Evaluating Multilingual Transformers under Low-Resource, Augmented, and Cross-Lingual Settings

## Research Problem
How effectively can multilingual transformer models perform Hindi and Bengali sentiment classification under limited labelled-data conditions, and to what extent can data augmentation and cross-lingual transfer improve their performance?

## Hypotheses (to test, not assume)
- H1: Macro-F1 degrades as the labelled-data budget shrinks from 100% to 10%, but the degree of degradation differs by model.
- H2: Back-translation augmentation improves Macro-F1 at low data budgets (10%, 50%) relative to the unaugmented baseline — not assumed to hold; may show no gain or regression.
- H3: Cross-lingual transfer (Hindi→Bengali, Bengali→Hindi) provides some transfer benefit over a from-scratch low-resource model, but is bounded by the domain/register mismatch between the two primary datasets.
- H4: Few-shot fine-tuning (16/32 examples/class) underperforms the 10% data-budget setting, quantifying the real cost of extreme label scarcity.

## Datasets (verified)
| Role | Dataset | HF id | Size | Labels | License |
|---|---|---|---|---|---|
| Primary Hindi | IITP Movie/Product Reviews | `ai4bharat/indic_glue` (`iitp-mr.hi`) | 3,100 | neg/neu/pos | cite IndicNLPSuite (Kakwani et al., 2020) |
| Primary Bengali | SentNoB | `khondoker/SentNoB` | 15,728 | neg/neu/pos | CC-BY-ND-4.0 |
| Cross-lingual eval (parallel) | IndicSentiment | `ai4bharat/IndicSentiment` (`hi`,`bn`) | ~1,156/lang | pos/neg/neu | unconfirmed, eval-only (no train split) |

**Documented limitation:** IITP (Hindi) is formal movie/product review text; SentNoB (Bengali) is noisy, informal social-media text. Both share the same 3-class scheme, but domain/register differs — a genuine confound for cross-lingual transfer, reported as a limitation rather than treated as a controlled comparison.

**Critical verified finding — label encodings are NOT aligned across the two raw datasets:**
- Hindi IITP (`ai4bharat/indic_glue`, `iitp-mr.hi`): confirmed via the dataset's own `ClassLabel` schema — `0=negative, 1=neutral, 2=positive`.
- Bengali SentNoB (`khondoker/SentNoB`): confirmed via the HF dataset card text ("positive (1), negative (2), or neutral (0)") — `0=neutral, 1=positive, 2=negative`.

These are opposite orderings. Training or evaluating across languages on the raw integer labels without remapping would silently corrupt every cross-lingual and combined experiment. Both datasets are therefore remapped to a shared canonical scheme (`negative=0, neutral=1, positive=2`, see `configs/config.yaml: canonical_name_to_id`) during preprocessing, before any split, augmentation, or training.

**Correction to an earlier (secondhand, incorrect) claim about IndicSentiment:** an initial research pass reported IndicSentiment as a 3-class (positive/negative/neutral) parallel benchmark. Direct inspection of the downloaded data (`data/raw/indicsentiment/`) shows this was wrong — the dataset is **strictly binary** (`LABEL` ∈ {"Negative", "Positive"} only; no "Neutral" ever appears across dev or test, for either language), and the test split additionally has some missing (`None`) labels that must be dropped before evaluation. This is corrected in `configs/config.yaml` (`cross_lingual_eval`). Consequence: when evaluating a 3-class-trained model on this benchmark, any "neutral" prediction is automatically wrong by construction — this is reported as an explicit scope limitation of the cross-lingual eval set, not treated as a fair 3-way comparison.

Datasets rejected after audit: `SentiHin-2500` (LLM-synthetic, not real text), `Mahadih534/Bangla-Large-Sentiment_Analysis` (heavily templated/duplicated), MTEB-repackaged subsamples (undocumented/inconsistent label definitions).

## Literature Review Summary
See table below for the 8 verified papers underpinning this work. Full citations maintained here; do not fabricate or extend beyond what was independently verified against ACL Anthology / arXiv / publisher pages.

| # | Paper | Authors | Venue/Year | Role in this project |
|---|---|---|---|---|
| 1 | IndicNLPSuite | Kakwani et al. | Findings EMNLP 2020 | IndicGLUE benchmark/corpus foundation |
| 2 | IndicBERT v2 / IndicXTREME | Doddapaneni et al. | ACL 2023 | Primary backbone candidate |
| 3 | How Multilingual is Multilingual BERT? | Pires et al. | ACL 2019 | Motivates mBERT cross-lingual baseline |
| 4 | Unsupervised Cross-lingual Representation Learning at Scale (XLM-R) | Conneau et al. | ACL 2020 | Strong cross-lingual backbone |
| 5 | MuRIL | Khanuja et al. | arXiv 2021 (preprint, not peer-reviewed) | India-specific backbone, stretch goal only |
| 6 | Few-shot classification w/ triplet nets + curriculum augmentation | Wei et al. | NAACL 2021 | Few-shot + augmentation methodology basis |
| 7 | Understanding Back-Translation at Scale | Edunov et al. | EMNLP 2018 | Back-translation augmentation technique |
| 8 | Cross-lingual sentiment via transliteration + char embeddings | Roychoudhury et al. | ICON 2022 | Most directly relevant prior Hindi–Bengali transfer work |

**Research gap:** No verified peer-reviewed work benchmarks modern transformer backbones (IndicBERT v2 / mBERT / XLM-R, optionally MuRIL) head-to-head on Hindi–Bengali sentiment specifically under controlled low-label budgets, jointly combined with augmentation and transformer-based (not transliteration-based) cross-lingual transfer. This project's contribution is that controlled experimental framework and its results — not a new architecture.

## Experimental Scope (reduced from the original full matrix — feasibility decision for a minor project on a 6GB-VRAM laptop GPU)
- Models: mBERT, XLM-R-base, IndicBERT v2 (core); MuRIL as stretch goal only.
- Data budgets: 10%, 50%, 100% (25% dropped).
- Random seeds: 2 (42, 123) instead of 3 — variance analysis reported with this caveat.
- Few-shot: 16 and 32 examples/class (8 and 64 dropped).
- Augmentation: back-translation only (contextual MLM substitution is a stretch goal).
- Cross-lingual transfer: Hindi→Bengali and Bengali→Hindi (no combined-train variant in core scope).

Full experiment definitions and tracked fields are in [configs/experiments.yaml](configs/experiments.yaml).

## Research Integrity Commitments
- No fabricated datasets, metrics, citations, or outcomes.
- No method is claimed novel merely because it was implemented.
- Test data is never used for model selection (validation Macro-F1 only).
- Augmented data is never present in validation/test splits.
- Cross-lingual results are reported honestly as supervised transfer (both languages have labelled training data used at some point), never mislabeled as zero-shot unless a specific experiment genuinely withholds target-language labels.
- Negative or null results (e.g., augmentation not helping) will be reported, not hidden.
