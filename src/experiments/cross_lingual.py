"""Experiment F: cross-lingual transfer. Train on one language's full labelled
training set; evaluate zero-shot (target labels never used for training or
model selection) on the target language's own test split AND on the parallel
IndicSentiment benchmark for that target language.

IndicSentiment is binary-only (Negative/Positive, no Neutral) for both languages
(verified directly from the data) — reported as an explicit limitation, not
glossed over as a fair 3-way comparison.
"""

import argparse

from src.data.loader import load_indicsentiment, load_processed
from src.training.trainer import run_one, save_run
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

DIRECTIONS = {
    "hi_to_bn": ("hindi", "bengali"),
    "bn_to_hi": ("bengali", "hindi"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--directions", nargs="+", default=None)
    args = parser.parse_args()

    config = load_config()
    models_config = load_config("models.yaml")
    exp_config = load_config("experiments.yaml")["experiment_F_cross_lingual"]

    model_keys = args.models or exp_config["models"]
    directions = args.directions or exp_config["directions"]

    for direction in directions:
        source_lang, target_lang = DIRECTIONS[direction]
        target_text_col = config["languages"][target_lang]["text_column"]
        target_splits = load_processed(target_lang, config)

        target_code = config["languages"][target_lang]["code"]
        indic_lang_config = f"translation-{target_code}"
        indic_splits = load_indicsentiment(indic_lang_config, config)
        indic_text_col = config["cross_lingual_eval"]["text_column"]

        for model_key in model_keys:
            for seed in exp_config["seeds"]:
                logger.info(f"=== CROSS-LINGUAL: {direction} | {model_key} | seed={seed} ===")
                extra_eval = {
                    f"{target_lang}_test": (target_splits["test"], target_text_col),
                    f"indicsentiment_{target_code}": (indic_splits["test"], indic_text_col),
                }
                res = run_one(
                    source_lang, model_key, data_pct=1.0, seed=seed,
                    cfg=config, models_cfg=models_config,
                    run_tag=f"xling_{direction}", extra_eval=extra_eval,
                )
                save_run(res, config, csv_name="cross_lingual.csv")
                logger.info(
                    f"DONE: source_test_macro_f1={res['metrics']['macro_f1']:.4f} | "
                    f"{target_lang}_test_macro_f1={res['cross_eval'][f'{target_lang}_test']['macro_f1']:.4f} | "
                    f"indicsentiment_{target_code}_macro_f1={res['cross_eval'][f'indicsentiment_{target_code}']['macro_f1']:.4f}"
                )
