"""Experiment D: back-translation augmentation at restricted data budgets (10%, 50%),
compared against the unaugmented Experiment C baseline at the same budgets.

Augmentation is applied ONCE per (language, data_pct, seed) and reused across all
models (it doesn't depend on which classifier will be trained), then each model is
fine-tuned on original+augmented data. Never touches val/test.
"""

import argparse

import pandas as pd

from src.augmentation.back_translation import augment_dataframe, load_translation_model
from src.data.loader import load_processed
from src.data.sampling import stratified_subset
from src.training.trainer import run_one, save_run
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    config = load_config()
    models_config = load_config("models.yaml")
    exp_config = load_config("experiments.yaml")["experiment_D_augmentation"]

    languages = args.languages or exp_config["languages"]
    model_keys = args.models or exp_config["models"]
    max_ratio = config["augmentation"]["max_augmented_ratio"]

    translation_model, translation_tokenizer, device = load_translation_model()

    for language in languages:
        text_col = config["languages"][language]["text_column"]
        train_full = load_processed(language, config)["train"]

        for data_pct in exp_config["data_budgets"]:
            for seed in exp_config["seeds"]:
                subset = stratified_subset(train_full, "canonical_label", data_pct, seed)
                augmented = augment_dataframe(
                    subset, text_col, language, translation_model, translation_tokenizer, device, max_ratio
                )
                combined = pd.concat([subset[[text_col, "canonical_label"]], augmented]).reset_index(drop=True)
                logger.info(
                    f"{language} pct={data_pct} seed={seed}: original={len(subset)} + augmented={len(augmented)} = {len(combined)}"
                )

                for model_key in model_keys:
                    logger.info(f"=== AUGMENTED: {language} | {model_key} | pct={data_pct} | seed={seed} ===")
                    res = run_one(
                        language, model_key, data_pct=data_pct, seed=seed,
                        cfg=config, models_cfg=models_config,
                        train_df_override=combined, run_tag="bt_augmented",
                    )
                    save_run(res, config, csv_name="augmentation.csv")
                    logger.info(f"DONE: macro_f1={res['metrics']['macro_f1']:.4f}")
