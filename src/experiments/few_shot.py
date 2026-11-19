"""Experiment E: few-shot supervised fine-tuning (16 / 32 examples per class).

Reproducible, class-balanced sampling via src.data.sampling.few_shot_subset.
Not zero-shot: these are real labelled examples used for supervised training,
just very few of them. Reported separately from Experiment C low-resource runs.
"""

import argparse

from src.data.loader import load_processed
from src.data.sampling import few_shot_subset
from src.training.trainer import _already_done, run_one, save_run
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config()
    models_config = load_config("models.yaml")
    exp_config = load_config("experiments.yaml")["experiment_E_few_shot"]

    languages = args.languages or exp_config["languages"]
    model_keys = args.models or exp_config["models"]

    for language in languages:
        splits = load_processed(language, config)
        for model_key in model_keys:
            for shots in exp_config["shots_per_class"]:
                for seed in exp_config["seeds"]:
                    run_tag = f"fewshot{shots}"
                    if args.resume and _already_done(language, model_key, f"fewshot_{shots}", seed, config, run_tag):
                        logger.info(f"SKIP (already done): {language} | {model_key} | shots={shots} | seed={seed}")
                        continue
                    train_df = few_shot_subset(splits["train"], "canonical_label", shots, seed)
                    logger.info(f"=== FEW-SHOT: {language} | {model_key} | shots={shots} | seed={seed} ===")
                    res = run_one(
                        language, model_key, data_pct=f"fewshot_{shots}", seed=seed,
                        cfg=config, models_cfg=models_config,
                        train_df_override=train_df, run_tag=f"fewshot{shots}",
                    )
                    save_run(res, config, csv_name="few_shot.csv")
                    logger.info(f"DONE: macro_f1={res['metrics']['macro_f1']:.4f}")
