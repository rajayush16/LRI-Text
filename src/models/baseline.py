"""Experiment A: TF-IDF + Logistic Regression / Linear SVM baselines.

Runs per language x model x seed, at the requested data budgets. Writes one row
per run to results/metrics/baselines.csv and the full metrics (incl. confusion
matrix) to results/metrics/raw/baseline_<language>_<model>_<pct>_<seed>.json.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.data.loader import load_processed
from src.data.sampling import stratified_subset
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging import get_logger
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed

logger = get_logger(__name__)

MODEL_BUILDERS = {
    "tfidf_logreg": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "tfidf_svm": lambda seed: LinearSVC(random_state=seed),
}


def run_one(language: str, model_key: str, data_pct: float, seed: int, cfg: dict, models_cfg: dict) -> dict:
    set_seed(seed)
    lang_cfg = cfg["languages"][language]
    text_col = lang_cfg["text_column"]
    label_col = "canonical_label"

    splits = load_processed(language, cfg)
    train_df = stratified_subset(splits["train"], label_col, data_pct, seed)

    tfidf_cfg = models_cfg["baselines"]["tfidf_logreg" if "logreg" in model_key else "tfidf_svm"]
    vectorizer = TfidfVectorizer(max_features=tfidf_cfg["max_features"], ngram_range=tuple(tfidf_cfg["ngram_range"]))

    X_train = vectorizer.fit_transform(train_df[text_col])
    X_test = vectorizer.transform(splits["test"][text_col])

    clf = MODEL_BUILDERS[model_key](seed)

    t0 = time.time()
    clf.fit(X_train, train_df[label_col])
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(X_test)
    inference_time = time.time() - t0

    metrics = compute_metrics(splits["test"][label_col], y_pred, cfg["canonical_label_names"])

    return {
        "experiment": "A_baselines",
        "language": language,
        "model": model_key,
        "data_pct": data_pct,
        "seed": seed,
        "train_examples": len(train_df),
        "train_time_sec": round(train_time, 3),
        "inference_time_sec": round(inference_time, 3),
        "metrics": metrics,
    }


def save_run(result: dict, cfg: dict) -> None:
    raw_dir = Path(PROJECT_ROOT) / cfg["paths"]["results_metrics"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fname = f"baseline_{result['language']}_{result['model']}_{result['data_pct']}_{result['seed']}.json"
    with open(raw_dir / fname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    summary_path = Path(PROJECT_ROOT) / cfg["paths"]["results_metrics"] / "baselines.csv"
    row = {
        "language": result["language"],
        "model": result["model"],
        "data_pct": result["data_pct"],
        "seed": result["seed"],
        "train_examples": result["train_examples"],
        "train_time_sec": result["train_time_sec"],
        "inference_time_sec": result["inference_time_sec"],
        "accuracy": result["metrics"]["accuracy"],
        "macro_f1": result["metrics"]["macro_f1"],
        "weighted_f1": result["metrics"]["weighted_f1"],
    }
    df_row = pd.DataFrame([row])
    if summary_path.exists():
        df_row.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df_row.to_csv(summary_path, mode="w", header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=["hindi", "bengali"])
    args = parser.parse_args()

    config = load_config()
    models_config = load_config("models.yaml")
    exp_config = load_config("experiments.yaml")
    exp_a = exp_config["experiment_A_baselines"]

    for language in args.languages:
        for model_key in exp_a["models"]:
            for data_pct in exp_a["data_budgets"]:
                for seed in exp_a["seeds"]:
                    res = run_one(language, model_key, data_pct, seed, config, models_config)
                    save_run(res, config)
                    logger.info(
                        f"{language} | {model_key} | pct={data_pct} seed={seed} "
                        f"-> macro_f1={res['metrics']['macro_f1']:.4f} acc={res['metrics']['accuracy']:.4f}"
                    )
