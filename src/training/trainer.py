"""Experiments B & C: transformer fine-tuning at full and reduced data budgets.

Usage:
    python -m src.training.trainer --experiment experiment_B_full_finetune
    python -m src.training.trainer --experiment experiment_C_low_resource
    python -m src.training.trainer --experiment experiment_B_full_finetune --models mbert --languages hindi

Writes one row per run to results/metrics/transformers.csv, full metrics JSON
to results/metrics/raw/, and the best checkpoint to checkpoints/.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.loader import load_processed
from src.data.sampling import stratified_subset
from src.models.model_loader import load_model_and_tokenizer
from src.training.evaluation import make_compute_metrics
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def build_dataset(df, tokenizer, text_col: str, max_len: int) -> Dataset:
    ds = Dataset.from_pandas(df[[text_col, "canonical_label"]].rename(columns={text_col: "text", "canonical_label": "labels"}))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    return ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def run_one(language: str, model_key: str, data_pct: float, seed: int, cfg: dict, models_cfg: dict, train_df_override=None, run_tag: str = "", extra_eval: dict | None = None) -> dict:
    """extra_eval: optional {name: (eval_df, text_col)} evaluated zero-shot after training
    (labels never used for training or model selection, only for reporting metrics)."""
    set_seed(seed)
    lang_cfg = cfg["languages"][language]
    text_col = lang_cfg["text_column"]
    train_cfg = cfg["training"]

    splits = load_processed(language, cfg)
    train_df = train_df_override if train_df_override is not None else stratified_subset(splits["train"], "canonical_label", data_pct, seed)

    model, tokenizer = load_model_and_tokenizer(model_key, lang_cfg["num_classes"], models_cfg)

    train_ds = build_dataset(train_df, tokenizer, text_col, train_cfg["max_seq_length"])
    val_ds = build_dataset(splits["val"], tokenizer, text_col, train_cfg["max_seq_length"])
    test_ds = build_dataset(splits["test"], tokenizer, text_col, train_cfg["max_seq_length"])

    collator = DataCollatorWithPadding(tokenizer)
    run_id = f"{language}_{model_key}_{data_pct}_{seed}{('_' + run_tag) if run_tag else ''}"
    output_dir = Path(PROJECT_ROOT) / cfg["paths"]["checkpoints"] / run_id

    use_fp16 = train_cfg["fp16"] and torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"] * 2,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        num_train_epochs=train_cfg["num_epochs"],
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=train_cfg["eval_metric_for_best_model"],
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        report_to=[],
        disable_tqdm=True,
        # Some checkpoints (e.g. IndicBERT v2) load with non-contiguous tensors that
        # safetensors refuses to serialize; .bin via torch.save has no such restriction.
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(cfg["canonical_label_names"]),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_cfg["early_stopping_patience"])],
        tokenizer=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    t0 = time.time()
    test_metrics = trainer.evaluate(test_ds)
    inference_time = time.time() - t0

    test_metrics = {k.replace("eval_", ""): v for k, v in test_metrics.items()}

    result = {
        "language": language,
        "model": model_key,
        "data_pct": data_pct,
        "seed": seed,
        "run_tag": run_tag,
        "train_examples": len(train_df),
        "train_time_sec": round(train_time, 2),
        "inference_time_sec": round(inference_time, 2),
        "checkpoint_path": str(output_dir),
        "metrics": test_metrics,
    }

    if extra_eval:
        result["cross_eval"] = {}
        for name, (eval_df, eval_text_col) in extra_eval.items():
            eval_ds = build_dataset(eval_df, tokenizer, eval_text_col, train_cfg["max_seq_length"])
            m = trainer.evaluate(eval_ds)
            result["cross_eval"][name] = {k.replace("eval_", ""): v for k, v in m.items()}

    return result


def save_run(result: dict, cfg: dict, csv_name: str = "transformers.csv") -> None:
    import pandas as pd

    raw_dir = Path(PROJECT_ROOT) / cfg["paths"]["results_metrics"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fname = f"transformer_{result['language']}_{result['model']}_{result['data_pct']}_{result['seed']}{('_' + result['run_tag']) if result['run_tag'] else ''}.json"
    with open(raw_dir / fname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    summary_path = Path(PROJECT_ROOT) / cfg["paths"]["results_metrics"] / csv_name
    row = {
        "language": result["language"],
        "model": result["model"],
        "data_pct": result["data_pct"],
        "seed": result["seed"],
        "run_tag": result["run_tag"],
        "train_examples": result["train_examples"],
        "train_time_sec": result["train_time_sec"],
        "inference_time_sec": result["inference_time_sec"],
        "accuracy": result["metrics"]["accuracy"],
        "macro_f1": result["metrics"]["macro_f1"],
        "weighted_f1": result["metrics"]["weighted_f1"],
        "checkpoint_path": result["checkpoint_path"],
    }
    for eval_name, eval_metrics in result.get("cross_eval", {}).items():
        row[f"{eval_name}_macro_f1"] = eval_metrics["macro_f1"]
        row[f"{eval_name}_accuracy"] = eval_metrics["accuracy"]
    df_row = pd.DataFrame([row])
    if summary_path.exists():
        df_row.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(summary_path, mode="w", header=True, index=False)


def _already_done(language: str, model_key: str, data_pct: float, seed: int, cfg: dict, run_tag: str = "") -> bool:
    raw_dir = Path(PROJECT_ROOT) / cfg["paths"]["results_metrics"] / "raw"
    fname = f"transformer_{language}_{model_key}_{data_pct}_{seed}{('_' + run_tag) if run_tag else ''}.json"
    return (raw_dir / fname).exists()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["experiment_B_full_finetune", "experiment_C_low_resource"])
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip (language, model, pct, seed) combos with an existing raw result JSON")
    args = parser.parse_args()

    config = load_config()
    models_config = load_config("models.yaml")
    exp_config = load_config("experiments.yaml")[args.experiment]

    languages = args.languages or exp_config["languages"]
    model_keys = args.models or exp_config["models"]

    for language in languages:
        for model_key in model_keys:
            for data_pct in exp_config["data_budgets"]:
                for seed in exp_config["seeds"]:
                    if args.resume and _already_done(language, model_key, data_pct, seed, config):
                        logger.info(f"SKIP (already done): {language} | {model_key} | pct={data_pct} | seed={seed}")
                        continue
                    logger.info(f"=== RUN: {language} | {model_key} | pct={data_pct} | seed={seed} ===")
                    res = run_one(language, model_key, data_pct, seed, config, models_config)
                    save_run(res, config)
                    logger.info(
                        f"DONE: {language} | {model_key} | pct={data_pct} seed={seed} "
                        f"-> macro_f1={res['metrics']['macro_f1']:.4f} train_time={res['train_time_sec']}s"
                    )
