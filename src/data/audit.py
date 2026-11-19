"""Audit raw datasets: availability, label distribution, missing/duplicate
values, text-length stats, and script consistency. Writes one JSON report
per language to results/logs/. Never mutates data/raw/.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

SCRIPT_RANGES = {
    "hindi": (0x0900, 0x097F),   # Devanagari
    "bengali": (0x0980, 0x09FF),  # Bengali
}


def script_fraction(texts: pd.Series, language: str) -> float:
    lo, hi = SCRIPT_RANGES[language]
    total_alpha = 0
    in_script = 0
    for text in texts.dropna().astype(str):
        for ch in text:
            if ch.isalpha():
                total_alpha += 1
                if lo <= ord(ch) <= hi:
                    in_script += 1
    return in_script / total_alpha if total_alpha else 0.0


def audit_split(df: pd.DataFrame, text_col: str, label_col: str, language: str) -> dict:
    if text_col not in df.columns or label_col not in df.columns:
        return {
            "error": f"expected columns '{text_col}'/'{label_col}' not found; actual columns: {list(df.columns)}"
        }

    text_na_mask = df[text_col].isna()
    texts = df[text_col].astype(str)
    missing_text = int((text_na_mask | (texts.str.strip() == "")).sum())
    missing_label = int(df[label_col].isna().sum())
    duplicate_text_count = int(texts.duplicated().sum())
    label_distribution = df[label_col].value_counts().to_dict()
    word_lengths = texts.str.split().str.len()

    return {
        "num_examples": len(df),
        "missing_text": missing_text,
        "missing_label": missing_label,
        "duplicate_text_count": duplicate_text_count,
        "label_distribution": {str(k): int(v) for k, v in label_distribution.items()},
        "text_length_words": {
            "mean": float(word_lengths.mean()),
            "min": int(word_lengths.min()),
            "max": int(word_lengths.max()),
            "p95": float(word_lengths.quantile(0.95)),
        },
        "script_fraction_expected": round(script_fraction(texts, language), 4),
    }


def audit_language(cfg: dict, language: str) -> dict:
    lang_cfg = cfg["languages"][language]
    raw_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_raw"] / language
    if not raw_dir.exists():
        logger.warning(f"No raw data found for {language} at {raw_dir} — run src/data/download.py first")
        return {"error": "raw data not found"}

    report = {"language": language, "source": lang_cfg["dataset_name"], "splits": {}}
    for parquet_file in sorted(raw_dir.glob("*.parquet")):
        split_name = parquet_file.stem
        df = pd.read_parquet(parquet_file)
        report["splits"][split_name] = audit_split(
            df, lang_cfg["text_column"], lang_cfg["label_column"], language
        )
    return report


def save_report(report: dict, language: str, cfg: dict) -> Path:
    out_dir = Path(PROJECT_ROOT) / cfg["paths"]["results_logs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"audit_{language}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Audit report written to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=["hindi", "bengali"])
    args = parser.parse_args()

    config = load_config()
    for lang in args.languages:
        rep = audit_language(config, lang)
        save_report(rep, lang, config)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
