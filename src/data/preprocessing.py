"""Clean raw datasets and remap labels to the canonical scheme.

Rules enforced (see research_methodology.md):
- Unicode NFC normalization + whitespace normalization only; no stemming/stopword removal.
- Exact-duplicate texts are dropped before any split is created, to prevent leakage.
- Raw integer labels are remapped to the shared canonical scheme BEFORE any split/augmentation.
- Bengali (SentNoB) ships with only a 'train' split; a stratified val/test split is carved
  out here, once, with a fixed seed, and never regenerated differently later.
- Never mutates data/raw/. Writes cleaned parquet + a JSON report to data/processed/.
"""

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split())


def remap_to_canonical(df: pd.DataFrame, label_col: str, raw_label_to_name: dict, name_to_id: dict) -> pd.DataFrame:
    df = df.copy()
    df["canonical_label"] = df[label_col].map(raw_label_to_name).map(name_to_id)
    unmapped = df["canonical_label"].isna().sum()
    if unmapped:
        logger.warning(f"{unmapped} rows have a raw label not present in raw_label_to_name; dropping them")
        df = df[df["canonical_label"].notna()]
    df["canonical_label"] = df["canonical_label"].astype(int)
    return df


def clean_and_dedup(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df[df[text_col].notna()]
    df[text_col] = df[text_col].map(clean_text)
    df = df[df[text_col].str.len() > 0]
    df = df.drop_duplicates(subset=[text_col], keep="first")
    removed = before - len(df)
    if removed:
        logger.info(f"Removed {removed} empty/duplicate rows ({before} -> {len(df)})")
    return df.reset_index(drop=True)


def process_hindi(cfg: dict, seed: int) -> dict:
    lang_cfg = cfg["languages"]["hindi"]
    raw_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_raw"] / "hindi"
    out_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_processed"] / "hindi"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"language": "hindi", "splits": {}}
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(raw_dir / f"{split}.parquet")
        df = clean_and_dedup(df, lang_cfg["text_column"])
        df = remap_to_canonical(df, lang_cfg["label_column"], lang_cfg["raw_label_to_name"], cfg["canonical_name_to_id"])
        out_name = "val" if split == "validation" else split
        df.to_parquet(out_dir / f"{out_name}.parquet")
        report["splits"][out_name] = {"num_examples": len(df)}
    return report


def process_bengali(cfg: dict, seed: int) -> dict:
    lang_cfg = cfg["languages"]["bengali"]
    raw_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_raw"] / "bengali"
    out_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_processed"] / "bengali"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(raw_dir / "train.parquet")
    df = clean_and_dedup(df, lang_cfg["text_column"])
    df = remap_to_canonical(df, lang_cfg["label_column"], lang_cfg["raw_label_to_name"], cfg["canonical_name_to_id"])

    # Carve out a leakage-safe stratified 80/10/10 split (only 'train' is provided upstream).
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df["canonical_label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["canonical_label"]
    )

    assert set(train_df.index) & set(val_df.index) == set()
    assert set(train_df.index) & set(test_df.index) == set()
    assert set(val_df.index) & set(test_df.index) == set()

    report = {"language": "bengali", "splits": {}}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet")
        report["splits"][name] = {"num_examples": len(split_df)}
    report["note"] = "val/test carved from original 'train' via stratified 80/10/10 split, seed=" + str(seed)
    return report


def process_indicsentiment(cfg: dict) -> dict:
    cl_cfg = cfg["cross_lingual_eval"]
    report = {"dataset": "indicsentiment", "configs": {}}
    for lang_config in cl_cfg["configs"]:
        raw_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_raw"] / "indicsentiment" / lang_config
        out_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_processed"] / "indicsentiment" / lang_config
        out_dir.mkdir(parents=True, exist_ok=True)
        report["configs"][lang_config] = {}
        for split in ["validation", "test"]:
            df = pd.read_parquet(raw_dir / f"{split}.parquet")
            df = df[df[cl_cfg["label_column"]].notna()]
            df = clean_and_dedup(df, cl_cfg["text_column"])
            canonical_name = df[cl_cfg["label_column"]].map(cl_cfg["raw_label_to_canonical_name"])
            df["canonical_label"] = canonical_name.map(cfg["canonical_name_to_id"])
            df = df[df["canonical_label"].notna()]
            df["canonical_label"] = df["canonical_label"].astype(int)
            out_name = "val" if split == "validation" else split
            df.to_parquet(out_dir / f"{out_name}.parquet")
            report["configs"][lang_config][out_name] = {"num_examples": len(df)}
    return report


def save_report(report: dict, language: str, cfg: dict) -> None:
    out_dir = Path(PROJECT_ROOT) / cfg["paths"]["results_logs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"preprocessing_{language}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Preprocessing report written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    set_seed(args.seed)

    hi_report = process_hindi(config, args.seed)
    save_report(hi_report, "hindi", config)
    print(json.dumps(hi_report, indent=2))

    bn_report = process_bengali(config, args.seed)
    save_report(bn_report, "bengali", config)
    print(json.dumps(bn_report, indent=2))

    cl_report = process_indicsentiment(config)
    save_report(cl_report, "indicsentiment", config)
    print(json.dumps(cl_report, indent=2))
