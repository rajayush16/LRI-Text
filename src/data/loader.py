from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT


def load_processed(language: str, cfg: dict) -> dict:
    proc_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_processed"] / language
    return {split: pd.read_parquet(proc_dir / f"{split}.parquet") for split in ["train", "val", "test"]}


def load_indicsentiment(lang_config: str, cfg: dict) -> dict:
    proc_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_processed"] / "indicsentiment" / lang_config
    return {split: pd.read_parquet(proc_dir / f"{split}.parquet") for split in ["val", "test"]}
