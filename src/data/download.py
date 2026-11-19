"""Download raw Hindi/Bengali sentiment datasets and the cross-lingual eval set.

Raw datasets are never overwritten once downloaded (use --force to re-download).
Saves each split as parquet under data/raw/<language>/ plus a metadata.json
recording source, config, split sizes, and download timestamp.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _save_dataset_dict(dataset_dict, out_dir: Path, source_name: str, source_config: str | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_sizes = {}
    for split_name, split_data in dataset_dict.items():
        split_data.to_parquet(str(out_dir / f"{split_name}.parquet"))
        split_sizes[split_name] = len(split_data)

    metadata = {
        "source_name": source_name,
        "source_config": source_config,
        "split_sizes": split_sizes,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {source_name} ({source_config}) -> {out_dir} | splits={split_sizes}")
    return metadata


def download_hindi(cfg: dict, force: bool = False) -> None:
    hi_cfg = cfg["languages"]["hindi"]
    out_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_raw"] / "hindi"
    if out_dir.exists() and not force:
        logger.info(f"Hindi raw data already exists at {out_dir}, skipping (use --force to re-download)")
        return
    ds = load_dataset(hi_cfg["dataset_name"], hi_cfg["dataset_config"])
    _save_dataset_dict(ds, out_dir, hi_cfg["dataset_name"], hi_cfg["dataset_config"])


def download_bengali(cfg: dict, force: bool = False) -> None:
    bn_cfg = cfg["languages"]["bengali"]
    out_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_raw"] / "bengali"
    if out_dir.exists() and not force:
        logger.info(f"Bengali raw data already exists at {out_dir}, skipping (use --force to re-download)")
        return
    ds = load_dataset(bn_cfg["dataset_name"], bn_cfg["dataset_config"])
    _save_dataset_dict(ds, out_dir, bn_cfg["dataset_name"], bn_cfg["dataset_config"])


def download_cross_lingual_eval(cfg: dict, force: bool = False) -> None:
    cl_cfg = cfg["cross_lingual_eval"]
    for lang_config in cl_cfg["configs"]:
        out_dir = Path(PROJECT_ROOT) / cfg["paths"]["data_raw"] / "indicsentiment" / lang_config
        if out_dir.exists() and not force:
            logger.info(f"IndicSentiment[{lang_config}] already exists at {out_dir}, skipping")
            continue
        ds = load_dataset(cl_cfg["dataset_name"], lang_config, trust_remote_code=True)
        _save_dataset_dict(ds, out_dir, cl_cfg["dataset_name"], lang_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if raw data already exists")
    parser.add_argument(
        "--only",
        choices=["hindi", "bengali", "cross_lingual", "all"],
        default="all",
    )
    args = parser.parse_args()

    config = load_config()

    if args.only in ("hindi", "all"):
        download_hindi(config, force=args.force)
    if args.only in ("bengali", "all"):
        download_bengali(config, force=args.force)
    if args.only in ("cross_lingual", "all"):
        download_cross_lingual_eval(config, force=args.force)
