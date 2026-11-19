from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "config.yaml") -> dict:
    path = PROJECT_ROOT / "configs" / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
