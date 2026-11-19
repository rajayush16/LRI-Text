"""Generate all figures from whatever results currently exist in results/metrics/.
Safe to run with partial results — each plot is skipped (with a log message) if
its required data isn't present yet, rather than crashing.
"""

import json
from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging import get_logger
from src.utils.visualization import (
    plot_budget_vs_macrof1,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_training_time,
)

logger = get_logger(__name__)

if __name__ == "__main__":
    cfg = load_config()
    fig_dir = Path(PROJECT_ROOT) / cfg["paths"]["results_figures"]
    metrics_dir = Path(PROJECT_ROOT) / cfg["paths"]["results_metrics"]

    transformers_path = metrics_dir / "transformers.csv"
    if transformers_path.exists():
        df = pd.read_csv(transformers_path)

        for language in df["language"].unique():
            plot_budget_vs_macrof1(df, language, fig_dir / f"budget_vs_macrof1_{language}.png")
            logger.info(f"Saved budget_vs_macrof1_{language}.png")

        for pct in sorted(df["data_pct"].unique()):
            plot_model_comparison(df, pct, fig_dir / f"model_comparison_{pct}.png")
            logger.info(f"Saved model_comparison_{pct}.png")

        plot_training_time(df, fig_dir / "training_time.png")
        logger.info("Saved training_time.png")
    else:
        logger.warning("No transformers.csv found; skipping transformer-based figures")

    # Class distribution from audit reports
    label_counts = {}
    for language in ["hindi", "bengali"]:
        audit_path = Path(PROJECT_ROOT) / cfg["paths"]["results_logs"] / f"audit_{language}.json"
        if audit_path.exists():
            with open(audit_path, encoding="utf-8") as f:
                audit = json.load(f)
            train_dist = audit["splits"]["train"]["label_distribution"]
            label_counts[language] = {str(k): v for k, v in train_dist.items()}
    if label_counts:
        plot_class_distribution(label_counts, ["0", "1", "2"], fig_dir / "class_distribution_raw.png")
        logger.info("Saved class_distribution_raw.png (raw label ints, pre-canonical-remap)")

    # Confusion matrix for the single best available run (highest macro_f1) as an example figure
    raw_dir = metrics_dir / "raw"
    if raw_dir.exists():
        best = None
        for f in raw_dir.glob("transformer_*.json"):
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
            if best is None or r["metrics"]["macro_f1"] > best["metrics"]["macro_f1"]:
                best = r
        if best:
            plot_confusion_matrix(
                best["metrics"]["confusion_matrix"], cfg["canonical_label_names"],
                fig_dir / "confusion_matrix_best_run.png",
                title=f"Confusion matrix — best run ({best['language']}/{best['model']}/{best['data_pct']})",
            )
            logger.info(f"Saved confusion_matrix_best_run.png (best run: {best['language']}/{best['model']}/{best['data_pct']}, seed={best['seed']})")

    print("Done.")
