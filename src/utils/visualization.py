"""Publication-quality figures for LRI-Text, saved to results/figures/.

Color policy (colorblind-safe Okabe-Ito palette, assigned in FIXED order per
entity so a filtered view never repaints survivors):
  models:    mbert=blue, xlmr_base=vermillion, indicbert_v2=green, muril_base=purple
  languages: hindi=orange, bengali=sky-blue
  sequential (confusion matrices, magnitude): single-hue 'Blues' colormap.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_COLORS = {
    "mbert": "#0072B2",
    "xlmr_base": "#D55E00",
    "indicbert_v2": "#009E73",
    "muril_base": "#CC79A7",
}
LANGUAGE_COLORS = {"hindi": "#E69F00", "bengali": "#56B4E9"}
MODEL_ORDER = ["mbert", "xlmr_base", "indicbert_v2", "muril_base"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(label_counts: dict[str, dict[str, int]], label_names: list[str], out_path: Path):
    """label_counts: {language: {label_name: count}}"""
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    x = range(len(label_names))
    for i, (language, counts) in enumerate(label_counts.items()):
        vals = [counts.get(name, 0) for name in label_names]
        offset = (i - (len(label_counts) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], vals, width, label=language, color=LANGUAGE_COLORS.get(language, "#999"))
    ax.set_xticks(list(x))
    ax.set_xticklabels(label_names)
    ax.set_ylabel("Number of examples")
    ax.set_title("Class distribution (train split)")
    ax.legend(frameon=False)
    _save(fig, out_path)


def plot_budget_vs_macrof1(df: pd.DataFrame, language: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = df[(df["language"] == language) & (df["run_tag"].fillna("") == "")]
    agg = sub.groupby(["model", "data_pct"])["macro_f1"].mean().reset_index()
    for model in MODEL_ORDER:
        m = agg[agg["model"] == model].sort_values("data_pct")
        if m.empty:
            continue
        ax.plot(m["data_pct"], m["macro_f1"], marker="o", markersize=7, linewidth=2,
                color=MODEL_COLORS[model], label=model)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Macro-F1 (test set, mean over seeds)")
    ax.set_title(f"Data budget vs. Macro-F1 — {language}")
    ax.legend(frameon=False)
    _save(fig, out_path)


def plot_model_comparison(df: pd.DataFrame, data_pct: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = df[(df["data_pct"] == data_pct) & (df["run_tag"].fillna("") == "")]
    agg = sub.groupby(["language", "model"])["macro_f1"].mean().reset_index()
    languages = sorted(agg["language"].unique())
    width = 0.8 / max(len(MODEL_ORDER), 1)
    for i, model in enumerate(MODEL_ORDER):
        vals = [agg[(agg["language"] == lang) & (agg["model"] == model)]["macro_f1"].mean() for lang in languages]
        xpos = [j + i * width for j in range(len(languages))]
        ax.bar(xpos, vals, width, label=model, color=MODEL_COLORS[model])
    ax.set_xticks([j + width * (len(MODEL_ORDER) - 1) / 2 for j in range(len(languages))])
    ax.set_xticklabels(languages)
    ax.set_ylabel("Macro-F1 (test set, mean over seeds)")
    ax.set_title(f"Model comparison at {int(data_pct*100)}% data budget")
    ax.legend(frameon=False)
    _save(fig, out_path)


def plot_augmentation_vs_baseline(baseline_df: pd.DataFrame, augmented_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    base = baseline_df[baseline_df["run_tag"].fillna("") == ""]
    base_agg = base.groupby(["language", "model", "data_pct"])["macro_f1"].mean().reset_index()
    aug_agg = augmented_df.groupby(["language", "model", "data_pct"])["macro_f1"].mean().reset_index()
    merged = base_agg.merge(aug_agg, on=["language", "model", "data_pct"], suffixes=("_baseline", "_augmented"))
    labels = [f"{r.language}\n{r.model}\n{r.data_pct}" for r in merged.itertuples()]
    x = range(len(merged))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], merged["macro_f1_baseline"], width, label="no augmentation", color="#999999")
    ax.bar([xi + width / 2 for xi in x], merged["macro_f1_augmented"], width, label="+ back-translation", color="#0072B2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Macro-F1 (test set)")
    ax.set_title("Back-translation augmentation vs. baseline")
    ax.legend(frameon=False)
    _save(fig, out_path)


def plot_confusion_matrix(cm, label_names: list[str], out_path: Path, title: str = "Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    vmax = max(max(row) for row in cm) if cm else 1
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            ax.text(j, i, str(val), ha="center", va="center",
                     color="white" if val > vmax / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_path)


def plot_per_class_f1(per_class: dict, out_path: Path, title: str = "Per-class F1"):
    fig, ax = plt.subplots(figsize=(5, 4))
    names = list(per_class.keys())
    f1s = [per_class[n]["f1"] for n in names]
    ax.bar(names, f1s, color="#0072B2")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title(title)
    _save(fig, out_path)


def plot_training_time(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = df[df["run_tag"].fillna("") == ""]
    agg = sub.groupby(["model", "data_pct"])["train_time_sec"].mean().reset_index()
    for model in MODEL_ORDER:
        m = agg[agg["model"] == model].sort_values("data_pct")
        if m.empty:
            continue
        ax.plot(m["data_pct"], m["train_time_sec"], marker="o", markersize=7, linewidth=2,
                color=MODEL_COLORS[model], label=model)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Training time (sec)")
    ax.set_title("Training time vs. data budget")
    ax.legend(frameon=False)
    _save(fig, out_path)


def plot_cross_lingual(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    metric_cols = [c for c in df.columns if c.endswith("_macro_f1") and c != "macro_f1"]
    agg = df.groupby("model")[["macro_f1"] + metric_cols].mean().reset_index()
    x = range(len(agg))
    width = 0.8 / (len(metric_cols) + 1)
    ax.bar([xi - width for xi in x], agg["macro_f1"], width, label="source-language test", color="#999999")
    for i, col in enumerate(metric_cols):
        ax.bar([xi + i * width for xi in x], agg[col], width, label=col.replace("_macro_f1", ""),
               color=list(MODEL_COLORS.values())[i % len(MODEL_COLORS)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg["model"])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Cross-lingual transfer")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_path)
