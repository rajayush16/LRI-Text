"""Reproducible, class-balanced sampling utilities.

Used for: data-budget subsets (10/50/100%) and few-shot subsets (16/32 per class).
Only ever applied to the TRAIN split — val/test stay fixed and full-size.
"""

import pandas as pd


def stratified_subset(df: pd.DataFrame, label_col: str, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return df.reset_index(drop=True)
    parts = []
    for _, group in df.groupby(label_col):
        n = max(1, round(len(group) * fraction))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def few_shot_subset(df: pd.DataFrame, label_col: str, shots_per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for label, group in df.groupby(label_col):
        if len(group) < shots_per_class:
            raise ValueError(
                f"Class {label} has only {len(group)} examples, fewer than requested {shots_per_class}-shot"
            )
        parts.append(group.sample(n=shots_per_class, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
