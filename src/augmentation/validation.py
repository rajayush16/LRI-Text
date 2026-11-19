"""Filters for back-translated augmentation candidates.

Rejects: empty/too-short output, output identical to the source (no new signal),
output in the wrong script (translation failed/fell back to source language leakage
or pivot-language leftovers), and duplicates against already-accepted augmented text.
"""

import pandas as pd

from src.data.audit import script_fraction

MIN_SCRIPT_FRACTION = 0.85
MIN_LENGTH_RATIO = 0.3
MAX_LENGTH_RATIO = 3.0


def is_valid_augmented_text(original: str, candidate: str, language: str) -> bool:
    candidate = candidate.strip()
    if not candidate:
        return False
    if candidate == original.strip():
        return False
    if script_fraction(pd.Series([candidate]), language) < MIN_SCRIPT_FRACTION:
        return False
    orig_len, cand_len = len(original.split()), len(candidate.split())
    if orig_len == 0:
        return False
    ratio = cand_len / orig_len
    if ratio < MIN_LENGTH_RATIO or ratio > MAX_LENGTH_RATIO:
        return False
    return True


def filter_augmented(df: pd.DataFrame, original_col: str, augmented_col: str, language: str) -> pd.DataFrame:
    mask = df.apply(lambda row: is_valid_augmented_text(row[original_col], row[augmented_col], language), axis=1)
    kept = df[mask].copy()
    kept = kept.drop_duplicates(subset=[augmented_col], keep="first")
    return kept
