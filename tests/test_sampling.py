import pandas as pd
import pytest

from src.data.sampling import few_shot_subset, stratified_subset


def _make_df(n_per_class=100, n_classes=3):
    rows = []
    for label in range(n_classes):
        for i in range(n_per_class):
            rows.append({"text": f"class{label}_item{i}", "label": label})
    return pd.DataFrame(rows)


def test_stratified_subset_preserves_class_proportions():
    df = _make_df(n_per_class=100, n_classes=3)
    out = stratified_subset(df, "label", fraction=0.1, seed=42)
    counts = out["label"].value_counts()
    assert set(counts.index) == {0, 1, 2}
    for c in counts:
        assert c == 10


def test_stratified_subset_full_fraction_returns_all():
    df = _make_df(n_per_class=10, n_classes=2)
    out = stratified_subset(df, "label", fraction=1.0, seed=42)
    assert len(out) == len(df)


def test_stratified_subset_reproducible_with_same_seed():
    df = _make_df(n_per_class=50, n_classes=3)
    out1 = stratified_subset(df, "label", fraction=0.2, seed=7)
    out2 = stratified_subset(df, "label", fraction=0.2, seed=7)
    assert sorted(out1["text"].tolist()) == sorted(out2["text"].tolist())


def test_few_shot_subset_exact_count_per_class():
    df = _make_df(n_per_class=50, n_classes=3)
    out = few_shot_subset(df, "label", shots_per_class=16, seed=42)
    counts = out["label"].value_counts()
    assert len(out) == 48
    for c in counts:
        assert c == 16


def test_few_shot_subset_raises_when_class_too_small():
    df = _make_df(n_per_class=5, n_classes=2)
    with pytest.raises(ValueError):
        few_shot_subset(df, "label", shots_per_class=16, seed=42)
