import pandas as pd

from src.data.preprocessing import clean_and_dedup, clean_text, remap_to_canonical


def test_clean_text_normalizes_whitespace():
    assert clean_text("  hello   world  \n") == "hello world"


def test_clean_text_unicode_nfc():
    # 'e' + combining acute accent (NFD) should normalize to the same as precomposed 'é' (NFC)
    nfd = "é"
    nfc = "é"
    assert clean_text(nfd) == clean_text(nfc)


def test_clean_and_dedup_removes_empty_and_duplicates():
    df = pd.DataFrame({"text": ["hello", "hello", "  ", "world", None]})
    out = clean_and_dedup(df, "text")
    assert sorted(out["text"].tolist()) == ["hello", "world"]


def test_remap_to_canonical_hindi_direction():
    # Hindi: 0=negative, 1=neutral, 2=positive -> canonical negative=0, neutral=1, positive=2 (identity)
    df = pd.DataFrame({"label": [0, 1, 2]})
    raw_map = {0: "negative", 1: "neutral", 2: "positive"}
    name_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    out = remap_to_canonical(df, "label", raw_map, name_to_id)
    assert out["canonical_label"].tolist() == [0, 1, 2]


def test_remap_to_canonical_bengali_direction_is_not_identity():
    # Bengali SentNoB: 0=neutral, 1=positive, 2=negative -> must remap to negative=0, neutral=1, positive=2
    df = pd.DataFrame({"label": [0, 1, 2]})
    raw_map = {0: "neutral", 1: "positive", 2: "negative"}
    name_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    out = remap_to_canonical(df, "label", raw_map, name_to_id)
    # raw 0 (neutral) -> canonical 1; raw 1 (positive) -> canonical 2; raw 2 (negative) -> canonical 0
    assert out["canonical_label"].tolist() == [1, 2, 0]


def test_remap_drops_unmapped_labels():
    df = pd.DataFrame({"label": [0, 1, 99]})
    raw_map = {0: "negative", 1: "neutral"}
    name_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    out = remap_to_canonical(df, "label", raw_map, name_to_id)
    assert len(out) == 2
