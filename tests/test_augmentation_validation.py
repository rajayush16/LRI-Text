import pandas as pd

from src.augmentation.validation import filter_augmented, is_valid_augmented_text


def test_rejects_empty_candidate():
    assert not is_valid_augmented_text("यह अच्छा है", "   ", "hindi")


def test_rejects_identical_candidate():
    text = "यह फिल्म बहुत अच्छी थी"
    assert not is_valid_augmented_text(text, text, "hindi")


def test_accepts_plausible_paraphrase():
    original = "यह फिल्म बहुत अच्छी थी और मुझे पसंद आई"
    candidate = "यह फिल्म बहुत बढ़िया थी और मुझे अच्छी लगी"
    assert is_valid_augmented_text(original, candidate, "hindi")


def test_rejects_wrong_script():
    original = "यह फिल्म बहुत अच्छी थी"
    candidate = "this film was very good"  # translation pipeline failed to translate back
    assert not is_valid_augmented_text(original, candidate, "hindi")


def test_rejects_extreme_length_ratio():
    original = "अच्छा"
    candidate = " ".join(["शब्द"] * 20)
    assert not is_valid_augmented_text(original, candidate, "hindi")


def test_filter_augmented_drops_duplicates_across_rows():
    df = pd.DataFrame({
        "orig": ["यह फिल्म अच्छी थी", "यह फिल्म खराब थी"],
        "cand": ["यह मूवी बढ़िया थी", "यह मूवी बढ़िया थी"],  # identical candidates across rows
        "canonical_label": [2, 0],
    })
    out = filter_augmented(df, "orig", "cand", "hindi")
    assert len(out) == 1
