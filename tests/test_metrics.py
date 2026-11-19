from src.utils.metrics import compute_metrics

LABEL_NAMES = ["negative", "neutral", "positive"]


def test_compute_metrics_perfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y_true, y_pred, LABEL_NAMES)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    for cls in LABEL_NAMES:
        assert m["per_class"][cls]["f1"] == 1.0


def test_compute_metrics_all_wrong():
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    m = compute_metrics(y_true, y_pred, LABEL_NAMES)
    assert m["accuracy"] == 0.0
    assert m["per_class"]["negative"]["recall"] == 0.0


def test_compute_metrics_confusion_matrix_shape():
    y_true = [0, 1, 2]
    y_pred = [0, 2, 1]
    m = compute_metrics(y_true, y_pred, LABEL_NAMES)
    assert len(m["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in m["confusion_matrix"])


def test_compute_metrics_handles_missing_class_in_batch():
    # class 'neutral' (1) never appears in this small batch; must not crash
    y_true = [0, 0, 2]
    y_pred = [0, 2, 2]
    m = compute_metrics(y_true, y_pred, LABEL_NAMES)
    assert m["per_class"]["neutral"]["support"] == 0
