import numpy as np

from src.utils.metrics import compute_metrics


def make_compute_metrics(label_names: list[str]):
    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics(labels, preds, label_names)

    return _compute
