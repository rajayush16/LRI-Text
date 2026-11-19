from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, label_names: list[str]) -> dict:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(len(label_names)))
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            label_names[i]: {
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "f1": float(per_class_f1[i]),
                "support": int(per_class_support[i]),
            }
            for i in range(len(label_names))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(label_names)))).tolist(),
    }
