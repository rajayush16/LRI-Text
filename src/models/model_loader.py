"""Shared model/tokenizer loading with architecture verification.

Never silently substitutes a different checkpoint. For MLM-only checkpoints
(e.g. IndicBERT v2), AutoModelForSequenceClassification attaches a fresh,
randomly-initialized classification head on top of the pretrained backbone —
this is expected and logged, not an error.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_model_and_tokenizer(model_key: str, num_labels: int, models_cfg: dict):
    model_cfg = models_cfg["transformers"][model_key]
    checkpoint = model_cfg["checkpoint"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

    actual_class = model.__class__.__name__
    logger.info(
        f"Loaded '{model_key}' from checkpoint '{checkpoint}' as {actual_class} "
        f"(requires_custom_head={model_cfg.get('requires_custom_head', False)})"
    )
    return model, tokenizer
