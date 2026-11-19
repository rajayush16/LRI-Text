"""Back-translation augmentation via a single pivot model (NLLB-200-distilled-600M),
pivoting through English: hi/bn -> en -> hi/bn.

Only ever applied to TRAIN data, never val/test (enforced by callers, not here).
Generated examples are validated (src/augmentation/validation.py) before use.
"""

from typing import List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.logging import get_logger

logger = get_logger(__name__)

NLLB_CHECKPOINT = "facebook/nllb-200-distilled-600M"
NLLB_LANG_CODES = {"hindi": "hin_Deva", "bengali": "ben_Beng", "english": "eng_Latn"}


def load_translation_model() -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(NLLB_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_CHECKPOINT).to(device)
    model.eval()
    logger.info(f"Loaded {NLLB_CHECKPOINT} on {device}")
    return model, tokenizer, device


def _translate_batch(
    texts: List[str], src_code: str, tgt_code: str, model, tokenizer, device: str,
    max_length: int = 128, batch_size: int = 16,
) -> List[str]:
    tokenizer.src_lang = src_code
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_code)
    outputs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            gen = model.generate(**enc, forced_bos_token_id=tgt_id, max_length=max_length, num_beams=4)
            outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def back_translate(texts: List[str], language: str, model, tokenizer, device: str) -> List[str]:
    src = NLLB_LANG_CODES[language]
    pivot = NLLB_LANG_CODES["english"]
    pivoted = _translate_batch(texts, src, pivot, model, tokenizer, device)
    back = _translate_batch(pivoted, pivot, src, model, tokenizer, device)
    return back


def augment_dataframe(df, text_col: str, language: str, model, tokenizer, device: str, max_ratio: float = 1.0):
    import pandas as pd

    from src.augmentation.validation import filter_augmented

    texts = df[text_col].tolist()
    candidates = back_translate(texts, language, model, tokenizer, device)

    cand_df = df.copy().reset_index(drop=True)
    cand_df["_augmented_text"] = candidates
    valid = filter_augmented(cand_df, text_col, "_augmented_text", language)

    max_n = int(len(df) * max_ratio)
    if len(valid) > max_n:
        valid = valid.sample(n=max_n, random_state=0)

    logger.info(f"Back-translation: {len(df)} source -> {len(candidates)} candidates -> {len(valid)} accepted")

    out = valid[["_augmented_text", "canonical_label"]].rename(columns={"_augmented_text": text_col})
    return pd.DataFrame(out).reset_index(drop=True)
