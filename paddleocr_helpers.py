"""
Doc-Worker — Shared PaddleOCR helpers
======================================

Provides language code mapping, singleton model loading, and OCR extraction
used by both the FastAPI server (`server.py`) and the worker pipeline
(`worker.py`).
"""

from __future__ import annotations

import os
from typing import Any

# ── Config (mirrors server.py and worker.py) ───────────────────────────
OCR_LANG = os.getenv("OCR_LANG", "deu")
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() in ("true", "1", "yes")
PADDLEOCR_MODELS = os.getenv("PADDLEOCR_MODELS", "/app/models")

# ── Language mapping ───────────────────────────────────────────────────
def paddleocr_lang_code() -> str:
    """Map ocrmypdf/Tesseract language codes to PaddleOCR lang codes."""
    mapping = {
        "eng": "en",
        "deu": "german",
        "fra": "french",
        "jpn": "japan",
        "kor": "korean",
        "ita": "italian",
        "por": "portuguese",
        "spa": "spanish",
        "rus": "russian",
        "nld": "dutch",
        "pol": "polish",
        "tur": "turkish",
        "chs": "ch",
        "cht": "chinese_cht",
    }
    return mapping.get(OCR_LANG.lower(), "en")


# ── Singleton model loader ─────────────────────────────────────────────
def _get_paddleocr_model() -> Any:
    """Return a cached (singleton) PaddleOCR instance.

    Model loading is expensive; reusing the same instance avoids repeated
    GPU memory allocation and disk I/O.
    """
    if not hasattr(_get_paddleocr_model, "_model"):
        from paddleocr import PaddleOCR

        _get_paddleocr_model._model = PaddleOCR(  # type: ignore[attr-defined]
            use_textline_orientation=True,
            lang=paddleocr_lang_code(),
            device="gpu" if OCR_USE_GPU else "cpu",
            text_detection_model_dir=f"{PADDLEOCR_MODELS}/PP-OCRv6_medium_det_infer",
            text_recognition_model_dir=f"{PADDLEOCR_MODELS}/PP-OCRv6_medium_rec_infer",
        )
    return _get_paddleocr_model._model  # type: ignore[attr-defined]


# ── OCR extraction ─────────────────────────────────────────────────────
def run_paddleocr(file_path: str) -> list[dict]:
    """Run PaddleOCR and return list of page dicts with text + blocks.

    Each page dict contains:
        page (int): 1-based page number
        text (str): joined text for the page
        blocks (list[dict]): per-block text, bbox, confidence
    """
    model = _get_paddleocr_model()
    result = model.ocr(file_path, cls=True)

    pages: list[dict] = []
    for page_idx, page_result in enumerate(result or []):
        if page_result is None:
            pages.append({"page": page_idx + 1, "text": "", "blocks": []})
            continue

        blocks: list[dict] = []
        text_parts: list[str] = []
        for line in page_result:
            bbox, (text, confidence) = line[0], line[1]
            blocks.append(
                {
                    "text": text,
                    "bbox": bbox,
                    "confidence": round(confidence, 4),
                }
            )
            text_parts.append(text)

        pages.append(
            {
                "page": page_idx + 1,
                "text": "\n".join(text_parts),
                "blocks": blocks,
            }
        )
    return pages
