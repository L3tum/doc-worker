"""
Doc-Worker — Shared PaddleOCR helpers
======================================

Provides language code mapping, singleton model loading, and OCR extraction
used by both the FastAPI server (`server.py`) and the worker pipeline
(`worker.py`).
"""

from __future__ import annotations

import os
import threading
import warnings
from pathlib import Path
from typing import Any

# ── Config (mirrors server.py and worker.py) ───────────────────────────
OCR_LANG = os.getenv("OCR_LANG", "deu")
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() in ("true", "1", "yes")
PADDLEOCR_MODELS = os.getenv("PADDLEOCR_MODELS", "/app/models")
DEFAULT_PADDLE_PDX_CACHE_HOME = "/tmp/.paddlex"

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
_PADDLEOCR_MODEL_LOCK = threading.Lock()


def get_paddleocr_init_exception() -> BaseException | None:
    """Return a cached PaddleOCR initialization exception, if one occurred."""
    if hasattr(_get_paddleocr_model, "_model"):
        return None

    exc = getattr(_get_paddleocr_model, "_init_exception", None)
    return exc if isinstance(exc, BaseException) else None


def _assert_writable_directory(path: Path) -> None:
    """Create *path* and verify the current user can write inside it."""
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write-test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def _ensure_paddlex_cache_home() -> None:
    """Set PADDLE_PDX_CACHE_HOME to a writable directory before PaddleOCR import."""
    configured_cache = Path(
        os.environ.get("PADDLE_PDX_CACHE_HOME") or DEFAULT_PADDLE_PDX_CACHE_HOME
    )
    try:
        _assert_writable_directory(configured_cache)
    except OSError:
        fallback_cache = Path(DEFAULT_PADDLE_PDX_CACHE_HOME)
        _assert_writable_directory(fallback_cache)
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(fallback_cache)
    else:
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(configured_cache)


def _get_paddleocr_model() -> Any:
    """Return a cached (singleton) PaddleOCR instance.

    Model loading is expensive; reusing the same instance avoids repeated
    GPU memory allocation and disk I/O.
    """
    with _PADDLEOCR_MODEL_LOCK:
        if hasattr(_get_paddleocr_model, "_model"):
            return _get_paddleocr_model._model  # type: ignore[attr-defined]

        # If a previous call failed to initialize, re-raise the stored exception
        # instead of attempting re-initialization (which would fail with
        # "PDX has already been initialized. Reinitialization is not supported.")
        init_exception = get_paddleocr_init_exception()
        if init_exception is not None:
            raise init_exception

        # Prevent PaddleX from trying to write to / or other read-only locations.
        # This must run before importing PaddleOCR because PaddleX reads the env
        # var during import/initialization.
        _ensure_paddlex_cache_home()

        # Validate that local model directories actually exist before attempting
        # to initialize PaddleOCR. This prevents the confusing "No model hoster"
        # error that occurs when PaddleOCR tries to download missing models.
        det_model_dir = Path(f"{PADDLEOCR_MODELS}/PP-OCRv6_medium_det_infer")
        rec_model_dir = Path(f"{PADDLEOCR_MODELS}/PP-OCRv6_medium_rec_infer")
        if not det_model_dir.is_dir() or not rec_model_dir.is_dir():
            missing = []
            if not det_model_dir.is_dir():
                missing.append(f"text detection ({det_model_dir})")
            if not rec_model_dir.is_dir():
                missing.append(f"text recognition ({rec_model_dir})")
            raise FileNotFoundError(
                f"PaddleOCR model directories are missing: {', '.join(missing)}. "
                "Set PADDLEOCR_MODELS to a directory containing "
                "PP-OCRv6_medium_det_infer/ and PP-OCRv6_medium_rec_infer/ "
                "subdirectories, or use a container image that includes these models."
            )

        try:
            from paddleocr import PaddleOCR
            import paddleocr._utils.logging as paddleocr_logging

            # Suppress PaddleOCR's internal logging noise (e.g. "Creating model", "No model hoster")
            # We rely on the app's own logging for errors via the exception handling below.
            paddleocr_logging.logger.setLevel(100)  # above DEBUG/ERROR
            # Suppress PaddleOCR's UserWarning about lang/ocr_version being ignored when model dirs are provided
            warnings.filterwarnings(
                "ignore",
                message=r"`lang` and `ocr_version` will be ignored when model names or model directories are not `None`",
            )

            # Disable orientation correction modules that would require additional models
            # (doc orientation classification, textline orientation, doc unwarping).
            # These are not needed for our use-case (flat document OCR) and would cause
            # runtime errors when models aren't present in PADDLEOCR_MODELS.
            _get_paddleocr_model._model = PaddleOCR(  # type: ignore[attr-defined]
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=paddleocr_lang_code(),
                device="gpu" if OCR_USE_GPU else "cpu",
                text_detection_model_dir=f"{PADDLEOCR_MODELS}/PP-OCRv6_medium_det_infer",
                text_recognition_model_dir=f"{PADDLEOCR_MODELS}/PP-OCRv6_medium_rec_infer",
            )
        except Exception as exc:
            # Store the exception to avoid "PDX has already been initialized"
            # on subsequent retries.
            _get_paddleocr_model._init_exception = exc  # type: ignore[attr-defined]
            raise

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
    result = model.ocr(file_path)

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
