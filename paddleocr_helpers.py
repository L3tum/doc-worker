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
TEXT_DETECTION_MODEL = "PP-OCRv6_medium_det_infer"
TEXT_RECOGNITION_MODEL = "PP-OCRv6_medium_rec_infer"
TEXTLINE_ORIENTATION_MODEL = "PP-LCNet_x1_0_textline_ori_infer"

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


def _model_dir(model_name: str) -> Path:
    """Return the local directory for a bundled PaddleOCR model."""
    return Path(PADDLEOCR_MODELS) / model_name


def validate_paddleocr_models() -> None:
    """Validate that required PaddleOCR model directories are present.

    PaddleOCR 3.x inference archives contain model directories with
    ``inference.pdiparams``, ``inference.yml``, and a model-definition file
    (usually ``inference.json``). Validate those files directly instead of
    relying on older ``.pdiparams.info`` metadata.

    Also check that the ``model_name`` in each ``inference.yml`` matches the
    directory name to catch the "model name mismatch" error before PaddleOCR
    tries to load.
    """
    models_dir = Path(PADDLEOCR_MODELS)
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"PADDLEOCR_MODELS directory does not exist: {models_dir}"
        )

    # Rename the incorrect directory name used by the previous image, if a user
    # mounted or copied models with that name manually.
    legacy_orientation_dir = models_dir / "PP-OCRv6_lcnet_x1_0_textline_ori_infer"
    orientation_dir = _model_dir(TEXTLINE_ORIENTATION_MODEL)
    if legacy_orientation_dir.is_dir() and not orientation_dir.exists():
        os.rename(str(legacy_orientation_dir), str(orientation_dir))
        print(
            f"Fixed model directory name: {legacy_orientation_dir.name} -> {orientation_dir.name}",
            flush=True,
        )

    import re

    missing: list[str] = []
    yml_mismatches: list[str] = []
    for model_name in (
        TEXT_DETECTION_MODEL,
        TEXT_RECOGNITION_MODEL,
        TEXTLINE_ORIENTATION_MODEL,
    ):
        model_dir = _model_dir(model_name)
        if not model_dir.is_dir():
            missing.append(f"{model_name}/")
            continue

        required_files = ["inference.pdiparams", "inference.yml"]
        for required_file in required_files:
            if not (model_dir / required_file).is_file():
                missing.append(f"{model_name}/{required_file}")

        if not any(
            (model_dir / model_file).is_file()
            for model_file in ("inference.json", "inference.pdmodel")
        ):
            missing.append(f"{model_name}/inference.json or inference.pdmodel")

        # Check model_name in inference.yml matches directory name
        yml_path = model_dir / "inference.yml"
        if yml_path.is_file():
            yml_content = yml_path.read_text(encoding="utf-8")
            match = re.search(r"^  model_name:\s*(.+)$", yml_content, re.MULTILINE)
            if not match:
                yml_mismatches.append(f"{model_name}: inference.yml missing model_name")
                continue

            yml_model_name = match.group(1).strip()
            if yml_model_name != model_name:
                yml_mismatches.append(
                    f"{model_name}: inference.yml declares '{yml_model_name}' "
                    f"(expected '{model_name}')"
                )

    if missing:
        raise FileNotFoundError(
            f"Missing PaddleOCR model files under {models_dir}: "
            f"{', '.join(missing)}. "
            "Rebuild the container image with the full model set or set "
            "PADDLEOCR_MODELS to a complete model directory."
        )

    if yml_mismatches:
        raise ValueError(
            f"PaddleOCR model name mismatches (inference.yml vs directory name): "
            f"{', '.join(yml_mismatches)}. "
            "The Dockerfile should patch these, or the model archive is broken. "
            "Fix: edit inference.yml and set model_name to match the directory."
        )


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
        validate_paddleocr_models()

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

            # Orientation model is loaded locally from the pre-downloaded directory,
            # so no network download attempts occur.
            _get_paddleocr_model._model = PaddleOCR(  # type: ignore[attr-defined]
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=paddleocr_lang_code(),
                device="gpu" if OCR_USE_GPU else "cpu",
                text_detection_model_dir=str(_model_dir(TEXT_DETECTION_MODEL)),
                text_recognition_model_dir=str(_model_dir(TEXT_RECOGNITION_MODEL)),
                textline_orientation_model_dir=str(_model_dir(TEXTLINE_ORIENTATION_MODEL)),
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
