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
from pathlib import Path
from typing import Any

# ── Config (mirrors server.py and worker.py) ───────────────────────────
OCR_LANG = os.getenv("OCR_LANG", "deu")
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() in ("true", "1", "yes")
PADDLEOCR_MODELS = os.getenv("PADDLEOCR_MODELS", "/app/models")
DEFAULT_PADDLE_PDX_CACHE_HOME = "/tmp/.paddlex"

# PaddleOCR/PaddleX distinguishes the logical model name from the extracted
# inference directory name.  The official tarballs extract to ``*_infer``
# directories, but their ``inference.yml`` files declare the model name without
# that suffix.  Passing/patching the directory name as the model name triggers
# PaddleX's "model name mismatch" guard.
TEXT_DETECTION_MODEL = "PP-OCRv6_medium_det"
TEXT_RECOGNITION_MODEL = "PP-OCRv6_medium_rec"
TEXTLINE_ORIENTATION_MODEL = "PP-LCNet_x1_0_textline_ori"

PADDLEOCR_MODEL_DIRS = {
    TEXT_DETECTION_MODEL: "PP-OCRv6_medium_det_infer",
    TEXT_RECOGNITION_MODEL: "PP-OCRv6_medium_rec_infer",
    TEXTLINE_ORIENTATION_MODEL: "PP-LCNet_x1_0_textline_ori_infer",
}

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
    return Path(PADDLEOCR_MODELS) / PADDLEOCR_MODEL_DIRS.get(model_name, model_name)


def validate_paddleocr_models() -> None:
    """Validate that required PaddleOCR model directories are present.

    PaddleOCR 3.x inference archives contain model directories with
    ``inference.pdiparams``, ``inference.yml``, and a model-definition file
    (usually ``inference.json``). Validate those files directly instead of
    relying on older ``.pdiparams.info`` metadata.

    Also check that the ``model_name`` in each ``inference.yml`` matches the
    logical PaddleOCR/PaddleX model name (without the ``_infer`` directory
    suffix) to catch the "model name mismatch" error before PaddleOCR tries to
    load.
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

        # Check model_name in inference.yml matches the logical Paddle model name.
        yml_path = model_dir / "inference.yml"
        if yml_path.is_file():
            yml_content = yml_path.read_text(encoding="utf-8")
            match = re.search(r"^[ \t]*model_name:\s*(.+)$", yml_content, re.MULTILINE)
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
            f"PaddleOCR model name mismatches (inference.yml vs logical model name): "
            f"{', '.join(yml_mismatches)}. "
            f"Files checked under {models_dir}. "
            "Likely cause: stale model files from an older image patched "
            "model_name to the *_infer directory name, or mixed PP-OCR model "
            "generations. Rebuild the image so inference.yml keeps Paddle's "
            "logical model names (for example 'model_name: PP-OCRv6_medium_det')."
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


def create_paddleocr_model(*, use_textline_orientation: bool = False) -> Any:
    """Create a PaddleOCR instance pinned to the bundled local PP-OCRv6 models."""
    # Prevent PaddleX from trying to write to / or other read-only locations.
    # This must run before importing PaddleOCR because PaddleX reads the env var
    # during import/initialization.
    _ensure_paddlex_cache_home()
    validate_paddleocr_models()

    from paddleocr import PaddleOCR
    import paddleocr._utils.logging as paddleocr_logging

    # Suppress PaddleOCR's internal logging noise (e.g. "Creating model",
    # "No model hoster"). We rely on the app's own logging for errors via the
    # exception handling below.
    paddleocr_logging.logger.setLevel(100)  # above DEBUG/ERROR

    # The official PP-OCRv6 tarballs extract to *_infer directories, but the
    # logical model names are the same strings without *_infer.  Pass both
    # explicitly; otherwise PaddleOCR/PaddleX can combine a default PP-OCRv5
    # model_name with our PP-OCRv6 model_dir for Latin languages.
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=use_textline_orientation,
        device="gpu" if OCR_USE_GPU else "cpu",
        text_detection_model_name=TEXT_DETECTION_MODEL,
        text_detection_model_dir=str(_model_dir(TEXT_DETECTION_MODEL)),
        text_recognition_model_name=TEXT_RECOGNITION_MODEL,
        text_recognition_model_dir=str(_model_dir(TEXT_RECOGNITION_MODEL)),
        textline_orientation_model_name=TEXTLINE_ORIENTATION_MODEL,
        textline_orientation_model_dir=str(_model_dir(TEXTLINE_ORIENTATION_MODEL)),
    )


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

        try:
            _get_paddleocr_model._model = create_paddleocr_model(  # type: ignore[attr-defined]
                use_textline_orientation=False
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
    result = model.predict(file_path)

    def _jsonable(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    pages: list[dict] = []
    for page_idx, page_result in enumerate(result or []):
        if not page_result:
            pages.append({"page": page_idx + 1, "text": "", "blocks": []})
            continue

        rec_texts = page_result.get("rec_texts", [])
        rec_scores = page_result.get("rec_scores", [])
        rec_boxes = page_result.get("rec_boxes", [])
        rec_polys = page_result.get("rec_polys", page_result.get("dt_polys", []))

        blocks: list[dict] = []
        text_parts: list[str] = []
        for idx, text in enumerate(rec_texts):
            if not str(text).strip():
                continue

            confidence = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
            bbox = rec_boxes[idx] if idx < len(rec_boxes) else None
            if bbox is None and idx < len(rec_polys):
                bbox = rec_polys[idx]

            blocks.append(
                {
                    "text": str(text),
                    "bbox": _jsonable(bbox),
                    "confidence": round(confidence, 4),
                }
            )
            text_parts.append(str(text))

        pages.append(
            {
                "page": page_idx + 1,
                "text": "\n".join(text_parts),
                "blocks": blocks,
            }
        )
    return pages
