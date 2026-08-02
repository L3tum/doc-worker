"""
Doc-Worker — PaddleX helpers
=============================

Provides General OCR pipeline, PP-StructureV3 structured layout parsing,
and semantic markdown building — all based on PaddleX 3.0 pipelines.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# ── Config ────────────────────────────────────────────────────────────────
OCR_LANG = os.getenv("OCR_LANG", "deu")
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() in ("true", "1", "yes")
PADDLEOCR_MODELS = os.getenv("PADDLEOCR_MODELS", "/app/models")
DEFAULT_PADDLE_PDX_CACHE_HOME = "/tmp/.paddlex"
USE_STRUCTURE_V3 = os.getenv("USE_STRUCTURE_V3", "true").lower() in ("true", "1", "yes")

# PaddleX model names (logical names, without the *_infer directory suffix)
TEXT_DETECTION_MODEL = "PP-OCRv6_medium_det"
TEXT_RECOGNITION_MODEL = "PP-OCRv6_medium_rec"
TEXTLINE_ORIENTATION_MODEL = "PP-LCNet_x1_0_textline_ori"
LAYOUT_DETECTION_MODEL = "PP-DocLayout-L"

# Model directories (extracted tarball names)
PADDLEX_MODEL_DIRS = {
    TEXT_DETECTION_MODEL: "PP-OCRv6_medium_det_infer",
    TEXT_RECOGNITION_MODEL: "PP-OCRv6_medium_rec_infer",
    TEXTLINE_ORIENTATION_MODEL: "PP-LCNet_x1_0_textline_ori_infer",
    LAYOUT_DETECTION_MODEL: "PP-DocLayout-L_infer",
}


# ── Language mapping ──────────────────────────────────────────────────────
def paddleocr_lang_code() -> str:
    """Map Tesseract/ocrmypdf language codes to PaddleOCR/PaddleX language codes."""
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


# ── Singleton model managers ──────────────────────────────────────────────
_PADDLEX_MODEL_LOCK = threading.Lock()


def get_paddlex_init_exception() -> BaseException | None:
    """Return a cached PaddleX initialization exception, if one occurred."""
    exc = getattr(_get_paddlex_model, "_init_exception", None)
    return exc if isinstance(exc, BaseException) else None


def _assert_writable_directory(path: Path) -> None:
    """Create *path* and verify the current user can write inside it."""
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write-test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def _model_dir(model_name: str) -> Path:
    """Return the local directory for a bundled PaddleX model."""
    return Path(PADDLEOCR_MODELS) / PADDLEX_MODEL_DIRS.get(model_name, model_name)


def validate_paddlex_models() -> None:
    """Validate that required PaddleX model directories are present."""
    models_dir = Path(PADDLEOCR_MODELS)
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"PADDLEOCR_MODELS directory does not exist: {models_dir}"
        )

    # Legacy rename: rename the incorrect orientation directory name if present
    legacy_orientation_dir = models_dir / "PP-OCRv6_lcnet_x1_0_textline_ori_infer"
    orientation_dir = _model_dir(TEXTLINE_ORIENTATION_MODEL)
    if legacy_orientation_dir.is_dir() and not orientation_dir.exists():
        os.rename(str(legacy_orientation_dir), str(orientation_dir))
        logging.getLogger("doc-worker.paddlex_helpers").info(
            "Renamed legacy model directory: %s -> %s",
            legacy_orientation_dir,
            orientation_dir,
        )

    missing: list[str] = []
    yml_mismatches: list[str] = []
    for model_name in (
        TEXT_DETECTION_MODEL,
        TEXT_RECOGNITION_MODEL,
        TEXTLINE_ORIENTATION_MODEL,
        LAYOUT_DETECTION_MODEL,
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
            (model_dir / f).is_file() for f in ("inference.json", "inference.pdmodel")
        ):
            missing.append(f"{model_name}/inference.json or inference.pdmodel")

        # Check model_name in inference.yml
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
            f"Missing PaddleX model files under {models_dir}: "
            f"{', '.join(missing)}. "
            "Rebuild the container image with the full model set."
        )

    if yml_mismatches:
        raise ValueError(
            f"PaddleX model name mismatches (inference.yml vs logical model name): "
            f"{', '.join(yml_mismatches)}. "
            f"Files checked under {models_dir}. "
            "Rebuild the image so inference.yml keeps Paddle's logical model names."
        )


# ── Offline / air-gapped compatibility ────────────────────────────────────
_PADDLEX_PATCHED = False


def _patch_paddlex_official_models() -> None:
    """Monkey-patch PaddleX official_models to return our local model dirs.

    PaddleX 3.x always calls ``official_models[model_name]`` internally, even when
    explicit ``model_dir`` parameters are passed to ``create_pipeline()``.  In
    air-gapped environments (or when the hosting platforms are geo-blocked), this
    fails with ``"No available model hosting platforms detected"`` because
    PaddleX cannot reach its hosting platforms (HuggingFace, ModelScope,
    AIStudio, BOS).

    This patch intercepts the ``official_models.__getitem__`` lookup so that
    model names we know about resolve to our pre-bundled local directories.
    Unknown names fall back to the original PaddleX behavior (which may trigger
    a download attempt if network is available).

    See: PaddleX#4578, PaddleOCR#16620, PaddleOCR#16639
    """
    global _PADDLEX_PATCHED

    if _PADDLEX_PATCHED:
        return  # Already patched — idempotent

    try:
        from paddlex.inference.utils import official_models as _om_module
    except ImportError:
        # PaddleX may not be installed (e.g. test env without paddlex)
        return

    official_models = _om_module.official_models
    original_getitem = official_models.__class__.__getitem__

    def _patched_getitem(self, model_name: str):  # type: ignore[no-untyped-def]
        # Check our local model dirs first
        local_dir = _model_dir(model_name)
        if local_dir.is_dir():
            return str(local_dir)
        # Fall back to original behavior (may trigger download or raise)
        return original_getitem(self, model_name)

    official_models.__class__.__getitem__ = _patched_getitem  # type: ignore[attr-defined]
    _PADDLEX_PATCHED = True
    logging.getLogger("doc-worker.paddlex_helpers").debug(
        "Patched paddlex.inference.utils.official_models for offline operation"
    )


def _ensure_paddlex_cache_home() -> None:
    """Set PADDLE_PDX_CACHE_HOME to a writable directory before PaddleX import."""
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


def _ensure_paddlex_offline_compat() -> None:
    """Ensure PaddleX is ready for offline / air-gapped operation.

    Calls ``_ensure_paddlex_cache_home()`` and then monkey-patches PaddleX's
    ``official_models`` registry so that model lookups resolve to our
    pre-bundled local directories instead of trying to download from the
    hosting platforms.

    This must be called before any ``create_pipeline()`` invocation.
    """
    _ensure_paddlex_cache_home()
    _patch_paddlex_official_models()


def _is_permanent_model_init_error(exc: Exception) -> bool:
    """Return True if the model initialization error is permanent (not retryable).

    Permanent errors indicate conditions that won't be fixed by retrying:
    - PaddleX hosting platforms unreachable when models should be local
    - PDX internal state conflict (already initialized)

    Transient errors (network timeout, temporary file access issues) return False
    so they can be retried.
    """
    msg = str(exc).lower()
    return "no available model hosting" in msg or "already been initialized" in msg


# ── General OCR pipeline ──────────────────────────────────────────────────
def _create_paddlex_ocr_pipeline(use_textline_orientation: bool = True) -> Any:
    """Create a PaddleX General OCR pipeline with bundled models.

    Args:
        use_textline_orientation: If True, textline orientation detection is enabled.
            Improves accuracy for rotated text at a slight performance cost.
            Default is True (improved accuracy), set to False for faster processing.
    """
    _ensure_paddlex_offline_compat()

    from paddlex import create_pipeline

    # Suppress PaddleX internal logging noise
    logging.getLogger("paddlex").setLevel(100)

    return create_pipeline(
        "OCR",
        text_detection_model_dir=str(_model_dir(TEXT_DETECTION_MODEL)),
        text_recognition_model_dir=str(_model_dir(TEXT_RECOGNITION_MODEL)),
        textline_orientation_model_dir=str(_model_dir(TEXTLINE_ORIENTATION_MODEL)),
        text_detection_model_name=TEXT_DETECTION_MODEL,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL,
        textline_orientation_model_name=TEXTLINE_ORIENTATION_MODEL,
        use_textline_orientation=use_textline_orientation,
        lang=paddleocr_lang_code(),
    )


def create_paddleocr_model(*, use_textline_orientation: bool = True) -> Any:
    """Create a PaddleX General OCR pipeline (for ocrmypdf plugin compatibility).

    Exposes the same API as the old create_paddleocr_model() from paddleocr.

    Args:
        use_textline_orientation: If True, textline orientation detection is enabled.
            Default is True. Set to False for faster processing.
    """
    return _create_paddlex_ocr_pipeline(
        use_textline_orientation=use_textline_orientation
    )


def _get_paddlex_model() -> Any:
    """Return a cached (singleton) PaddleX General OCR pipeline.

    If initialization fails, the exception is cached and re-raised on subsequent
    calls. To allow retry after a transient failure, use destroy_paddlex_model()
    to clear the cache.
    """
    with _PADDLEX_MODEL_LOCK:
        if hasattr(_get_paddlex_model, "_model"):  # type: ignore[attr-defined]
            return _get_paddlex_model._model  # type: ignore[attr-defined]

        exc = get_paddlex_init_exception()
        if exc is not None:
            raise exc

        # Retry mechanism: up to 3 attempts with exponential backoff.
        # Permanent errors (hosting unreachable, PDX conflict) skip retry.
        max_retries = 3
        base_delay = 0.1  # seconds
        for attempt in range(max_retries):
            try:
                _get_paddlex_model._model = _create_paddlex_ocr_pipeline()  # type: ignore[attr-defined]
            except Exception as e:
                # Permanent errors: don't retry — cache and fail fast
                if _is_permanent_model_init_error(e):
                    _get_paddlex_model._init_exception = e  # type: ignore[attr-defined]
                    raise
                if attempt == max_retries - 1:
                    _get_paddlex_model._init_exception = e  # type: ignore[attr-defined]
                    raise
                delay = base_delay * (2**attempt)  # 0.1s, 0.2s, 0.4s
                logging.getLogger("doc-worker.paddlex_helpers").warning(
                    "PaddleX model initialization failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)
            else:
                break

        return _get_paddlex_model._model  # type: ignore[attr-defined]


# ── Model destruction ────────────────────────────────────────────────────
def destroy_paddlex_model() -> None:
    """Destroy cached PaddleX models and reclaim memory."""
    import gc
    import logging

    logger = logging.getLogger("doc-worker.paddlex_helpers")

    try:
        with _PADDLEX_MODEL_LOCK:
            if hasattr(_get_paddlex_model, "_model"):
                del _get_paddlex_model._model
            if hasattr(_get_paddlex_model, "_init_exception"):
                del _get_paddlex_model._init_exception
            # Also clear Structure V3 model if it exists
            if hasattr(_get_paddlex_structure_v3_model, "_model"):
                del _get_paddlex_structure_v3_model._model
            if hasattr(_get_paddlex_structure_v3_model, "_init_exception"):
                del _get_paddlex_structure_v3_model._init_exception

        # Clear OCRmyPDF plugin's singleton
        try:
            engine_module = __import__(
                "ocrmypdf_paddleocr.engine", fromlist=["_reset_paddle_engine"]
            )
            if hasattr(engine_module, "_reset_paddle_engine"):
                engine_module._reset_paddle_engine()
        except ImportError:
            pass

        gc.collect()
        if OCR_USE_GPU:
            try:
                import paddle.device

                paddle.device.cuda.empty_cache()
                logger.info("GPU memory cache cleared")
            except Exception:
                logger.exception("Failed to clear CUDA cache")

        logger.info("PaddleX models destroyed — memory reclaimed")

    except Exception:
        logger.exception("Error during PaddleX model destruction")


# ── PDF → image conversion helper ─────────────────────────────────────────
def _pdf_to_images(pdf_path: str, tmp_dir: str) -> list[str]:
    """Convert a PDF file to a list of PNG images (one per page).

    Uses poppler-utils (pdftoppm) which is installed in the Docker image.
    The caller is responsible for the tmp_dir lifecycle.
    Returns the list of PNG file paths inside tmp_dir.
    """
    from pathlib import Path

    tmp_path = Path(tmp_dir) / "page"
    cmd = [
        "pdftoppm",
        "-png",
        "-r",
        "150",  # reasonable DPI for OCR
        pdf_path,
        str(tmp_path),
    ]
    # Timeout: 60 seconds per PDF to prevent hanging on corrupted/complex files
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"pdftoppm timed out after 60 seconds converting {pdf_path}. "
            "The PDF may be too large or corrupted."
        )

    # Warn on stderr output, but elevate to error if no pages were produced
    if result.stderr:
        logging.getLogger("doc-worker.paddlex_helpers").warning(
            "pdftoppm stderr: %s", result.stderr
        )

    png_files = sorted(Path(tmp_dir).glob("page-*.png"))
    if not png_files:
        raise RuntimeError(
            f"pdftoppm failed to produce any pages from {pdf_path}. "
            f"stderr: {result.stderr.strip() if result.stderr else 'none'}"
        )
    return [str(p) for p in png_files]


# ── Structure V3 pipeline ────────────────────────────────────────────────
def _create_structure_v3_pipeline() -> Any:
    """Create a PaddleX PP-StructureV3 pipeline with bundled models."""
    _ensure_paddlex_offline_compat()

    from paddlex import create_pipeline

    # Suppress logging
    logging.getLogger("paddlex").setLevel(100)

    return create_pipeline(
        "layout_parsing",
        layout_detection_model_dir=str(_model_dir(LAYOUT_DETECTION_MODEL)),
        layout_detection_model_name=LAYOUT_DETECTION_MODEL,
        text_detection_model_dir=str(_model_dir(TEXT_DETECTION_MODEL)),
        text_recognition_model_dir=str(_model_dir(TEXT_RECOGNITION_MODEL)),
        textline_orientation_model_dir=str(_model_dir(TEXTLINE_ORIENTATION_MODEL)),
        text_detection_model_name=TEXT_DETECTION_MODEL,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL,
        textline_orientation_model_name=TEXTLINE_ORIENTATION_MODEL,
        use_textline_orientation=True,
        lang=paddleocr_lang_code(),
        # Disable table recognition (not yet needed, can be added later)
        use_table_recognition=False,
    )


def _get_paddlex_structure_v3_model() -> Any:
    """Return a cached (singleton) PaddleX PP-StructureV3 pipeline.

    If initialization fails, the exception is cached and re-raised on subsequent
    calls. Uses retry logic with exponential backoff for transient errors.
    """
    with _PADDLEX_MODEL_LOCK:
        if hasattr(_get_paddlex_structure_v3_model, "_model"):  # type: ignore[attr-defined]
            return _get_paddlex_structure_v3_model._model  # type: ignore[attr-defined]

        exc = getattr(_get_paddlex_structure_v3_model, "_init_exception", None)
        if exc is not None:
            raise exc

        # Retry mechanism: up to 3 attempts with exponential backoff.
        # Permanent errors skip retry.
        max_retries = 3
        base_delay = 0.1  # seconds
        for attempt in range(max_retries):
            try:
                _get_paddlex_structure_v3_model._model = _create_structure_v3_pipeline()  # type: ignore[attr-defined]
            except Exception as e:
                # Permanent errors: don't retry — cache and fail fast
                if _is_permanent_model_init_error(e):
                    _get_paddlex_structure_v3_model._init_exception = e  # type: ignore[attr-defined]
                    raise
                if attempt == max_retries - 1:
                    _get_paddlex_structure_v3_model._init_exception = e  # type: ignore[attr-defined]
                    raise
                delay = base_delay * (2**attempt)  # 0.1s, 0.2s, 0.4s
                logging.getLogger("doc-worker.paddlex_helpers").warning(
                    "PaddleX Structure V3 model initialization failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)
            else:
                break

        return _get_paddlex_structure_v3_model._model  # type: ignore[attr-defined]


# ── OCR extraction (backward compatible) ─────────────────────────────────
def run_paddleocr(file_path: str) -> list[dict]:
    """Run PaddleX General OCR and return list of page dicts.

    Backward-compatible with the old `run_paddleocr` API — returns page dicts
    with 'page' (int), 'text' (str), and 'blocks' (list[dict] with text, bbox,
    confidence).
    """
    model = _get_paddlex_model()
    result = model.predict(file_path)

    # Safeguard: model.predict should return a list, but check anyway
    if not isinstance(result, (list, tuple)):
        raise TypeError(
            f"Expected list or tuple from model.predict, got {type(result).__name__}"
        )

    def _jsonable(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    pages: list[dict] = []
    for page_idx, page_result in enumerate(result):
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


# ── Structure V3 extraction ─────────────────────────────────────────────
def run_paddlex_structure_v3(file_path: str) -> list[dict]:
    """Run PP-StructureV3 and return list of page dicts with structured blocks.

    Each page dict contains:
        page (int): 1-based page number
        text (str): raw text from OCR (same as General OCR output)
        blocks (list[dict]): per-block text, bbox, confidence
        structured_blocks (list[dict]): blocks with type, bbox, content, confidence
    """

    model = _get_paddlex_structure_v3_model()

    # If file_path is a PDF, convert pages to temporary PNG images.
    # The temp directory must stay alive while we process the images.
    if file_path.lower().endswith(".pdf"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            images = _pdf_to_images(file_path, tmp_dir)
            if not images:
                raise RuntimeError(
                    f"Failed to convert PDF to images: {file_path}. "
                    "Check that pdftoppm is installed and the file is a valid PDF."
                )
            try:
                return _process_structure_v3_pages(model, images)
            except Exception as exc:
                raise RuntimeError(
                    f"PP-StructureV3 prediction failed on {file_path}: {exc}"
                ) from exc
    else:
        try:
            return _process_structure_v3_pages(model, [file_path])
        except Exception as exc:
            raise RuntimeError(
                f"PP-StructureV3 prediction failed on {file_path}: {exc}"
            ) from exc


def _process_structure_v3_pages(model: Any, images: list[str]) -> list[dict]:
    """Process a list of image files through the Structure V3 model.

    Internal helper used by run_paddlex_structure_v3.
    """
    pages: list[dict] = []
    for page_idx, img_path in enumerate(images):
        page_result = model.predict(img_path)

        # Extract flat text/blocks from the overall OCR result
        overall = page_result.get("overall_ocr_res", {})
        rec_texts = overall.get("rec_texts", [])
        rec_scores = overall.get("rec_scores", [])
        rec_boxes = overall.get("rec_boxes", [])
        rec_polys = overall.get("rec_polys", [])

        blocks: list[dict] = []
        text_parts: list[str] = []
        for idx, text in enumerate(rec_texts):
            if not str(text).strip():
                continue

            confidence = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
            bbox = rec_boxes[idx] if idx < len(rec_boxes) else None
            if bbox is None and idx < len(rec_polys):
                bbox = rec_polys[idx]

            # Safely convert bbox to list (avoiding mypy errors)
            safe_bbox = (
                bbox.tolist() if hasattr(bbox, "tolist") and bbox is not None else bbox
            )

            blocks.append(
                {
                    "text": str(text),
                    "bbox": safe_bbox,
                    "confidence": round(confidence, 4),
                }
            )
            text_parts.append(str(text))

        # Extract structured blocks from parsing_res_list
        # Build confidence lookup: index layout_det_res boxes by label (O(1) per block)
        layout_confidence: dict[str, float] = {}
        for layout_box in page_result.get("layout_det_res", {}).get("boxes", []):
            label = layout_box.get("label")
            if label and label not in layout_confidence:
                layout_confidence[label] = layout_box.get("score", 0.0)

        structured_blocks: list[dict] = []
        for block in page_result.get("parsing_res_list", []):
            block_label = block.get("block_label", "unknown")
            block_content = block.get("block_content", "")
            block_bbox = block.get("block_bbox", [0, 0, 0, 0])
            confidence = layout_confidence.get(block_label, 0.0)

            structured_blocks.append(
                {
                    "type": block_label,
                    "bbox": block_bbox,
                    "text": str(block_content),
                    "confidence": round(confidence, 4),
                }
            )

        pages.append(
            {
                "page": page_idx + 1,
                "text": "\n".join(text_parts),
                "blocks": blocks,
                "structured_blocks": structured_blocks,
            }
        )

    return pages


# ── Markdown builder ─────────────────────────────────────────────────────
def _convert_table_html_to_markdown(html_content: str) -> str:
    """Convert simple HTML table to Markdown table using html.parser."""
    import html as html_module
    from html.parser import HTMLParser

    class TableParser(HTMLParser):
        """Minimal HTML table parser that handles tr, td, colspan.

        Note: rowspan is not supported — if the source HTML contains rowspan,
        cells will be duplicated vertically. This is acceptable because
        PaddleX Structure V3 rarely produces rowspan in table output.
        """

        def __init__(self) -> None:
            super().__init__()
            self.in_table = False
            self.in_row = False
            self.in_cell = False
            self.table_rows: list[list[str]] = []
            self.current_row: list[str] = []
            self.current_cell = ""
            self.colspan = 1

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            attr_dict = dict(attrs)
            if tag.lower() == "table":
                self.in_table = True
            elif tag.lower() == "tr":
                self.in_row = True
            elif tag.lower() in ("td", "th"):
                if self.in_cell:
                    # Close the previous cell
                    self.current_row.append(
                        html_module.unescape(self.current_cell.strip())
                    )
                colspan_value = attr_dict.get("colspan") or 1
                self.colspan = int(colspan_value)
                self.in_cell = True
                self.current_cell = ""
                # If this is a header cell, we'll bold the content later
                self.is_header = tag.lower() == "th"

        def handle_endtag(self, tag: str) -> None:
            if tag.lower() == "table":
                self.in_table = False
            elif tag.lower() == "tr":
                # Append any unclosed cell, then finish the row
                if self.in_cell:
                    cell_text = html_module.unescape(self.current_cell.strip())
                    if getattr(self, "is_header", False):
                        cell_text = f"**{cell_text}**"
                    self.current_row.append(cell_text)
                    self.in_cell = False
                self.table_rows.append(list(self.current_row))  # snapshot
                self.current_row = []
                self.in_row = False
            elif tag.lower() in ("td", "th"):
                # Append cell content, bold if it was a header
                cell_text = html_module.unescape(self.current_cell.strip())
                if getattr(self, "is_header", False):
                    cell_text = f"**{cell_text}**"
                colspan = getattr(self, "colspan", 1)
                for _ in range(colspan):
                    self.current_row.append(cell_text)
                self.colspan = 1  # reset for next cell
                self.in_cell = False

        def handle_data(self, data: str) -> None:
            if self.in_cell:
                self.current_cell += data

    parser = TableParser()
    try:
        parser.feed(html_content)
    except Exception:
        # Graceful fallback — return empty string on parse errors
        return ""

    rows = parser.table_rows
    if not rows:
        return ""

    # Expand colspans into empty placeholder columns and determine max column count
    # Note: The HTML parser duplicates text for colspan > 1, so we use that duplicated content.
    expanded_rows: list[list[str]] = list(rows)

    max_cols = max(len(r) for r in expanded_rows) if expanded_rows else 1

    # Pad all rows to max_cols
    padded_rows = []
    for row in expanded_rows:
        padded_rows.append(row + [""] * (max_cols - len(row)))

    def markdown_row(row: list[str]) -> str:
        return "|" + "|".join(f" {c} " for c in row) + "|"

    md_lines = []
    for i, row in enumerate(padded_rows):
        md_lines.append(markdown_row(row))
        if i == 0:
            md_lines.append("|" + "|".join(["------"] * max_cols) + "|")

    return "\n".join(md_lines)


def _compute_reading_order_key(
    block: dict, column_clusters: list[float] | None
) -> tuple:
    """Return a sort key for a single block given pre-computed column clusters.

    If column_clusters is None or has only one column, uses simple (y, x) sort.
    Otherwise assigns the block to its nearest column and returns (y, col_index).
    """
    bx = block["bbox"][0]
    by = block["bbox"][1]
    if column_clusters is None or len(column_clusters) <= 1:
        return (by, bx)

    col_idx = min(
        range(len(column_clusters)), key=lambda i: abs(column_clusters[i] - bx)
    )
    return (by, col_idx)


def _compute_column_clusters(
    lefts: list[float], threshold: float = 50.0
) -> list[float] | None:
    """Cluster left x-coordinates into column centers.

    Returns None if there's only one column or no blocks.
    """
    if not lefts:
        return None
    lefts_sorted = sorted(lefts)
    columns: list[float] = [lefts_sorted[0]]
    for lx in lefts_sorted[1:]:
        if lx - columns[-1] > threshold:
            columns.append(lx)
    return columns if len(columns) > 1 else None


def _sort_blocks_by_reading_order(
    structured_blocks: list[dict], column_threshold: float = 50.0
) -> list[dict]:
    """Sort blocks by estimated reading order (handles multi-column layouts).

    Efficiently clusters columns once, then assigns each block to a column in O(1).
    Total complexity: O(n log n) instead of O(n²).

    Args:
        structured_blocks: List of block dicts with 'bbox' keys.
        column_threshold: Min x-distance to consider as separate columns (default 50px).

    Returns:
        New list of blocks sorted by reading order.
    """
    if len(structured_blocks) < 2:
        return sorted(structured_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    # Cluster left coordinates into column centers (single pass)
    lefts = [b["bbox"][0] for b in structured_blocks]
    column_clusters = _compute_column_clusters(lefts, threshold=column_threshold)

    # Sort using pre-computed clusters — O(n log n) with O(1) column assignment
    return sorted(
        structured_blocks,
        key=lambda b: _compute_reading_order_key(b, column_clusters),
    )


def blocks_to_markdown(structured_blocks: list[dict]) -> str:
    """Convert structured blocks to semantic Markdown.

    Mapping:
        - title / paragraph_title → # Heading / ## Subheading
        - figure_title → captured and used for following image alt-text
        - text → regular paragraph
        - table → Markdown table (HTML converted)
        - image → [Image] or [Figure: caption] if a figure_title precedes it

    Sorting: blocks are ordered by estimated reading order, accounting for
    multi-column layouts by clustering on x-coordinate.
    """
    lines: list[str] = []
    last_figure_title = ""

    # Sort blocks by estimated reading order (handles multi-column layouts)
    sorted_blocks = _sort_blocks_by_reading_order(
        structured_blocks, column_threshold=50.0
    )

    for block in sorted_blocks:
        block_type = block["type"]
        content = block["text"]

        # Image blocks may have empty content — we still render them
        if block_type != "image" and not content.strip():
            continue

        if block_type in ("title", "paragraph_title"):
            # Title
            level = 1 if block_type == "title" else 2
            prefix = "#" * level
            lines.append(f"\n{prefix} {content}\n")
            last_figure_title = ""

        elif block_type == "figure_title":
            # Capture for a following image block; if none follows, it will
            # be emitted as italic text after the loop
            last_figure_title = content

        elif block_type == "text":
            lines.append(f"\n{content}\n")

        elif block_type == "table":
            # Table: use block_content which may contain HTML
            if content.startswith(("<html", "<table")):
                table_md = _convert_table_html_to_markdown(content)
                lines.append(f"\n{table_md}\n")
            else:
                lines.append(f"\n{content}\n")

        elif block_type == "image":
            # Use preceding figure_title as alt text, or generic placeholder
            figure_text = last_figure_title if last_figure_title else "Image"
            lines.append(f"\n![{figure_text}]\n")
            last_figure_title = ""

        else:
            # Unknown type, just add as plain text
            lines.append(f"\n{content}\n")

    # If there's a leftover figure_title that was never consumed by an image,
    # emit it as italic text so the caption is preserved.
    if last_figure_title:
        lines.append(f"\n*{last_figure_title}*\n")

    return "\n".join(lines)
