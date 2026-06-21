"""
Stage 2: OCR Processing
========================

Runs the selected OCR model and produces structured output:
- OCR'd PDF (searchable PDF with text layer)
- Markdown (structured text extraction)
- JSON metadata (layout, tables, confidence scores)

Currently uses OCRmyPDF + RapidOCR for backward compatibility.
Will be updated in Phase 4 to use PaddleOCR directly.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from server.companion import (
    FileIO, get_config, get_logger, get_model_manager,
)
from server.models import Job, JobContext, ProcessingMode
from server.orchestrator import StageError


def ocr(job: Job, ctx: JobContext) -> JobContext:
    """Run OCR processing on the document.

    Args:
        job: The current job.
        ctx: The job context.

    Returns:
        Updated context with OCR outputs.

    Raises:
        StageError: If OCR processing fails.
    """
    logger = get_logger("doc-worker.stages.ocr")
    config = get_config()
    filename = job.input.filename

    logger.info(f"OCR processing: {filename} (mode={ctx.processing_mode.value})")

    # Read input content
    content = FileIO.read(job.input.path, job.input.data)

    try:
        if ctx.processing_mode == ProcessingMode.SKIP:
            # Text-based PDF — pass through, extract text directly
            ctx.outputs.ocr_pdf = content
            ctx.outputs.markdown = _extract_text_simple(content, filename)
            ctx.outputs.json_metadata = {
                "processing_mode": "skip",
                "document_type": ctx.document_type.value,
                "filename": filename,
            }
            logger.info(f"OCR skipped (text PDF): {filename}")

        elif ctx.processing_mode in (ProcessingMode.FULL_OCR, ProcessingMode.OVERLAY):
            # Run OCR pipeline
            result = _run_ocr_pipeline(content, filename, config)
            ctx.outputs.ocr_pdf = result.get("pdf")
            ctx.outputs.markdown = result.get("markdown")
            ctx.outputs.json_metadata = result.get("metadata")

            logger.info(f"OCR complete: {filename}")

        else:
            raise StageError(
                f"Unknown processing mode: {ctx.processing_mode}",
                stage="ocr",
            )

    except StageError:
        raise
    except Exception as exc:
        raise StageError(
            f"OCR processing failed: {exc}",
            retryable=True,
            stage="ocr",
        )

    return ctx


def _run_ocr_pipeline(
    content: bytes, filename: str, config
) -> dict[str, object]:
    """Run the OCR pipeline.

    Currently uses OCRmyPDF + RapidOCR via the existing worker.py logic.
    Will be replaced with direct PaddleOCR in Phase 4.
    """
    logger = get_logger("doc-worker.stages.ocr")

    # Write to temp file for OCRmyPDF
    with tempfile.NamedTemporaryFile(
        suffix=".pdf" if filename.lower().endswith(".pdf") else ".png",
        delete=False
    ) as tmp_in:
        tmp_in.write(content)
        tmp_in_path = Path(tmp_in.name)

    tmp_out = tmp_in.with_suffix("_ocr.pdf")

    try:
        # Import and run OCRmyPDF
        # We import here to avoid import errors if ocrmypdf is not installed
        # (e.g., during Phase 4 migration)
        try:
            import ocrmypdf
        except ImportError:
            logger.warning(
                "ocrmypdf not installed — returning input as-is. "
                "Install ocrmypdf or wait for Phase 4 PaddleOCR integration."
            )
            return {
                "pdf": content,
                "markdown": "",
                "metadata": {
                    "processing_mode": "passthrough",
                    "reason": "ocrmypdf not installed",
                    "filename": filename,
                },
            }

        # Configure RapidOCR runtime (from existing worker.py)
        _configure_rapidocr_runtime()

        # Run OCR
        ocrmypdf.ocr(
            str(tmp_in_path),
            str(tmp_out),
            plugins=["ocrmypdf_rapidocr"],
            language=config.OCR_LANG,
            force_ocr=True,
            rapidocr_config_path=os.environ.get("RAPIDOCR_CONFIG"),
        )

        # Read output
        ocr_pdf = tmp_out.read_bytes() if tmp_out.exists() else content

        return {
            "pdf": ocr_pdf,
            "markdown": "",  # Will be populated by PaddleOCR-VL in Phase 4
            "metadata": {
                "processing_mode": ctx.processing_mode.value,
                "document_type": "scanned",
                "engine": "ocrmypdf+rapidocr",
                "language": config.OCR_LANG,
                "filename": filename,
            },
        }

    finally:
        # Clean up temp files
        for p in (tmp_in_path, tmp_out):
            if p.exists():
                p.unlink()


# ---------------------------------------------------------------------------
# RapidOCR runtime configuration (ported from worker.py)
# ---------------------------------------------------------------------------

_rapidocr_configured = False
_rapidocr_params: dict[str, object] = {}


def _configure_rapidocr_runtime() -> None:
    """Configure RapidOCR for the selected runtime backend."""
    global _rapidocr_configured, _rapidocr_params
    if _rapidocr_configured:
        return
    _rapidocr_configured = True

    config = get_config()
    runtime = config.OCR_RUNTIME

    BACKENDS = {
        "cpu": ("CPUExecutionProvider", None),
        "cuda": ("CUDAExecutionProvider", "onnxruntime-gpu"),
        "openvino": ("OpenVINOExecutionProvider", "onnxruntime-openvino"),
        "rocm": ("ROCMExecutionProvider", "onnxruntime-rocm"),
    }

    if runtime not in BACKENDS:
        runtime = "cpu"

    target_provider, _ = BACKENDS[runtime]

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if target_provider not in available:
            runtime = "cpu"
            target_provider = "CPUExecutionProvider"
    except Exception:
        runtime = "cpu"
        target_provider = "CPUExecutionProvider"

    if runtime == "cuda":
        _rapidocr_params = {"EngineConfig.onnxruntime.use_cuda": True}
    elif runtime in ("openvino", "rocm"):
        os.environ["RAPIDOCR_ONNXRUNTIME_PROVIDER"] = target_provider

    # Patch RapidOCR if available
    try:
        import rapidocr
        _orig_init = rapidocr.RapidOCR.__init__

        def _patched_init(
            self, config_path=None, params=None
        ):
            merged = dict(_rapidocr_params)
            if params:
                merged.update(params)
            _orig_init(self, config_path=config_path, params=merged or None)

        rapidocr.RapidOCR.__init__ = _patched_init
    except ImportError:
        pass


def _extract_text_simple(content: bytes, filename: str) -> str:
    """Simple text extraction for text-based PDFs.

    Placeholder — will use PaddleOCR-VL in Phase 4.
    """
    return ""