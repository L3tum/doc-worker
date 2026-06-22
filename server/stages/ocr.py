"""
Stage 2: OCR Processing
========================

Runs the selected OCR model and produces structured output:
- OCR'd PDF (searchable PDF with text layer)
- Markdown (structured text extraction)
- JSON metadata (layout, tables, confidence scores)

Phase 4: Uses PaddleOCR (PP-OCRv6 + PaddleOCR-VL) as the primary engine.
OCRmyPDF is used as a wrapper to embed text layers into PDFs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

from server.companion import (
    FileIO,
    get_config,
    get_logger,
    get_model_manager,
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
            ctx.outputs.markdown = _extract_text_from_pdf(content)
            ctx.outputs.json_metadata = {
                "processing_mode": "skip",
                "document_type": ctx.document_type.value,
                "filename": filename,
            }
            logger.info(f"OCR skipped (text PDF): {filename}")

        elif ctx.processing_mode in (ProcessingMode.FULL_OCR, ProcessingMode.OVERLAY):
            # Run PaddleOCR pipeline
            result = _run_paddle_ocr_pipeline(content, filename, ctx, config)
            ctx.outputs.ocr_pdf = cast(bytes | None, result.get("pdf"))
            ctx.outputs.markdown = cast(str | None, result.get("markdown"))
            ctx.outputs.json_metadata = cast(
                dict[str, Any] | None, result.get("metadata")
            )

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


def _run_paddle_ocr_pipeline(
    content: bytes, filename: str, ctx: JobContext, config
) -> dict[str, Any]:
    """Run the PaddleOCR pipeline.

    Steps:
    1. Run PP-OCR for text detection and recognition
    2. Optionally run PaddleOCR-VL for document understanding
    3. Embed text layer into PDF using OCRmyPDF or direct embedding
    4. Generate markdown output

    Returns:
        Dict with pdf bytes, markdown string, and metadata dict.
    """
    logger = get_logger("doc-worker.stages.ocr")
    model_mgr = get_model_manager()

    # Write input to temp file
    suffix = ".pdf" if filename.lower().endswith(".pdf") else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(content)
        tmp_in_path = Path(tmp_in.name)

    tmp_out = tmp_in.with_suffix("_ocr.pdf")

    try:
        # Step 1: Run PP-OCR for text detection/recognition
        logger.info("Running PP-OCR text detection...")
        ocr_result = model_mgr.run_ocr(tmp_in_path, ctx)

        # Step 2: Run PaddleOCR-VL for document understanding (if configured)
        markdown = ""
        vl_result = None
        if config.PROCESSING_MODE in ("paddle_vl", "auto"):
            logger.info("Running PaddleOCR-VL document understanding...")
            try:
                vl_result = model_mgr.run_vl_understanding(tmp_in_path, ctx)
                markdown = vl_result.get("markdown", "")
            except Exception as exc:
                logger.warning(f"PaddleOCR-VL failed, using PP-OCR text only: {exc}")
                markdown = ocr_result.get("text", "")

        if not markdown:
            markdown = ocr_result.get("text", "")

        # Step 3: Embed text layer into PDF
        ocr_pdf = _embed_text_layer(tmp_in_path, tmp_out, ocr_result, config)

        # Step 4: Build metadata
        metadata = {
            "processing_mode": ctx.processing_mode.value,
            "document_type": ctx.document_type.value,
            "engine": "paddleocr",
            "pp_ocr": {
                "block_count": ocr_result.get("block_count", 0),
                "avg_confidence": ocr_result.get("avg_confidence", 0.0),
                "model": "pp_ocr",
            },
            "language": config.OCR_LANG,
            "filename": filename,
        }

        if vl_result:
            metadata["paddle_vl"] = {
                "tables_count": len(vl_result.get("tables", [])),
                "formulas_count": len(vl_result.get("formulas", [])),
                "layout_blocks": len(vl_result.get("layout", [])),
                "model": "paddle_vl",
            }

        return {
            "pdf": ocr_pdf,
            "markdown": markdown,
            "metadata": metadata,
        }

    finally:
        # Clean up temp files
        for p in (tmp_in_path, tmp_out):
            if p.exists():
                p.unlink()


def _embed_text_layer(
    input_path: Path, output_path: Path, ocr_result: dict, config
) -> bytes:
    """Embed the OCR text layer into the PDF.

    Uses OCRmyPDF with the PaddleOCR engine if available,
    otherwise falls back to direct text embedding.
    """
    logger = get_logger("doc-worker.stages.ocr")

    # Try OCRmyPDF first (it can use PaddleOCR as a backend)
    try:
        import ocrmypdf

        ocrmypdf.ocr(
            str(input_path),
            str(output_path),
            language=config.OCR_LANG,
            force_ocr=True,
            # Skip deskew, clean, and optimize for speed
            skip_textual=True,  # Don't re-OCR pages that already have text
            output_type="pdfa",  # PDF/A for archival
        )

        if output_path.exists():
            return output_path.read_bytes()

    except ImportError:
        logger.warning("ocrmypdf not installed, using direct text embedding")
    except Exception as exc:
        logger.warning(f"ocrmypdf failed, using direct text embedding: {exc}")

    # Fallback: return the original content with text metadata
    # In a production environment, you'd want a proper PDF text embedding library
    logger.warning(
        "Returning original PDF without embedded text layer. "
        "Install ocrmypdf for proper text embedding."
    )
    return input_path.read_bytes()


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from a text-based PDF.

    Uses PyPDF2/pypdf for text extraction.
    """
    try:
        try:
            from pypdf import PdfReader as _PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader as _PdfReader
            except ImportError:
                _PdfReader = None  # type: ignore[assignment]

        import io

        if _PdfReader is None:
            get_logger("doc-worker.stages.ocr").warning(
                "pypdf/PyPDF2 not installed, cannot extract text from PDF"
            )
            return ""

        reader = _PdfReader(io.BytesIO(content))
        text_parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        return "\n\n".join(text_parts)

    except Exception as exc:
        get_logger("doc-worker.stages.ocr").warning(f"Text extraction failed: {exc}")
        return ""
