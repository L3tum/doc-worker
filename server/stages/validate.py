"""
Stage 1: Input Validation & Triage
====================================

Verifies the document is processable and determines the processing path:
- Validate file type (PDF, image formats)
- Check file size against MAX_FILE_SIZE
- Determine document type (text PDF, scanned, hybrid, image)
- Set processing_mode for downstream stages
"""

from __future__ import annotations

from pathlib import Path

from server.companion import FileIO, get_config, get_logger
from server.models import (
    DocumentType, Job, JobContext, ProcessingMode,
)
from server.orchestrator import StageValidationError


def validate(job: Job, ctx: JobContext) -> JobContext:
    """Run input validation and triage.

    Raises:
        StageValidationError: If the input is invalid (non-retryable).
    """
    logger = get_logger("doc-worker.stages.validate")
    config = get_config()
    filename = job.input.filename

    logger.info(f"Validating: {filename}")

    # 1. Validate file type
    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".webp"):
        raise StageValidationError(
            f"Unsupported file type: {ext}",
            stage="validate",
        )

    # 2. Read content and check size
    content = FileIO.read(job.input.path, job.input.data)
    size_mb = len(content) / (1024 * 1024)

    if size_mb > config.MAX_FILE_SIZE_MB:
        raise StageValidationError(
            f"File too large: {size_mb:.1f}MB > {config.MAX_FILE_SIZE_MB}MB",
            stage="validate",
        )

    # 3. Determine document type
    if ext == ".pdf":
        doc_type = _classify_pdf(content)
    else:
        doc_type = DocumentType.IMAGE

    # 4. Set processing mode
    if doc_type == DocumentType.TEXT_PDF:
        mode = ProcessingMode.SKIP
    elif doc_type == DocumentType.SCANNED_PDF or doc_type == DocumentType.IMAGE:
        mode = ProcessingMode.FULL_OCR
    elif doc_type == DocumentType.HYBRID_PDF:
        mode = ProcessingMode.OVERLAY
    else:
        mode = ProcessingMode.FULL_OCR

    # 5. Update context
    ctx.document_type = doc_type
    ctx.processing_mode = mode

    logger.info(
        f"Validation OK: {filename} — type={doc_type.value}, "
        f"mode={mode.value}, size={size_mb:.1f}MB"
    )

    return ctx


def _classify_pdf(content: bytes) -> DocumentType:
    """Classify a PDF as text-based, scanned, or hybrid.

    Uses a simple heuristic: check if the PDF contains extractable text
    by looking for text operators in the raw content.
    """
    # Check for basic PDF markers
    if not content.startswith(b"%PDF"):
        raise StageValidationError(
            "File does not appear to be a valid PDF",
            stage="validate",
        )

    # Simple heuristic: search for text-related PDF operators
    # BT...ET blocks contain text, Tj and TJ operators render text
    has_text_operators = (
        b"Tj" in content or b"TJ" in content or b"' " in content
    )

    # Check for /Text in resource dictionaries (indicates text content)
    has_text_resources = b"/Text" in content

    if has_text_operators or has_text_resources:
        # Could be hybrid — check for image content too
        has_images = b"/Image" in content or b"/XObject" in content
        if has_images and has_text_operators:
            return DocumentType.HYBRID_PDF
        return DocumentType.TEXT_PDF

    return DocumentType.SCANNED_PDF