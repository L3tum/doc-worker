"""
Tests for server.stages.validate — input validation and triage.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from server.models import DocumentInput, DocumentType, Job, JobContext, ProcessingMode
from server.orchestrator import StageValidationError
from server.stages.validate import validate, _classify_pdf


class TestValidate:
    """Test the validate stage."""

    def _make_job(self, temp_dir, content, filename="test.pdf", source="folder"):
        """Helper to create a test job."""
        if source == "folder":
            file_path = temp_dir / filename
            file_path.write_bytes(content)
            return Job(input=DocumentInput(
                filename=filename,
                source=source,
                path=file_path,
            ))
        else:
            return Job(input=DocumentInput(
                filename=filename,
                source=source,
                data=content,
            ))

    def test_valid_pdf(self, temp_dir, sample_pdf_bytes):
        """Test validation of a valid PDF."""
        job = self._make_job(temp_dir, sample_pdf_bytes)
        ctx = JobContext(job_id=job.job_id)

        result = validate(job, ctx)

        assert result.document_type == DocumentType.SCANNED_PDF
        assert result.processing_mode == ProcessingMode.FULL_OCR

    def test_valid_image_png(self, temp_dir, sample_png_bytes):
        """Test validation of a valid PNG image."""
        job = self._make_job(temp_dir, sample_png_bytes, filename="test.png")
        ctx = JobContext(job_id=job.job_id)

        result = validate(job, ctx)

        assert result.document_type == DocumentType.IMAGE
        assert result.processing_mode == ProcessingMode.FULL_OCR

    def test_unsupported_file_type(self, temp_dir):
        """Test rejection of unsupported file type."""
        job = self._make_job(temp_dir, b"content", filename="test.docx")
        ctx = JobContext(job_id=job.job_id)

        with pytest.raises(StageValidationError, match="Unsupported file type"):
            validate(job, ctx)

    def test_file_too_large(self, temp_dir, monkeypatch):
        """Test rejection of files exceeding size limit."""
        # Create a large file
        large_content = b"x" * (101 * 1024 * 1024)  # 101 MB
        job = self._make_job(temp_dir, large_content)
        ctx = JobContext(job_id=job.job_id)

        # Patch config to have 100 MB limit
        with patch("server.stages.validate.get_config") as mock_config:
            mock_config.return_value.MAX_FILE_SIZE_MB = 100
            with pytest.raises(StageValidationError, match="File too large"):
                validate(job, ctx)

    def test_text_pdf_classification(self, temp_dir):
        """Test classification of text-based PDF."""
        # PDF with text operators
        text_pdf = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
  /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
trailer << /Size 6 /Root 1 0 R >>
startxref
0
%%EOF"""

        job = self._make_job(temp_dir, text_pdf)
        ctx = JobContext(job_id=job.job_id)

        result = validate(job, ctx)

        assert result.document_type == DocumentType.TEXT_PDF
        assert result.processing_mode == ProcessingMode.SKIP

    def test_invalid_pdf_header(self, temp_dir):
        """Test rejection of invalid PDF."""
        job = self._make_job(temp_dir, b"not a pdf file")
        ctx = JobContext(job_id=job.job_id)

        with pytest.raises(StageValidationError, match="not appear to be a valid PDF"):
            validate(job, ctx)


class TestClassifyPDF:
    """Test _classify_pdf() function."""

    def test_scanned_pdf(self):
        """Test classification of scanned PDF (no text operators)."""
        scanned_pdf = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
xref
0 4
trailer << /Size 4 /Root 1 0 R >>
startxref
0
%%EOF"""

        result = _classify_pdf(scanned_pdf)
        assert result == DocumentType.SCANNED_PDF

    def test_text_pdf(self):
        """Test classification of text PDF."""
        text_pdf = b"""%PDF-1.4
stream
BT /F1 12 Tf (Hello) Tj ET
endstream"""

        result = _classify_pdf(text_pdf)
        assert result == DocumentType.TEXT_PDF

    def test_hybrid_pdf(self):
        """Test classification of hybrid PDF (text + images)."""
        hybrid_pdf = b"""%PDF-1.4
stream
BT /F1 12 Tf (Hello) Tj ET
endstream
/XObject /Image"""

        result = _classify_pdf(hybrid_pdf)
        assert result == DocumentType.HYBRID_PDF

    def test_invalid_pdf_raises(self):
        """Test that invalid PDF raises StageValidationError."""
        with pytest.raises(StageValidationError, match="not appear to be a valid PDF"):
            _classify_pdf(b"not a pdf")