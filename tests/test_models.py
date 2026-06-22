"""
Tests for server.models — data models and enums.
"""

from __future__ import annotations

import time

from server.models import (
    DocumentInput, DocumentType, Job, JobContext, JobState,
    ProcessingMode, StageOutput,
)


class TestJobState:
    """Test JobState enum transitions."""

    def test_all_states_exist(self):
        assert JobState.QUEUED.value == "queued"
        assert JobState.VALIDATING.value == "validating"
        assert JobState.PROCESSING.value == "processing"
        assert JobState.DELIVERING.value == "delivering"
        assert JobState.COMPLETED.value == "completed"
        assert JobState.FAILED.value == "failed"


class TestDocumentType:
    """Test DocumentType enum."""

    def test_all_types_exist(self):
        assert DocumentType.UNKNOWN.value == "unknown"
        assert DocumentType.TEXT_PDF.value == "text_pdf"
        assert DocumentType.SCANNED_PDF.value == "scanned_pdf"
        assert DocumentType.HYBRID_PDF.value == "hybrid_pdf"
        assert DocumentType.IMAGE.value == "image"


class TestProcessingMode:
    """Test ProcessingMode enum."""

    def test_all_modes_exist(self):
        assert ProcessingMode.SKIP.value == "skip"
        assert ProcessingMode.FULL_OCR.value == "full_ocr"
        assert ProcessingMode.OVERLAY.value == "overlay"


class TestDocumentInput:
    """Test DocumentInput dataclass."""

    def test_from_path(self, temp_dir, sample_pdf_bytes):
        """Test DocumentInput from filesystem path."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(sample_pdf_bytes)

        doc = DocumentInput(
            filename="test.pdf",
            source="folder",
            path=pdf_path,
        )

        assert doc.filename == "test.pdf"
        assert doc.source == "folder"
        assert doc.path == pdf_path
        assert doc.data is None
        assert doc.content_hash != "unknown"

    def test_from_bytes(self, sample_pdf_bytes):
        """Test DocumentInput from bytes."""
        doc = DocumentInput(
            filename="test.pdf",
            source="api",
            data=sample_pdf_bytes,
        )

        assert doc.filename == "test.pdf"
        assert doc.source == "api"
        assert doc.path is None
        assert doc.data == sample_pdf_bytes
        assert doc.content_hash != "unknown"

    def test_content_hash_consistency(self, sample_pdf_bytes):
        """Test that content_hash is consistent for same data."""
        doc1 = DocumentInput(filename="a.pdf", source="api", data=sample_pdf_bytes)
        doc2 = DocumentInput(filename="b.pdf", source="api", data=sample_pdf_bytes)

        assert doc1.content_hash == doc2.content_hash


class TestJobContext:
    """Test JobContext dataclass."""

    def test_defaults(self):
        ctx = JobContext(job_id="test-123")

        assert ctx.job_id == "test-123"
        assert ctx.document_type == DocumentType.UNKNOWN
        assert ctx.processing_mode == ProcessingMode.FULL_OCR
        assert isinstance(ctx.outputs, StageOutput)
        assert ctx.errors == []
        assert ctx.timings == {}
        assert ctx.extra == {}


class TestJob:
    """Test Job dataclass."""

    def test_creation(self):
        job = Job()

        assert len(job.job_id) == 12
        assert job.state == JobState.QUEUED
        assert job.retry_count == 0
        assert job.max_retries == 3
        assert job.started_at is None
        assert job.completed_at is None

    def test_transition(self):
        job = Job()

        job.transition(JobState.VALIDATING)
        assert job.state == JobState.VALIDATING
        assert job.updated_at >= job.created_at

        job.transition(JobState.COMPLETED)
        assert job.state == JobState.COMPLETED
        assert job.completed_at is not None

    def test_elapsed_seconds(self):
        job = Job()
        time.sleep(0.1)
        job.transition(JobState.COMPLETED)

        assert job.elapsed_seconds >= 0.1

    def test_post_init_context_link(self):
        """Test that __post_init__ links context to job."""
        doc = DocumentInput(filename="test.pdf", source="folder")
        job = Job(input=doc)

        assert job.context.job_id == job.job_id


class TestStageOutput:
    """Test StageOutput dataclass."""

    def test_defaults(self):
        output = StageOutput()

        assert output.ocr_pdf is None
        assert output.markdown is None
        assert output.json_metadata is None
        assert output.manifest is None

    def test_with_data(self):
        output = StageOutput(
            ocr_pdf=b"%PDF-1.4...",
            markdown="# Test",
            json_metadata={"key": "value"},
            manifest={"job_id": "123"},
        )

        assert output.ocr_pdf == b"%PDF-1.4..."
        assert output.markdown == "# Test"
        assert output.json_metadata == {"key": "value"}
        assert output.manifest == {"job_id": "123"}