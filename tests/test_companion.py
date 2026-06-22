"""
Tests for server.companion — configuration, logging, metrics, file I/O.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from server.companion import (
    Config,
    FileIO,
    HealthStatus,
    JsonFormatter,
    Metrics,
)


class TestConfig:
    """Test Config dataclass and from_env()."""

    def test_defaults(self):
        config = Config()

        assert config.INBOX == Path("/work/inbox")
        assert config.PROCESSING == Path("/work/processing")
        assert config.DONE == Path("/work/done")
        assert config.ERROR == Path("/work/error")
        assert config.OUTPUT_DIR == Path("/work/output")
        assert config.PAPERLESS_CONSUME == Path("/paperless-consume")
        assert config.OCR_LANG == "deu"
        assert config.OCR_RUNTIME == "cpu"
        assert config.PROCESSING_MODE == "auto"
        assert config.POLL_INTERVAL == 5
        assert config.MAX_RETRIES == 3
        assert config.MAX_FILE_SIZE_MB == 100
        assert config.API_HOST == "0.0.0.0"
        assert config.API_PORT == 8000

    def test_from_env(self, temp_dir, monkeypatch):
        """Test Config.from_env() reads from environment variables."""
        monkeypatch.setenv("INBOX", str(temp_dir / "inbox"))
        monkeypatch.setenv("OCR_LANG", "eng")
        monkeypatch.setenv("PROCESSING_MODE", "paddle_vl")
        monkeypatch.setenv("MAX_FILE_SIZE_MB", "50")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        config = Config.from_env()

        assert config.INBOX == temp_dir / "inbox"
        assert config.OCR_LANG == "eng"
        assert config.PROCESSING_MODE == "paddle_vl"
        assert config.MAX_FILE_SIZE_MB == 50
        assert config.LOG_LEVEL == "DEBUG"

    def test_ensure_directories(self, temp_dir):
        """Test ensure_directories() creates all required directories."""
        config = Config(
            INBOX=temp_dir / "inbox",
            PROCESSING=temp_dir / "processing",
            DONE=temp_dir / "done",
            ERROR=temp_dir / "error",
            OUTPUT_DIR=temp_dir / "output",
            PAPERLESS_CONSUME=temp_dir / "paperless",
        )

        config.ensure_directories()

        assert (temp_dir / "inbox").exists()
        assert (temp_dir / "processing").exists()
        assert (temp_dir / "done").exists()
        assert (temp_dir / "error").exists()
        assert (temp_dir / "output").exists()
        assert (temp_dir / "paperless").exists()


class TestFileIO:
    """Test FileIO abstraction."""

    def test_read_from_path(self, temp_dir, sample_pdf_bytes):
        """Test reading from a filesystem path."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(sample_pdf_bytes)

        data = FileIO.read(pdf_path, None)
        assert data == sample_pdf_bytes

    def test_read_from_bytes(self, sample_pdf_bytes):
        """Test reading from in-memory bytes."""
        data = FileIO.read(None, sample_pdf_bytes)
        assert data == sample_pdf_bytes

    def test_read_no_data_raises(self):
        """Test that reading with no data raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="No input data available"):
            FileIO.read(None, None)

    def test_write(self, temp_dir):
        """Test writing bytes to a file."""
        test_data = b"Hello, World!"
        output_path = temp_dir / "output.bin"

        FileIO.write(output_path, test_data)

        assert output_path.read_bytes() == test_data

    def test_write_creates_parent_dirs(self, temp_dir):
        """Test that write creates parent directories."""
        test_data = b"Hello, World!"
        output_path = temp_dir / "nested" / "dir" / "output.bin"

        FileIO.write(output_path, test_data)

        assert output_path.read_bytes() == test_data

    def test_write_text(self, temp_dir):
        """Test writing text to a file."""
        test_text = "Hello, World!"
        output_path = temp_dir / "output.txt"

        FileIO.write_text(output_path, test_text)

        assert output_path.read_text() == test_text

    def test_atomic_write(self, temp_dir):
        """Test atomic write (write to .tmp then rename)."""
        test_data = b"Hello, World!"
        output_path = temp_dir / "output.bin"

        FileIO.atomic_write(output_path, test_data)

        assert output_path.read_bytes() == test_data
        assert not (temp_dir / "output.bin.tmp").exists()

    def test_atomic_write_failure_cleanup(self, temp_dir):
        """Test that atomic write cleans up on failure."""
        test_data = b"Hello, World!"
        output_path = temp_dir / "nonexistent" / "output.bin"

        # This should fail because parent doesn't exist for atomic write
        # (the .tmp file can't be created)
        try:
            FileIO.atomic_write(output_path, test_data)
        except Exception:
            pass

        # No .tmp file should be left behind
        tmp_path = output_path.with_suffix(".bin.tmp")
        assert not tmp_path.exists()


class TestMetrics:
    """Test Metrics tracker."""

    def test_initial_state(self):
        metrics = Metrics()

        assert metrics.files_processed == 0
        assert metrics.files_failed == 0
        assert metrics.total_processing_time == 0.0
        assert metrics.stage_timings == {}
        assert metrics.errors == []

    def test_record_stage_timing(self):
        metrics = Metrics()

        metrics.record_stage_timing("validate", 0.5)
        metrics.record_stage_timing("validate", 0.3)
        metrics.record_stage_timing("ocr", 2.0)

        assert metrics.stage_timings["validate"] == [0.5, 0.3]
        assert metrics.stage_timings["ocr"] == [2.0]

    def test_record_success(self):
        metrics = Metrics()

        metrics.record_success(1.5)
        metrics.record_success(2.5)

        assert metrics.files_processed == 2
        assert metrics.total_processing_time == 4.0

    def test_record_failure(self):
        metrics = Metrics()

        metrics.record_failure("Error 1")
        metrics.record_failure("Error 2")

        assert metrics.files_failed == 2
        assert metrics.errors == ["Error 1", "Error 2"]

    def test_summary(self):
        metrics = Metrics()
        metrics.record_success(2.0)
        metrics.record_success(4.0)
        metrics.record_failure("Test error")

        summary = metrics.summary()

        assert summary["files_processed"] == 2
        assert summary["files_failed"] == 1
        assert summary["total_processing_time"] == 6.0
        assert summary["avg_processing_time"] == 3.0
        assert summary["errors"] == ["Test error"]


class TestJsonFormatter:
    """Test JSON log formatter."""

    def test_format_basic(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        log_entry = json.loads(output)

        assert log_entry["level"] == "INFO"
        assert log_entry["logger"] == "test.logger"
        assert log_entry["message"] == "Test message"
        assert "timestamp" in log_entry

    def test_format_with_extra_fields(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        # Add extra fields as attributes on the record
        record.job_id = "abc123"
        record.stage = "ocr"
        record.filename = "test.pdf"

        output = formatter.format(record)
        log_entry = json.loads(output)

        assert log_entry["job_id"] == "abc123"
        assert log_entry["stage"] == "ocr"
        assert log_entry["filename"] == "test.pdf"


class TestHealthStatus:
    """Test HealthStatus dataclass."""

    def test_defaults(self):
        health = HealthStatus()

        assert health.status == "healthy"
        assert health.ready is True
        assert health.model_loaded is False
        assert health.uptime == 0.0
        assert health.metrics == {}
