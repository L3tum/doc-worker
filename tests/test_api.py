"""
Tests for server.api — FastAPI endpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from server.api import app, set_orchestrator
from server.orchestrator import Orchestrator


@pytest.fixture
def client(mock_config, mock_model_manager):
    """Create a test client with mocked orchestrator."""
    # Mock the orchestrator
    orch = MagicMock(spec=Orchestrator)
    orch.can_accept = True
    orch._running = True

    def mock_run_pipeline(job):
        """Mock pipeline that produces test outputs."""
        from server.models import JobState, DocumentType, ProcessingMode

        job.transition(JobState.VALIDATING)
        job.context.document_type = DocumentType.SCANNED_PDF
        job.context.processing_mode = ProcessingMode.FULL_OCR

        job.transition(JobState.PROCESSING)
        job.context.outputs.ocr_pdf = b"%PDF-1.4 test output"
        job.context.outputs.markdown = "# Test Document\n\nHello World"
        job.context.outputs.json_metadata = {
            "engine": "paddleocr",
            "block_count": 1,
        }

        job.transition(JobState.DELIVERING)
        job.transition(JobState.COMPLETED)

    orch._run_pipeline = mock_run_pipeline

    set_orchestrator(orch)

    # Patch both init_companion and security hardening to avoid /work access
    with patch("server.api.init_companion"):
        with patch("server.api.apply_security_hardening"):
            with patch("server.api.get_config", return_value=mock_config):
                with TestClient(app) as test_client:
                    yield test_client


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime" in data


class TestReadyEndpoint:
    """Test /ready endpoint."""

    def test_ready(self, client):
        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["orchestrator_running"] is True


class TestMetricsEndpoint:
    """Test /metrics endpoint."""

    def test_metrics(self, client):
        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "files_processed" in data
        assert "files_failed" in data


class TestConvertEndpoint:
    """Test /api/v1/convert endpoint."""

    def test_convert_pdf(self, client, sample_pdf_bytes):
        """Test converting a PDF file."""
        response = client.post(
            "/api/v1/convert",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            data={"mode": "auto"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "job_id" in data
        assert data["filename"] == "test.pdf"
        assert data["document_type"] == "scanned_pdf"
        assert data["processing_mode"] == "full_ocr"
        assert "pdf" in data
        assert "markdown" in data
        assert data["markdown"]["content"] == "# Test Document\n\nHello World"

    def test_convert_image(self, client, sample_png_bytes):
        """Test converting an image file."""
        response = client.post(
            "/api/v1/convert",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
            data={"mode": "pp_ocr"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["filename"] == "test.png"

    def test_convert_unsupported_type(self, client):
        """Test rejection of unsupported file type."""
        response = client.post(
            "/api/v1/convert",
            files={"file": ("test.docx", b"content", "application/octet-stream")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "Unsupported file type" in data["detail"]

    def test_convert_file_too_large(self, client, mock_config):
        """Test rejection of files exceeding size limit."""
        mock_config.MAX_FILE_SIZE_MB = 0  # 0 MB limit

        response = client.post(
            "/api/v1/convert",
            files={"file": ("test.pdf", b"x" * 1000, "application/pdf")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "File too large" in data["detail"]

    def test_convert_no_orchestrator(self, mock_config):
        """Test 503 when orchestrator is not initialized."""
        # Clear the global orchestrator set by previous tests
        import server.api as api_module

        api_module._orchestrator = None

        with patch("server.api.init_companion"):
            with patch("server.api.apply_security_hardening"):
                with patch("server.api.get_config", return_value=mock_config):
                    with TestClient(app, raise_server_exceptions=False) as test_client:
                        response = test_client.post(
                            "/api/v1/convert",
                            files={
                                "file": ("test.pdf", b"%PDF-1.4", "application/pdf")
                            },
                        )

                        assert response.status_code == 503

    def test_convert_mode_parameter(self, client, sample_pdf_bytes):
        """Test that mode parameter is passed through."""
        response = client.post(
            "/api/v1/convert",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            data={"mode": "paddle_vl"},
        )

        assert response.status_code == 200
