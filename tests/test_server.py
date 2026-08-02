"""
Doc-Worker — Integration tests for server.py endpoints.

Tests /layout-parsing, /extract, and /health with mocked PaddleX pipelines.
"""

from __future__ import annotations

import base64
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import server
from server import app


# ── Test client setup ────────────────────────────────────────────────────
@pytest.fixture
def test_client():
    """Return a TestClient for the FastAPI app."""
    return TestClient(app)


# ── Health endpoint tests ─────────────────────────────────────────────────
class TestHealth:
    def test_health_ok(self, test_client):
        """Health endpoint should return 200 when no init exception."""
        with patch("server.get_paddlex_init_exception", return_value=None):
            response = test_client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "paddlex_lang" in data

    def test_health_unhealthy(self, test_client):
        """Health endpoint should return 503 when model failed to initialize."""
        with patch(
            "server.get_paddlex_init_exception", return_value=Exception("Test error")
        ):
            response = test_client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["detail"]["status"] == "unhealthy"
            assert data["detail"]["component"] == "paddlex"


# ── Layout-parsing endpoint tests ────────────────────────────────────────
class TestLayoutParsing:
    def test_layout_parsing_with_structure(self, test_client):
        """Test /layout-parsing with PP-StructureV3 enabled returns structured markdown."""
        mock_pages = [
            {
                "page": 1,
                "text": "Title of Document\nThis is paragraph text.",
                "blocks": [
                    {
                        "text": "Title of Document",
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.99,
                    }
                ],
                "structured_blocks": [
                    {
                        "type": "title",
                        "text": "Title of Document",
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.99,
                    },
                    {
                        "type": "text",
                        "text": "This is paragraph text.",
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.95,
                    },
                ],
            }
        ]

        with patch("server.run_paddlex_structure_v3", return_value=mock_pages):
            with patch(
                "server.blocks_to_markdown",
                return_value="# Title of Document\n\nThis is paragraph text.",
            ):
                with patch("server.os") as mock_os:
                    mock_os.getenv = lambda key, default="true": (
                        "true" if key == "USE_STRUCTURE_V3" else default
                    )
                    with patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"):
                        file_bytes = b"%PDF-1.4 Fake PDF content"
                        file_b64 = base64.b64encode(file_bytes).decode()

                        response = test_client.post(
                            "/layout-parsing",
                            headers={"Authorization": "Bearer test-token"},
                            json={
                                "file": file_b64,
                                "fileType": 0,
                                "useDocOrientationClassify": False,
                            },
                        )

        assert response.status_code == 200
        data = response.json()
        results = data["result"]["layoutParsingResults"]
        assert len(results) == 1
        assert "markdown" in results[0]
        assert "# Title of Document" in results[0]["markdown"]["text"]
        assert "structuredBlocks" in results[0]
        assert len(results[0]["structuredBlocks"]) == 2
        assert results[0]["structuredBlocks"][0]["type"] == "title"

    def test_layout_parsing_fallback_to_ocr(self, test_client):
        """Test /layout-parsing falls back to plain OCR when USE_STRUCTURE_V3=false."""
        mock_pages = [{"page": 1, "text": "Plain OCR text", "blocks": []}]

        with patch("server.run_paddleocr", return_value=mock_pages):
            with patch("server.os") as mock_os:
                mock_os.getenv = lambda key, default="true": (
                    "false" if key == "USE_STRUCTURE_V3" else default
                )
                with patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"):
                    file_b64 = base64.b64encode(b"%PDF-1.4 fake pdf").decode()

                    response = test_client.post(
                        "/layout-parsing",
                        headers={"Authorization": "Bearer test-token"},
                        json={"file": file_b64, "fileType": 0},
                    )

        assert response.status_code == 200
        data = response.json()
        results = data["result"]["layoutParsingResults"]
        assert len(results) == 1
        assert "Plain OCR text" in results[0]["markdown"]["text"]
        assert results[0].get("structuredBlocks") is None

    def test_layout_parsing_missing_file(self, test_client):
        """Test /layout-parsing returns 422 when file field is missing (Pydantic validation)."""
        response = test_client.post("/layout-parsing", json={})
        assert response.status_code == 422

    def test_layout_parsing_auth_required(self, test_client):
        """Test /layout-parsing returns 403 when token is invalid."""
        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", "secret"):
            response = test_client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer wrong"},
                json={"file": base64.b64encode(b"fake").decode()},
            )
        assert response.status_code == 403

    def test_layout_parsing_model_unhealthy(self, test_client):
        """Test /layout-parsing returns 503 when model init failed."""
        import server

        with (
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
            patch(
                "server.get_paddlex_init_exception",
                return_value=Exception("Model init failed"),
            ),
        ):
            response = test_client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": base64.b64encode(b"fake").decode()},
            )
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["status"] == "unhealthy"
        assert data["detail"]["component"] == "paddlex"


# ── Extract endpoint tests ────────────────────────────────────────────────
class TestExtract:
    def test_extract_with_structure(self, test_client):
        """Test /extract returns structured markdown and blocks."""
        mock_pages = [
            {
                "page": 1,
                "text": "Title\nParagraph",
                "blocks": [{"text": "Title", "bbox": [0, 0, 0, 0], "confidence": 0.99}],
                "structured_blocks": [
                    {
                        "type": "title",
                        "text": "Title",
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.99,
                    },
                    {
                        "type": "text",
                        "text": "Paragraph",
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.95,
                    },
                ],
            }
        ]

        with patch("server.run_paddlex_structure_v3", return_value=mock_pages):
            with patch(
                "server.blocks_to_markdown", return_value="# Title\n\nParagraph"
            ):
                with patch("server.os") as mock_os:
                    mock_os.getenv = lambda key, default="true": (
                        "true" if key == "USE_STRUCTURE_V3" else default
                    )
                    with patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"):
                        # Create a fake file upload with PDF signature to bypass text detection
                        response = test_client.post(
                            "/extract",
                            headers={"Authorization": "Bearer test-token"},
                            files={
                                "file": (
                                    "test.pdf",
                                    b"%PDF-1.4 fake pdf content",
                                    "application/pdf",
                                )
                            },
                        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert len(data["pages"]) == 1
        assert "markdown" in data["pages"][0]
        assert "# Title" in data["pages"][0]["markdown"]
        assert "structured_blocks" in data["pages"][0]
        assert len(data["pages"][0]["structured_blocks"]) == 2
        assert data["pages"][0]["structured_blocks"][0]["type"] == "title"

    def test_extract_structure_v3_error_fallback(self, test_client):
        """Test /extract returns 500 when Structure V3 prediction fails (not raw text)."""
        mock_pages = [
            {
                "page": 1,
                "text": "Title\nParagraph",
                "blocks": [],
                "structured_blocks": [],
            }
        ]

        def failing_run(*args, **kwargs):
            raise RuntimeError("Structure V3 model crashed")

        with patch("server.run_paddlex_structure_v3", failing_run):
            with patch("server.run_paddleocr", return_value=mock_pages):
                with patch("server.os") as mock_os:
                    mock_os.getenv = lambda key, default="true": (
                        "true" if key == "USE_STRUCTURE_V3" else default
                    )
                    with patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"):
                        response = test_client.post(
                            "/extract",
                            headers={"Authorization": "Bearer test-token"},
                            files={
                                "file": (
                                    "test.pdf",
                                    b"%PDF-1.4 fake pdf content",
                                    "application/pdf",
                                )
                            },
                        )

        # Should return 500 with the error message, not fallback to text
        assert response.status_code == 500
        data = response.json()
        assert "Extraction failed" in data["detail"]
        assert "Structure V3 model crashed" in data["detail"]

    def test_extract_fallback(self, test_client):
        """Test /extract falls back to plain OCR when USE_STRUCTURE_V3=false."""
        mock_pages = [{"page": 1, "text": "Plain text", "blocks": []}]

        with patch("server.run_paddleocr", return_value=mock_pages):
            with patch("server.os") as mock_os:
                mock_os.getenv = lambda key, default="true": (
                    "false" if key == "USE_STRUCTURE_V3" else default
                )
                with patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"):
                    response = test_client.post(
                        "/extract",
                        headers={"Authorization": "Bearer test-token"},
                        files={
                            "file": (
                                "test.pdf",
                                b"%PDF-1.4 fake pdf content",
                                "application/pdf",
                            )
                        },
                    )

        assert response.status_code == 200
        data = response.json()
        assert "Plain text" in data["full_text"]
        assert data["pages"][0].get("structured_blocks") is None

    def test_extract_missing_file(self, test_client):
        """Test /extract returns 422 when file is not provided (Pydantic validation)."""
        response = test_client.post("/extract")  # no file field
        assert (
            response.status_code == 422
        )  # FastAPI validation error, not our custom 400

    def test_extract_model_unhealthy(self, test_client):
        """Test /extract returns 503 when model init failed."""
        import server

        with (
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
            patch(
                "server.get_paddlex_init_exception",
                return_value=Exception("Model init failed"),
            ),
        ):
            response = test_client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("test.pdf", b"fake pdf", "application/pdf")},
            )
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["status"] == "unhealthy"
        assert data["detail"]["component"] == "paddlex"


# ── Extract: text file fallback tests ────────────────────────────────────
class TestExtractTextFileFallback:
    """Tests for text file detection and fallback in /layout-parsing and /extract."""

    def test_layout_parsing_text_file_returns_as_is(self, test_client):
        """Text files sent to /layout-parsing are returned without OCR."""
        import base64

        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", ""):
            # Pure text content (no binary signatures)
            text_content = b"This is a plain text document with some content."
            text_b64 = base64.b64encode(text_content).decode()

            response = test_client.post(
                "/layout-parsing",
                json={"file": text_b64, "fileType": 0},
            )

        assert response.status_code == 200
        data = response.json()
        results = data["result"]["layoutParsingResults"]
        assert len(results) == 1
        assert "This is a plain text document" in results[0]["markdown"]["text"]

    def test_extract_text_file_returns_as_is(self, test_client):
        """Text files sent to /extract are returned without OCR."""
        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", ""):
            # Pure text content (no binary signatures)
            text_content = b"This is a plain text document with some content."

            response = test_client.post(
                "/extract",
                files={"file": ("test.txt", text_content, "text/plain")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"
        pages = data["pages"]
        assert len(pages) == 1
        assert "This is a plain text document" in pages[0]["text"]
        assert "This is a plain text document" in data["full_text"]

    def test_layout_parsing_text_file_auth(self, test_client):
        """Text file endpoint respects auth token requirement."""
        import base64

        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", "secret"):
            text_content = b"Just some text content."
            text_b64 = base64.b64encode(text_content).decode()

            response = test_client.post(
                "/layout-parsing",
                json={"file": text_b64, "fileType": 0},
            )

        assert response.status_code == 401  # Missing auth header


# ── Health check endpoint detail tests ───────────────────────────────────
class TestHealthCheckEndpoint:
    """Tests for /health endpoint details."""

    def test_health_includes_use_structure_v3(self, test_client):
        """Health endpoint reports Structure V3 status."""
        with patch("server.get_paddlex_init_exception", return_value=None):
            response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "use_structure_v3" in data
        assert isinstance(data["use_structure_v3"], bool)

    def test_health_includes_endpoints(self, test_client):
        """Health endpoint lists available endpoints."""
        with patch("server.get_paddlex_init_exception", return_value=None):
            response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert "/layout-parsing" in data["endpoints"]
        assert "/extract" in data["endpoints"]
        assert "/health" in data["endpoints"]
