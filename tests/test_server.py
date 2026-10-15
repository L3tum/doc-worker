"""
Doc-Worker — Integration tests for server.py endpoints.

Tests /layout-parsing, /extract, and /health with mocked PaddleX pipelines.
"""

from __future__ import annotations

import base64
import os
import struct
from pathlib import Path
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
                with patch.dict(os.environ, {"USE_STRUCTURE_V3": "true"}):
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
            with patch.dict(os.environ, {"USE_STRUCTURE_V3": "false"}):
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

    def test_layout_parsing_auth_wrong_scheme(self, test_client):
        """A non-Bearer Authorization header must return 403, not 401."""
        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", "secret"):
            response = test_client.post(
                "/layout-parsing",
                headers={"Authorization": "Basic secret"},
                json={"file": base64.b64encode(b"fake").decode()},
            )
        assert response.status_code == 403

    def test_layout_parsing_auth_trailing_content_rejected(self, test_client):
        """Only the token after 'Bearer ' is compared; extra trailing
        content in the header must not authenticate (prefix-parse pin)."""
        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", "secret"):
            response = test_client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer secret extra"},
                json={"file": base64.b64encode(b"fake").decode()},
            )
        assert response.status_code == 403

    def test_layout_parsing_auth_empty_token_rejected(self, test_client):
        """'Bearer ' with an empty token must return 403 (prefix-parse pin)."""
        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", "secret"):
            response = test_client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer "},
                json={"file": base64.b64encode(b"fake").decode()},
            )
        assert response.status_code == 403

    def test_layout_parsing_auth_double_space_rejected(self, test_client):
        """'Bearer  secret' (double space) parses the token as ' secret',
        which must not authenticate (prefix-parse pin)."""
        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", "secret"):
            response = test_client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer  secret"},
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

    def test_layout_parsing_returns_500_on_processing_error(self, test_client):
        """Test /layout-parsing returns 500 when processing fails (not raw bytes)."""
        import server

        def failing_run(*args, **kwargs):
            raise RuntimeError("pdfium: failed to load PDF")

        with patch("server.run_paddlex_structure_v3", failing_run):
            with patch.dict(os.environ, {"USE_STRUCTURE_V3": "true"}):
                with patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"):
                    file_b64 = base64.b64encode(b"%PDF-1.4 fake pdf content").decode()
                    response = test_client.post(
                        "/layout-parsing",
                        headers={"Authorization": "Bearer test-token"},
                        json={"file": file_b64, "fileType": 0},
                    )

        assert response.status_code == 500
        data = response.json()
        assert "Layout parsing failed" in data["detail"]
        assert "pdfium: failed to load PDF" in data["detail"]
        # Crucially: no success-shaped body on error
        assert "layoutParsingResults" not in data

    def test_layout_parsing_no_garbled_output_on_error(self, test_client):
        """Regression test: error response must NOT contain garbled binary text.

        Sends a realistic FlateDecode PDF that, if its raw bytes were decoded
        as UTF-8, would produce the garble pattern (â, Ŋ, U+FFFD, etc.).
        """
        import zlib

        import server

        stream_data = zlib.compress(b"Hello World " * 100)
        pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
            b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
            b"3 0 obj <</Type /Page /MediaBox [0 0 612 792] /Parent 2 0 R>> endobj\n"
            b"4 0 obj <</Length "
            + str(len(stream_data)).encode()
            + b" /Filter /FlateDecode>>\n"
            b"stream\n" + stream_data + b"\nendstream\nendobj\n"
            b"trailer <</Size 5 /Root 1 0 R>>\n%%EOF"
        )

        def failing_run(*args, **kwargs):
            raise RuntimeError("simulated processing failure")

        with patch("server.run_paddlex_structure_v3", failing_run):
            with patch.dict(os.environ, {"USE_STRUCTURE_V3": "true"}):
                with patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"):
                    file_b64 = base64.b64encode(pdf).decode()
                    response = test_client.post(
                        "/layout-parsing",
                        headers={"Authorization": "Bearer test-token"},
                        json={"file": file_b64, "fileType": 0},
                    )

        assert response.status_code == 500
        body_text = response.text
        # Error shape, not success shape
        data = response.json()
        assert "detail" in data
        assert "layoutParsingResults" not in data
        # The garble fingerprint must be absent
        assert "\ufffd" not in body_text  # U+FFFD replacement character
        assert "\xe2\x82\xac" not in body_text  # â (common garble char)

    def test_layout_parsing_empty_file_returns_400(self, test_client):
        """Empty file content must return 400, not 500 or garbled 200."""
        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"):
            # Empty string is caught by the 'Missing file field' guard
            response = test_client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": "", "fileType": 0},
            )
        assert response.status_code == 400
        assert "file" in response.json()["detail"].lower()

    def test_layout_parsing_german_text_passthrough(self, test_client):
        """German text files must be returned as-is (200), not 500.

        This tests the coupling between the _is_text_content fix and the
        error fallback fix. Without the UTF-8-aware _is_text_content, German
        text would be misclassified as binary, sent to OCR, and if OCR
        fails, would hit the error path.
        """
        import server

        german_text = "Müllerstraße über große höfe für deutsche Größe\n"
        file_b64 = base64.b64encode(german_text.encode("utf-8")).decode()

        with patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"):
            response = test_client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )

        assert response.status_code == 200
        data = response.json()
        results = data["result"]["layoutParsingResults"]
        assert len(results) == 1
        assert german_text in results[0]["markdown"]["text"]


# ── Extract: empty file test ────────────────────────────────────────────────
class TestExtractEmptyFile:
    def test_extract_empty_file_returns_400(self, test_client):
        """Empty file upload must return 400."""
        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"):
            response = test_client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("test.pdf", b"", "application/pdf")},
            )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()


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
                with patch.dict(os.environ, {"USE_STRUCTURE_V3": "true"}):
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
                with patch.dict(os.environ, {"USE_STRUCTURE_V3": "true"}):
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
            with patch.dict(os.environ, {"USE_STRUCTURE_V3": "false"}):
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


# ── Magic-byte classifier pins ──────────────────────────────────────────
class TestDetectFileTypeSignatures:
    """Pure-function pins for server.detect_file_type (no TestClient)."""

    @staticmethod
    def _bmp(file_size: int, dib_size: int) -> bytes:
        # "BM" + file-size DWORD (offset 2) + reserved DWORD (offset 6) +
        # placeholder DWORD (offset 10) + DIB header-size DWORD (offset 14)
        # + padding so the sample is >= 18 bytes.
        return (
            b"BM"
            + struct.pack("<I", file_size)
            + struct.pack("<I", 0)
            + struct.pack("<I", 14)
            + struct.pack("<I", dib_size)
            + b"\x00" * 4
        )

    def test_webp_valid_is_image(self):
        """RIFF-anchored WebP classifies as image."""
        assert server.detect_file_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image"

    def test_webp_not_riff_anchored_is_text(self):
        """Regression: a >=12-byte non-RIFF file with "WEBP" at offset 8
        must not classify as image (it used to, and died with HTTP 500 in
        the image pipeline; the bytes passed through as text before)."""
        assert server.detect_file_type(b"12345678WEBP" + b"!" * 8) == "text"

    def test_riff_non_webp_is_not_image(self):
        """Other RIFF content (WAVE fourcc) is not an image; this sample's
        NULs make it binary under the explicit NUL check, so it is unknown
        (pre-change the NULs decoded as valid UTF-8 and it leaked through
        as text — the 7-bit-clean-binary regression the NUL check closes)."""
        assert server.detect_file_type(b"RIFF\x00\x00\x00\x00WAVEfmt ") == "unknown"

    def test_bmp_bitmapinfoheader_is_image(self):
        """Regression: DIB size 40 (BITMAPINFOHEADER) with a real file size
        classifies as image."""
        assert server.detect_file_type(self._bmp(100, 40)) == "image"

    def test_bmp_v4_unspecified_filesize_is_image(self):
        """DIB size 108 (BITMAPV4HEADER) with file-size DWORD 0 (unspecified)
        classifies as image."""
        assert server.detect_file_type(self._bmp(0, 108)) == "image"

    def test_bmp_unknown_dib_size_is_text(self):
        """'BM' text with a DIB slot value outside the known set is not an
        image. The sample is NUL-free (the NUL check now classifies any
        NUL-laden file as binary), so the text check decides: text."""
        data = b"BM" + b"this is plain text" * 2
        assert server.detect_file_type(data) == "text"


# ── Request-body size policy (chunked + oversized) ────────────────────────
class TestBodySizePolicy:
    """_body_size_policy: the body is fully materialized before processing,
    so an unbounded (chunked, no Content-Length) POST is a DoS vector."""

    def test_post_without_content_length_rejected(self):
        assert server._body_size_policy("POST", None) is not None

    def test_post_with_content_length_ok(self):
        assert server._body_size_policy("POST", "100") is None

    def test_post_oversized_rejected(self):
        with patch.object(server, "MAX_REQUEST_SIZE", 100):
            assert server._body_size_policy("POST", "101") is not None
            assert server._body_size_policy("POST", "100") is None

    def test_non_post_exempt(self):
        assert server._body_size_policy("GET", None) is None
        assert server._body_size_policy("GET", "999999999") is None

    def test_chunked_post_rejected_413(self, test_client):
        """A POST arriving without Content-Length (chunked transfer)
        reaches the middleware as content-length=None → 413 before the
        body is touched."""
        body = b'{"file": "eA==", "fileType": 0}'
        resp = test_client.post(
            "/layout-parsing",
            content=iter([body]),  # httpx sends iterables chunked (no length)
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 413
        assert "chunked" in resp.json()["detail"]


# ── PDF page counting (page-limit guard) ─────────────────────────────────
class TestCountPdfPages:
    def test_counts_real_pdf(self):
        """_count_pdf_pages reads the true page count via pypdfium2
        (the same rasterizer the worker uses for AcroForm flatten)."""
        import pypdfium2  # hard dependency (worker.py AcroForm path)

        fixture = Path(__file__).parent / "fixtures" / "two_column.pdf"
        assert server._count_pdf_pages(str(fixture)) == 1
        with pypdfium2.PdfDocument(str(fixture)) as doc:  # noqa: F841
            pass  # fixture must be a valid, openable PDF

    def test_uncountable_pdf_returns_none(self):
        """A non-PDF path fails soft to None (the pipeline then reports
        its own error instead of a 413)."""
        assert server._count_pdf_pages("/nonexistent/nope.pdf") is None
