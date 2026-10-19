"""
Doc-Worker — Integration tests for server.py endpoints.

Tests /layout-parsing, /extract, and /health with mocked PaddleX pipelines.
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import os
import struct
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException
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
        """Health endpoint should return 503 when model failed to
        initialize — with the SAME generic detail as the OCR endpoints'
        _require_model_ready guard: the raw exception text (model/temp
        paths, download diagnostics) must not leak to unauthenticated
        callers."""
        with patch(
            "server.get_paddlex_init_exception",
            return_value=Exception(
                "Test error: model dir /tmp/leak-models download failed"
            ),
        ):
            response = test_client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["detail"] == {
            "status": "unhealthy",
            "component": "paddlex",
            "error": "PaddleX model initialization failed; see server logs",
        }
        # No raw exception text in the response body.
        assert "Test error" not in response.text
        assert "/tmp/leak-models" not in response.text


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

    def test_check_token_scheme_case_insensitive(self):
        """RFC 6750: the auth-scheme is case-insensitive (RFC 7235) —
        'bearer'/'BEARER' authenticate the same token. The TOKEN itself
        is not case-folded (compare_digest stays case-sensitive)."""
        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", "secret"):
            for scheme in ("bearer", "BEARER", "BeArEr"):
                server._check_token(f"{scheme} secret")  # no raise
            with pytest.raises(HTTPException) as exc:
                server._check_token("Bearer SECRET")  # token case matters
            assert exc.value.status_code == 403

    def test_check_token_non_ascii_bearer_no_500(self):
        """A non-ASCII bearer token must 403, never 500.

        hmac.compare_digest raises TypeError on non-ASCII STR inputs; the
        implementation encodes both sides to UTF-8 bytes first, so ANY
        token value is safe. Pin that contract (a 500 here would DoS the
        endpoint for a crafted header), and that a matching non-ASCII
        configured token still authenticates.
        """
        import server

        with patch.object(server, "PADDLEOCR_VL_TOKEN", "expected-secret"):
            with pytest.raises(HTTPException) as exc:
                server._check_token("Bearer héllo-wörld")
            assert exc.value.status_code == 403
        with patch.object(server, "PADDLEOCR_VL_TOKEN", "expektert-ßecret"):
            server._check_token("Bearer expektert-ßecret")  # no raise

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
        # Generic error string: the full init exception stays in the logs.
        assert (
            data["detail"]["error"]
            == "PaddleX model initialization failed; see server logs"
        )

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
        # Generic detail: the full error text stays in the server logs only.
        assert data["detail"] == "Layout parsing failed"
        assert "pdfium: failed to load PDF" not in data["detail"]
        # Crucially: no success-shaped body on error
        assert "layoutParsingResults" not in data

    def test_layout_parsing_no_garbled_output_on_error(self, test_client):
        """Regression test: error response must NOT contain garbled binary text.

        Sends a realistic FlateDecode PDF that, if its raw bytes were decoded
        as UTF-8, would produce the garble pattern (â, Ŋ, U+FFFD, etc.).
        """
        import zlib

        import server

        # Well-formed (xref + startxref complete) so the request passes
        # _enforce_pdf_caps and actually reaches the (mock-failing)
        # pipeline — the test's subject is the 500 error shape, not the
        # caps. (A startxref-less variant of these bytes 413s at the cap
        # now — the parser-disagreement fail-closed shape; see
        # TestEnforcePdfCaps::test_pypdf_constructor_failure_413_unparseable.)
        stream_data = zlib.compress(b"Hello World " * 100)
        objs = [
            b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n",
            b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n",
            b"3 0 obj <</Type /Page /MediaBox [0 0 612 792] /Parent 2 0 R>> endobj\n",
            b"4 0 obj <</Length "
            + str(len(stream_data)).encode()
            + b" /Filter /FlateDecode>>\nstream\n"
            + stream_data
            + b"\nendstream endobj\n",
        ]
        body = b"%PDF-1.4\n" + b"".join(objs)
        xref_pos = len(body)
        offsets: list[int] = []
        pos = len(b"%PDF-1.4\n")
        for o in objs:
            offsets.append(pos)
            pos += len(o)
        xref = b"xref\n0 5\n0000000000 65535 f \n"
        for off in offsets:
            xref += f"{off:010d} 00000 n \n".encode()
        pdf = (
            body
            + xref
            + b"trailer <</Size 5 /Root 1 0 R>>\n"
            + b"startxref\n"
            + str(xref_pos).encode()
            + b"\n%%EOF\n"
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

        # Should return 500 with the generic error message, not fallback to
        # text — the full error text stays in the server logs only.
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Extraction failed"
        assert "Structure V3 model crashed" not in data["detail"]

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
        # Generic error string: the full init exception stays in the logs.
        assert (
            data["detail"]["error"]
            == "PaddleX model initialization failed; see server logs"
        )


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
    so an unbounded (chunked, no Content-Length) POST is a DoS vector.
    Returns (status_code, detail) for a rejected body, None if it may
    proceed."""

    def test_post_without_content_length_rejected(self):
        status, detail = server._body_size_policy("POST", None, None)
        assert status == 413
        assert "chunked" in detail

    def test_post_with_content_length_ok(self):
        assert server._body_size_policy("POST", "100", None) is None

    def test_post_with_transfer_encoding_rejected(self):
        """Any Transfer-Encoding makes the body unbounded: per RFC 7230
        §3.3.3 the transport honors Transfer-Encoding and IGNORES
        Content-Length when both are present, so a lying small length
        (the H1 DoS bypass) must not resize the enforced limit — 413 even
        with a small, otherwise-valid Content-Length."""
        status, detail = server._body_size_policy("POST", "10", "chunked")
        assert status == 413
        assert "Transfer-Encoding" in detail

    def test_transfer_encoding_precedes_content_length_checks(self):
        """The TE check runs BEFORE the Content-Length validation: a
        malformed CL on a TE request is still a 413 (the body is
        unbounded either way), not a 400."""
        status, detail = server._body_size_policy("POST", "abc", "chunked")
        assert status == 413
        assert "Transfer-Encoding" in detail

    def test_post_oversized_rejected(self):
        with patch.object(server, "MAX_REQUEST_SIZE", 100):
            status, detail = server._body_size_policy("POST", "101", None)
            assert status == 413
            assert "too large" in detail.lower()
            assert server._body_size_policy("POST", "100", None) is None

    def test_post_non_numeric_content_length_rejected(self):
        """A malformed (non-numeric) Content-Length is a 400, not a 500 —
        the int() parse must not escape the policy. (httpx will not let
        TestClient forge a non-numeric Content-Length header through the
        middleware, so the pure-function check is the coverage.)"""
        status, detail = server._body_size_policy("POST", "abc", None)
        assert status == 400
        assert detail == "Invalid Content-Length header"

    def test_post_negative_content_length_rejected(self):
        """A negative Content-Length is not a 1*DIGIT token (RFC 7230) —
        the strict digits guard 400s it instead of letting a negative
        size through the policy (previously it parsed as a negative int
        and was accepted)."""
        status, detail = server._body_size_policy("POST", "-5", None)
        assert status == 400
        assert detail == "Invalid Content-Length header"

    def test_post_underscored_content_length_rejected(self):
        """Python int() accepts underscores ('1_000' → 1000), but the
        header is not a 1*DIGIT token — pin the strict guard so an
        underscored length can't silently resize the enforced limit."""
        status, detail = server._body_size_policy("POST", "1_000", None)
        assert status == 400
        assert detail == "Invalid Content-Length header"

    def test_post_unicode_digit_content_length_rejected(self):
        """str.isdigit() alone would accept non-ASCII decimal digits
        ('٥' → 5) — the isascii() half of the guard keeps the token
        strictly 1*DIGIT."""
        status, detail = server._body_size_policy("POST", "\u0665", None)
        assert status == 400
        assert detail == "Invalid Content-Length header"

    def test_post_too_long_digit_content_length_rejected(self):
        """int() raises ValueError for digit strings longer than 4300
        characters (CPython 3.11+ int-str digit limit; the repo pins
        3.12) — pin the 400 so a ~4 KB all-digit header cannot 500 the
        middleware, per the policy's 'the app must not 500' contract.
        (The digit guard above admits any LENGTH of digits, so this is
        the case that used to escape as an uncaught ValueError.)"""
        status, detail = server._body_size_policy("POST", "9" * 5000, None)
        assert status == 400
        assert detail == "Invalid Content-Length header"

    @staticmethod
    def _raw_asgi_post(
        content_length_value: bytes,
        transfer_encoding_value: bytes | None = None,
    ) -> tuple[int, bytes]:
        """Drive a POST through the REAL ASGI middleware with a forged
        Content-Length (and optionally a forged Transfer-Encoding),
        returning (status, response body). httpx/TestClient
        normalize the header before the app sees it, so a raw scope is
        the only way to pin the policy end-to-end (not just the pure
        _body_size_policy function). The probe stops at the size
        middleware, which short-circuits before call_next (the
        disconnect receive is never consumed)."""
        headers = [
            (b"host", b"testserver"),
            (b"content-length", content_length_value),
            (b"content-type", b"application/pdf"),
        ]
        if transfer_encoding_value is not None:
            headers.append((b"transfer-encoding", transfer_encoding_value))
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/extract",
            "raw_path": b"/extract",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
        sent: list[dict] = []

        async def receive() -> dict:
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            sent.append(message)

        async def run() -> None:
            await server.app(scope, receive, send)

        asyncio.run(run())

        start = next(m for m in sent if m["type"] == "http.response.start")
        body = b"".join(
            m.get("body", b"") for m in sent if m["type"] == "http.response.body"
        )
        return start["status"], body

    def test_non_post_exempt(self):
        assert server._body_size_policy("GET", None, None) is None
        assert server._body_size_policy("GET", "999999999", None) is None
        # A Transfer-Encoding on a non-POST is exempt too (no body to bound).
        assert server._body_size_policy("GET", "10", "chunked") is None

    def test_chunked_post_rejected_413(self, test_client):
        """A POST arriving without Content-Length (httpx sends the
        iterable chunked, so the request carries Transfer-Encoding:
        chunked and no length) → 413 before the body is touched. (A
        no-CL request with NO Transfer-Encoding — unreachable through
        httpx — is pinned by test_post_without_content_length_rejected
        on the pure policy.)"""
        body = b'{"file": "eA==", "fileType": 0}'
        resp = test_client.post(
            "/layout-parsing",
            content=iter([body]),  # httpx sends iterables chunked (no length)
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 413
        assert "Transfer-Encoding" in resp.json()["detail"]

    def test_non_numeric_content_length_400_via_raw_asgi(self):
        """A malformed (non-numeric) Content-Length is a 400 through the
        REAL ASGI middleware end-to-end (see _raw_asgi_post for why a raw
        scope is required)."""
        status, body = self._raw_asgi_post(b"abc")
        assert status == 400
        assert b"Invalid Content-Length header" in body

    def test_too_long_content_length_400_via_raw_asgi(self):
        """A 5000-digit Content-Length is a 400 through the REAL ASGI
        middleware, not a 500: int() on a >4300-digit string raises
        ValueError under CPython 3.11+ (the repo pins 3.12). Real-world
        exposure is low (both uvicorn HTTP backends reject >20-digit
        Content-Lengths first), but raw-ASGI transports can deliver the
        value to the app, and the policy's contract is 'must not 500'."""
        status, body = self._raw_asgi_post(b"9" * 5000)
        assert status == 400
        assert b"Invalid Content-Length header" in body

    def test_chunked_with_lying_content_length_413_via_raw_asgi(self):
        """H1 regression, end-to-end: a small lying Content-Length
        alongside Transfer-Encoding: chunked is a 413 through the REAL
        ASGI middleware. Per RFC 7230 §3.3.3 the transport (h11) honors
        Transfer-Encoding and IGNORES Content-Length when both are
        present, so the chunked body is unbounded no matter what the
        length claims — pre-fix the policy trusted the CL and the app
        fully materialized the body (multi-GB RAM on /layout-parsing,
        unbounded /tmp fill on /extract), trivially crafted with curl.
        (The probe stops at the size middleware; pre-fix this request
        sailed past the 413 policy to fastapi body validation, observed
        as 422.)"""
        status, body = self._raw_asgi_post(b"10", b"chunked")
        assert status == 413
        assert b"Transfer-Encoding" in body


# ── PDF request caps (page count + rendered pixels, single pass) ────────
class TestEnforcePdfCaps:
    """_enforce_pdf_caps: the page-count cap (pypdf page-tree walk —
    authoritative, since the /Count entry is attacker-writable and
    both parsers' count APIs are /Count-gated; pypdfium2's count is
    the fallback) and the per-page rendered-pixel cap (pypdf pass),
    enforced over the in-memory BYTES at RENDER_DPI BEFORE any
    temp-file staging — a rejected job never stages its temp file, and
    a tiny PDF with a huge MediaBox — page-level, inherited from the
    page tree, or with unnormalized corners — must not stage hundreds
    of GB of RGB.

    The pixel cap is measured with pypdf because it resolves the
    EFFECTIVE boxes the way poppler does (page-tree inheritance +
    corner normalization) while pypdfium2's page-level box APIs see
    neither (PDFium crbug 1786) — see server._enforce_pdf_caps.
    Tests that exercise the pixel cap therefore need BOTH pypdfium2 and
    pypdf importable (the house importorskip pattern).
    """

    @staticmethod
    def _pdf_with_pages(
        pages: list[tuple[tuple | None, tuple | str | None]],
        pages_media: tuple[float, float, float, float] | None = None,
        count: int | None = None,
    ) -> bytes:
        """Build an N-page minimal PDF with a REAL xref table — openable
        by BOTH pypdfium2 and pypdf (the old fixture was a trailer-only
        fake that pypdfium2 tolerates but pypdf rejects, and the pixel
        cap is measured with pypdf). Each page spec is
        (mediabox 4-tuple or None = no /MediaBox entry, cropbox 4-tuple
        or None, or a RAW /CropBox entry string for the malformed-shape
        tests, e.g. "0 0 612" — a 3-element array pypdf's cropbox
        property rejects); ``pages_media`` optionally puts a /MediaBox
        on the /Pages node (the page-tree-inheritance shape); ``count``
        optionally lies about the page count in /Count (spec-optional,
        attacker-writable — the lying-/Count regression shape).
        """
        n = len(pages)
        pm = (
            f" /MediaBox [{pages_media[0]} {pages_media[1]} "
            f"{pages_media[2]} {pages_media[3]}]"
            if pages_media
            else ""
        )
        objs: dict[int, bytes] = {1: b"<< /Type /Catalog /Pages 2 0 R >>"}
        kids = " ".join(f"{3 + i} 0 R" for i in range(n))
        objs[2] = (
            f"<< /Type /Pages /Kids [{kids}] /Count {count if count is not None else n}{pm} >>".encode()
        )
        for i, (mb, cb) in enumerate(pages):
            media = f" /MediaBox [{mb[0]} {mb[1]} {mb[2]} {mb[3]}]" if mb else ""
            if isinstance(cb, str):
                crop = f" /CropBox {cb}"
            elif cb:
                crop = f" /CropBox [{cb[0]} {cb[1]} {cb[2]} {cb[3]}]"
            else:
                crop = ""
            objs[3 + i] = f"<< /Type /Page{media}{crop} /Parent 2 0 R >>".encode()
        objs[3 + n] = b"<< /Length 20 >> stream\n01234567890123456789\nendstream"
        out = bytearray(b"%PDF-1.4\n")
        offsets: dict[int, int] = {}
        for num in sorted(objs):
            offsets[num] = len(out)
            out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
        xref_pos = len(out)
        out += f"xref\n0 {len(objs) + 1}\n".encode()
        out += b"0000000000 65535 f \n"
        for num in range(1, len(objs) + 1):
            out += f"{offsets[num]:010d} 00000 n \n".encode()
        out += (
            f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n".encode()
        )
        return bytes(out)

    @staticmethod
    def _minimal_pdf(
        width: float | None,
        height: float | None,
        cropbox: tuple[float, float] | str | None = None,
        pages_media: tuple[float, float, float, float] | None = None,
        mediabox: tuple[float, float, float, float] | None = None,
        count: int | None = None,
    ) -> bytes:
        """Single-page minimal PDF (see _pdf_with_pages) with MediaBox
        ``mediabox`` (explicit 4-tuple, e.g. inverted corners) or the
        default ``[0 0 width height]`` (width/height None = no /MediaBox
        entry at all), an optional CropBox (4-value tuple, or a RAW
        entry string for the malformed-shape tests), an optional
        /Pages-node MediaBox, and an optional lying /Count (``count``).
        Valid enough for BOTH pypdfium2 (page count) and pypdf (box
        measurement) to open."""
        if mediabox is not None:
            mb = mediabox
        elif width is not None:
            mb = (0.0, 0.0, float(width), float(height))
        else:
            mb = None
        if cropbox is None:
            cb: tuple | str | None = None
        elif isinstance(cropbox, str):
            cb = cropbox
        else:
            cb = (0.0, 0.0, float(cropbox[0]), float(cropbox[1]))
        return TestEnforcePdfCaps._pdf_with_pages([(mb, cb)], pages_media, count)

    @staticmethod
    def _rendered_px(width: float, height: float) -> float:
        """Rendered pixel count, computed with the EXACT same expression
        and operation order as server._enforce_pdf_caps (points *
        RENDER_DPI / 72 per axis) — bit-identical floats, so boundary
        assertions hold."""
        w_px = width * server.RENDER_DPI / 72.0
        h_px = height * server.RENDER_DPI / 72.0
        return w_px * h_px

    # ── page cap ──

    def test_page_cap_over_limit_413(self, monkeypatch):
        """A REAL 2-page PDF (twopage.pdf) over MAX_OCR_PAGES=1 413s with
        the exact count — the count is the true page count read from the
        in-memory bytes via pypdfium2 (no mock)."""
        fixture = (Path(__file__).parent / "fixtures" / "twopage.pdf").read_bytes()
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 1)
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(fixture, "pdf")
        assert exc.value.status_code == 413
        assert exc.value.detail == "Too many pages: 2 (max 1)"

    def test_page_cap_boundary_passes(self, monkeypatch):
        """count == MAX_OCR_PAGES passes (the cap rejects count > MAX,
        not >=)."""
        fixture = (Path(__file__).parent / "fixtures" / "twopage.pdf").read_bytes()
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 2)
        server._enforce_pdf_caps(fixture, "pdf")  # 2 pages == cap 2: no raise

    def test_page_cap_precedes_pixel_cap(self, monkeypatch):
        """When both caps are violated, the page cap wins — it is checked
        first, before any per-page work, so an over-cap document 413s
        'Too many pages' without page enumeration."""
        fixture = (Path(__file__).parent / "fixtures" / "twopage.pdf").read_bytes()
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 1)
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", 1)
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(fixture, "pdf")
        assert exc.value.status_code == 413
        assert exc.value.detail == "Too many pages: 2 (max 1)"

    # ── page cap: the /Count lie (spec-optional, attacker-writable) ──

    @staticmethod
    def _encrypt_and_lie_count(
        plain_pdf: bytes, true_count: int, lying_count: int
    ) -> bytes:
        """Encrypt a _pdf_with_pages-built PDF (owner-only: empty USER
        password, so pdftoppm opens it — the renderable encrypted
        shape), then lie about /Count. pypdf's writer rewrites /Count
        from the page tree on add_page, so the lie is injected AFTER
        encryption as a first-occurrence byte replacement: object
        dictionaries are not encrypted (only streams and strings), so
        the replacement lands on the /Pages node (written before any
        stream object). Raises AssertionError if the needle is missing
        — a serialization change fails the test loudly instead of
        silently no-op'ing the lie."""
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter()
        reader = PdfReader(io.BytesIO(plain_pdf), strict=False)
        for page in reader.pages:
            writer.add_page(page)
        writer.encrypt(user_password="", owner_password="test-owner")
        buf = io.BytesIO()
        writer.write(buf)
        data = buf.getvalue()
        needle = f"/Count {true_count}".encode()
        assert needle in data, (
            "encrypted output no longer exposes /Count in clear — update the injection"
        )
        replacement = f"/Count {lying_count}".encode()
        # The byte replacement shifts every byte after it: xref offsets
        # stay valid ONLY when old and new /Count have equal length
        # (single-digit → single-digit). All current call sites do; a
        # length mismatch must fail loudly here, not silently corrupt
        # the fixture's xref (which would surface later as an opaque
        # parse failure in the test it feeds).
        assert len(replacement) == len(needle), (
            f"/Count byte replacement is length-preserving only for "
            f"equal-length values ({true_count} → {lying_count} changes "
            "the document length) — use equal-digit counts or re-anchor "
            "the xref (see _encrypted_inline_kids_pdf)"
        )
        return data.replace(needle, replacement, 1)

    def test_lying_count_understated_413(self, monkeypatch):
        """BYPASS REGRESSION (lying /Count, understated): /Count is
        spec-optional and attacker-writable, and pypdfium2's
        CountPages() returns the raw /Count whenever 0 < /Count <
        0xFFFFF without cross-checking /Kids — this 3-page PDF declares
        /Count 1, so the /Count-gated count reads 1 (self-verified
        below; the pre-fix gate would have passed it) while pdftoppm
        renders ALL 3 tree pages. The cap counts the page tree with
        pypdf (get_num_pages → _flatten): 3 > cap 1 → 413 naming the
        TRUE count."""
        import pypdfium2

        pdf = self._pdf_with_pages([((0.0, 0.0, 612.0, 792.0), None)] * 3, count=1)
        # Self-verify the lie is effective: the /Count-gated count API
        # (the pre-fix gate) reads 1.
        with pypdfium2.PdfDocument(pdf) as doc:
            assert len(doc) == 1
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 1)
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(pdf, "pdf")
        assert exc.value.status_code == 413
        assert exc.value.detail == "Too many pages: 3 (max 1)"

    def test_lying_count_understated_cap_between_counts_413(self, monkeypatch):
        """The divergence discriminator: /Count=1, 3 real pages, cap 2.
        The lying count (1) sits at/under the cap — the pre-fix gate
        passed this document; the TRUE count (3) exceeds it → 413
        naming 3."""
        import pypdfium2

        pdf = self._pdf_with_pages([((0.0, 0.0, 612.0, 792.0), None)] * 3, count=1)
        with pypdfium2.PdfDocument(pdf) as doc:
            assert len(doc) == 1
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 2)
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(pdf, "pdf")
        assert exc.value.status_code == 413
        assert exc.value.detail == "Too many pages: 3 (max 2)"

    def test_lying_count_understated_at_cap_passes(self, monkeypatch):
        """count == MAX_OCR_PAGES passes on the TRUE count: /Count=1,
        3 real pages, cap 3 — the tree count (3) sits exactly at the
        cap (the gate rejects count > MAX, not >=)."""
        pdf = self._pdf_with_pages([((0.0, 0.0, 612.0, 792.0), None)] * 3, count=1)
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 3)
        server._enforce_pdf_caps(pdf, "pdf")  # 3 pages == cap 3: no raise

    def test_overstated_count_no_spurious_413(self, monkeypatch):
        """INVERTED /Count (overstated): a 1-page PDF declaring /Count
        10 read 10 to the /Count-gated count (self-verified) and would
        have 413'd spuriously at cap 5 — a valid-but-malformed document
        rejected. The page-tree count (1) is authoritative now: no
        raise."""
        import pypdfium2

        pdf = self._minimal_pdf(612.0, 792.0, count=10)
        with pypdfium2.PdfDocument(pdf) as doc:
            assert len(doc) == 10
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 5)
        server._enforce_pdf_caps(pdf, "pdf")  # 1 real page <= cap 5: no raise

    def test_overstated_count_true_count_413(self, monkeypatch):
        """INVERTED /Count (overstated) over the cap: 3 real pages,
        /Count 10, cap 2 → 413 naming the TRUE count (3), not the lying
        /Count (10) — the detail must not advertise the attacker's
        field."""
        import pypdfium2

        pdf = self._pdf_with_pages([((0.0, 0.0, 612.0, 792.0), None)] * 3, count=10)
        with pypdfium2.PdfDocument(pdf) as doc:
            assert len(doc) == 10
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 2)
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(pdf, "pdf")
        assert exc.value.status_code == 413
        assert exc.value.detail == "Too many pages: 3 (max 2)"

    def test_lying_count_encrypted_413(self, monkeypatch):
        """BYPASS REGRESSION (lying /Count, ENCRYPTED): pypdf's own page
        count is /Count-gated when is_encrypted (get_num_pages returns
        the raw /Count without a tree walk) and its page sequence is
        truncated to the same /Count — so an encrypted 3-page PDF
        declaring /Count 1 reads 1 to BOTH parsers' count APIs
        (self-verified), and every /Count-gated consumer — poppler's
        render loop included, which is clamped to the raw /Count —
        sees only the first /Count page. The cap decrypts with the
        empty user password, forces pypdf's guarded tree walk
        (_flatten), and 413s on the TRUE count."""
        import pypdf
        import pypdfium2

        plain = self._pdf_with_pages([((0.0, 0.0, 612.0, 792.0), None)] * 3)
        pdf = self._encrypt_and_lie_count(plain, 3, 1)
        # Self-verify the lie is visible to BOTH /Count-gated count
        # APIs — the pre-fix gates:
        with pypdfium2.PdfDocument(pdf) as doc:
            assert len(doc) == 1
        assert len(pypdf.PdfReader(io.BytesIO(pdf), strict=False).pages) == 1
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 1)
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(pdf, "pdf")
        assert exc.value.status_code == 413
        assert exc.value.detail == "Too many pages: 3 (max 1)"

    def test_lying_count_encrypted_pixel_cap_sees_all_tree_pages(self, monkeypatch):
        """Encrypted lying-/Count companion (pixel cap): the per-page
        cap must measure ALL tree pages, not the /Count-gated prefix.
        A 2-page encrypted PDF declaring /Count 1 — page 1 with its own
        small box, page 2 inheriting a huge MediaBox from the /Pages
        node: the /Count-gated page sequence stops after page 1 and
        would pass. Cap set between the two rendered sizes (same
        construction as the inheritance pixel-cap tests —
        DPI-independent) → 413 naming page 2."""
        pages = [
            ((0.0, 0.0, 612.0, 792.0), None),  # page 1: own small box
            (None, (0.0, 0.0, 612.0, 792.0)),  # page 2: inherits the huge
        ]
        plain = self._pdf_with_pages(pages, pages_media=(0.0, 0.0, 100000.0, 100000.0))
        pdf = self._encrypt_and_lie_count(plain, 2, 1)
        crop_px = self._rendered_px(612.0, 792.0)
        media_px = self._rendered_px(100000.0, 100000.0)
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 10)
        monkeypatch.setattr(
            server, "MAX_RENDERED_PIXELS", int(crop_px + (media_px - crop_px) / 2)
        )
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(pdf, "pdf")
        assert exc.value.status_code == 413
        assert "page 2" in exc.value.detail

    def test_single_open_per_parser_serves_both_caps(self, monkeypatch):
        """Each parser opens the request bytes EXACTLY ONCE per
        _enforce_pdf_caps call: pypdfium2 (page count) and pypdf
        (per-page boxes). The original two-cap split parsed the same
        up-to-100-MB bytes twice with the SAME parser; the current
        design trades that for one open per parser (the pypdf open is
        required to see the effective boxes pdftoppm renders — see the
        class docstring). Counting constructions through the real
        classes pins the single-open-per-parser contract end-to-end.
        """
        import pypdfium2

        pytest.importorskip("pypdf")
        import pypdf

        fixture = (Path(__file__).parent / "fixtures" / "twopage.pdf").read_bytes()
        opens = {"pdfium": 0, "pypdf": 0}
        real_pdf_document = pypdfium2.PdfDocument
        real_pdf_reader = pypdf.PdfReader

        def counting_pdf_document(*args, **kwargs):
            opens["pdfium"] += 1
            return real_pdf_document(*args, **kwargs)

        def counting_pdf_reader(*args, **kwargs):
            opens["pypdf"] += 1
            return real_pdf_reader(*args, **kwargs)

        monkeypatch.setattr(pypdfium2, "PdfDocument", counting_pdf_document)
        monkeypatch.setattr(pypdf, "PdfReader", counting_pdf_reader)
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 10)
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", 10**12)
        server._enforce_pdf_caps(fixture, "pdf")
        assert opens == {"pdfium": 1, "pypdf": 1}

    # ── page cap: pypdf's OWN tree guards (fail-closed 413) ──

    @staticmethod
    def _degenerate_tree_pdf(shape: str, encrypted: bool = False) -> bytes:
        """Build a degenerate page-tree fixture that pypdf's OWN safety
        guards refuse to measure (verified against pinned pypdf 6.21.1,
        which the cap's fail-closed 413 depends on):

        - ``'deep'``:   101-level /Pages chain (guard: depth > 100) —
                    ~9 KB, one 100000x100000 leaf, honest /Count 1
        - ``'wide'``:   one /Pages node with 100,001 kids (guard:
                    array/element limit > 100,000) — ~8 MB, first leaf
                    huge, lying /Count 1
        - ``'cyclic'``: /Pages cycle A→B→A (guard: cycle detection) —
                    ~0.5 KB, no reachable leaves at all

        The huge first leaf is the point: the pre-fix fail-soft
        fallback (pdfium's /Count-gated count + pixel-cap skip) passed
        every shape, and the leaf is what made the pass a ~521 GB
        render. Built with pypdf's writer low-level API — the
        high-level add_page/append walk reader.pages and trip the same
        guards (a degenerate tree is unroundtrippable by design). The
        writer's default root catalog is repurposed (its /Pages is
        re-pointed; no private writer state assigned). ``encrypted`` =
        owner-only encryption (empty USER password — the renderable
        shape), pinning the ENCRYPTED branch (decrypt("") →
        _flatten(True)) to the same 413.
        """
        pytest.importorskip("pypdf")
        from pypdf import PdfWriter
        from pypdf.generic import (
            ArrayObject,
            DictionaryObject,
            NameObject,
            NumberObject,
        )

        def _dict(*pairs: tuple[str, object]) -> DictionaryObject:
            d = DictionaryObject()
            for key, value in pairs:
                d[NameObject(key)] = value
            return d

        writer = PdfWriter()
        catalog = writer.root_object
        if shape == "deep":
            # 101 intermediate /Pages nodes in a chain, then one huge
            # leaf (pypdf's default depth limit is 100).
            node_refs = [
                writer._add_object(
                    _dict(
                        ("/Type", NameObject("/Pages")),
                        ("/Count", NumberObject(1)),
                    )
                )
                for _ in range(101)
            ]
            leaf = writer._add_object(
                _dict(
                    ("/Type", NameObject("/Page")),
                    ("/Parent", node_refs[-1]),
                    (
                        "/MediaBox",
                        ArrayObject(
                            [
                                NumberObject(0),
                                NumberObject(0),
                                NumberObject(100000),
                                NumberObject(100000),
                            ]
                        ),
                    ),
                )
            )
            for i, ref in enumerate(node_refs):
                nxt = node_refs[i + 1] if i + 1 < len(node_refs) else leaf
                ref.get_object()[NameObject("/Kids")] = ArrayObject([nxt])
            catalog[NameObject("/Pages")] = node_refs[0]
        elif shape == "wide":
            # One /Pages node with 100,001 kids (pypdf's array/element
            # limit is 100,000), first leaf huge, lying /Count 1.
            pages = writer._add_object(
                _dict(
                    ("/Type", NameObject("/Pages")),
                    ("/Count", NumberObject(1)),
                )
            )
            leaf_refs: list = []
            for i in range(100001):
                leaf_pairs = [
                    ("/Type", NameObject("/Page")),
                    ("/Parent", pages),
                ]
                if i == 0:
                    leaf_pairs.append(
                        (
                            "/MediaBox",
                            ArrayObject(
                                [
                                    NumberObject(0),
                                    NumberObject(0),
                                    NumberObject(100000),
                                    NumberObject(100000),
                                ]
                            ),
                        )
                    )
                leaf_refs.append(writer._add_object(_dict(*leaf_pairs)))
            pages.get_object()[NameObject("/Kids")] = ArrayObject(leaf_refs)
            catalog[NameObject("/Pages")] = pages
        elif shape == "cyclic":
            # /Pages cycle: A→B→A (no reachable leaves at all).
            d2 = writer._add_object(_dict(("/Type", NameObject("/Pages"))))
            d3 = writer._add_object(_dict(("/Type", NameObject("/Pages"))))
            d2.get_object()[NameObject("/Kids")] = ArrayObject([d3])
            d3.get_object()[NameObject("/Kids")] = ArrayObject([d2])
            d2.get_object()[NameObject("/Parent")] = d3
            d3.get_object()[NameObject("/Parent")] = d2
            catalog[NameObject("/Pages")] = d2
        else:
            raise ValueError(f"unknown shape: {shape!r}")
        if encrypted:
            # OWNER-only password (empty user password — the renderable
            # shape; decrypt("") succeeds, as in _encrypt_and_lie_count).
            writer.encrypt(user_password="", owner_password="test-owner")
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    @pytest.mark.parametrize(
        ("shape", "encrypted"),
        [
            ("deep", False),
            ("deep", True),
            ("wide", False),
            ("wide", True),
            ("cyclic", False),
            ("cyclic", True),
        ],
        ids=[
            "deep",
            "deep-encrypted",
            "wide",
            "wide-encrypted",
            "cyclic",
            "cyclic-encrypted",
        ],
    )
    def test_page_tree_guard_413(
        self, monkeypatch, caplog, shape: str, encrypted: bool
    ) -> None:
        """pypdf's OWN page-tree guard (depth / entry count / cycle)
        tripping is a fail-CLOSED 413, not fail-soft: the fallback's
        /Count-gated count is exactly the attacker-writable value the
        cap distrusts, and the huge first leaf of these fixtures is
        what the pre-fix fallback let through to a ~521 GB render.

        Self-verifying: pypdf trips the guard on a fresh read (all
        shapes: LimitReachedError for depth/entries, PdfReadError for
        the cycle — the encrypted variants via the ENCRYPTED branch's
        _flatten(True) after decrypt("")), and pypdfium2's count (the
        pre-fix gate) is 1 on the renderable shapes — i.e. under the
        old code every one of these sailed through.
        """
        pytest.importorskip("pypdf")
        from pypdf import PdfReader
        from pypdf import errors as pypdf_errors

        pdf = self._degenerate_tree_pdf(shape, encrypted)

        # Self-verify the guard trip (the 413 depends on it — a pypdf
        # version where the guard no longer trips fails LOUDLY here
        # instead of the fixture silently passing through).
        reader = PdfReader(io.BytesIO(pdf), strict=False)

        def _walk() -> None:
            if reader.is_encrypted:
                # Owner-only fixture: the empty user password MUST
                # decrypt (the renderable encrypted shape).
                assert reader.decrypt("") == 1
                reader._flatten(True)
            else:
                len(reader.pages)

        with pytest.raises((pypdf_errors.LimitReachedError, pypdf_errors.PdfReadError)):
            _walk()

        # Self-verify the pre-fix gate: pdfium's /Count-gated count is
        # the tiny value the old fallback would have trusted — except
        # the cyclic tree, which pdfium rejects outright (the 413 there
        # does not depend on any pdfium state).
        import pypdfium2

        try:
            with pypdfium2.PdfDocument(pdf) as doc:
                assert len(doc) == 1
        except Exception:
            assert shape == "cyclic", (
                "pdfium failed to open a renderable guard-trip fixture — "
                "the pre-fix gate it represents no longer matches"
            )

        monkeypatch.setattr(server, "MAX_OCR_PAGES", 500)
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", 100_000_000)
        with caplog.at_level("WARNING", logger="doc-worker.api"):
            with pytest.raises(HTTPException) as exc_info:
                server._enforce_pdf_caps(pdf, "pdf")
        assert exc_info.value.status_code == 413
        assert "Page tree unmeasurable" in exc_info.value.detail
        # The fail-closed branch must be visible in the logs (the
        # fallback's silent degradation is the failure mode this
        # closes).
        guard_logs = [
            r
            for r in caplog.records
            if "pypdf page-tree guard tripped" in r.getMessage()
        ]
        assert guard_logs, "guard-trip 413 was not logged"
        assert guard_logs[0].exc_info is not None

    def test_encrypted_user_password_fail_soft(self, monkeypatch) -> None:
        """A REAL user password (decrypt("") fails) is fail-soft — no
        raise: pdftoppm cannot open the document either, so the
        pipeline's own error applies (the pre-slot gate must not turn
        an undecryptable document into a spurious 413). The tiny caps
        below make any regression to measuring the document 413 loudly
        (a 612x792 page is ~2.55 Mpx at 300 dpi, > 1)."""
        pytest.importorskip("pypdf")
        from pypdf import PdfReader, PdfWriter

        plain = self._minimal_pdf(612, 792)
        reader = PdfReader(io.BytesIO(plain), strict=False)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.encrypt(user_password="secret-user", owner_password="owner")
        buf = io.BytesIO()
        writer.write(buf)
        pdf = buf.getvalue()
        # Self-verify: the empty user password FAILS to decrypt (the
        # branch under test) — a serialization change that made it
        # succeed would route this fixture through the measurement
        # path, and the tiny caps below would 413 loudly.
        assert not PdfReader(io.BytesIO(pdf), strict=False).decrypt("")
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 1)
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", 1)
        server._enforce_pdf_caps(pdf, "pdf")  # no raise

    # ── per-page pixel cap ──

    def test_non_pdf_kind_untouched(self, monkeypatch):
        """Images/text are single-page by construction: the caps are a PDF
        guard and must not raise for other kinds, even with the smallest
        PRODUCTION-REACHABLE caps (import clamps MAX_OCR_PAGES to >= 1,
        so 0 is not a reachable state — pin the reachable floor)."""
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 1)
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", 1)
        # PNG signature + padding — would be enormous if treated as a page
        server._enforce_pdf_caps(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16, "image")
        server._enforce_pdf_caps(b"plain text document", "text")

    def test_garbage_bytes_fail_soft(self, monkeypatch):
        """Uncountable (unopenable) PDF bytes fail soft — no raise (the
        open failure precedes both caps); the pipeline then reports its
        own error instead of a spurious 413."""
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 1)
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", 1)
        server._enforce_pdf_caps(b"%PDF-1.4 not really a pdf", "pdf")
        server._enforce_pdf_caps(b"this is not a pdf", "pdf")

    def test_page_count_failure_fails_soft(self, monkeypatch):
        """A pypdfium2 document that OPENS (the constructor works) but
        whose page COUNT raises fails soft — no raise; the pipeline
        reports its own error instead of a spurious 413. The count is
        the ONLY pypdfium2 read in the current design, so this pins the
        fail-soft branch of the first pass (the pypdf-side equivalents
        are below)."""
        import pypdfium2

        class _LenFailsDoc:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __len__(self):
                raise RuntimeError("simulated len failure")

        monkeypatch.setattr(pypdfium2, "PdfDocument", _LenFailsDoc)
        # Tiny caps: a regression to hard-fail (or a spurious 413) raises.
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 1)
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", 1)
        server._enforce_pdf_caps(b"%PDF-1.4 stub", "pdf")  # no raise

    def test_pypdf_constructor_failure_413_unparseable(self, monkeypatch, caplog):
        """pypdf's CONSTRUCTOR rejects the bytes while pypdfium2 opens
        the same ones → fail-CLOSED 413 "Unparseable PDF" (pre-fix
        residual, now closed): failing soft here would skip the
        per-page pixel cap ENTIRELY for a renderer-openable document —
        a ~51 KB 500-page × 6000 pt file would render ≈1.88 GB of RGB
        per page (~19× the default envelope) for the full render
        timeout, per unauthenticated request — and the /Count-gated
        pypdfium2 count the old behavior gated the page cap on is
        exactly the attacker-writable value the page-tree gate exists
        to distrust. (The REAL-parser shape of this branch — a
        spec-legal PDF missing startxref — is pinned by
        test_pypdf_constructor_failure_real_missing_startxref_413.)"""
        pytest.importorskip("pypdf")
        import pypdf

        fixture = (Path(__file__).parent / "fixtures" / "twopage.pdf").read_bytes()

        class _BrokenReader:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("simulated pypdf parse failure")

        monkeypatch.setattr(pypdf, "PdfReader", _BrokenReader)
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 10)
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", 1)
        with caplog.at_level("WARNING", logger="doc-worker.api"):
            with pytest.raises(HTTPException) as exc:
                server._enforce_pdf_caps(fixture, "pdf")
        assert exc.value.status_code == 413
        assert exc.value.detail == "Unparseable PDF"
        # The branch decides a security invariant — it must be visible
        # in the logs (with the original traceback).
        failclosed = [
            r for r in caplog.records if "but pypdfium2 opens it" in r.getMessage()
        ]
        assert failclosed, "fail-closed constructor branch was not logged"
        assert failclosed[0].exc_info is not None

    def test_pypdf_constructor_failure_real_missing_startxref_413(self):
        """BYPASS REGRESSION with REAL parsers (no mocks): a spec-legal
        PDF with a valid xref table + trailer but NO startxref section
        — the verified residual shape. Self-verifying both sides of the
        parser disagreement so a pypdf/pypdfium2 behavior change fails
        LOUD here instead of silently re-opening the hole: pypdf's
        constructor raises PdfReadError ("startxref not found"),
        pypdfium2 opens the same bytes from the xref table. Pre-fix,
        the cap failed soft here (page cap on the /Count-gated count
        only, pixel cap skipped) and this fixture's huge page 1
        (100000×100000 pt) would have rendered ~521 GB at 300 dpi from
        a ~1 KB file. Now: 413 "Unparseable PDF"."""
        pytest.importorskip("pypdf")
        import io
        import re

        import pypdf
        import pypdfium2

        # 3 pages: page 1 has the huge box (the pre-fix hole), pages
        # 2-3 normal. Honest /Count (= 3).
        pdf = self._pdf_with_pages(
            [
                ((0.0, 0.0, 100000.0, 100000.0), None),
                ((0.0, 0.0, 612.0, 792.0), None),
                ((0.0, 0.0, 612.0, 792.0), None),
            ]
        )
        assert b"startxref" in pdf  # sanity: the builder emits it
        fixture = re.sub(rb"startxref\n\d+\n", b"", pdf)
        assert b"startxref" not in fixture

        # Self-verify the parser disagreement (the premise of the fix):
        with pytest.raises(pypdf.errors.PdfReadError):
            pypdf.PdfReader(io.BytesIO(fixture), strict=False)
        with pypdfium2.PdfDocument(fixture) as doc:
            assert len(doc) == 3

        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(fixture, "pdf")
        assert exc.value.status_code == 413
        assert exc.value.detail == "Unparseable PDF"

    def test_pypdf_mid_iteration_failure_fails_soft(self, monkeypatch, caplog):
        """A pypdf reader that raises MID-ITERATION (page enumeration
        explodes after a successful open) fails soft the same way — the
        fail-soft branch must not turn a mid-parse error into a 500 or a
        spurious 413. The fake is shape-complete for every branch that
        runs before the pixel loop (is_encrypted, a decrypt stand-in,
        a len()-able pages) so the failure ACTUALLY fires in the loop
        it claims to guard: page 1 is yielded and measured (it passes
        — the default 100 Mpx is far above the fixture's 200x200 pt
        page), then the SECOND enumeration step raises. The pinned
        assertions prove the path: exactly two generator resumptions
        (not a death at open or len()), and the failure reached the
        documented fail-soft branch (warning logged with the original
        traceback)."""
        pytest.importorskip("pypdf")
        import pypdf

        fixture = (Path(__file__).parent / "fixtures" / "twopage.pdf").read_bytes()
        real_pdf_reader = pypdf.PdfReader
        constructed: list = []

        class _MidIterFailPages:
            """len()-able page sequence: yields page 1, then raises on
            the SECOND next() — the mid-iteration shape. ``steps``
            counts generator resumptions so the test can assert the
            failure fired on the second step, after page 1 was
            yielded and measured."""

            def __init__(self, inner):
                self._inner = inner
                self.steps = 0

            def __len__(self):
                return len(self._inner)

            def __iter__(self):
                self.steps += 1
                yield self._inner[0]
                self.steps += 1
                raise RuntimeError("simulated mid-iteration failure")

        class _MidIterFailReader:
            """Shape-complete fake: without is_encrypted the old fake
            died with AttributeError BEFORE the pixel loop (the test
            then passed for the wrong reason — it only re-pinned the
            parse-level fail-soft), and without __len__ the failure
            would fire at len(reader.pages), not mid-iteration."""

            def __init__(self, *args, **kwargs):
                self._inner = real_pdf_reader(*args, **kwargs)
                self.is_encrypted = False
                self._pages = None
                constructed.append(self)

            def decrypt(self, _password):
                # Stand-in (unreachable: is_encrypted is False) — keeps
                # the fake shape-complete for branches that call it.
                return 4

            @property
            def pages(self):
                if self._pages is None:
                    self._pages = _MidIterFailPages(self._inner.pages)
                return self._pages

        monkeypatch.setattr(pypdf, "PdfReader", _MidIterFailReader)
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 10)
        with caplog.at_level("WARNING", logger="doc-worker.api"):
            server._enforce_pdf_caps(fixture, "pdf")  # no raise
        assert len(constructed) == 1
        assert constructed[0]._pages.steps == 2, (
            "failure did not fire on the second enumeration step — "
            "the guarded tree walk (len()) died first"
        )
        residual = [
            r
            for r in caplog.records
            if "failed after opening the document" in r.getMessage()
        ]
        assert residual, "mid-iteration failure did not reach the fail-soft branch"
        assert residual[0].exc_info is not None

    def test_two_page_fixture_per_page_check(self, monkeypatch):
        """The cap is checked PER PAGE (1-based, in document order): with a
        cap between the two fixture pages' rendered sizes, page 2 (the
        larger one) is the one named in the 413; a cap below both names
        page 1; a cap above the largest passes.

        twopage.pdf: page 1 = 200x200 pt, page 2 = 300x400 pt.
        """
        pytest.importorskip("pypdf")
        fixture = (Path(__file__).parent / "fixtures" / "twopage.pdf").read_bytes()
        page1_px = self._rendered_px(200, 200)  # ≈694,444 px at 300 dpi
        page2_px = self._rendered_px(300, 400)  # ≈2,083,333 px at 300 dpi
        assert page1_px < page2_px

        # Cap between the two pages' pixel counts → page 1 passes, page 2 413s.
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", int(page1_px) + 1)
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(fixture, "pdf")
        assert exc.value.status_code == 413
        assert "page 2" in exc.value.detail
        assert "page 1" not in exc.value.detail

        # Cap below both pages' pixel counts → the first page is named.
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", int(page1_px) - 1)
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(fixture, "pdf")
        assert exc.value.status_code == 413
        assert "page 1" in exc.value.detail

        # Cap at/above the largest page's pixel count → no raise.
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", int(page2_px) + 1)
        server._enforce_pdf_caps(fixture, "pdf")

    def test_boundary_exact(self, monkeypatch):
        """120x120 pt at the CURRENT RENDER_DPI: cap == the rendered
        pixel count is legal (the check is strict >), cap one below is a
        413 naming page 1. DPI-independent on purpose: the caps are
        derived from the rendered size at server.RENDER_DPI (a documented
        override must not break the boundary pin)."""
        pytest.importorskip("pypdf")
        pdf = self._minimal_pdf(120, 120)
        exact_px = self._rendered_px(120, 120)
        passing_cap = math.ceil(exact_px)
        failing_cap = passing_cap - 1

        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", passing_cap)
        server._enforce_pdf_caps(pdf, "pdf")  # no raise at the boundary

        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", failing_cap)
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(pdf, "pdf")
        assert exc.value.status_code == 413
        assert "page 1" in exc.value.detail

    def test_huge_mediabox_with_normal_cropbox_413(self, monkeypatch):
        """Bypass regression: pypdfium2's get_width/get_height report the
        CropBox when one is set, but the renderer (pdftoppm, no -cropbox
        flag) rasterizes the MediaBox — a normal-CropBox/huge-MediaBox
        page must be measured by the MediaBox and 413, not pass on the
        CropBox size. Every earlier fixture set only MediaBox (no
        CropBox), which is exactly why this variant slipped through. The
        cap sits BETWEEN the crop-rendered and media-rendered sizes, so
        a CropBox-only measurement would pass and a max(crop, media)
        measurement 413s — DPI-independent (both sizes derive from
        server.RENDER_DPI)."""
        pytest.importorskip("pypdf")
        pdf = self._minimal_pdf(100000, 100000, cropbox=(612, 792))
        crop_px = self._rendered_px(612, 792)
        media_px = self._rendered_px(100000, 100000)
        assert crop_px < media_px
        monkeypatch.setattr(
            server, "MAX_RENDERED_PIXELS", int(crop_px + (media_px - crop_px) / 2)
        )
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(pdf, "pdf")
        assert exc.value.status_code == 413
        assert "page 1" in exc.value.detail

    def test_cropbox_smaller_than_mediabox_passes(self, monkeypatch):
        """A normal page whose CropBox is smaller than its MediaBox must
        NOT 413 when the cap sits at the LARGER (MediaBox) rendered
        size — the measurement is max(crop, media) per axis, so no
        false positive from the MediaBox (or the ANSI-A fallback a
        MediaBox-less page gets)."""
        pytest.importorskip("pypdf")
        pdf = self._minimal_pdf(612, 792, cropbox=(306, 396))
        cap = math.ceil(self._rendered_px(612, 792))  # the larger box, exactly
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", cap)
        server._enforce_pdf_caps(pdf, "pdf")  # no raise

    def test_cropbox_exception_falls_back_to_mediabox(self, monkeypatch):
        """The per-page CropBox-exception fallback branch (previously
        UNTESTED): a 3-element /CropBox makes pypdf's cropbox property
        raise (verified: ValueError 'Expected four values for /CropBox')
        while the MediaBox stays resolvable — the cap must then run on
        the MediaBox ALONE. A huge MediaBox behind a malformed CropBox
        still 413s (no silent skip of the pixel cap); a normal one
        passes (no spurious 413)."""
        pytest.importorskip("pypdf")
        # Huge MediaBox + malformed (3-element) CropBox → 413 on the
        # MediaBox alone (the fallback, not a pixel-cap skip).
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(
                self._minimal_pdf(100000, 100000, cropbox="0 0 612"), "pdf"
            )
        assert exc.value.status_code == 413
        assert "exceeds" in exc.value.detail
        # Normal MediaBox + malformed CropBox → passes.
        server._enforce_pdf_caps(self._minimal_pdf(612, 792, cropbox="0 0 612"), "pdf")

    def test_page_tree_inherited_mediabox_413(self, monkeypatch):
        """Bypass regression (theme-verified against the pinned
        pypdfium2 5.14.0 + poppler source): a huge MediaBox on the
        /PAGES NODE with a normal CropBox on the page is invisible to
        pypdfium2's page-level box APIs — FPDFPage_GetMediaBox does not
        walk /Parent (PDFium crbug 1786) and get_width()/get_mediabox()
        both report the 612x792 CropBox — so the old single-pass
        measurement saw ~8.4 Mpx and passed, while pdftoppm (no
        -cropbox) inherits the parent's MediaBox and renders
        416667x416667 px at 300 dpi (~521 GB). The cap now measures the
        EFFECTIVE box via pypdf, which resolves the same inheritance.
        The cap sits BETWEEN the crop-rendered and inherited-media-
        rendered sizes, so a no-inheritance measurement provably passes
        while the effective-box measurement 413s — DPI-independent.
        """
        pytest.importorskip("pypdf")
        pdf = self._minimal_pdf(
            None, None, cropbox=(612, 792), pages_media=(0, 0, 100000, 100000)
        )
        crop_px = self._rendered_px(612, 792)
        media_px = self._rendered_px(100000, 100000)
        assert crop_px < media_px
        monkeypatch.setattr(
            server, "MAX_RENDERED_PIXELS", int(crop_px + (media_px - crop_px) / 2)
        )
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(pdf, "pdf")
        assert exc.value.status_code == 413
        assert "page 1" in exc.value.detail

    def test_inverted_mediabox_corners_413(self, monkeypatch):
        """Bypass regression: PDF 32000-1 allows inverted corners. The
        old measurement took RAW corner deltas (box[2]-box[0]) from
        get_mediabox() — negative here — and max()ed them with
        get_width(), which reported the 612x792 CropBox, so a page with
        MediaBox [0 0 -100000 -100000] behind a normal CropBox passed
        every cap while pdftoppm normalizes the corners and renders the
        full 100000 pt (~521 GB at 300 dpi). The cap now takes the
        abs() of the effective corner deltas (pypdf keeps the corners as
        written; poppler swaps them — same rendered size). Cap-between
        construction as in the sibling regression: a raw-delta or
        CropBox-only measurement provably passes."""
        pytest.importorskip("pypdf")
        pdf = self._minimal_pdf(
            None, None, cropbox=(612, 792), mediabox=(0, 0, -100000, -100000)
        )
        crop_px = self._rendered_px(612, 792)
        media_px = self._rendered_px(100000, 100000)
        assert crop_px < media_px
        monkeypatch.setattr(
            server, "MAX_RENDERED_PIXELS", int(crop_px + (media_px - crop_px) / 2)
        )
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(pdf, "pdf")
        assert exc.value.status_code == 413
        assert "page 1" in exc.value.detail

    def test_swapped_mediabox_corners_413(self, monkeypatch):
        """Bypass regression (x-swap variant): MediaBox [100000 0 0 792]
        behind a normal CropBox — the raw delta box[2]-box[0] is
        -100000 (negative, ignored by max()) while get_width() reports
        the 612 CropBox width, so the old measurement passed; pdftoppm
        renders the 100000 pt width (~4.1 GB per page at 300 dpi).
        Abs-of-deltas on the effective box 413s."""
        pytest.importorskip("pypdf")
        pdf = self._minimal_pdf(
            None, None, cropbox=(612, 792), mediabox=(100000, 0, 0, 792)
        )
        crop_px = self._rendered_px(612, 792)
        media_px = self._rendered_px(100000, 792)
        assert crop_px < media_px
        monkeypatch.setattr(
            server, "MAX_RENDERED_PIXELS", int(crop_px + (media_px - crop_px) / 2)
        )
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(pdf, "pdf")
        assert exc.value.status_code == 413
        assert "page 1" in exc.value.detail

    def test_unmeasurable_page_does_not_disable_other_pages(self, monkeypatch):
        """Fail-soft is PER PAGE, not per document: page 1 has no
        resolvable MediaBox anywhere (pypdf raises; poppler renders the
        612x792 spec default — bounded) and is skipped, but page 2's
        huge MediaBox must STILL 413. A regression to document-level
        fail-soft on the first unmeasurable page would let a 500-page
        document with one malformed page skip the cap for the other 499.
        The cap sits above the 612x792 default (so a mistaken measurement
        of the skipped page would still pass) and far below page 2.
        """
        pytest.importorskip("pypdf")
        pdf = self._pdf_with_pages(
            [
                (None, (0, 0, 300, 400)),  # no MediaBox anywhere → unmeasurable
                ((0, 0, 100000, 100000), None),  # huge MediaBox
            ]
        )
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 10)
        monkeypatch.setattr(
            server, "MAX_RENDERED_PIXELS", self._rendered_px(612, 792) + 1
        )
        with pytest.raises(HTTPException) as exc:
            server._enforce_pdf_caps(pdf, "pdf")
        assert exc.value.status_code == 413
        assert "page 2" in exc.value.detail

    @staticmethod
    def _encrypted_inline_kids_pdf() -> bytes:
        """Build the ENCRYPTED + INLINE-/KIDS-LEAF shape: a 1-page
        owner-only-encrypted PDF (empty USER password — the renderable
        shape) whose single /Page is an INLINE dict inside the /Kids
        array rather than an indirect object. On this shape the
        production encrypted branch (decrypt("") → _flatten(True),
        list_only) yields a PageObject with NO own entries — pypdf's
        list-only suppression applies ONLY to inline kids (indirect
        leaves are pre-populated by PageObject) — so page.mediabox
        raises and the per-page pixel cap skips the page; the bound
        then rests on poppler, which the base-image poppler 22.09
        rejects inline-kids pages (≈1 px render, source-verified).

        Built in four steps because pypdf's writer re-serializes the
        page tree (an add_page'd inline kid comes back indirect):
        plain writer → owner-only encrypt → byte-substitute the /Kids
        reference with the inline page dict → re-anchor the xref (the
        substitution shifts every byte after it). Every substitution
        asserts its needle landed — a pypdf writer serialization change
        fails LOUDLY instead of silently no-op'ing the shape.
        """
        pytest.importorskip("pypdf")
        import re

        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        writer.write(buf)
        plain = buf.getvalue()
        writer2 = PdfWriter()
        reader = PdfReader(io.BytesIO(plain), strict=False)
        for page in reader.pages:
            writer2.add_page(page)
        writer2.encrypt(user_password="", owner_password="test-owner")
        buf2 = io.BytesIO()
        writer2.write(buf2)
        enc = buf2.getvalue().decode("latin-1")
        page_num = next(iter(writer2.root_object["/Pages"].get_object()["/Kids"])).idnum
        m = re.search(rf"{page_num}\s+0\s+obj\s*\n(.*?)\nendobj", enc, re.DOTALL)
        assert m, "page object not found in the encrypted output"
        page_body = m.group(1).strip()
        out, n = re.subn(
            rf"/Kids\s*\[\s*{page_num}\s+0\s+R\s*\]",
            lambda _mm: f"/Kids [{page_body}]",
            enc,
            count=1,
        )
        assert n == 1, (
            "the /Kids reference no longer matches — pypdf writer "
            "serialization changed; update the substitution"
        )
        # The substitution shifted every byte after it (later objects,
        # the xref table, startxref): re-anchor the xref offsets by
        # re-locating each "N 0 obj" so the fixture stays FULLY valid —
        # both parsers open it directly, without parser recovery.
        header = re.search(r"xref\n(\d+)\s+(\d+)\n", out)
        assert header, "xref table layout changed — update the re-anchor"
        first, count = int(header.group(1)), int(header.group(2))
        entries = re.findall(
            r"(\d{10}) (\d{5}) ([nf])",
            out[header.end() : header.end() + 20 * count],
        )
        assert len(entries) == count, "xref entry layout changed — update it"
        rebuilt = []
        for k, (off, gen, flag) in enumerate(entries):
            if flag == "f":
                rebuilt.append(f"{off} {gen} f \n")
            else:
                obj_num = first + k
                mm = re.search(rf"\n{obj_num} 0 obj", out)
                assert mm, f"object {obj_num} missing during xref re-anchor"
                rebuilt.append(f"{mm.start() + 1:010d} {gen} n \n")
        out = out[: header.end()] + "".join(rebuilt) + out[out.index("trailer\n") :]
        new_startxref = out.index("xref\n")
        out = re.sub(r"startxref\n\d+\n", f"startxref\n{new_startxref}\n", out, count=1)
        return out.encode("latin-1")

    def test_encrypted_inline_kids_leaf_pixel_cap_skips_bounded(
        self, monkeypatch, caplog
    ):
        """PIN (encrypted inline-/Kids leaf): the ONE shape where the
        per-page pixel cap silently SKIPS the page it COUNTS. The
        single /Page is an inline dict inside /Kids; on the encrypted
        path production flattens with list_only=True, and pypdf
        applies its own-entry suppression ONLY to inline kids — the
        page comes out of _flatten with no entries, page.mediabox
        raises, and the pixel cap skips the page (debug-logged) while
        the page cap still counts it. The bound for the skipped page
        is poppler-side: the base-image poppler 22.09 rejects
        inline-kids pages (≈1 px render, source-verified) — bounded
        either way; exploitable only if a future poppler accepts
        inline kids and renders their box.

        Self-verifies the mechanism on the pinned pypdf 6.x: the
        production encrypted-branch sequence (decrypt("") →
        _flatten(True)) yields exactly one page with no /MediaBox
        entry and a RAISING mediabox — a pypdf change that made
        inline entries stick would fail LOUDLY here (the page would
        then be measured, and the 1-px cap 413s)."""
        pytest.importorskip("pypdf")
        import pypdfium2
        from pypdf import PdfReader

        pdf = self._encrypted_inline_kids_pdf()

        # Self-verify the shape (the skip depends on it): renderable
        # (empty user password decrypts), the /Count-gated pdfium count
        # is 1, and the production encrypted-branch sequence yields
        # exactly one page with NO resolvable box.
        assert PdfReader(io.BytesIO(pdf), strict=False).decrypt("") == 1
        with pypdfium2.PdfDocument(pdf) as doc:
            assert len(doc) == 1
        reader = PdfReader(io.BytesIO(pdf), strict=False)
        assert reader.decrypt("") == 1
        reader._flatten(True)
        assert reader.flattened_pages is not None
        assert len(reader.flattened_pages) == 1
        page = reader.flattened_pages[0]
        assert "/MediaBox" not in page

        def _resolve_box():
            # The property access itself is the act under test (a pypdf
            # change that made inline entries stick stops raising).
            return page.mediabox

        with pytest.raises(ValueError):
            _resolve_box()

        # The cap: the page is COUNTED (1 == cap 1: pass) and the 1-px
        # pixel cap must NOT 413 — a measured 612x792 page is ~2.55 Mpx
        # at RENDER_DPI and would 413 loudly, so no-raise proves the
        # page was SKIPPED, not measured as the spec default.
        monkeypatch.setattr(server, "MAX_OCR_PAGES", 1)
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", 1)
        with caplog.at_level("DEBUG", logger="doc-worker.api"):
            server._enforce_pdf_caps(pdf, "pdf")  # no raise
        skips = [
            r
            for r in caplog.records
            if "box unresolvable, skipping pixel-cap check" in r.getMessage()
        ]
        assert skips, "the pixel-cap skip branch did not fire"
        assert skips[0].exc_info is not None


# ── MAX_CONCURRENT_OCR_JOBS env contract (import-time clamp) ─────────────
# ── image canvas cap (metadata-only, pre-slot) ───────────────────────────
class TestEnforceImagePixelCap:
    """_enforce_image_pixel_cap: the IMAGE upload's header-declared
    canvas against MAX_RENDERED_PIXELS — the PDF huge-MediaBox DoS shape
    in image form (a few-KB file declaring a canvas whose decode is tens
    of GB). Pins both over-cap branches (direct w*h, PIL bomb), the
    at-cap boundary, the undecodable-header fail-soft, and the
    cap-named detail (the e2e tests assert on it)."""

    @staticmethod
    def _png(width: int, height: int) -> bytes:
        """Minimal valid PNG (signature + IHDR + IEND, no pixels) — the
        size comes from the IHDR, which Image.open reads without
        decoding (the cap is header-only)."""
        import struct
        import zlib

        def _chunk(tag: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + tag
                + payload
                + struct.pack(">I", zlib.crc32(tag + payload))
            )

        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IEND", b"")

    def test_huge_canvas_direct_branch_413(self):
        """Canvas between PIL's warning and bomb thresholds (the size is
        readable → the direct w*h check fires): 413 with the cap named
        and the dimensions in the detail."""
        png = self._png(13000, 13000)  # 1.69e8 px, default cap 1e8
        with pytest.raises(HTTPException) as exc:
            server._enforce_image_pixel_cap(png)
        assert exc.value.status_code == 413
        assert "exceeds" in exc.value.detail
        assert str(server.MAX_RENDERED_PIXELS) in exc.value.detail
        assert "13000x13000" in exc.value.detail

    def test_huge_canvas_pil_bomb_branch_413(self):
        """Canvas above PIL's decompression-bomb ERROR threshold (PIL
        raises before the size is readable) — the cap 413s whenever the
        threshold sits at/above MAX_RENDERED_PIXELS (the default case)."""
        from PIL import Image

        assert Image.MAX_IMAGE_PIXELS is not None
        assert 2 * Image.MAX_IMAGE_PIXELS > server.MAX_RENDERED_PIXELS
        png = self._png(100000, 100000)  # 1e10 px
        with pytest.raises(HTTPException) as exc:
            server._enforce_image_pixel_cap(png)
        assert exc.value.status_code == 413
        assert "exceeds" in exc.value.detail

    def test_at_and_under_cap_passes(self, monkeypatch):
        """w*h == cap passes (strict >), under the cap passes."""
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", 13000 * 13000)
        server._enforce_image_pixel_cap(self._png(13000, 13000))  # exactly at cap
        server._enforce_image_pixel_cap(self._png(612, 792))  # well under

    def test_just_over_cap_413(self, monkeypatch):
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", 13000 * 13000 - 1)
        with pytest.raises(HTTPException) as exc:
            server._enforce_image_pixel_cap(self._png(13000, 13000))
        assert exc.value.status_code == 413

    def test_undecodable_header_fail_soft(self):
        """PIL cannot identify the bytes → fail-soft (the pipeline's own
        error applies; no spurious 413)."""
        server._enforce_image_pixel_cap(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        server._enforce_image_pixel_cap(b"not an image at all")

    def test_bomb_branch_fail_soft_when_cap_above_threshold(self, monkeypatch) -> None:
        """An operator who raises MAX_RENDERED_PIXELS above PIL's own
        error threshold keeps PIL's bounded error from the pipeline
        instead of a 413 (no allocation past the threshold either
        way)."""
        from PIL import Image

        monkeypatch.setattr(
            server,
            "MAX_RENDERED_PIXELS",
            2 * Image.MAX_IMAGE_PIXELS + 1,
        )
        server._enforce_image_pixel_cap(self._png(100000, 100000))  # no raise


# ── server-side model idle-timer stamping ────────────────────────────────
class TestRunOcrPipelineStamping:
    """The _mark_model_used stamping on the SERVER side of a job
    (worker-side stamping is pinned by TestWorkerStamping): a job longer
    than MODEL_IDLE_TIMEOUT must restart the idle clock instead of being
    destroyed on the first poll after it finishes — the server stamps
    before AND after the pipeline run."""

    def test_pipeline_stamped_before_and_after(self, monkeypatch):
        stamps: list[str] = []

        def fake_mark() -> None:
            stamps.append("mark")

        def fake_pipeline(_tmp_path: str) -> list[dict]:
            stamps.append("pipeline")
            return [{"markdown": {"text": "ok"}}]

        monkeypatch.setattr(server, "_mark_model_used", fake_mark)
        monkeypatch.setattr(server, "run_paddlex_structure_v3", fake_pipeline)

        pages, used_v3 = asyncio.run(server._run_ocr_pipeline(b"text content", "text"))
        assert used_v3 is True
        assert pages == [{"markdown": {"text": "ok"}}]
        assert stamps == ["mark", "pipeline", "mark"]


# ── concurrency env contracts (import-time clamps) ──────────────────────
class TestJobSlotsEnvContract:
    """Both concurrency knobs (MAX_CONCURRENT_OCR_JOBS /
    MAX_CONCURRENT_PDF_PARSSES) are clamped to >= 1 at import, matching
    the sibling knobs (RENDER_DPI / PAPDFTOPPM_TIMEOUT): 0 would 503
    every OCR request "busy" forever with no import-time signal, and a
    negative value makes asyncio.Semaphore raise ValueError at the first
    request — an opaque generic 500. Subprocess-based on purpose (same
    pattern as the RENDER_DPI tests in test_paddlex_helpers.py): the
    clamp runs at import time, so only a fresh interpreter proves the
    env contract. (The 0 and negative cases hit the same max(1, ...) floor;
    0 pins it — the negative twin was dropped to save a 10-30 s
    subprocess per CI run on the CPU image.)"""

    @pytest.mark.parametrize(
        ("env_name", "env_value", "expected"),
        [
            ("MAX_CONCURRENT_OCR_JOBS", "0", 1),
            ("MAX_CONCURRENT_OCR_JOBS", "5", 5),
            ("MAX_CONCURRENT_PDF_PARSSES", "0", 1),
            ("MAX_CONCURRENT_PDF_PARSSES", "5", 5),
        ],
        ids=[
            "jobs-zero",
            "jobs-explicit",
            "parses-zero",
            "parses-explicit",
        ],
    )
    def test_concurrency_knobs_clamp_in_fresh_interpreter(
        self, env_name: str, env_value: str, expected: int
    ):
        from conftest import run_fresh_interpreter_ok

        script = (
            "import server as s\n"
            f"assert s.{env_name} == {expected}, s.{env_name}\n"
            "print('OK')\n"
        )
        run_fresh_interpreter_ok(script, env_overrides={env_name: env_value})


class TestPdfParseSlot:
    """The pre-slot parse semaphore (MAX_CONCURRENT_PDF_PARSSES):
    _enforce_pdf_caps runs BEFORE the job slot (a rejected job must not
    take a slot), so its fan-out is bounded here — at most
    MAX_CONCURRENT_PDF_PARSSES concurrent cap-check parses, 503 when
    saturated (consistent with the job slot). The slot must also RELEASE
    on the 413 rejection path (a leaked slot would 503 every later PDF
    request until restart). Direct slot-level tests: the per-loop
    wiring (constant → semaphore → 503) is pinned at the endpoint by
    test_vl_api_e2e's parse-slot tests."""

    def test_503_when_saturated(self, monkeypatch):
        sem = asyncio.Semaphore(0)  # fully saturated: locked() is True
        monkeypatch.setattr(server, "_pdf_parse_semaphore", lambda: sem)
        exc: HTTPException | None = None

        async def scenario():
            nonlocal exc
            with pytest.raises(HTTPException) as info:
                async with server._pdf_parse_slot():
                    pytest.fail("a saturated parse slot must not admit")
            exc = info.value

        asyncio.run(scenario())
        assert exc is not None and exc.status_code == 503
        assert "parse slots" in exc.detail

    def test_slot_held_while_admitted_and_released_on_exit(self, monkeypatch):
        sem = asyncio.Semaphore(1)
        monkeypatch.setattr(server, "_pdf_parse_semaphore", lambda: sem)

        async def scenario():
            async with server._pdf_parse_slot():
                # The one permit is held for the duration of the parse.
                assert sem.locked()
            assert not sem.locked()  # released on normal exit

        asyncio.run(scenario())

    def test_released_on_rejection(self, monkeypatch):
        """The 413 path: the slot is released even when the cap check
        raises (finally), so the next parse is admitted — the same
        leak class the job slot's test_failed_job_releases_slot pins."""
        sem = asyncio.Semaphore(1)
        monkeypatch.setattr(server, "_pdf_parse_semaphore", lambda: sem)

        async def scenario():
            with pytest.raises(HTTPException):
                async with server._pdf_parse_slot():
                    raise HTTPException(413, "Too many pages: 9 (max 5)")
            # The slot is free again: a second parse is admitted.
            async with server._pdf_parse_slot():
                pass

        asyncio.run(scenario())

    def test_capacity_driven_by_max_concurrent_pdf_parses(self, monkeypatch):
        """Constant → semaphore link: the per-loop semaphore is created
        with the MAX_CONCURRENT_PDF_PARSSES constant — patched to 3 here
        (with a fresh registry), a FRESH loop (asyncio.run) must admit
        exactly 3 concurrent holds and 503 the 4th while all 3 are
        held. Distinguishes ignoring the constant (default 2 → the
        third hold 503s), a hardcoded 1 (two 503s), or a hardcoded 3+
        (the fourth admitted). Deterministic like the job slot's barrier
        test: semaphore admits take the synchronous fast path (no yield)
        while permits remain, so by the time any task first yields
        (holding its slot) all four tasks have already reached the
        slot — the 4th deterministically sees all three permits held."""
        monkeypatch.setattr(server, "_PDF_PARSE_SEMAPHORES", {})
        monkeypatch.setattr(server, "MAX_CONCURRENT_PDF_PARSSES", 3)
        codes: list[int] = []

        async def scenario():
            release = asyncio.Event()

            async def hold():
                try:
                    async with server._pdf_parse_slot():
                        codes.append(200)
                        await release.wait()  # hold the slot until release
                except HTTPException as exc:
                    codes.append(exc.status_code)

            tasks = [asyncio.create_task(hold()) for _ in range(4)]
            # One loop tick lets all four tasks reach the slot (acquire is
            # the synchronous fast path; release.wait() is where each
            # admitted task first yields) — the 4th deterministically
            # sees all three permits held.
            await asyncio.sleep(0)
            assert sorted(codes) == [200, 200, 200, 503], codes
            release.set()
            await asyncio.gather(*tasks)

        asyncio.run(scenario())


# ── admission-cap clamp env contracts (import-time clamps) ──────────────
class TestOcrCapClampContract:
    """MAX_OCR_PAGES / MAX_RENDERED_PIXELS / MAX_REQUEST_SIZE are clamped
    to >= 1 at import, matching the sibling knobs
    (MAX_CONCURRENT_OCR_JOBS / RENDER_DPI / PAPDFTOPPM_TIMEOUT). The env
    contract has two halves, pinned here in a FRESH interpreter (the
    clamp runs at import, so only a fresh interpreter proves it; same
    subprocess pattern as TestJobSlotsEnvContract above):
    - zero clamps to 1 — 0 would silently 413 every OCR request with no
      import-time signal;
    - an explicit positive value passes through UNCHANGED — the clamp is
      a floor, not an assignment (pre-fix only the zero case was
      exercised, so a broken passthrough — e.g. an accidental
      assignment — would have gone undetected)."""

    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            pytest.param(
                {
                    "MAX_OCR_PAGES": "0",
                    "MAX_RENDERED_PIXELS": "0",
                    "MAX_REQUEST_SIZE": "0",
                },
                {
                    "MAX_OCR_PAGES": 1,
                    "MAX_RENDERED_PIXELS": 1,
                    "MAX_REQUEST_SIZE": 1,
                },
                id="zero-clamped-to-1",
            ),
            pytest.param(
                {
                    "MAX_OCR_PAGES": "5",
                    "MAX_RENDERED_PIXELS": "1234",
                    "MAX_REQUEST_SIZE": "999",
                },
                {
                    "MAX_OCR_PAGES": 5,
                    "MAX_RENDERED_PIXELS": 1234,
                    "MAX_REQUEST_SIZE": 999,
                },
                id="explicit-passthrough",
            ),
        ],
    )
    def test_admission_caps_env_contract_in_fresh_interpreter(self, env, expected):
        from conftest import run_fresh_interpreter_ok

        script = (
            "import server as s\n"
            f"assert s.MAX_OCR_PAGES == {expected['MAX_OCR_PAGES']}, "
            f"s.MAX_OCR_PAGES\n"
            f"assert s.MAX_RENDERED_PIXELS == {expected['MAX_RENDERED_PIXELS']}, "
            f"s.MAX_RENDERED_PIXELS\n"
            f"assert s.MAX_REQUEST_SIZE == {expected['MAX_REQUEST_SIZE']}, "
            f"s.MAX_REQUEST_SIZE\n"
            "print('OK')\n"
        )
        run_fresh_interpreter_ok(script, env_overrides=env)
