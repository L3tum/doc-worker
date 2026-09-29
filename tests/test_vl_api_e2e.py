"""
Doc-Worker — E2E tests for the PaddleOCR-VL compatible API (Open-WebUI).

Exercises the full request path end-to-end:

    HTTP request → server endpoint → real paddlex_helpers pipeline code
    → JSON response

Only the model boundary (``pipeline.predict``) and the PDF→image conversion
(``_pdf_to_images``) are mocked. The mock payloads are shaped like real
PaddleX output — numpy arrays and float32 scalars, including ``block_bbox``
as a *list* of np.float32. That is the exact payload class that caused the
production incident ("Object of type float32 is not JSON serializable");
the plain-Python mock fixtures used elsewhere in the suite cannot catch
that kind of regression.

Endpoints covered:
    POST /layout-parsing  — Open-WebUI PaddleOCR-VL contract
    POST /extract         — direct upload API
"""

from __future__ import annotations

import base64
import json
import os
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import paddlex_helpers
import server
from server import SafeJSONResponse, app


# ── Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture
def client() -> TestClient:
    """TestClient for the FastAPI app (no lifespan → no background threads)."""
    return TestClient(app)


# Minimal FlateDecode PDF — passes _is_text_content() as binary, so the
# request goes through the OCR path instead of the text passthrough.
_FAKE_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
    b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
    b"3 0 obj <</Type /Page /MediaBox [0 0 612 792] /Parent 2 0 R>> endobj\n"
    b"4 0 obj <</Length 20>> stream\n01234567890123456789\nendstream endobj\n"
    b"trailer <</Size 5 /Root 1 0 R>>\n%%EOF"
)

_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


# ── Realistic PaddleX payloads (numpy-flavored, like production) ─────────
def _numpy_page_result(page_no: int) -> dict:
    """Build one PP-StructureV3 page result with real PaddleX data types.

    Mirrors production output: rec scores/layout scores as np.float32,
    rec boxes as ndarrays, and ``block_bbox`` as a *list* of np.float32
    scalars (the shape that broke JSON serialization in the incident).
    """
    return {
        "overall_ocr_res": {
            "rec_texts": [f"Title {page_no}", f"Body text {page_no}."],
            "rec_scores": [np.float32(0.99), np.float32(0.95)],
            "rec_boxes": [
                np.array([50.0, 50.0, 200.0, 70.0], dtype=np.float32),
                np.array([50.0, 90.0, 300.0, 130.0], dtype=np.float32),
            ],
            "rec_polys": [
                np.array([[50, 50], [200, 50], [200, 70], [50, 70]], dtype=np.int32),
                np.array([[50, 90], [300, 90], [300, 130], [50, 130]], dtype=np.int32),
            ],
        },
        "parsing_res_list": [
            {
                "block_label": "title",
                "block_content": f"Title {page_no}",
                "block_bbox": [
                    np.float32(50.0),
                    np.float32(50.0),
                    np.float32(200.0),
                    np.float32(70.0),
                ],
            },
            {
                "block_label": "text",
                "block_content": f"Body text {page_no}.",
                "block_bbox": np.array([50.0, 90.0, 300.0, 130.0], dtype=np.float32),
            },
        ],
        "layout_det_res": {
            "boxes": [
                {
                    "label": "title",
                    "score": np.float32(0.99),
                    "coordinate": np.array([50, 50, 200, 70], dtype=np.int32),
                },
                {
                    "label": "text",
                    "score": np.float32(0.97),
                    "coordinate": np.array([50, 90, 300, 130], dtype=np.int32),
                },
            ]
        },
    }


def _numpy_ocr_result(page_no: int) -> dict:
    """Build one General OCR result with numpy types (USE_STRUCTURE_V3=false)."""
    return {
        "rec_texts": [f"Plain OCR line {page_no}"],
        "rec_scores": [np.float32(0.97)],
        "rec_boxes": [np.array([10.0, 10.0, 100.0, 30.0], dtype=np.float32)],
        "rec_polys": [],
    }


class _FakeV3Model:
    """PP-StructureV3 pipeline stand-in: one predict() result per image."""

    def __init__(self, results: list[dict]) -> None:
        self._results = list(results)

    def predict(self, _path: str):
        if not self._results:
            return iter(())
        yield self._results.pop(0)


class _FakeOcrModel:
    """General OCR pipeline stand-in: all pages from one predict() call."""

    def __init__(self, results: list[dict]) -> None:
        self._results = list(results)

    def predict(self, _path: str):
        return iter(self._results)


# ── POST /layout-parsing — Open-WebUI PaddleOCR-VL contract ──────────────
class TestLayoutParsingE2E:
    def test_pdf_single_page_contract(self, client: TestClient) -> None:
        """1-page PDF → 200 with the Open-WebUI response contract.

        Regression for the production incident: a numpy-flavored PaddleX
        payload must produce valid JSON, not
        "TypeError: Object of type float32 is not JSON serializable".
        """
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={
                    "file": file_b64,
                    "fileType": 0,
                    "useDocOrientationClassify": True,
                    "useDocUnwarping": True,
                    "useChartRecognition": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        results = data["result"]["layoutParsingResults"]
        assert len(results) == 1
        # Open-WebUI reads markdown.text
        assert "# Title 1" in results[0]["markdown"]["text"]
        assert "Body text 1." in results[0]["markdown"]["text"]
        # structuredBlocks must carry plain Python numerics (JSON round-trip)
        blocks = results[0]["structuredBlocks"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "title"
        assert all(isinstance(v, (int, float)) for v in blocks[0]["bbox"])
        assert isinstance(blocks[0]["confidence"], float)

    def test_pdf_two_pages(self, client: TestClient) -> None:
        """A 2-page PDF yields two layoutParsingResults, one per page."""
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model(
                    [_numpy_page_result(1), _numpy_page_result(2)]
                ),
            ),
            patch.object(
                paddlex_helpers,
                "_pdf_to_images",
                return_value=["/tmp/page-1.png", "/tmp/page-2.png"],
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert len(results) == 2
        assert "Title 1" in results[0]["markdown"]["text"]
        assert "Title 2" in results[1]["markdown"]["text"]

    def test_image_file_type(self, client: TestClient) -> None:
        """fileType=1 (image) skips PDF conversion and goes straight to the model."""
        file_b64 = base64.b64encode(_FAKE_PNG).decode()
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 1},
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert len(results) == 1
        assert "Title 1" in results[0]["markdown"]["text"]

    def test_null_page_fields_no_500(self, client: TestClient) -> None:
        """PaddleX null page sections (F1) must yield empty results, not a 500.

        Regression-locks the `or {}`/`or []` guards through the full HTTP path.
        """
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        null_page = {
            "overall_ocr_res": None,
            "layout_det_res": None,
            "parsing_res_list": None,
        }
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([null_page]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert len(results) == 1
        assert results[0]["structuredBlocks"] == []

    def test_plain_ocr_fallback(self, client: TestClient) -> None:
        """USE_STRUCTURE_V3=false → plain OCR path, structuredBlocks=None."""
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_model",
                return_value=_FakeOcrModel([_numpy_ocr_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.dict(os.environ, {"USE_STRUCTURE_V3": "false"}),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert len(results) == 1
        assert "Plain OCR line 1" in results[0]["markdown"]["text"]
        assert results[0]["structuredBlocks"] is None

    def test_auth_wrong_token(self, client: TestClient) -> None:
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        with patch.object(server, "PADDLEOCR_VL_TOKEN", "secret"):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer wrong-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 403

    def test_auth_missing_token(self, client: TestClient) -> None:
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        with patch.object(server, "PADDLEOCR_VL_TOKEN", "secret"):
            response = client.post(
                "/layout-parsing", json={"file": file_b64, "fileType": 0}
            )
        assert response.status_code == 401

    def test_invalid_base64(self, client: TestClient) -> None:
        with patch.object(server, "PADDLEOCR_VL_TOKEN", ""):
            response = client.post(
                "/layout-parsing", json={"file": "a==b", "fileType": 0}
            )
        assert response.status_code == 400
        assert "base64" in response.json()["detail"].lower()

    def test_oversized_request(self, client: TestClient) -> None:
        with patch.object(server, "MAX_REQUEST_SIZE", 64):
            response = client.post(
                "/layout-parsing",
                json={"file": base64.b64encode(_FAKE_PDF).decode(), "fileType": 0},
            )
        assert response.status_code == 413


# ── POST /extract — direct upload API ────────────────────────────────────
class TestExtractE2E:
    def test_extract_pdf_full_flow(self, client: TestClient) -> None:
        """Multipart upload → structured pages + blocks, numpy payload safe."""
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _FAKE_PDF, "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "doc.pdf"
        assert len(data["pages"]) == 1
        page = data["pages"][0]
        assert page["page"] == 1
        assert "# Title 1" in page["markdown"]
        assert "Body text 1." in page["text"]
        assert "Body text 1." in data["full_text"]
        blocks = page["structured_blocks"]
        assert blocks[0]["type"] == "title"
        assert all(isinstance(v, (int, float)) for v in blocks[0]["bbox"])
        # Raw OCR blocks must also be JSON-safe (numpy bboxes scrubbed)
        assert all(isinstance(v, (int, float)) for v in page["blocks"][0]["bbox"])

    def test_extract_image_full_flow(self, client: TestClient) -> None:
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("scan.png", _FAKE_PNG, "image/png")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "scan.png"
        assert "Title 1" in data["pages"][0]["markdown"]

    def test_extract_plain_ocr_fallback(self, client: TestClient) -> None:
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_model",
                return_value=_FakeOcrModel([_numpy_ocr_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.dict(os.environ, {"USE_STRUCTURE_V3": "false"}),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _FAKE_PDF, "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "Plain OCR line 1" in data["full_text"]
        assert data["pages"][0]["structured_blocks"] is None


# ── Helper layer (covers the worker.py sidecar path too) ─────────────────
class TestHelperLayerE2E:
    def test_structure_v3_pages_are_json_serializable(self, tmp_path) -> None:
        """run_paddlex_structure_v3 output must survive plain json.dumps.

        This is the exact failure mode from the production incident — the
        worker's sidecar JSON and both server endpoints serialize these
        pages without a default= handler at the call site.
        """
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(_FAKE_PDF)
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
        ):
            pages = paddlex_helpers.run_paddlex_structure_v3(str(pdf))

        json.dumps(pages)  # no default= — must not raise TypeError
        assert pages[0]["structured_blocks"][0]["bbox"] == [50.0, 50.0, 200.0, 70.0]
        assert all(type(v) is float for v in pages[0]["structured_blocks"][0]["bbox"])

    def test_ocr_pages_are_json_serializable(self, tmp_path) -> None:
        img = tmp_path / "page.png"
        img.write_bytes(_FAKE_PNG)
        with patch.object(
            paddlex_helpers,
            "_get_paddlex_model",
            return_value=_FakeOcrModel([_numpy_ocr_result(1)]),
        ):
            pages = paddlex_helpers.run_paddleocr(str(img))

        json.dumps(pages)  # no default= — must not raise TypeError
        assert all(type(v) is float for v in pages[0]["blocks"][0]["bbox"])


# ── SafeJSONResponse backstop ────────────────────────────────────────────
class TestSafeJSONResponse:
    def test_numpy_leak_serialized_not_crash(self) -> None:
        """A numpy type that slips past the source-level fix must still serialize.

        This is the defense-in-depth backstop (serialization.json_default):
        the endpoint must never return a 500 because of a non-JSON type.
        """
        response = SafeJSONResponse(
            content={
                "scalar_list": [np.float32(1.5)],
                "ndarray": np.array([1, 2], dtype=np.int32),
            }
        )
        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["scalar_list"] == [1.5]
        assert body["ndarray"] == [1, 2]

    def test_unknown_type_still_raises(self) -> None:
        """The backstop must not silently string-ify unknown objects."""
        with pytest.raises(TypeError):
            SafeJSONResponse(content={"obj": object()})
