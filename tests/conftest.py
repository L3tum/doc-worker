from __future__ import annotations

import sys
from pathlib import Path
import tempfile
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Mock PaddleX pipelines ────────────────────────────────────────────────
@pytest.fixture
def mock_paddlex_ocr_result() -> dict:
    """Mock PaddleX General OCR pipeline result."""
    return {
        "rec_texts": ["Hello World", "This is a test", ""],
        "rec_scores": [0.98, 0.95, 0.0],
        "rec_boxes": [[10, 10, 100, 20], [10, 30, 100, 40], None],
        "rec_polys": [
            [10, 10, 100, 10, 100, 20, 10, 20],
            [10, 30, 100, 30, 100, 40, 10, 40],
            None,
        ],
    }


@pytest.fixture
def mock_paddlex_structure_v3_result() -> dict:
    """Mock PP-StructureV3 pipeline result."""
    return {
        "overall_ocr_res": {
            "rec_texts": ["Title of Document", "This is paragraph text.", "Subheading"],
            "rec_scores": [0.99, 0.95, 0.92],
            "rec_boxes": [[50, 50, 200, 70], [50, 90, 300, 130], [50, 150, 180, 170]],
            "rec_polys": [],
        },
        "parsing_res_list": [
            {
                "block_label": "title",
                "block_content": "Title of Document",
                "block_bbox": [50, 50, 200, 70],
            },
            {
                "block_label": "text",
                "block_content": "This is paragraph text.",
                "block_bbox": [50, 90, 300, 130],
            },
            {
                "block_label": "paragraph_title",
                "block_content": "Subheading",
                "block_bbox": [50, 150, 180, 170],
            },
        ],
        "layout_det_res": {
            "boxes": [
                {"label": "title", "score": 0.99},
                {"label": "text", "score": 0.97},
                {"label": "paragraph_title", "score": 0.96},
            ],
        },
    }


@pytest.fixture
def mock_paddlex_ocr_pipeline(
    mock_paddlex_ocr_result: dict,
) -> Generator[MagicMock, None, None]:
    """Create and patch a mock PaddleX General OCR pipeline."""
    pipeline: MagicMock = MagicMock()
    pipeline.predict.return_value = [mock_paddlex_ocr_result]  # single page
    with patch("paddlex_helpers._get_paddlex_model", return_value=pipeline):
        yield pipeline


@pytest.fixture
def mock_paddlex_structure_v3_pipeline(
    mock_paddlex_structure_v3_result: dict,
) -> Generator[MagicMock, None, None]:
    """Create and patch a mock PaddleX PP-StructureV3 pipeline."""
    pipeline: MagicMock = MagicMock()
    pipeline.predict.return_value = mock_paddlex_structure_v3_result
    with patch(
        "paddlex_helpers._get_paddlex_structure_v3_model", return_value=pipeline
    ):
        yield pipeline


@pytest.fixture
def mock_create_pipeline(
    mock_paddlex_ocr_pipeline: MagicMock,
    mock_paddlex_structure_v3_pipeline: MagicMock,
) -> MagicMock:
    """Mock PaddleX create_pipeline to return appropriate pipelines."""

    def mock_create(name: str, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        if name == "layout_parsing":
            return mock_paddlex_structure_v3_pipeline
        elif name == "ocr":
            return mock_paddlex_ocr_pipeline
        else:
            return mock_paddlex_ocr_pipeline

    return mock_create  # type: ignore[return-value]


# ── Test files ────────────────────────────────────────────────────────────
@pytest.fixture
def test_pdf_file() -> Path:
    """Return a temporary PDF file (minimal, valid-ish)."""
    pdf_content = b"""%PDF-1.4
1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj
2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj
3 0 obj <</Type /Page /MediaBox [0 0 612 792] /Parent 2 0 R>> endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer <</Size 4 /Root 1 0 R>>
startxref
190
%%EOF"""
    tmp_path = Path(tempfile.gettempdir()) / "test_document.pdf"
    with open(tmp_path, "wb") as f:
        f.write(pdf_content)
    return tmp_path


@pytest.fixture
def mock_pdf_to_images() -> Generator[MagicMock, None, None]:
    """Mock the _pdf_to_images function to return a single dummy image path."""
    with patch("paddlex_helpers._pdf_to_images") as mock:
        mock.return_value = [
            str(Path(tempfile.gettempdir()) / "page-1.png"),
        ]
        yield mock
