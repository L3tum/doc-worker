"""
Pytest configuration and shared fixtures for Doc-Worker tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_pdf_bytes():
    """Return minimal valid PDF bytes for testing."""
    # Minimal PDF 1.4 with one empty page
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer
<< /Size 4 /Root 1 0 R >>
startxref
190
%%EOF"""


@pytest.fixture
def sample_png_bytes():
    """Return minimal valid PNG bytes for testing."""
    # 1x1 red pixel PNG
    return bytes(
        [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,  # PNG signature
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,  # IHDR chunk
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,
            0x08,
            0x02,
            0x00,
            0x00,
            0x00,
            0x90,
            0x77,
            0x53,
            0xDE,
            0x00,
            0x00,
            0x00,
            0x0C,
            0x49,
            0x44,
            0x41,  # IDAT chunk
            0x54,
            0x08,
            0xD7,
            0x63,
            0xF8,
            0xFF,
            0xFF,
            0xFF,
            0x00,
            0x05,
            0xFE,
            0x02,
            0xFE,
            0xA7,
            0x9E,
            0x9D,
            0x89,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,  # IEND chunk
            0x44,
            0xAE,
            0x42,
            0x60,
            0x82,
        ]
    )


@pytest.fixture
def mock_config(temp_dir):
    """Return a mock Config object with test directories."""
    from server.companion import Config

    config = Config(
        INBOX=temp_dir / "inbox",
        PROCESSING=temp_dir / "processing",
        DONE=temp_dir / "done",
        ERROR=temp_dir / "error",
        OUTPUT_DIR=temp_dir / "output",
        PAPERLESS_CONSUME=temp_dir / "paperless",
        OCR_LANG="deu",
        OCR_RUNTIME="cpu",
        PROCESSING_MODE="auto",
        PADDLE_OCR_LANG="ch",
        PADDLE_VL_MODEL="PaddleOCR-VL-1.5B",
        PADDLE_DEVICE="cpu",
        POLL_INTERVAL=1,
        MAX_RETRIES=1,
        RETRY_DELAY=1,
        MAX_FILE_SIZE_MB=100,
        MAX_CONCURRENT_JOBS=1,
        API_HOST="127.0.0.1",
        API_PORT=8765,
        API_ENABLED=True,
        LOG_LEVEL="WARNING",
        LOG_JSON=False,
    )
    config.ensure_directories()
    return config


@pytest.fixture
def mock_model_manager():
    """Return a mock ModelManager that doesn't actually load models."""
    manager = MagicMock()
    manager.is_loaded = True
    manager._pp_ocr_engine = MagicMock()
    manager._vl_model = None
    manager._vl_processor = None

    def mock_run_ocr(input_data, ctx):
        return {
            "text": "Hello World",
            "text_blocks": [
                {
                    "text": "Hello World",
                    "bbox": [[0, 0], [100, 0], [100, 20], [0, 20]],
                    "confidence": 0.95,
                }
            ],
            "block_count": 1,
            "avg_confidence": 0.95,
            "model": "pp_ocr",
        }

    def mock_run_vl(input_data, ctx):
        return {
            "markdown": "# Hello World\n\nThis is a test document.",
            "layout": [
                {"type": "heading", "level": 1, "text": "Hello World", "line": 0}
            ],
            "tables": [],
            "formulas": [],
            "text": "Hello World This is a test document.",
            "confidence": 0.9,
            "model": "paddle_vl",
        }

    manager.run_ocr = mock_run_ocr
    manager.run_vl_understanding = mock_run_vl
    return manager


@pytest.fixture(autouse=True)
def reset_companion_singletons():
    """Reset companion module singletons between tests."""
    import server.companion as companion

    companion._config = None
    companion._metrics = None
    companion._model_manager = None

    yield

    companion._config = None
    companion._metrics = None
    companion._model_manager = None
