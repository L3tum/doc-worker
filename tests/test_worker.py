"""
Doc-Worker — Tests for worker.py sidecar generation.

Tests generate_native_sidecar with mocked PaddleX pipelines.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from worker import generate_native_sidecar, handle_docling


# ── generate_native_sidecar tests ─────────────────────────────────────────
class TestGenerateNativeSidecar:
    def test_sidecar_with_structure(self):
        """Test sidecar generation produces structured markdown + JSON."""
        mock_pages = [
            {
                "page": 1,
                "text": "Title of Document\nThis is a paragraph.",
                "blocks": [{"text": "Title", "bbox": [0, 0, 0, 0], "confidence": 0.99}],
                "structured_blocks": [
                    {
                        "type": "title",
                        "text": "Title of Document",
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.99,
                    },
                    {
                        "type": "text",
                        "text": "This is a paragraph.",
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.95,
                    },
                ],
            },
            {
                "page": 2,
                "text": "Page 2 content",
                "blocks": [],
                "structured_blocks": [
                    {
                        "type": "text",
                        "text": "Page 2 content",
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.90,
                    },
                ],
            },
        ]

        with patch("worker.run_paddlex_structure_v3", return_value=mock_pages):
            with patch("worker.blocks_to_markdown") as mock_md:
                mock_md.return_value = "# Title of Document\n\nThis is a paragraph."
                pdf_path = Path(tempfile.gettempdir()) / "test.pdf"
                pdf_path.write_text("fake")

                # Temporarily set DOCLING_OUT to a temp dir
                import worker

                old = worker.DOCLING_OUT
                worker.DOCLING_OUT = Path(tempfile.gettempdir()) / "sidecar_test"
                try:
                    result = generate_native_sidecar(pdf_path)

                    assert result is True

                    # Check JSON sidecar — output goes to DOCLING_OUT/{stem}/{stem}.json
                    sidecar_dir = worker.DOCLING_OUT / "test"
                    json_path = sidecar_dir / "test.json"
                    assert json_path.exists()
                    with open(json_path) as f:
                        data = json.load(f)
                    assert len(data["pages"]) == 2
                    assert data["pages"][0]["page"] == 1
                    assert "markdown" in data["pages"][0]
                    assert "structured_blocks" in data["pages"][0]
                    assert len(data["pages"][0]["structured_blocks"]) == 2

                    # Check Markdown sidecar
                    md_path = sidecar_dir / "test.md"
                    assert md_path.exists()
                    md_content = md_path.read_text()
                    assert "## Page 1" in md_content
                    assert "# Title of Document" in md_content
                    assert "## Page 2" in md_content

                finally:
                    worker.DOCLING_OUT = old

    def test_sidecar_failure(self):
        """Test sidecar generation returns False on exception."""
        with patch(
            "paddlex_helpers.run_paddlex_structure_v3",
            side_effect=Exception("Test error"),
        ):
            pdf_path = Path(tempfile.gettempdir()) / "test.pdf"
            pdf_path.write_text("fake")

            old = os.getenv("DOCLING_DIR", "")
            os.environ["DOCLING_DIR"] = tempfile.gettempdir() + "/sidecar_test_fail"
            try:
                result = generate_native_sidecar(pdf_path)
                assert result is False
            finally:
                if old:
                    os.environ["DOCLING_DIR"] = old
                else:
                    del os.environ["DOCLING_DIR"]

    def test_handle_docling_native(self):
        """Test handle_docling with native mode."""
        import worker

        old_mode = worker.DOCLING_MODE
        worker.DOCLING_MODE = "native"
        try:
            with patch(
                "worker.generate_native_sidecar", return_value=True
            ) as mock_sidecar:
                result = handle_docling(Path("test.pdf"))
                assert result is True
                mock_sidecar.assert_called_once()
        finally:
            worker.DOCLING_MODE = old_mode

    def test_handle_docling_off(self):
        """Test handle_docling skips when mode is off."""
        import worker

        old_mode = worker.DOCLING_MODE
        worker.DOCLING_MODE = "off"
        try:
            with patch("worker.generate_native_sidecar") as mock_sidecar:
                result = handle_docling(Path("test.pdf"))
                assert result is True
                mock_sidecar.assert_not_called()
        finally:
            worker.DOCLING_MODE = old_mode

    def test_handle_docling_required_fail(self):
        """Test handle_docling returns False when required mode fails."""
        import worker

        old_mode = worker.DOCLING_MODE
        worker.DOCLING_MODE = "required"
        try:
            with patch("worker.call_docling_convert", return_value=False):
                result = handle_docling(Path("test.pdf"))
                assert result is False
        finally:
            worker.DOCLING_MODE = old_mode

    def test_handle_docling_best_effort_fail(self):
        """Test handle_docling returns True (continues) when best_effort fails."""
        with patch("worker.call_docling_convert", return_value=False):
            old = os.getenv("DOCLING_MODE", "")
            os.environ["DOCLING_MODE"] = "best_effort"
            try:
                result = handle_docling(Path("test.pdf"))
                assert result is True  # continues despite failure
            finally:
                if old:
                    os.environ["DOCLING_MODE"] = old
                else:
                    del os.environ["DOCLING_MODE"]
