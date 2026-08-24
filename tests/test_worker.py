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

import pytest

from worker import (
    generate_native_sidecar,
    handle_docling,
    move_to_error,
    process_file,
    process_with_retries,
    recover_leftover_files,
)


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

    def test_sidecar_with_numpy_blocks(self):
        """Sidecar generation handles numpy-typed structured_blocks.

        Exercises the _json_default safety net: even if a numpy type leaks past
        the source-level fix, json.dump(default=...) must not crash.
        """
        import numpy as np

        mock_pages = [
            {
                "page": 1,
                "text": "test",
                "blocks": [],
                "structured_blocks": [
                    {
                        "type": "text",
                        "text": "test",
                        "bbox": np.array([0.0, 0.0, 100.0, 50.0], dtype=np.float32),
                        "confidence": np.float32(0.85),
                    }
                ],
            }
        ]

        with patch("worker.run_paddlex_structure_v3", return_value=mock_pages):
            with patch("worker.blocks_to_markdown", return_value="# heading"):
                pdf_path = Path(tempfile.gettempdir()) / "test_np.pdf"
                pdf_path.write_text("fake")

                import worker

                old = worker.DOCLING_OUT
                worker.DOCLING_OUT = Path(tempfile.gettempdir()) / "sidecar_test_np"
                try:
                    assert generate_native_sidecar(pdf_path) is True
                    json_path = worker.DOCLING_OUT / "test_np" / "test_np.json"
                    assert json_path.exists()
                    with open(json_path) as f:
                        data = json.load(f)
                    sb = data["pages"][0]["structured_blocks"][0]
                    assert sb["bbox"] == [0.0, 0.0, 100.0, 50.0]
                    # .tolist() on the float32 scalar yields the float64 of the
                    # nearest float32, so compare against that exact value.
                    assert sb["confidence"] == float(np.float32(0.85))
                finally:
                    worker.DOCLING_OUT = old

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


# ── move_to_error tests ──────────────────────────────────────────────────
class TestMoveToError:
    def test_move_to_error_creates_unique_name(self):
        """No overwrite when file already exists in ERROR/."""

        import worker

        tmp_dir = Path(tempfile.mkdtemp())
        worker.ERROR = tmp_dir

        src = Path(tempfile.mktemp(suffix=".pdf"))
        src.write_text("test pdf content")
        dest = tmp_dir / src.name
        dest.write_text("existing error file")

        move_to_error(src, "test reason")

        # Original dest should be untouched
        assert dest.read_text() == "existing error file"
        # New file with timestamp should exist
        error_files = [f for f in tmp_dir.glob("*.pdf") if f != dest]
        assert len(error_files) == 1
        assert error_files[0].read_text() == "test pdf content"

    def test_move_to_error_missing_source(self):
        """Gracefully handles missing source file."""
        import sys
        from io import StringIO

        import worker

        tmp_dir = Path(tempfile.mkdtemp())
        worker.ERROR = tmp_dir

        # Capture stderr to check for error log
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            move_to_error(Path("/nonexistent/path.pdf"), "missing file")
            stderr_output = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr

        # Should log an error about the missing file
        assert "Cannot move missing file to ERROR/" in stderr_output
        # No files should be created
        assert not any(tmp_dir.iterdir())

    def test_move_to_error_log_reason(self):
        """Reason is logged in the output."""
        import sys
        from io import StringIO

        import worker

        tmp_dir = Path(tempfile.mkdtemp())
        worker.ERROR = tmp_dir

        src = Path(tempfile.mktemp(suffix=".pdf"))
        src.write_text("test")

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            move_to_error(src, "my custom reason")
            stdout_output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "my custom reason" in stdout_output


# ── recover_leftover_files tests ──────────────────────────────────────────
class TestRecoverLeftoverFiles:
    def test_recovers_leftovers(self):
        """Files in PROCESSING/ are moved to ERROR/."""
        import worker

        tmp_processing = Path(tempfile.mkdtemp())
        tmp_error = Path(tempfile.mkdtemp())
        worker.PROCESSING = tmp_processing
        worker.ERROR = tmp_error

        leftover = tmp_processing / "leftover.pdf"
        leftover.write_text("leftover content")

        recover_leftover_files()

        assert not leftover.exists()
        recovered = tmp_error / "leftover.pdf"
        assert recovered.exists()
        assert recovered.read_text() == "leftover content"

    def test_no_leftovers(self):
        """Does nothing when PROCESSING/ is empty."""
        import worker

        tmp_processing = Path(tempfile.mkdtemp())
        tmp_error = Path(tempfile.mkdtemp())
        worker.PROCESSING = tmp_processing
        worker.ERROR = tmp_error

        recover_leftover_files()

        # No files should be moved
        assert not any(tmp_error.iterdir())

    def test_no_overwrite_in_error(self):
        """Uses timestamped name if file already in ERROR/."""
        import worker

        tmp_processing = Path(tempfile.mkdtemp())
        tmp_error = Path(tempfile.mkdtemp())
        worker.PROCESSING = tmp_processing
        worker.ERROR = tmp_error

        # Create existing file in ERROR/
        existing = tmp_error / "conflict.pdf"
        existing.write_text("already here")

        # Create leftover in PROCESSING/ with same name
        leftover = tmp_processing / "conflict.pdf"
        leftover.write_text("leftover content")

        recover_leftover_files()

        # Original file should be untouched
        assert existing.read_text() == "already here"
        # New timestamped file should exist
        error_pdfs = list(tmp_error.glob("*.pdf"))
        assert len(error_pdfs) == 2
        # One should have the timestamp suffix
        timestamped = [f for f in error_pdfs if "_20" in f.name]
        assert len(timestamped) == 1


# ── process_file with model init failure ─────────────────────────────────
class TestProcessFileModelInitFailure:
    def test_model_init_failure_returns_false(self):
        """process_file() returns False when model init fails during native sidecar.

        'native' mode makes a sidecar failure non-fatal (handle_docling
        returns True), so the pipeline proceeds to OCR. OCR is mocked to
        succeed here so the test isolates the sidecar-failure-then-continue
        path without running a real OCR pass.
        """
        import worker

        with patch.object(worker, "DOCLING_MODE", "native"):
            with patch(
                "worker.run_paddlex_structure_v3",
                side_effect=RuntimeError(
                    "No available model hosting platforms detected."
                ),
            ):
                with (
                    patch("worker.run_ocrmypdf") as mock_ocr,
                    patch("worker.push_to_paperless", return_value=True),
                ):
                    tmp_inbox = Path(tempfile.mkdtemp())
                    tmp_processing = Path(tempfile.mkdtemp())
                    worker.INBOX = tmp_inbox
                    worker.PROCESSING = tmp_processing
                    worker.DONE = Path(tempfile.mkdtemp())
                    worker.ERROR = Path(tempfile.mkdtemp())
                    worker.DOCLING_OUT = Path(tempfile.mkdtemp())
                    worker.PAPERLESS_CONSUME = Path(tempfile.mkdtemp())

                    test_pdf = tmp_inbox / "test.pdf"
                    test_pdf.write_bytes(b"%PDF-1.4 fake")

                    result = process_file(test_pdf)
                    # Sidecar failed but native mode continues; mocked OCR +
                    # push succeed, so the pipeline completes.
                    assert result is True
                    mock_ocr.assert_called_once()

    def test_process_file_ocr_failure_raises(self):
        """process_file() re-raises a ZeroDivisionError from OCR (non-retryable).

        Previously OCR failures were swallowed (returned False) so the retry
        loop couldn't classify them. Now they propagate for fast-fail.
        """
        import worker

        with patch.object(worker, "DOCLING_MODE", "off"):
            with (
                patch(
                    "worker.run_ocrmypdf",
                    side_effect=ZeroDivisionError("float division by zero"),
                ),
                patch("worker.push_to_paperless", return_value=True),
            ):
                tmp_inbox = Path(tempfile.mkdtemp())
                tmp_processing = Path(tempfile.mkdtemp())
                worker.INBOX = tmp_inbox
                worker.PROCESSING = tmp_processing
                worker.DONE = Path(tempfile.mkdtemp())
                worker.ERROR = Path(tempfile.mkdtemp())
                worker.DOCLING_OUT = Path(tempfile.mkdtemp())
                worker.PAPERLESS_CONSUME = Path(tempfile.mkdtemp())

                test_pdf = tmp_inbox / "test.pdf"
                test_pdf.write_bytes(b"%PDF-1.4 fake")

                with pytest.raises(ZeroDivisionError):
                    process_file(test_pdf)

    def test_native_sidecar_failure_continues_pipeline(self):
        """Native sidecar failure doesn't abort pipeline in best_effort mode."""
        import worker

        old_mode = worker.DOCLING_MODE
        worker.DOCLING_MODE = "best_effort"
        try:
            with patch("worker.call_docling_convert", return_value=False):
                # Sidecar generation failed but best_effort continues
                result = handle_docling(Path("test.pdf"))
                assert result is True  # continues despite sidecar failure
        finally:
            worker.DOCLING_MODE = old_mode


# ── process_with_retries (non-retryable fast-fail) ─────────────────────────
class TestProcessWithRetries:
    def test_non_retryable_error_skips_retries(self):
        """A non-retryable error (ZeroDivisionError) stops immediately.

        process_file is called exactly once (no retry sleep), and the file is
        moved to ERROR/ with a reason containing 'non-retryable'.
        """
        import worker

        tmp_inbox = Path(tempfile.mkdtemp())
        tmp_processing = Path(tempfile.mkdtemp())
        tmp_error = Path(tempfile.mkdtemp())
        old_inbox, old_proc, old_err = worker.INBOX, worker.PROCESSING, worker.ERROR
        worker.INBOX = tmp_inbox
        worker.PROCESSING = tmp_processing
        worker.ERROR = tmp_error
        try:
            test_pdf = tmp_inbox / "test.pdf"
            test_pdf.write_bytes(b"%PDF-1.4 fake")

            with (
                patch(
                    "worker.process_file",
                    side_effect=ZeroDivisionError("float division by zero"),
                ) as mock_pf,
                patch("worker.move_to_error") as mock_move,
            ):
                # retry_delay=0 so no real sleep even if a retry were attempted
                process_with_retries(test_pdf, max_retries=3, retry_delay=0)

            # Non-retryable → exactly one attempt, no retry loop.
            assert mock_pf.call_count == 1
            # Landed in ERROR/ with a non-retryable reason.
            assert mock_move.call_count == 1
            _, reason = mock_move.call_args[0]
            assert "non-retryable" in reason
        finally:
            worker.INBOX, worker.PROCESSING, worker.ERROR = old_inbox, old_proc, old_err

    def test_retryable_error_retries_then_gives_up(self):
        """A retryable (non-exception) failure is retried up to max_retries.

        process_file returns False (not raising) → all attempts run, then the
        file is moved to ERROR/ with 'max retries reached'.
        """
        import worker

        tmp_inbox = Path(tempfile.mkdtemp())
        tmp_processing = Path(tempfile.mkdtemp())
        tmp_error = Path(tempfile.mkdtemp())
        old_inbox, old_proc, old_err = worker.INBOX, worker.PROCESSING, worker.ERROR
        worker.INBOX = tmp_inbox
        worker.PROCESSING = tmp_processing
        worker.ERROR = tmp_error
        try:
            test_pdf = tmp_inbox / "test.pdf"
            test_pdf.write_bytes(b"%PDF-1.4 fake")

            with (
                patch("worker.process_file", return_value=False) as mock_pf,
                patch("worker.move_to_error") as mock_move,
            ):
                process_with_retries(test_pdf, max_retries=3, retry_delay=0)

            assert mock_pf.call_count == 3
            assert mock_move.call_count == 1
            _, reason = mock_move.call_args[0]
            assert reason == "max retries reached"
        finally:
            worker.INBOX, worker.PROCESSING, worker.ERROR = old_inbox, old_proc, old_err
