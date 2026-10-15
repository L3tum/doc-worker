"""
Doc-Worker — Tests for worker.py sidecar generation.

Tests generate_native_sidecar with mocked PaddleX pipelines.

Moved from tests/test_worker.py to tests/unit/test_worker.py: CI
(docker.yaml) runs `python -m pytest -q tests/`, so this file is now
executed in CI in addition to local dev runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from worker import (
    flatten_acroform_if_needed,
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

        Exercises the json_default safety net (serialization.json_default):
        even if a numpy type leaks past the source-level fix,
        json.dump(default=...) must not crash.
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

    def test_stale_flattened_intermediate_unlinked(self, monkeypatch):
        """A stale marker-named intermediate is unlinked (not promoted) —
        but only when its sibling original (foo.pdf) is also in
        PROCESSING/.

        With the original present, the worker-owned AcroForm pre-flatten
        copy is re-creatable from it, so it must not be promoted to ERROR/
        as if it were a user file; the original is handled by the normal
        recovery. (Without the original the intermediate is preserved —
        see test_flattened_without_original_preserved.)
        """
        import worker

        tmp_processing = Path(tempfile.mkdtemp())
        tmp_error = Path(tempfile.mkdtemp())
        monkeypatch.setattr(worker, "PROCESSING", tmp_processing)
        monkeypatch.setattr(worker, "ERROR", tmp_error)

        intermediate_name = "foo__acroform_flattened__deadbeef.pdf"
        (tmp_processing / "foo.pdf").write_text("original")
        (tmp_processing / intermediate_name).write_text("worker intermediate")

        recover_leftover_files()

        # The original is recovered as usual...
        assert (tmp_error / "foo.pdf").exists()
        # ...and the intermediate is deleted rather than promoted anywhere.
        assert not (tmp_processing / intermediate_name).exists()
        flattened_in_error = [
            f for f in tmp_error.iterdir() if worker._FLATTENED_MARKER in f.name
        ]
        assert not flattened_in_error

    def test_user_file_named_like_old_intermediate_preserved(self, monkeypatch):
        """An old-style *_flattened.pdf user file is preserved, not unlinked.

        The retired fixed-name scheme (<stem>_flattened.pdf) made recovery
        unlink such a file whenever the sibling original was also present
        — silent data loss. Under the marker scheme a plain user file is
        just data: it must flow to ERROR/ with its content intact.
        """
        import worker

        tmp_processing = Path(tempfile.mkdtemp())
        tmp_error = Path(tempfile.mkdtemp())
        monkeypatch.setattr(worker, "PROCESSING", tmp_processing)
        monkeypatch.setattr(worker, "ERROR", tmp_error)

        (tmp_processing / "foo.pdf").write_text("original")
        (tmp_processing / "foo_flattened.pdf").write_text("user document")

        recover_leftover_files()

        # Both are user files: both must be preserved in ERROR/, intact.
        assert (tmp_error / "foo.pdf").exists()
        recovered = tmp_error / "foo_flattened.pdf"
        assert recovered.exists()
        assert recovered.read_text() == "user document"

    def test_flattened_without_original_preserved(self, monkeypatch):
        """A marker-named intermediate whose original is missing is preserved.

        Without foo.pdf next to it, the worker intermediate may be the
        only surviving copy of the user's document — it must flow through
        to ERROR/ with its content intact, not be silently unlinked.
        """
        import worker

        tmp_processing = Path(tempfile.mkdtemp())
        tmp_error = Path(tempfile.mkdtemp())
        monkeypatch.setattr(worker, "PROCESSING", tmp_processing)
        monkeypatch.setattr(worker, "ERROR", tmp_error)

        intermediate_name = "foo__acroform_flattened__deadbeef.pdf"
        (tmp_processing / intermediate_name).write_text("worker intermediate")

        recover_leftover_files()

        # Gone from PROCESSING/, preserved in ERROR/ with content intact.
        assert not (tmp_processing / intermediate_name).exists()
        recovered = tmp_error / intermediate_name
        assert recovered.exists()
        assert recovered.read_text() == "worker intermediate"

    def test_marker_named_directory_skipped(self, monkeypatch):
        """A marker-named *directory* in PROCESSING/ must not crash startup.

        Path.unlink() raises IsADirectoryError on a directory; recovery must
        skip non-files (same guard as the AcroForm flatten cleanup) instead
        of crash-looping the worker at boot.
        """
        import worker

        tmp_processing = Path(tempfile.mkdtemp())
        tmp_error = Path(tempfile.mkdtemp())
        monkeypatch.setattr(worker, "PROCESSING", tmp_processing)
        monkeypatch.setattr(worker, "ERROR", tmp_error)

        intermediate_name = "foo__acroform_flattened__deadbeef.pdf"
        (tmp_processing / intermediate_name).mkdir()  # stray directory, not a file
        (tmp_processing / "foo.pdf").write_text("original")

        recover_leftover_files()  # must not raise IsADirectoryError

        # The stray directory is left in place (never silently deleted)...
        assert (tmp_processing / intermediate_name).is_dir()
        # ...and the real leftover is recovered as usual.
        assert (tmp_error / "foo.pdf").exists()

    def test_leftovers_recovered_in_name_order(self, monkeypatch):
        """Recovery processes leftovers in name order, so the outcome is
        deterministic when an original and its intermediate both survive a
        crash (iterdir() order is filesystem-dependent)."""
        import worker

        tmp_processing = Path(tempfile.mkdtemp())
        tmp_error = Path(tempfile.mkdtemp())
        recorded: list[str] = []
        monkeypatch.setattr(worker, "PROCESSING", tmp_processing)
        monkeypatch.setattr(worker, "ERROR", tmp_error)
        monkeypatch.setattr(worker, "log", lambda msg: recorded.append(str(msg)))

        (tmp_processing / "b.pdf").write_text("b")
        (tmp_processing / "a.pdf").write_text("a")

        recover_leftover_files()

        idx_a = next(i for i, m in enumerate(recorded) if "a.pdf" in m)
        idx_b = next(i for i, m in enumerate(recorded) if "b.pdf" in m)
        assert idx_a < idx_b
        assert (tmp_error / "a.pdf").exists()
        assert (tmp_error / "b.pdf").exists()


# ── AcroForm pre-flatten (Version-Specific Hack #5, AGENTS.md) ──────────────
class TestFlattenAcroform:
    """flatten_acroform_if_needed: bake filled form values in before OCR.

    The AcroForm fixture (tests/fixtures/acroform_filled.pdf) is generated
    by tests/fixtures/generate_acroform_pdf.py: one page with baked-in
    label text in the content stream plus two filled /Tx fields (Name /
    City) carrying /V values, /AP appearance streams and
    AcroForm.NeedAppearances=true — i.e. a realistic fill-in.
    """

    FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "acroform_filled.pdf"

    @pytest.fixture()
    def pdfium(self):
        return pytest.importorskip("pypdfium2")

    @staticmethod
    def _render_hash(path: Path, pdfium) -> str:
        doc = pdfium.PdfDocument(str(path))
        try:
            return hashlib.sha256(doc[0].render(scale=1).to_pil().tobytes()).hexdigest()
        finally:
            doc.close()

    def test_acroform_flattened_copy(self, tmp_path, pdfium):
        """Filled AcroForm → flattened copy without forms; original untouched.

        The red guard: a plain (no init_forms) render of the original must
        differ from a render of the flattened copy — the values are only
        visible because flatten() baked them into the content stream.
        """
        import worker

        src = tmp_path / "form.pdf"
        shutil.copy(self.FIXTURE, src)
        src_hash = self._render_hash(src, pdfium)

        out = flatten_acroform_if_needed(src)
        assert out is not None
        assert out.exists()
        # Marker + short-uuid naming: collision-free with user files.
        assert out.name.startswith("form")
        assert worker._FLATTENED_MARKER in out.name
        assert out.name.endswith(".pdf")
        assert out.name != "form_flattened.pdf"
        assert worker._ACROFORM_FLATTEN_STATE == "active"

        # The copy has no AcroForm left; the original still does.
        flat_doc = pdfium.PdfDocument(str(out))
        assert flat_doc.init_forms() is False
        flat_doc.close()
        src_doc = pdfium.PdfDocument(str(src))
        assert src_doc.init_forms() is True
        src_doc.close()

        # Red guard: values now visible in the flattened render only.
        assert self._render_hash(out, pdfium) != src_hash

    def test_no_form_pdf_returns_none(self, tmp_path, pdfium, monkeypatch):
        """A PDF without AcroForms → None (original OCR'd as-is).

        Also pins the unpatched → skipped state transition: a formless
        document sets _ACROFORM_FLATTEN_STATE to "skipped" (the sticky
        state is reset to "unpatched" first so the assertion is
        independent of test order).
        """
        import worker

        plain = tmp_path / "plain.pdf"
        doc = pdfium.PdfDocument.new()
        doc.new_page(612, 792)
        doc.save(str(plain))
        doc.close()

        monkeypatch.setattr(worker, "_ACROFORM_FLATTEN_STATE", "unpatched")
        assert flatten_acroform_if_needed(plain) is None
        assert not any(worker._FLATTENED_MARKER in p.name for p in tmp_path.iterdir())
        assert worker._ACROFORM_FLATTEN_STATE == "skipped"

    def test_corrupt_file_returns_none(self, tmp_path):
        """An unopenable file → None (fail-soft), not an exception.

        Passes via different branches per environment: pypdfium2 not
        importable → the ImportError path; importable → PdfDocument
        raises → the 'cannot open' path. Both must stay fail-soft None
        (no exception, no partial intermediate file).
        """
        import worker

        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"not a pdf at all")
        assert flatten_acroform_if_needed(corrupt) is None
        assert not any(worker._FLATTENED_MARKER in p.name for p in tmp_path.iterdir())

    def test_stray_directory_at_intermediate_path(self, tmp_path, monkeypatch):
        """A stray directory at the intermediate path cannot make cleanup raise.

        The fake pypdfium2 save() raises (what a directory at the target
        path would produce); the except-branch guard must only unlink when
        the path is a regular file, so the directory survives, no
        IsADirectoryError escapes, and the original is OCR'd as-is.
        """
        import sys
        import types
        from unittest.mock import MagicMock

        import worker

        # Fix the uuid so the intermediate name is deterministic.
        monkeypatch.setattr(
            worker,
            "uuid",
            types.SimpleNamespace(
                uuid4=lambda: types.SimpleNamespace(hex="deadbeefdeadbeef")
            ),
        )

        src = tmp_path / "form.pdf"
        src.write_bytes(b"%PDF-1.4 fake")
        stray_dir = tmp_path / "form__acroform_flattened__deadbeef.pdf"
        stray_dir.mkdir()

        fake_doc = MagicMock()
        fake_doc.init_forms.return_value = True
        fake_doc.save.side_effect = OSError("simulated save failure")
        fake_pypdfium2 = MagicMock(PdfDocument=MagicMock(return_value=fake_doc))

        with patch.dict(sys.modules, {"pypdfium2": fake_pypdfium2}):
            assert flatten_acroform_if_needed(src) is None
        assert stray_dir.is_dir()

    @staticmethod
    def _point_worker_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Point all worker pipeline globals at fresh tmp dirs.

        Uses monkeypatch so the module globals are restored automatically
        (the old direct assignments leaked into other tests).
        """
        import worker

        for attr, subdir in (
            ("INBOX", "inbox"),
            ("PROCESSING", "processing"),
            ("DONE", "done"),
            ("ERROR", "error"),
            ("DOCLING_OUT", "docling"),
            ("PAPERLESS_CONSUME", "paperless"),
        ):
            path = tmp_path / subdir
            path.mkdir(parents=True)
            monkeypatch.setattr(worker, attr, path)

    def test_process_file_feeds_flattened_copy_to_ocr(
        self, tmp_path, pdfium, monkeypatch
    ):
        """process_file OCRs the flattened copy; the original lands in DONE/.

        The flattened intermediate is unlinked on the success path.
        """
        import worker

        self._point_worker_at(tmp_path, monkeypatch)
        with (
            patch.object(worker, "DOCLING_MODE", "off"),
            patch("worker.run_ocrmypdf") as mock_ocr,
            patch("worker.push_to_paperless", return_value=True),
        ):
            src = worker.INBOX / "form.pdf"
            shutil.copy(self.FIXTURE, src)

            assert process_file(src) is True

        # OCR received the flattened copy (not the original).
        ocr_input: Path = mock_ocr.call_args.args[0]
        assert ocr_input.parent == worker.PROCESSING
        assert ocr_input.name.startswith("form")
        assert worker._FLATTENED_MARKER in ocr_input.name
        # The intermediate is cleaned up on the success path.
        assert not ocr_input.exists()
        # The original (unflattened) file is what lands in DONE/.
        done_pdfs = sorted(p.name for p in worker.DONE.glob("form*.pdf"))
        assert done_pdfs == ["form.pdf"]

    def test_process_file_ocr_failure_unlinks_flattened(
        self, tmp_path, pdfium, monkeypatch
    ):
        """OCR raising after flatten: intermediate unlinked, original kept.

        The flattened copy is a worker-owned intermediate — even when
        run_ocrmypdf raises it must not leak into PROCESSING/ (where
        crash recovery would otherwise promote it to ERROR/). The original
        stays in PROCESSING/ so the retry loop can find it; nothing lands
        in DONE/.
        """
        import worker

        self._point_worker_at(tmp_path, monkeypatch)
        with (
            patch.object(worker, "DOCLING_MODE", "off"),
            patch("worker.run_ocrmypdf", side_effect=RuntimeError("boom")) as mock_ocr,
            patch("worker.push_to_paperless", return_value=True),
        ):
            src = worker.INBOX / "form.pdf"
            shutil.copy(self.FIXTURE, src)

            with pytest.raises(RuntimeError, match="boom"):
                process_file(src)

        # OCR ran against the flattened copy (not the original).
        ocr_input: Path = mock_ocr.call_args.args[0]
        assert ocr_input.parent == worker.PROCESSING
        assert ocr_input.name.startswith("form")
        assert worker._FLATTENED_MARKER in ocr_input.name
        # The intermediate is unlinked on the failure path, the original is
        # left in PROCESSING/ for the retry loop, and DONE/ stays empty.
        assert sorted(p.name for p in worker.PROCESSING.iterdir()) == ["form.pdf"]
        assert not ocr_input.exists()
        assert not any(worker.DONE.iterdir())

    def test_process_file_flatten_none_uses_original(self, tmp_path, monkeypatch):
        """flatten → None (formless PDF / no pypdfium2): OCR the original.

        Success path: run_ocrmypdf receives the original processing path
        and the original filename lands in DONE/.
        """
        import worker

        self._point_worker_at(tmp_path, monkeypatch)
        with (
            patch.object(worker, "DOCLING_MODE", "off"),
            patch("worker.flatten_acroform_if_needed", return_value=None),
            patch("worker.run_ocrmypdf") as mock_ocr,
            patch("worker.push_to_paperless", return_value=True),
        ):
            src = worker.INBOX / "plain.pdf"
            src.write_bytes(b"%PDF-1.4 fake")

            assert process_file(src) is True

        # OCR ran against the original processing path (no flattened copy).
        ocr_input: Path = mock_ocr.call_args.args[0]
        assert ocr_input == worker.PROCESSING / "plain.pdf"
        # Success: the original filename lands in DONE/, PROCESSING/ empty.
        assert sorted(p.name for p in worker.DONE.iterdir()) == ["plain.pdf"]
        assert not any(worker.PROCESSING.iterdir())


# ── process_file: model-init failure (native sidecar) ──────────────────────
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


# ── process_file: sidecar ordering vs. AcroForm pre-flatten (Hack #5) ───────
class TestProcessFileSidecarFlattenOrdering:
    """The sidecar must run on the flattened copy (flatten before sidecar)
    so sidecar and OCR text layer agree for /V-only widgets, while the
    DOCLING_OUT key stays the original filename."""

    @staticmethod
    def _point_worker_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import worker

        for attr, subdir in (
            ("INBOX", "inbox"),
            ("PROCESSING", "processing"),
            ("DONE", "done"),
            ("ERROR", "error"),
            ("DOCLING_OUT", "docling"),
            ("PAPERLESS_CONSUME", "paperless"),
        ):
            path = tmp_path / subdir
            path.mkdir(parents=True)
            monkeypatch.setattr(worker, attr, path)

    def test_native_sidecar_runs_on_flattened_copy_keyed_by_original(
        self, tmp_path, monkeypatch
    ):
        """With flattening active and DOCLING_MODE=native, the native
        sidecar receives the flattened path as pdf_path and
        sidecar_name=<original filename> as kwarg — DOCLING_OUT stays
        keyed by the original stem."""
        import worker

        self._point_worker_at(tmp_path, monkeypatch)
        flattened = worker.PROCESSING / "form_flattened.pdf"
        flattened.write_bytes(b"flattened copy")

        with (
            patch.object(worker, "DOCLING_MODE", "native"),
            patch(
                "worker.flatten_acroform_if_needed", return_value=flattened
            ) as mock_flatten,
            patch("worker.generate_native_sidecar", return_value=True) as mock_sidecar,
            patch("worker.run_ocrmypdf"),
            patch("worker.push_to_paperless", return_value=True),
        ):
            src = worker.INBOX / "form.pdf"
            src.write_bytes(b"%PDF-1.4 fake")

            assert process_file(src) is True

        # Flatten ran on the original processing path...
        mock_flatten.assert_called_once_with(worker.PROCESSING / "form.pdf")
        # ...and the sidecar ran on the flattened copy, keyed by the
        # original name (DOCLING_OUT/<original stem>/).
        mock_sidecar.assert_called_once_with(flattened, sidecar_name="form.pdf")
        # The intermediate is cleaned up on the success path.
        assert not flattened.exists()

    def test_required_sidecar_failure_unlinks_flattened(self, tmp_path, monkeypatch):
        """Sidecar failure in 'required' mode while a flattened copy
        exists: the intermediate is unlinked, so PROCESSING/ holds only
        the original (a leaked copy would be promoted to ERROR/ by crash
        recovery on the next startup)."""
        import worker

        self._point_worker_at(tmp_path, monkeypatch)
        flattened = worker.PROCESSING / "form_flattened.pdf"
        flattened.write_bytes(b"flattened copy")

        with (
            patch.object(worker, "DOCLING_MODE", "required"),
            patch("worker.flatten_acroform_if_needed", return_value=flattened),
            patch("worker.call_docling_convert", return_value=False),
            patch("worker.run_ocrmypdf") as mock_ocr,
        ):
            src = worker.INBOX / "form.pdf"
            src.write_bytes(b"%PDF-1.4 fake")

            assert process_file(src) is False

        # Aborted before OCR; only the original remains in PROCESSING/.
        mock_ocr.assert_not_called()
        assert sorted(p.name for p in worker.PROCESSING.iterdir()) == ["form.pdf"]
        assert not flattened.exists()

    def test_handle_docling_raise_unlinks_flattened(self, tmp_path, monkeypatch):
        """A raising handle_docling backend must not leak the intermediate.

        e.g. call_docling_convert raising RuntimeError on a broken
        `requests` install: the exception propagates (upstream retry /
        ERROR/ classification), but the worker-owned intermediate is
        unlinked first, so it cannot sit in PROCESSING/ until the next
        startup recovery.
        """
        import worker

        self._point_worker_at(tmp_path, monkeypatch)
        flattened = worker.PROCESSING / "form__acroform_flattened__deadbeef.pdf"
        flattened.write_bytes(b"flattened copy")

        with (
            patch.object(worker, "DOCLING_MODE", "off"),
            patch("worker.flatten_acroform_if_needed", return_value=flattened),
            patch(
                "worker.handle_docling",
                side_effect=RuntimeError("broken requests install"),
            ),
        ):
            src = worker.INBOX / "form.pdf"
            src.write_bytes(b"%PDF-1.4 fake")

            with pytest.raises(RuntimeError, match="broken requests install"):
                process_file(src)

        # The intermediate is unlinked; the original stays in PROCESSING/
        # for the retry loop.
        assert not flattened.exists()
        assert sorted(p.name for p in worker.PROCESSING.iterdir()) == ["form.pdf"]


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


# ── run_ocrmypdf idle-timeout regression ─────────────────────────────────
class TestRunOcrmyPdfMarksModelUsed:
    """Regression: the OCRmyPDF plugin loads the shared PaddleX General-OCR
    model lazily inside ocrmypdf.ocr(). run_ocrmypdf must mark the model as
    used so the idle-unload clock tracks this load. In the default
    'best_effort' DOCLING mode this OCR path is the only model load for a
    file, so without the mark the model would never be unloaded (VRAM leak).
    """

    def test_marks_model_used_before_ocr(self, tmp_path):
        import sys
        from unittest.mock import MagicMock

        import worker

        input_pdf = tmp_path / "in.pdf"
        input_pdf.write_bytes(b"%PDF-1.4 fake")
        output_pdf = tmp_path / "out.pdf"
        output_pdf.write_bytes(b"%PDF-1.4 fake output")

        ocrmypdf_mock = MagicMock()
        calls: list[str] = []
        ocrmypdf_mock.ocr.side_effect = lambda *a, **k: calls.append("ocr")

        def fake_mark() -> None:
            calls.append("mark")
            worker._model_idle_tracker._last_used = 123.0

        with patch.dict(sys.modules, {"ocrmypdf": ocrmypdf_mock}):
            with patch.object(worker, "_mark_model_used", side_effect=fake_mark):
                worker.run_ocrmypdf(input_pdf, output_pdf)

        # The idle clock was advanced, and the mark happened before the load.
        assert worker._model_idle_tracker._last_used == 123.0
        assert ocrmypdf_mock.ocr.called
        assert calls.index("mark") < calls.index("ocr")


# ── run_ocrmypdf in-flight guard + kwargs regression ──────────────────────
class TestRunOcrmyPdfInFlightGuard:
    """run_ocrmypdf must hold the in-flight guard across the entire
    ocrmypdf.ocr() call (the plugin forces jobs=1, so one interval
    deterministically covers every plugin predict in get_deskew/generate_ocr)
    and stamp the idle clock both before and after the call."""

    def test_guard_held_across_ocr_with_double_stamp(self, tmp_path):
        import sys
        from unittest.mock import MagicMock

        import paddlex_helpers
        import worker

        input_pdf = tmp_path / "in.pdf"
        input_pdf.write_bytes(b"%PDF-1.4 fake")
        output_pdf = tmp_path / "out.pdf"
        output_pdf.write_bytes(b"%PDF-1.4 fake output")

        ocrmypdf_mock = MagicMock()
        events: list[tuple[str, int]] = []

        def fake_ocr(*args, **kwargs) -> None:
            events.append(("ocr", paddlex_helpers.model_in_flight_count()))

        ocrmypdf_mock.ocr.side_effect = fake_ocr

        def fake_mark() -> None:
            events.append(("mark", paddlex_helpers.model_in_flight_count()))

        with patch.dict(sys.modules, {"ocrmypdf": ocrmypdf_mock}):
            with patch.object(worker, "_mark_model_used", side_effect=fake_mark):
                worker.run_ocrmypdf(input_pdf, output_pdf)

        # Pre-stamp before the guard is up, the whole ocr() call under the
        # guard, completion stamp after release.
        assert events == [("mark", 0), ("ocr", 1), ("mark", 0)]
        assert paddlex_helpers.model_in_flight_count() == 0

    def test_ocr_kwargs_pin_no_paddle_use_gpu(self, tmp_path):
        """ocrmypdf.ocr() must not pass paddle_use_gpu= (the dead kwarg whose
        removal this change made — re-adding it would re-introduce the
        device-confusion bug) and must keep force_ocr=True."""
        import sys
        from unittest.mock import MagicMock

        import worker

        input_pdf = tmp_path / "in.pdf"
        input_pdf.write_bytes(b"%PDF-1.4 fake")
        output_pdf = tmp_path / "out.pdf"
        output_pdf.write_bytes(b"%PDF-1.4 fake output")

        ocrmypdf_mock = MagicMock()
        with patch.dict(sys.modules, {"ocrmypdf": ocrmypdf_mock}):
            worker.run_ocrmypdf(input_pdf, output_pdf)

        ocrmypdf_mock.ocr.assert_called_once()
        args, kwargs = ocrmypdf_mock.ocr.call_args
        assert args == (input_pdf, output_pdf)
        assert kwargs == {
            "plugins": ["ocrmypdf_paddleocr"],
            "language": worker.OCR_LANG,
            "force_ocr": True,
        }
        assert "paddle_use_gpu" not in kwargs
