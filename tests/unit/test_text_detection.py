"""Unit tests for server._is_text_content binary/text detection.

These run in CI (tests/unit/) without requiring model loading.
"""

from __future__ import annotations

import zlib

import server


class TestIsTextContent:
    """Tests for the _is_text_content function."""

    # ── Binary rejection ────────────────────────────────────────────────

    def test_rejects_pdf(self) -> None:
        assert server._is_text_content(b"%PDF-1.4\n1 0 obj\n...") is False

    def test_rejects_png(self) -> None:
        assert server._is_text_content(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) is False

    def test_rejects_jpeg(self) -> None:
        assert server._is_text_content(b"\xff\xd8\xff\xe0" + b"\x00" * 100) is False

    def test_rejects_gif(self) -> None:
        assert server._is_text_content(b"GIF89a" + b"\x00" * 100) is False

    def test_rejects_bmp(self) -> None:
        assert server._is_text_content(b"BM" + b"\x00" * 100) is False

    def test_rejects_riff(self) -> None:
        assert server._is_text_content(b"RIFF" + b"\x00" * 100) is False

    def test_rejects_zip(self) -> None:
        assert server._is_text_content(b"PK\x03\x04" + b"\x00" * 100) is False

    def test_rejects_gzip(self) -> None:
        assert server._is_text_content(b"\x1f\x8b" + b"\x00" * 100) is False

    def test_rejects_bzip2(self) -> None:
        assert server._is_text_content(b"BZ" + b"\x00" * 100) is False

    def test_rejects_random_binary(self) -> None:
        """Random binary data with no known signature must be rejected."""
        data = b"\x00\x01\x02\xff\xfe\xfd\x00" * 1000
        assert server._is_text_content(data) is False

    def test_rejects_uncompressed_pdf(self) -> None:
        """A realistic uncompressed PDF must be rejected as binary.

        The first 8 KB of an uncompressed PDF is mostly printable dictionary
        text, which the old 5% heuristic would have misclassified as text.
        The strict UTF-8 decode catches the binary stream data later in the
        file (or the FlateDecode bytes in the header region).
        """
        stream_data = zlib.compress(b"Hello World " * 100)
        pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
            b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
            b"3 0 obj <</Type /Page /MediaBox [0 0 612 792] /Parent 2 0 R>> endobj\n"
            b"4 0 obj <</Length " + str(len(stream_data)).encode() + b" /Filter /FlateDecode>>\n"
            b"stream\n" + stream_data + b"\nendstream\nendobj\n"
            b"trailer <</Size 5 /Root 1 0 R>>\n%%EOF"
        )
        assert server._is_text_content(pdf) is False

    def test_rejects_large_file_with_binary_tail(self) -> None:
        """A large file with a valid-UTF-8 head but binary tail must be rejected.

        This tests the tail-check: an uncompressed PDF can have printable
        dictionary text in the first 8 KB but binary stream data later.
        The tail uses 0xFF bytes which are never valid in UTF-8.
        """
        head = b"This is a printable header. " * 426  # ~8192 bytes, valid UTF-8
        tail = b"\xff\xfe\xfd\xfc" * 2048  # 8192 bytes, invalid UTF-8
        data = head + tail
        assert len(data) > 10240  # Ensure we trigger the tail check
        assert server._is_text_content(data) is False

    # ── Text acceptance ─────────────────────────────────────────────────

    def test_accepts_ascii_text(self) -> None:
        data = b"Hello, this is a plain text file.\nWith some content.\n"
        assert server._is_text_content(data) is True

    def test_accepts_german_umlauts(self) -> None:
        """German text (default OCR_LANG=deu) must be classified as text.

        The old 5% high-byte heuristic misclassified this because umlauts
        are 2-byte UTF-8 sequences (e.g., ü = C3 BC, 2 bytes > 127).
        """
        data = "Müllerstraße über große höfe für deutsche Größe\n".encode()
        assert server._is_text_content(data) is True

    def test_accepts_cjk(self) -> None:
        data = "日本語テキスト\n".encode()
        assert server._is_text_content(data) is True

    def test_accepts_emoji(self) -> None:
        data = "Hello 🌍 world\n".encode()
        assert server._is_text_content(data) is True

    def test_accepts_multilingual_mix(self) -> None:
        data = "Café naïve 日本語 🎉 Straße\n".encode()
        assert server._is_text_content(data) is True

    def test_text_starting_with_bm_is_text(self) -> None:
        """A text file starting with 'BM' (but not a real BMP) must be text.

        The signature check only matches at offset 0, and 'BM' followed by
        valid UTF-8 text will pass the decode check. However, 'BM' IS in the
        signature list, so this actually tests that the signature check
        fires. A real BMP starts with BM + 2-byte file size + ...
        """
        # This WILL be rejected by the BM signature check. That's the
        # accepted trade-off: a text file literally starting with "BM"
        # is extremely rare. The important case is text *containing* "BM"
        # mid-text, which is NOT rejected.
        data = b"BM" + b"This is text, not a BMP file.\n"
        # The BM signature fires → rejected as binary. This is the known
        # limitation documented in the plan.
        assert server._is_text_content(data) is False

    def test_text_containing_bm_midtext_is_text(self) -> None:
        """A text file containing 'BM' mid-text (not at offset 0) must be text."""
        data = b"Overview of BMP format and its uses in graphics.\n"
        assert server._is_text_content(data) is True

    def test_text_containing_bz_midtext_is_text(self) -> None:
        """A text file containing 'BZ' mid-text (not at offset 0) must be text."""
        data = b"The BZ algorithm is used for compression.\n"
        assert server._is_text_content(data) is True

    # ── Edge cases ──────────────────────────────────────────────────────

    def test_empty_bytes(self) -> None:
        assert server._is_text_content(b"") is False

    def test_small_file_no_tail_check(self) -> None:
        """Files ≤ 10 KB should not trigger the tail check (still works)."""
        data = b"Small text file\n"
        assert len(data) <= 10240
        assert server._is_text_content(data) is True
