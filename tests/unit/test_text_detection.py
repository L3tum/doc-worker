"""Unit tests for server._is_text_content binary/text detection.

These run in CI (tests/unit/) without requiring model loading.
"""

from __future__ import annotations

import struct
import zlib

import pytest

import server

# A realistic near-silent WAV: every byte < 0x80, so the whole file decodes
# as valid UTF-8 (sample rate 8000 keeps the fmt chunk 7-bit clean). NUL is
# valid UTF-8, so only the explicit NUL check can reject it — the canonical
# case the tightened WebP-only RIFF rule no longer covers (the pre-change
# bare-RIFF rule did).
WAV_7BIT_CLEAN = (
    b"RIFF"
    + struct.pack("<I", 36)
    + b"WAVE"
    + b"fmt "
    + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 8000, 1, 8)
    + b"data"
    + struct.pack("<I", 0)
)


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
        # A real minimal BMP: "BM" + file-size DWORD (>= 14) + 2 reserved
        # DWORDs + DIB header-size DWORD (40 = BITMAPINFOHEADER) + header.
        data = (
            b"BM"
            + struct.pack("<I", 66)
            + b"\x00" * 8
            + struct.pack("<I", 40)
            + b"\x00" * 48
        )
        assert server._is_text_content(data) is False

    def test_rejects_riff(self) -> None:
        # A real WebP: RIFF container with the "WEBP" fourcc at offset 8.
        data = b"RIFF" + struct.pack("<I", 4) + b"WEBP" + b"VP8X" + b"\x00" * 8
        assert server._is_text_content(data) is False

    def test_rejects_7bit_clean_wav(self) -> None:
        """A 7-bit-clean binary (valid UTF-8 end-to-end) must be rejected.

        NUL is *valid* UTF-8, so the decode check alone cannot catch this:
        a near-silent WAV with zeroed data is the canonical case the old
        bare-RIFF rule rejected and the tightened WebP-only rule does not.
        """
        assert all(b < 0x80 for b in WAV_7BIT_CLEAN)
        assert server._is_text_content(WAV_7BIT_CLEAN) is False

    @pytest.mark.parametrize(
        "name, data",
        [
            ("zip", b"PK\x03\x04" + b"\x00" * 100),
            ("gzip", b"\x1f\x8b" + b"\x00" * 100),
            ("bzip2", b"BZ" + b"\x00" * 100),
            ("lzip", b"x\xda\x03" + b"\x00" * 100),
        ],
    )
    def test_rejects_container_signatures(self, name: str, data: bytes) -> None:
        """Reject-only container signatures never route to a pipeline."""
        assert server._is_text_content(data) is False

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
            b"4 0 obj <</Length "
            + str(len(stream_data)).encode()
            + b" /Filter /FlateDecode>>\n"
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

    def test_rejects_large_file_with_nul_tail(self) -> None:
        """A printable head with a NUL-laden tail must be rejected.

        NUL is valid UTF-8, so the tail's decode passes — only the explicit
        NUL scan in the tail region catches this.
        """
        head = b"This is a printable header. " * 426  # ~8192 bytes, valid UTF-8
        tail = b"\x00" * 8192  # valid UTF-8 without the NUL check
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

        The BMP predicate requires a file-size DWORD >= 14 at offset 2 AND a
        known DIB header size at offset 14; 'BM' followed by plain words has
        a garbage DIB size, so the strict UTF-8 decode decides: text.
        """
        data = b"BM" + b"This is text, not a BMP file.\n"
        assert server._is_text_content(data) is True

    def test_accepts_text_starting_with_bmw(self) -> None:
        """T3 pin: 'BMW…' text is no longer rejected by the BMP signature."""
        data = b"BMW iX3 review, second line here\nmore text\n"
        assert server._is_text_content(data) is True

    def test_accepts_text_starting_with_riff(self) -> None:
        """T3 pin: 'RIFF…' text without a WEBP fourcc at offset 8 is no
        longer rejected by the RIFF container signature."""
        data = b"RIFF notes file, not webp\n"
        assert server._is_text_content(data) is True

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


# ── detect_file_type tests ───────────────────────────────────────────────
class TestDetectFileType:
    """Content-based classification (magic bytes), independent of caller hints."""

    @pytest.mark.parametrize(
        "content",
        [b"%PDF-1.7\n", b"%PDF-1.4\n1 0 obj <</Type /Catalog>> endobj"],
        ids=["header_only", "with_body"],
    )
    def test_pdf(self, content: bytes) -> None:
        assert server.detect_file_type(content) == "pdf"

    @pytest.mark.parametrize(
        "head",
        [
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,  # PNG
            b"\xff\xd8\xff\xe0" + b"\x00" * 16,  # JPEG
            b"GIF87a" + b"\x00" * 16,  # GIF87a
            b"GIF89a" + b"\x00" * 16,  # GIF89a
            (
                b"BM"
                + struct.pack("<I", 66)
                + b"\x00" * 8
                + struct.pack("<I", 40)
                + b"\x00" * 48
            ),  # BMP
            b"RIFF\x00\x00\x00\x00WEBP",  # WebP (RIFF container)
            b"II\x2a\x00" + b"\x00" * 16,  # TIFF, little-endian
            b"MM\x00\x2a" + b"\x00" * 16,  # TIFF, big-endian
        ],
        ids=["png", "jpeg", "gif87a", "gif89a", "bmp", "webp", "tiff_le", "tiff_be"],
    )
    def test_images(self, head: bytes) -> None:
        assert server.detect_file_type(head) == "image"

    def test_tiff_le_is_not_mistaken_for_text(self) -> None:
        # NUL is valid UTF-8: without the TIFF signature, "II*\x00" would
        # pass the text check in _is_text_content.
        assert server.detect_file_type(b"II\x2a\x00") == "image"

    @pytest.mark.parametrize(
        "text", ["Hello World\n", "line1\nline2\n", "\u00e4\u00f6\u00fc"]
    )
    def test_text(self, text: str) -> None:
        assert server.detect_file_type(text.encode("utf-8")) == "text"

    @pytest.mark.parametrize(
        "head",
        [
            b"",  # empty
            b"\x80\x81\x82" * 16,  # invalid UTF-8, no known signature
            b"PK\x03\x04" + b"\x00" * 16,  # ZIP/ODF (unsupported)
            b"\x1f\x8b\x08\x00" + b"\x00" * 16,  # GZIP (unsupported)
        ],
        ids=["empty", "garbage", "zip", "gzip"],
    )
    def test_unknown(self, head: bytes) -> None:
        assert server.detect_file_type(head) == "unknown"

    # ── RIFF/BMP tightening pins (T3) ─────────────────────────────────

    def test_text_starting_with_bmw_is_text(self) -> None:
        """Plain text starting with 'BMW…' is text, not an image.

        The size DWORD at offset 2 of this literal happens to be >= 14, so
        it is the DIB header-size check (offset 14) that excludes it.
        """
        data = b"BMW iX3 review, second line here\nmore text\n"
        assert server.detect_file_type(data) == "text"

    def test_text_starting_with_riff_is_text(self) -> None:
        """Plain text starting with 'RIFF…' with no WEBP fourcc at offset 8
        is text, not an image — the old bare-RIFF rule misclassified it."""
        data = b"RIFF notes file, not webp\n"
        assert server.detect_file_type(data) == "text"

    def test_minimal_webp_is_image(self) -> None:
        """Minimal WebP: RIFF + 4-byte LE size + WEBP fourcc at offset 8."""
        data = b"RIFF" + struct.pack("<I", 4) + b"WEBP"
        assert server.detect_file_type(data) == "image"

    def test_wav_is_unknown(self) -> None:
        """WAV shares the RIFF container but is not an image.

        A realistic WAV header (fmt chunk with binary sample-rate DWORDs)
        is not valid UTF-8, so it lands in "unknown" instead of the old
        bare-RIFF "image".
        """
        data = (
            b"RIFF"
            + struct.pack("<I", 44)
            + b"WAVE"
            + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 2, 44100, 88200, 2, 16)
        )
        assert server.detect_file_type(data) == "unknown"

    def test_7bit_clean_wav_is_unknown(self) -> None:
        """A 7-bit-clean WAV (valid UTF-8, NUL-laden) is unknown, not text —
        the silent-garbage regression the NUL check closes."""
        assert server.detect_file_type(WAV_7BIT_CLEAN) == "unknown"

    @pytest.mark.parametrize(
        "prefix",
        [b"\n", b"\r\n", b"\x00\x00", b"  \t"],
        ids=["lf", "crlf", "nul_nul", "whitespace"],
    )
    def test_pdf_with_leading_noise_is_pdf(self, prefix: bytes) -> None:
        """A %PDF header padded with leading whitespace/control bytes must
        still route to PDF (it OCR'd fine pre-change), not 415 or raw text."""
        assert server.detect_file_type(prefix + b"%PDF-1.4\n") == "pdf"

    def test_leading_whitespace_text_is_text(self) -> None:
        """Plain text with leading whitespace is still text: the anchor skip
        must not turn '   Hello' into a signature match."""
        assert server.detect_file_type(b"   Hello World\n") == "text"

    def test_minimal_bmp_is_image(self) -> None:
        """Minimal BMP: BM + file-size DWORD (>= 14) + 2 reserved DWORDs
        + DIB header-size DWORD (40) at offset 14."""
        data = b"BM" + struct.pack("<I", 18) + b"\x00" * 8 + struct.pack("<I", 40)
        assert server.detect_file_type(data) == "image"

    def test_corrupt_png_body_still_classified_by_header(self) -> None:
        """Classification is anchored at offset 0: a corrupt body after a
        valid header must not change the kind (corrupt files fail later in
        the pipeline, not in routing)."""
        data = b"\x89PNG\r\n\x1a\n" + b"\x00\x01\x02\xff\xfe" * 10
        assert server.detect_file_type(data) == "image"
