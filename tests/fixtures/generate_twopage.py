#!/usr/bin/env python3
"""
Doc-Worker — deterministic two-page minimal PDF fixture generator.

Generates tests/fixtures/twopage.pdf: a 432-byte hand-minimal PDF
(%PDF-1.4, no streams, no fonts, real xref table) with two pages that
have *deliberately different* MediaBoxes:

    page 1:  200 x 200 pt   (MediaBox [0 0 200 200])
    page 2:  300 x 400 pt   (MediaBox [0 0 300 400])

Why it exists
-------------
tests/test_vl_api_e2e.py (page-cap tests, MAX_OCR_PAGES=1 → 413 and
count == MAX_OCR_PAGES → 200) and tests/test_server.py
(test_two_page_fixture_per_page_check) load this *real* 2-page fixture
and count pages / compute per-page rendered pixel counts from the
in-memory request bytes via pypdfium2 — no page-count or render-size
mocks; the real page count is the thing under test. The
distinct per-page MediaBoxes are what make the per-page rendered-pixel
cap testable at all: with a cap between the two pages' pixel counts,
page 1 passes and page 2 is the one named in the 413 — a check that
only inspects the largest page would still pass page 2's cap logic
vacuously.

Deterministic
-------------
No randomness, no timestamps: the output is the exact concatenation of
fixed byte literals (header, four objects, xref table, trailer) with
the xref offsets computed from the assembled bytes, so the xref table
is always self-consistent with the object layout. A fresh build is
byte-identical across runs and machines. The canonical offsets (15 /
64 / 127 / 198, xref at 269) are pinned as a check so any accidental
layout drift in the literals fails loudly instead of producing a
silently-different fixture.

Unlike the pikepdf-based generators, this fixture is intentionally NOT
normalized through a PDF library: the point is a tiny, human-readable
minimal PDF with a valid xref table that pypdfium2 opens directly.

Usage
-----
    python3 tests/fixtures/generate_twopage.py

The script self-verifies the produced bytes with pypdfium2 (page count
== 2, per-page MediaBoxes) and raises if any check fails; it then
writes tests/fixtures/twopage.pdf and reads it back to confirm the
bytes landed. The output is byte-identical across runs and machines — a
drift check (regenerate + git diff) can catch hand-edits to the binary
fixture at any time (no CI step does this yet).

Requires pypdfium2 (in requirements.txt; hard dep of ocrmypdf's
built-in rasterizer).
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "tests" / "fixtures" / "twopage.pdf"

# Fixed object layout — the entire 432-byte file is these byte literals
# plus the computed xref entries. (Determinism is the convention, not
# the algorithm: a byte literal is perfectly fine for a 432-byte
# fixture.)
HEADER = b"%PDF-1.4\n"
# PDF spec asks for a comment line with high-bit bytes so downstream
# tools treat the file as binary.
BINARY_COMMENT = b"%\xe2\xe3\xcf\xd3\n"
OBJ_CATALOG = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
OBJ_PAGES = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n"
OBJ_PAGE_1 = (
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>\nendobj\n"
)
OBJ_PAGE_2 = (
    b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 400] >>\nendobj\n"
)
OBJECTS = (OBJ_CATALOG, OBJ_PAGES, OBJ_PAGE_1, OBJ_PAGE_2)

# Canonical layout check: with the literals above, the objects start at
# these byte offsets and the xref table at 269. If you edit an object
# literal on purpose, update these pins.
EXPECTED_OBJECT_OFFSETS = (15, 64, 127, 198)
EXPECTED_XREF_OFFSET = 269

# (page 1, page 2) — the deliberately different MediaBoxes.
EXPECTED_PAGE_SIZES = ((200.0, 200.0), (300.0, 400.0))


def build_pdf() -> bytes:
    """Assemble the fixture; xref offsets are computed from the
    assembled bytes so the table is self-consistent with the layout."""
    body = HEADER + BINARY_COMMENT
    offsets: list[int] = []
    for obj in OBJECTS:
        offsets.append(len(body))
        body += obj

    xref_offset = len(body)
    if tuple(offsets) != EXPECTED_OBJECT_OFFSETS or xref_offset != EXPECTED_XREF_OFFSET:
        raise AssertionError(
            f"object layout drift: object offsets {offsets} (expected "
            f"{list(EXPECTED_OBJECT_OFFSETS)}), xref at {xref_offset} "
            f"(expected {EXPECTED_XREF_OFFSET}) — update EXPECTED_* pins"
        )

    xref = b"xref\n0 5\n"
    xref += b"0000000000 65535 f \n"  # free-list head entry (20 bytes)
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode("ascii")  # 20 bytes each

    body += xref
    body += b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n"
    body += str(xref_offset).encode("ascii") + b"\n"
    body += b"%%EOF\n"
    return body


def verify(data: bytes) -> None:
    """Re-open the produced bytes in pypdfium2 and check the invariants
    the tests rely on (same in-memory-bytes path the server uses)."""
    doc = pdfium.PdfDocument(data)
    try:
        if len(doc) != 2:
            raise AssertionError(f"expected 2 pages, got {len(doc)}")
        for i, (w, h) in enumerate(EXPECTED_PAGE_SIZES):
            pw, ph = doc[i].get_width(), doc[i].get_height()
            if (pw, ph) != (w, h):
                raise AssertionError(
                    f"page {i + 1} size mismatch: expected {w:.0f} x {h:.0f}, "
                    f"got {pw:.0f} x {ph:.0f}"
                )
    finally:
        doc.close()


def main() -> None:
    data = build_pdf()
    verify(data)
    print("verification OK: 2 pages, MediaBoxes (200, 200) and (300, 400)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(data)
    # Read-back: the bytes that landed must equal the generated bytes.
    written = OUTPUT.read_bytes()
    if written != data:
        raise AssertionError(
            "read-back of the written file does not match the generated bytes"
        )
    print(f"wrote {OUTPUT} ({len(written)} bytes)")


if __name__ == "__main__":
    main()
