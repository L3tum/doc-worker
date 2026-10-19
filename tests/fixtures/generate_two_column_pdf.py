#!/usr/bin/env python3
"""
Doc-Worker — deterministic two-column PDF fixture generator (E2E).

Generates tests/fixtures/two_column.pdf: one US-Letter page, vector
text only (Helvetica base font — no font files), laid out as a
full-width title plus two text columns:

    TWOCOLUMNS                     (full-width title)
    LEFTONE       RIGHTONE
    LEFTTWO       RIGHTTWO
    LEFTTHREE     RIGHTTHREE
    LEFTFOUR      RIGHTFOUR

Why it exists
-------------
tests/e2e/variants_e2e.py runs this fixture through the real
run_paddlex_structure_v3 pipeline (300-dpi pdftoppm render → PP-StructureV3)
and asserts the reading order is column-first: every LEFT* marker must
appear in the markdown before every RIGHT* marker. With the legacy
(y, column) sort key the lines interleave L1 R1 L2 R2 …; the band/span
column-first ordering (paddlex_helpers._sort_blocks_by_reading_order)
must emit the entire left column first.

Deterministic
-------------
No randomness, no timestamps: a fresh write is byte-identical across
runs and machines (pikepdf derives the trailer /ID from content on a
fresh write; overwriting an existing file would inherit its old ID, so
the script deletes the output before saving).

Usage
-----
    python3 tests/fixtures/generate_two_column_pdf.py

The script re-verifies the result with pypdfium2 (one page, all marker
words present in the text layer) and exits non-zero if any check fails.
Regenerate after a pikepdf/pypdfium2 version bump and commit the artifact.

Requires pikepdf 10.x (transitive dep of ocrmypdf >= 17.4.1) and
pypdfium2 (both in the image and in CI via requirements.txt).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pikepdf
import pypdfium2 as pdfium
from pikepdf import Array, Name

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "tests" / "fixtures" / "two_column.pdf"

TITLE = "TWOCOLUMNS"
LEFT_MARKERS = ("LEFTONE", "LEFTTWO", "LEFTTHREE", "LEFTFOUR")
RIGHT_MARKERS = ("RIGHTONE", "RIGHTTWO", "RIGHTTHREE", "RIGHTFOUR")

PAGE_W, PAGE_H = 612, 792
LEFT_X, RIGHT_X = 72, 324
TOP_Y, LINE_STEP = 680, 40


def _dict(*pairs: tuple) -> pikepdf.objects.Object:
    d = pikepdf.Dictionary()
    for k, v in pairs:
        d[k] = v
    return d


def build_pdf() -> pikepdf.Pdf:
    """Build the two-column document in memory (no file I/O)."""
    pdf = pikepdf.Pdf.new()
    root = pdf.Root
    root[Name.Type] = Name.Catalog

    font_obj = pdf.make_indirect(
        _dict(
            (Name.Type, Name.Font),
            (Name.Subtype, Name.Type1),
            (Name.BaseFont, Name.Helvetica),
        )
    )

    lines = [
        f"BT /F1 18 Tf 72 740 Td ({TITLE}) Tj ET",
        f"BT /F1 16 Tf {LEFT_X} {TOP_Y} Td ({LEFT_MARKERS[0]}) Tj",
        *[f"0 -{LINE_STEP} Td ({w}) Tj" for w in LEFT_MARKERS[1:]],
        "ET",
        f"BT /F1 16 Tf {RIGHT_X} {TOP_Y} Td ({RIGHT_MARKERS[0]}) Tj",
        *[f"0 -{LINE_STEP} Td ({w}) Tj" for w in RIGHT_MARKERS[1:]],
        "ET",
    ]
    content = pdf.make_stream(("\n".join(lines) + "\n").encode())

    page = pdf.make_indirect(
        _dict(
            (Name.Type, Name.Page),
            (Name.MediaBox, Array([0, 0, PAGE_W, PAGE_H])),
            (Name.Resources, _dict((Name.Font, _dict((Name.F1, font_obj))))),
            (Name.Contents, content),
        )
    )
    pages = pdf.make_indirect(
        _dict((Name.Type, Name.Pages), (Name.Kids, Array([page])), (Name.Count, 1))
    )
    root[Name.Pages] = pages
    return pdf


def verify(path: Path) -> None:
    """Re-open the artifact and check the invariants the e2e relies on."""
    doc = pdfium.PdfDocument(str(path))
    try:
        if len(doc) != 1:
            raise AssertionError(f"expected 1 page, got {len(doc)}")
        text = doc[0].get_textpage().get_text_bounded()
        for word in (TITLE, *LEFT_MARKERS, *RIGHT_MARKERS):
            if word not in text:
                raise AssertionError(f"marker {word!r} missing from text layer")
    finally:
        doc.close()


def main() -> int:
    pdf = build_pdf()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Delete first: pikepdf inherits the trailer /ID from an existing
    # target, which would make re-runs non-byte-identical.
    OUTPUT.unlink(missing_ok=True)
    pdf.save(str(OUTPUT))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")

    try:
        verify(OUTPUT)
    except AssertionError as exc:
        print(f"ERROR: self-verification failed: {exc}", file=sys.stderr)
        return 1

    print("verification OK: one page, all markers present in text layer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
