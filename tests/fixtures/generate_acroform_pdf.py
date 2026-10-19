#!/usr/bin/env python3
"""
Doc-Worker — deterministic AcroForm PDF fixture generator (E2E).

Generates tests/fixtures/acroform_filled.pdf: one US-Letter page with
baked-in label text in the page content stream ("NAME:", "CITY:") plus
two filled /Tx AcroForm fields (Name = "Anna Mustermann",
City = "Berlin") carrying /V values, /AP appearance streams, /DA default
appearance and AcroForm.DR default resources — i.e. the shape a real
fill-in tool writes. NeedAppearances is set, as fill-in tools do.

Why it exists
-------------
The worker pre-flattens AcroForms before OCR (worker.flatten_acroform_if_needed,
AGENTS.md Version-Specific Hack #5). This fixture exercises that path
end-to-end: pypdfium2 without init_forms() does not draw the form layer,
so the red-guard pixel-diff in tests/e2e/variants_e2e.py holds whether or
not the widgets carry /AP streams, and the ocrmypdf output must contain
the values that were typed in.

Deterministic
-------------
No randomness, no timestamps: a fresh write is byte-identical across
runs and machines (pikepdf derives the trailer /ID from content on a
fresh write; overwriting an existing file would inherit its old ID, so
the script deletes the output before saving).

Usage
-----
    python3 tests/fixtures/generate_acroform_pdf.py

The script re-verifies the result with pypdfium2 (form detected, values
invisible without the form layer, visible after flatten, render pixels
differ) and exits non-zero if any check fails. Regenerate after a
pikepdf/pypdfium2 version bump and commit the artifact.

Requires pikepdf 10.x (transitive dep of ocrmypdf >= 17.4.1) and
pypdfium2 (both in the image and in CI via requirements.txt).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pikepdf
import pypdfium2 as pdfium
from pikepdf import Array, Name

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "tests" / "fixtures" / "acroform_filled.pdf"

NAME_VALUE = "Anna Mustermann"
CITY_VALUE = "Berlin"


def _dict(*pairs: tuple) -> pikepdf.objects.Object:
    d = pikepdf.Dictionary()
    for k, v in pairs:
        d[k] = v
    return d


def build_pdf() -> pikepdf.Pdf:
    """Build the AcroForm document in memory (no file I/O)."""
    pdf = pikepdf.Pdf.new()
    root = pdf.Root
    root[Name.Type] = Name.Catalog

    font_f1 = pdf.make_indirect(
        _dict(
            (Name.Type, Name.Font),
            (Name.Subtype, Name.Type1),
            (Name.BaseFont, Name.Helvetica),
        )
    )
    font_helv = pdf.make_indirect(
        _dict(
            (Name.Type, Name.Font),
            (Name.Subtype, Name.Type1),
            (Name.BaseFont, Name.Helvetica),
        )
    )
    # Labels are baked into the content stream (not form fields) so that
    # a plain render of the original already shows something and the
    # red-guard diff isolates the typed-in values.
    content = pdf.make_stream(
        b"BT /F1 12 Tf 72 720 Td (NAME:) Tj 0 -25 Td (CITY:) Tj ET\n"
    )

    def widget(name: str, value: str, y: float) -> pikepdf.objects.Object:
        ap = pdf.make_stream(
            f"BT /Helv 10 Tf 1 0 0 1 164 {y + 4} Tm ({value}) Tj ET\n".encode()
        )
        w = _dict(
            (Name.Type, Name.Annot),
            (Name.Subtype, Name.Widget),
            (Name.Rect, Array([160, y, 450, y + 18])),
            (Name.T, name),
            (Name.FT, Name.Tx),
            (Name.V, value),
            (Name.DA, "/Helv 0 Tf 0 g"),
            (Name.Borders, Array([0, 0, 0])),
            (Name.Ff, 0),
        )
        w[Name.AP] = _dict((Name.N, ap))
        return w

    w1 = pdf.make_indirect(widget("Name", NAME_VALUE, 706))
    w2 = pdf.make_indirect(widget("City", CITY_VALUE, 681))
    w1[Name.Parent] = w1
    w2[Name.Parent] = w2

    page = pdf.make_indirect(
        _dict(
            (Name.Type, Name.Page),
            (Name.MediaBox, Array([0, 0, 612, 792])),
            (
                Name.Resources,
                _dict((Name.Font, _dict((Name.F1, font_f1), (Name.Helv, font_helv)))),
            ),
            (Name.Contents, content),
            (Name.Annots, Array([w1, w2])),
        )
    )
    pages = pdf.make_indirect(
        _dict((Name.Type, Name.Pages), (Name.Kids, Array([page])), (Name.Count, 1))
    )
    root[Name.Pages] = pages
    root[Name.AcroForm] = _dict(
        (Name.Fields, Array([w1, w2])),
        (Name.DR, _dict((Name.Font, _dict((Name.Helv, font_helv))))),
        (Name.NeedAppearances, True),
    )
    return pdf


def _render_hash(path: Path) -> str:
    doc = pdfium.PdfDocument(str(path))
    try:
        return hashlib.sha256(doc[0].render(scale=1).to_pil().tobytes()).hexdigest()
    finally:
        doc.close()


def verify(original: Path) -> None:
    """Re-open the artifact and check the invariants the e2e relies on."""
    doc = pdfium.PdfDocument(str(original))
    try:
        if doc.init_forms() is not True:
            raise AssertionError("init_forms() must detect the AcroForm")
        plain_text = doc[0].get_textpage().get_text_bounded()
        if NAME_VALUE in plain_text or CITY_VALUE in plain_text:
            raise AssertionError(
                "typed-in values must NOT be visible without the form layer"
            )
        if "NAME:" not in plain_text:
            raise AssertionError("baked-in label missing from content stream")
    finally:
        doc.close()

    # Flatten a throwaway copy: values must appear, forms must disappear.
    doc = pdfium.PdfDocument(str(original))
    try:
        if doc.init_forms() is not True:
            raise AssertionError("init_forms() must detect the AcroForm (2nd open)")
        for page in doc:
            page.flatten()
        tmp = original.with_name(original.stem + "_verify.pdf")
        doc.save(str(tmp))
    finally:
        doc.close()
    try:
        flat = pdfium.PdfDocument(str(tmp))
        try:
            if flat.init_forms() is not False:
                raise AssertionError("flattened copy must have no AcroForm")
            flat_text = flat[0].get_textpage().get_text_bounded()
            for value in (NAME_VALUE, CITY_VALUE):
                if value not in flat_text:
                    raise AssertionError(f"value {value!r} missing after flatten")
        finally:
            flat.close()
        if _render_hash(tmp) == _render_hash(original):
            raise AssertionError("red guard: render hashes must differ")
    finally:
        tmp.unlink(missing_ok=True)


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

    print(
        "verification OK: form detected, values hidden w/o form layer, "
        "visible after flatten, render red-guard holds"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
