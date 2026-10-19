"""
Doc-Worker — In-image e2e: layout-variant regressions (AcroForm + reading order).

Runs inside the built Docker image (see the ``e2e-ocr`` job in
``.github/workflows/docker.yaml``, which runs this script after
``umlaut_e2e.py``):

Check A — AcroForm pre-flatten (AGENTS.md Version-Specific Hack #5):

    worker.flatten_acroform_if_needed(acroform_filled.pdf)
        → ocrmypdf.ocr (plugins=["ocrmypdf_paddleocr"], language="deu",
          force_ocr=True) → pdftotext → the typed-in field values
          ("Anna Mustermann", "Berlin") must extract as contiguous words.

    Red guard: the same ocrmypdf pass on the *unflattened* copy must NOT
    show the values — ocrmypdf 17.4.1's built-in pypdfium rasterizer calls
    page.render(may_draw_forms=True) but never doc.init_forms(), so PDFium
    does not draw the form layer (verified against the pinned wheel). If a
    future ocrmypdf starts initializing forms natively, the "absent"
    half of this guard will fail and its expectation must be revisited.

Check B — two-column reading order:

    run_paddlex_structure_v3(two_column.pdf) → blocks_to_markdown
    → every LEFT* marker must precede every RIGHT* marker (column-first
    band ordering, not the legacy (y, column) interleaving L1 R1 L2 R2).

Check C — Word-split-layout reading order:

    run_paddlex_structure_v3(word_split.pdf) → blocks_to_markdown
    → WORDHEADER first, then every WORDLEFT* before every WORDRIGHT*.

Exit code: 0 only when all checks pass; 1 with a clear message on any
failure. Like umlaut_e2e.py this is an in-image script: it requires
ocrmypdf, poppler-utils (pdftotext) and the bundled PaddleX models,
all of which exist in the doc-worker image.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Fixtures live in the repo: the CI job runs the script from the in-image
# repo copy (COPY . . -> /app) with no bind mounts, and local (non-Docker)
# runs resolve the same path repo-relative.
FIXTURES_IN_REPO = Path(__file__).resolve().parents[1] / "fixtures"

# Check A — AcroForm fixture field values (must stay in sync with
# tests/fixtures/generate_acroform_pdf.py).
ACROFORM_FIXTURE = "acroform_filled.pdf"
ACROFORM_VALUES = ("Anna Mustermann", "Berlin")

# Check B — two-column fixture markers (generate_two_column_pdf.py).
TWO_COLUMN_FIXTURE = "two_column.pdf"
TWO_COLUMN_TITLE = "TWOCOLUMNS"
LEFT_MARKERS = ("LEFTONE", "LEFTTWO", "LEFTTHREE", "LEFTFOUR")
RIGHT_MARKERS = ("RIGHTONE", "RIGHTTWO", "RIGHTTHREE", "RIGHTFOUR")

# Check C — Word-split fixture markers (generate_word_split_pdf.py).
WORD_SPLIT_FIXTURE = "word_split.pdf"
WORD_SPLIT_HEADER = "WORDHEADER"
WORD_LEFT_MARKERS = ("WORDLEFTONE", "WORDLEFTTWO")
WORD_RIGHT_MARKERS = ("WORDRIGHTONE", "WORDRIGHTTWO")


class E2EError(Exception):
    """Raised by e2e helpers; caught once in main() for a clean ERROR exit."""


def _find_fixture(name: str) -> Path:
    """Return the repo-relative fixture path, else raise E2EError."""
    candidate = FIXTURES_IN_REPO / name
    if candidate.is_file():
        return candidate
    raise E2EError(
        f"fixture {name!r} not found — looked in {FIXTURES_IN_REPO} "
        "(the repo copy of tests/fixtures; there is no in-image mount)."
    )


def _pdftotext(pdf: Path, txt: Path) -> str:
    """Run pdftotext (poppler-utils, installed by the Dockerfile)."""
    result = subprocess.run(
        ["pdftotext", str(pdf), str(txt)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise E2EError(
            f"pdftotext failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return txt.read_text(encoding="utf-8")


def _run_ocrmypdf(src: Path, dst: Path) -> None:
    """Run the same ocrmypdf.ocr call the worker uses (PaddleOCR backend)."""
    import ocrmypdf

    ocrmypdf.ocr(
        src,
        dst,
        plugins=["ocrmypdf_paddleocr"],
        language="deu",
        force_ocr=True,
    )


def _structure_markdown(pdf_path: Path) -> str:
    """Run the real PP-StructureV3 pipeline on a PDF, return page markdown.

    Mirrors server.py's /layout-parsing path: run_paddlex_structure_v3
    renders the PDF (pdftoppm) and runs the model. run_paddlex_structure_v3
    returns the blocks in reading order (DPI/page-width-aware);
    blocks_to_markdown serializes them.
    """
    import paddlex_helpers

    pages = paddlex_helpers.run_paddlex_structure_v3(str(pdf_path))
    if not pages:
        raise E2EError(f"run_paddlex_structure_v3 returned no pages for {pdf_path}")
    return "\n\n".join(
        paddlex_helpers.blocks_to_markdown(page.get("structured_blocks") or [])
        for page in pages
    )


def _normalized(text: str) -> str:
    """Collapse whitespace so word-boundary checks survive layout noise."""
    return " ".join(text.split())


def _find_all(normalized: str, marker: str) -> list[int]:
    """All start offsets of marker in normalized text (empty if absent)."""
    return [m.start() for m in re.finditer(re.escape(marker), normalized)]


def _check_acroform(work: Path) -> None:
    """Check A: flatten → ocrmypdf → pdftotext must show the typed values."""
    import worker

    work.mkdir(parents=True, exist_ok=True)
    fixture = _find_fixture(ACROFORM_FIXTURE)
    original = work / "original.pdf"
    shutil.copy(fixture, original)
    print(f"[e2e] check A: fixture {fixture} ({fixture.stat().st_size:,} bytes)")

    # Red guard: WITHOUT pre-flatten the values must be invisible to OCR
    # (ocrmypdf never calls init_forms on its pypdfium rasterizer).
    ocr_pre = work / "ocr_unflattened.pdf"
    _run_ocrmypdf(original, ocr_pre)
    pre_text = _normalized(_pdftotext(ocr_pre, work / "pre.txt"))
    leaked = [v for v in ACROFORM_VALUES if v in pre_text]
    if leaked:
        raise E2EError(
            f"red guard: values visible WITHOUT pre-flatten: {leaked} — "
            "the pinned ocrmypdf rasterizer must not draw the AcroForm "
            "layer; if ocrmypdf changed, revisit flatten_acroform_if_needed."
        )
    print("[e2e] check A: red guard OK (values invisible without pre-flatten)")

    flattened = worker.flatten_acroform_if_needed(original)
    if flattened is None:
        raise E2EError(
            "flatten_acroform_if_needed returned None — no AcroForm "
            "detected in the fixture (fixture or detector regressed)"
        )
    try:
        state = worker._ACROFORM_FLATTEN_STATE
    except AttributeError:
        state = "(state constant missing)"
    print(
        f"[e2e] check A: flattened copy: {flattened.name} (pre-flatten state: {state})"
    )
    if state != "active":
        raise E2EError(
            f"flatten succeeded but worker._ACROFORM_FLATTEN_STATE is "
            f"{state!r}, expected 'active'"
        )

    ocr_post = work / "ocr_flattened.pdf"
    _run_ocrmypdf(flattened, ocr_post)
    post_text = _normalized(_pdftotext(ocr_post, work / "post.txt"))
    missing = [v for v in ACROFORM_VALUES if v not in post_text]
    if missing:
        print("---- pdftotext (flattened) ----")
        print(post_text if post_text else "<empty>")
        print("--------------------------------")
        raise E2EError(f"typed values missing after flatten+OCR: {missing}")
    for value in ACROFORM_VALUES:
        print(f"[e2e] check A: OK: {value!r} extracted after pre-flatten")


def _check_two_column(work: Path) -> None:
    """Check B: every LEFT* marker must precede every RIGHT* marker."""
    work.mkdir(parents=True, exist_ok=True)
    fixture = _find_fixture(TWO_COLUMN_FIXTURE)
    pdf = work / "two_column.pdf"
    shutil.copy(fixture, pdf)
    print(f"[e2e] check B: fixture {fixture}")

    markdown = _structure_markdown(pdf)
    normalized = _normalized(markdown)
    all_markers = (TWO_COLUMN_TITLE, *LEFT_MARKERS, *RIGHT_MARKERS)
    missing = [w for w in all_markers if not _find_all(normalized, w)]
    if missing:
        print("---- structure-v3 markdown ----")
        print(markdown if markdown else "<empty>")
        print("--------------------------------")
        raise E2EError(f"markers not recognized by the model: {missing}")

    left_max = max(max(_find_all(normalized, w)) for w in LEFT_MARKERS)
    right_min = min(min(_find_all(normalized, w)) for w in RIGHT_MARKERS)
    if left_max >= right_min:
        print("---- structure-v3 markdown ----")
        print(markdown)
        print("--------------------------------")
        raise E2EError(
            f"two-column reading order is not column-first: last LEFT* at "
            f"offset {left_max}, first RIGHT* at offset {right_min} "
            "(expected the entire left column before the right column)"
        )
    if _find_all(normalized, TWO_COLUMN_TITLE)[0] > left_max:
        raise E2EError("full-width title did not precede the columns")
    print(
        f"[e2e] check B: OK: column-first order (title first, LEFT {left_max} < RIGHT {right_min})"
    )


def _check_word_split(work: Path) -> None:
    """Check C: WORDHEADER first, then all WORDLEFT* before all WORDRIGHT*."""
    work.mkdir(parents=True, exist_ok=True)
    fixture = _find_fixture(WORD_SPLIT_FIXTURE)
    pdf = work / "word_split.pdf"
    shutil.copy(fixture, pdf)
    print(f"[e2e] check C: fixture {fixture}")

    markdown = _structure_markdown(pdf)
    normalized = _normalized(markdown)
    all_markers = (WORD_SPLIT_HEADER, *WORD_LEFT_MARKERS, *WORD_RIGHT_MARKERS)
    missing = [w for w in all_markers if not _find_all(normalized, w)]
    if missing:
        print("---- structure-v3 markdown ----")
        print(markdown if markdown else "<empty>")
        print("--------------------------------")
        raise E2EError(f"markers not recognized by the model: {missing}")

    left_max = max(max(_find_all(normalized, w)) for w in WORD_LEFT_MARKERS)
    right_min = min(min(_find_all(normalized, w)) for w in WORD_RIGHT_MARKERS)
    if left_max >= right_min:
        print("---- structure-v3 markdown ----")
        print(markdown)
        print("--------------------------------")
        raise E2EError(
            f"word-split reading order is not column-first: last WORDLEFT* "
            f"at offset {left_max}, first WORDRIGHT* at offset {right_min}"
        )
    header_idx = min(_find_all(normalized, WORD_SPLIT_HEADER))
    if header_idx > left_max:
        raise E2EError("WORDHEADER did not precede the cells")
    print(
        f"[e2e] check C: OK: header first, WORDLEFT {left_max} < WORDRIGHT {right_min}"
    )


def main() -> int:
    """Run all three layout-variant checks inside the image."""
    with tempfile.TemporaryDirectory(prefix="variants_e2e_") as tmp:
        try:
            _check_acroform(Path(tmp) / "acroform")
            _check_two_column(Path(tmp) / "two_column")
            _check_word_split(Path(tmp) / "word_split")
            print("[e2e] PASS (acroform + two-column + word-split)")
        except E2EError as exc:
            print(f"ERROR: {exc}")
            return 1
        except ImportError as exc:
            print(f"ERROR: {exc}")
            return 1
        except Exception as exc:
            print(f"ERROR: unexpected failure: {exc!r}")
            print(f"[e2e] exception class: {type(exc).__name__}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
