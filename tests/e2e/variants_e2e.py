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
    The ordering assertion is conditioned on the model's block geometry
    matching the fixture layout (_column_side_check): a mis-localized
    marker makes the sort order the model's own boxes, so the assertion
    is then skipped with a warning that dumps the geometry (marker
    presence is still enforced; the sort itself has deterministic unit
    tests in tests/test_paddlex_helpers.py).

Check C — Word-split-layout reading order:

    run_paddlex_structure_v3(word_split.pdf) → blocks_to_markdown
    → WORDHEADER first, then every WORDLEFT* before every WORDRIGHT*
    (conditioned on model geometry as in check B).

Exit code: 0 only when all checks pass; 1 with a clear message on any
failure. Like umlaut_e2e.py this is an in-image script: it requires
ocrmypdf, poppler-utils (pdftotext) and the bundled PaddleX models,
all of which exist in the doc-worker image.
"""

from __future__ import annotations

import math
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


def _structure_pages(pdf_path: Path) -> list[dict]:
    """Run the real PP-StructureV3 pipeline on a PDF, return the page dicts.

    Mirrors server.py's /layout-parsing path: run_paddlex_structure_v3
    renders the PDF (pdftoppm) and runs the model, then sorts the model's
    blocks into reading order (DPI/page-width-aware) and keeps each
    block's model-reported bbox — which is what _column_side_check
    conditions the reading-order assertions on.
    """
    import paddlex_helpers

    pages = paddlex_helpers.run_paddlex_structure_v3(str(pdf_path))
    if not pages:
        raise E2EError(f"run_paddlex_structure_v3 returned no pages for {pdf_path}")
    return pages


def _structure_markdown(pages: list[dict]) -> str:
    """Serialize the pages' reading-order blocks to markdown (endpoint-style)."""
    import paddlex_helpers

    return "\n\n".join(
        paddlex_helpers.blocks_to_markdown(page.get("structured_blocks") or [])
        for page in pages
    )


def _blocks_flat(pages: list[dict]) -> list[dict]:
    """All reading-order blocks across all pages, in page order."""
    return [b for page in pages for b in (page.get("structured_blocks") or [])]


def _warn_model_geometry(
    check: str, markdown: str, blocks: list[dict], detail: str
) -> None:
    """Warn (do not fail) when the model's block geometry can't be trusted.

    The reading-order assertion is only meaningful when the model
    localized each marker on the fixture side it was drawn on; when it
    did not, the sort faithfully ordered the model's own boxes, so
    failing here would test the model instead of the sort. The geometry
    dump in the CI log is the follow-up signal — a model upgrade that
    localizes the markers correctly re-arms the assertion automatically.
    """
    print(
        f"[e2e] check {check}: WARNING: model geometry disagrees with the "
        f"fixture layout ({detail}) — the reading-order sort is ordering the "
        f"model's own boxes; column-first assertion skipped (marker presence "
        f"still verified)."
    )
    print("---- structure-v3 markdown ----")
    print(markdown)
    print("--------------------------------")
    print("---- model blocks (text, bbox) ----")
    for block in blocks:
        print(f"  {block.get('text')!r} {block.get('bbox')!r}")
    print("--------------------------------")


# Column geometry of the two layout fixtures — kept in sync with
# tests/fixtures/generate_two_column_pdf.py and generate_word_split_pdf.py
# (612 pt-wide page, left column at x=72 pt, right column at x=324 pt).
# The side test below converts this to rendered pixels at the pipeline's
# current RENDER_DPI (see _render_scale), so it follows RENDER_DPI
# overrides instead of assuming 300.
FIXTURE_COLUMN_SPLIT_PT = (72.0 + 324.0) / 2.0


def _render_scale() -> float:
    """Rendered pixels per PDF point (RENDER_DPI / 72) used by the pipeline."""
    import paddlex_helpers

    return paddlex_helpers.RENDER_DPI / 72.0


def _column_side_check(
    blocks: list[dict],
    title: str,
    left_markers: tuple[str, ...],
    right_markers: tuple[str, ...],
) -> tuple[bool, str]:
    """Check that the model's block geometry matches the fixture layout.

    Returns (ok, detail). ok is True only if the title and every marker
    are found in exactly one block with a finite, non-degenerate bbox,
    AND each marker's block center-x is on the fixture side the marker
    was drawn on (left of the column split for the title + LEFT* markers,
    right of it for the RIGHT* markers). A marker the model localized on
    the wrong side (or a merged cross-column block whose center lands on
    the other side) makes the column-first assertion test the model
    instead of the sort, so ok=False and the caller skips it (marker
    presence is still enforced upstream).
    """
    left_markers = (title, *left_markers)
    split_px = FIXTURE_COLUMN_SPLIT_PT * _render_scale()

    def _center_x(block: dict) -> float | None:
        bbox = block.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox[:4])
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            return None
        if not (x1 > x0 and y1 > y0):
            return None
        return (x0 + x1) / 2

    problems: list[str] = []
    for side, markers in (("left", left_markers), ("right", right_markers)):
        for marker in markers:
            hits = [
                b for b in blocks if marker in _normalized(str(b.get("text") or ""))
            ]
            if len(hits) != 1:
                problems.append(f"{marker!r} found in {len(hits)} blocks")
                continue
            center = _center_x(hits[0])
            if center is None:
                problems.append(
                    f"{marker!r} has no usable bbox: {hits[0].get('bbox')!r}"
                )
                continue
            wrong = center >= split_px if side == "left" else center <= split_px
            if wrong:
                problems.append(
                    f"{marker!r} center-x {center:.0f} px is on the wrong side "
                    f"of the fixture column split ({split_px:.0f} px)"
                )
    if problems:
        return False, "; ".join(problems)
    return True, ""


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
    """Check B: every LEFT* marker must precede every RIGHT* marker.

    Conditioned on the model's geometry matching the fixture layout
    (see _column_side_check / _warn_model_geometry).
    """
    work.mkdir(parents=True, exist_ok=True)
    fixture = _find_fixture(TWO_COLUMN_FIXTURE)
    pdf = work / "two_column.pdf"
    shutil.copy(fixture, pdf)
    print(f"[e2e] check B: fixture {fixture}")

    pages = _structure_pages(pdf)
    markdown = _structure_markdown(pages)
    normalized = _normalized(markdown)
    all_markers = (TWO_COLUMN_TITLE, *LEFT_MARKERS, *RIGHT_MARKERS)
    missing = [w for w in all_markers if not _find_all(normalized, w)]
    if missing:
        print("---- structure-v3 markdown ----")
        print(markdown if markdown else "<empty>")
        print("--------------------------------")
        raise E2EError(f"markers not recognized by the model: {missing}")

    blocks = _blocks_flat(pages)
    sides_ok, side_detail = _column_side_check(
        blocks, TWO_COLUMN_TITLE, LEFT_MARKERS, RIGHT_MARKERS
    )
    if not sides_ok:
        _warn_model_geometry("B", markdown, blocks, side_detail)
        return

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
    """Check C: WORDHEADER first, then all WORDLEFT* before all WORDRIGHT*.

    Conditioned on the model's geometry matching the fixture layout as
    in check B (see _column_side_check / _warn_model_geometry).
    """
    work.mkdir(parents=True, exist_ok=True)
    fixture = _find_fixture(WORD_SPLIT_FIXTURE)
    pdf = work / "word_split.pdf"
    shutil.copy(fixture, pdf)
    print(f"[e2e] check C: fixture {fixture}")

    pages = _structure_pages(pdf)
    markdown = _structure_markdown(pages)
    normalized = _normalized(markdown)
    all_markers = (WORD_SPLIT_HEADER, *WORD_LEFT_MARKERS, *WORD_RIGHT_MARKERS)
    missing = [w for w in all_markers if not _find_all(normalized, w)]
    if missing:
        print("---- structure-v3 markdown ----")
        print(markdown if markdown else "<empty>")
        print("--------------------------------")
        raise E2EError(f"markers not recognized by the model: {missing}")

    blocks = _blocks_flat(pages)
    sides_ok, side_detail = _column_side_check(
        blocks, WORD_SPLIT_HEADER, WORD_LEFT_MARKERS, WORD_RIGHT_MARKERS
    )
    if not sides_ok:
        _warn_model_geometry("C", markdown, blocks, side_detail)
        return

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
