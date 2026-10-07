"""
Doc-Worker — In-image e2e: umlaut regression for PaddleX word segmentation.

Runs inside the built Docker image (see the ``e2e-ocr`` job in
``.github/workflows/docker.yaml``):

    Pillow        → 1-page PDF from tests/fixtures/german_umlaut.png
    ocrmypdf.ocr  → plugins=["ocrmypdf_paddleocr"], language="deu", force_ocr=True
                    (mirrors the ocrmypdf call in worker.py:run_ocrmypdf)
    pdftotext     → assert "Äpfel" / "Bürger" / "Straße" survive as
                    contiguous words

Regression being pinned: PaddleX's word segmentation used an ASCII-only
classifier, so umlauts were emitted as their own word tokens and the OCRed
PDF's hidden text layer extracted as "Ä pfel" / "B ü rger". The runtime
patch ``paddlex_helpers._patch_paddlex_word_segmentation()`` must keep real
words contiguous in the extracted text.

Exit code: 0 only when the word-segmentation patch state is "applied"
AND every expected word extracts as a contiguous word; 1 with a clear
message on any failure. The state check is a hard gate (not just a
warning) because the pinned PaddleX 3.7.2 still carries the upstream
bug (PaddleX#5188), so a correct run must have state "applied"; when
upstream lands the fix, the pin is bumped and the patch is deleted per
its removal contract.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

# Inside the CI container the fixtures are mounted read-only at /fixtures.
FIXTURE_IN_IMAGE = Path("/fixtures/german_umlaut.png")
# Fallback for local (non-Docker) runs: repo-relative fixture.
FIXTURE_IN_REPO = Path(__file__).resolve().parents[1] / "fixtures" / "german_umlaut.png"

EXPECTED_WORDS = ("Äpfel", "Bürger", "Straße")


class E2EError(Exception):
    """Raised by e2e helpers; caught once in main() for a clean ERROR exit."""


def _find_fixture() -> Path:
    """Return the first existing fixture path, else raise E2EError."""
    for candidate in (FIXTURE_IN_IMAGE, FIXTURE_IN_REPO):
        if candidate.is_file():
            return candidate
    raise E2EError(
        f"fixture not found — looked for {FIXTURE_IN_IMAGE} "
        f"(in-image mount) and {FIXTURE_IN_REPO} (repo fallback)."
    )


def _build_pdf(png: Path, pdf: Path) -> None:
    """Build a 1-page image-only PDF from the PNG (Pillow, already in the image)."""
    from PIL import Image

    with Image.open(png) as img:
        img.convert("RGB").save(pdf, "PDF")


def _pdftotext(pdf: Path, txt: Path) -> str:
    """Run pdftotext (poppler-utils, installed by the Dockerfile).

    Returns the extracted text.
    """
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


def _print_runtime_diagnostics() -> str | None:
    """Print PaddleX version + patch state; return the patch state.

    Returns the word-segmentation patch state ("applied" / "skipped" /
    "unpatched"), or None if paddlex_helpers is not importable.  Both
    imports can fail outside the container; a silent gate-skip (state
    "skipped" / "unpatched") must be distinguishable from an OCR
    misread when the word check below fails.
    """
    try:
        import paddlex
    except ImportError as exc:
        print(f"[e2e] diagnostics: paddlex not importable: {exc}")
    else:
        version = getattr(paddlex, "__version__", "unknown")
        print(f"[e2e] diagnostics: PaddleX {version}")
    state: str | None = None
    try:
        import paddlex_helpers
    except ImportError as exc:
        print(f"[e2e] diagnostics: paddlex_helpers not importable: {exc}")
    else:
        state = paddlex_helpers._PADDLEX_WORD_SEGMENTATION_STATE
        print(f"[e2e] diagnostics: word-segmentation patch state: {state}")
        if state != "applied":
            print(
                "[e2e] WARNING: patch state is not 'applied' — the e2e "
                "may still pass (e.g. upstream already fixed the word "
                "segmentation), but if the word check below fails, check "
                "this patch state first."
            )
    return state


def main() -> int:
    """Run the full OCR pipeline and verify the words stayed contiguous."""
    try:
        fixture = _find_fixture()
    except E2EError as exc:
        print(f"ERROR: {exc}")
        return 1

    with tempfile.TemporaryDirectory(prefix="umlaut_e2e_") as tmp:
        tmp_path = Path(tmp)
        input_pdf = tmp_path / "input.pdf"
        ocr_pdf = tmp_path / "ocr.pdf"
        text_out = tmp_path / "text.txt"

        try:
            print(f"[e2e] fixture: {fixture} ({fixture.stat().st_size:,} bytes)")
            _build_pdf(fixture, input_pdf)
            print(f"[e2e] built 1-page PDF: {input_pdf.stat().st_size:,} bytes")

            patch_state = _print_runtime_diagnostics()
            if patch_state != "applied":
                raise E2EError(
                    f"patch not applied (state={patch_state!r}): refusing "
                    "to pass. If PaddleX was upgraded and fixed upstream, "
                    "bump the pin and delete the patch per its removal "
                    "contract."
                )

            import ocrmypdf

            print(
                "[e2e] running ocrmypdf.ocr (ocrmypdf_paddleocr, "
                "language=deu, force_ocr)"
            )
            ocrmypdf.ocr(
                input_pdf,
                ocr_pdf,
                plugins=["ocrmypdf_paddleocr"],
                language="deu",
                force_ocr=True,
            )
            print(f"[e2e] OCRed PDF: {ocr_pdf.stat().st_size:,} bytes")

            text = _pdftotext(ocr_pdf, text_out)
            # Collapse whitespace to single spaces; a spurious mid-word space
            # (e.g. "Ä pfel") must remain and fail the boundary check below.
            normalized = " ".join(text.split())

            missing = [
                word
                for word in EXPECTED_WORDS
                if re.search(rf"\b{re.escape(word)}\b", normalized) is None
            ]
            if missing:
                print(f"ERROR: expected words not found as contiguous words: {missing}")
                print("---- pdftotext output ----")
                print(text if text else "<empty>")
                print("----------------------------")
                return 1

            for word in EXPECTED_WORDS:
                print(f"[e2e] OK: '{word}' present as a contiguous word")
            print("[e2e] PASS")
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
