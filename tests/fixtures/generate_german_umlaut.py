#!/usr/bin/env python3
"""Generate tests/fixtures/german_umlaut.png.

Deterministic on a given machine (fixed font, size, layout), but not
byte-reproducible across machines (font resolution may fall back to
~/.fonts); the committed PNG is hash-pinned by a unit test.

Renders "Äpfel", "Bürger" and "Straße" — one word per line — in DejaVu Sans
(regular weight), black on white, so the in-image e2e (tests/e2e/umlaut_e2e.py)
has clean large glyphs to OCR.

Re-run this script whenever the fixture is ever edited by hand:

    python3 tests/fixtures/generate_german_umlaut.py

The CI e2e-ocr job (.github/workflows/docker.yaml) OCRs this exact image, so a
stale or hand-edited fixture silently breaks (or fakes) the umlaut regression
test.

The script exits non-zero when the DejaVu Sans TTF cannot be located, or when
a critical glyph (Ä, ü, ß) renders as the font's .notdef tofu box — the exact
failure mode that originally produced this fixture (Pillow's default bitmap
font has no Latin-1 umlauts and drew "□pfel" / "B□rger").
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LINES = ("Äpfel", "Bürger", "Straße")
CRITICAL_CHARS = ("Ä", "ü", "ß")
MAX_WIDTH = 1200
MARGIN = 80
FONT_SIZE_START = 96
FONT_SIZE_STEP = 4
FONT_SIZE_MIN = 72
# Filesystem candidate list only — fc-list is not available in the CI shells.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)
NOTDEF_CHAR = "\uffff"  # unassigned codepoint -> the font's .notdef glyph


def _find_dejavu_sans() -> Path:
    """Return the first existing DejaVuSans.ttf candidate, else exit(1)."""
    candidates = [Path(path) for path in FONT_CANDIDATES]
    home_fonts = Path.home() / ".fonts"
    if home_fonts.is_dir():
        candidates.extend(sorted(home_fonts.rglob("DejaVuSans.ttf")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(str(candidate) for candidate in candidates)
    sys.exit(f"ERROR: DejaVuSans.ttf not found. Tried:\n  {tried}")


def _mask_bytes(mask: object) -> bytes:
    """Rasterize an ImageFont.getmask() result to raw bytes.

    Pillow >= 9 returns a low-level ImagingCore (no tobytes()); older
    versions return an Image.Image. Pasting into an "L" image with an
    explicit 4-item region box works for both.
    """
    tobytes = getattr(mask, "tobytes", None)
    if callable(tobytes):
        return bytes(tobytes())
    image = Image.new("L", mask.size, 0)  # type: ignore[attr-defined]
    image.paste(
        mask,  # type: ignore[arg-type]
        (0, 0, mask.size[0], mask.size[1]),  # type: ignore[attr-defined]
    )
    return image.tobytes()


def _assert_glyphs_rendered(font: ImageFont.FreeTypeFont) -> None:
    """Exit(1) if a critical character renders as the font's .notdef box.

    .notdef is looked up with U+FFFF (an unassigned codepoint), so its mask
    is exactly what a *missing* glyph would render as; a byte-identical mask
    means the font drew tofu instead of the real glyph.
    """
    notdef_mask = font.getmask(NOTDEF_CHAR)
    notdef_bytes = _mask_bytes(notdef_mask)
    missing = []
    for char in CRITICAL_CHARS:
        mask = font.getmask(char)
        if mask.size == notdef_mask.size and _mask_bytes(mask) == notdef_bytes:
            missing.append(char)
    if missing:
        names = ", ".join(repr(char) for char in missing)
        sys.exit(
            f"ERROR: glyph(s) {names} render as the font's .notdef tofu box "
            "— the fixture would contain tofu instead of real glyphs. "
            "Check the font file."
        )


def _fit_font(font_path: Path) -> ImageFont.FreeTypeFont:
    """Largest DejaVu Sans font size (down from FONT_SIZE_START) whose longest
    line fits the canvas, so the image never exceeds MAX_WIDTH."""
    for size in range(FONT_SIZE_START, FONT_SIZE_MIN - 1, -FONT_SIZE_STEP):
        candidate = ImageFont.truetype(str(font_path), size)
        if max(candidate.getlength(line) for line in LINES) <= MAX_WIDTH - 2 * MARGIN:
            return candidate
    return ImageFont.truetype(str(font_path), FONT_SIZE_MIN)


def main() -> None:
    font_path = _find_dejavu_sans()
    font = _fit_font(font_path)
    print(f"[fixture] font: {font_path} ({font.size}px)")

    # Fail early on tofu: the e2e must OCR real umlauts, not .notdef boxes.
    _assert_glyphs_rendered(font)

    ascent, descent = font.getmetrics()
    line_height = int((ascent + descent) * 1.25)  # generous leading
    width = min(
        MAX_WIDTH, int(max(font.getlength(line) for line in LINES)) + 2 * MARGIN
    )
    height = 2 * MARGIN + len(LINES) * line_height

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(LINES):
        draw.text(
            (MARGIN, MARGIN + index * line_height),
            line,
            font=font,
            fill="black",
            anchor="la",
        )

    out_path = Path(__file__).resolve().parent / "german_umlaut.png"
    image.save(out_path, "PNG")
    print(f"[fixture] wrote {out_path} ({width}x{height})")


if __name__ == "__main__":
    main()
