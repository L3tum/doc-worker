# SPDX-License-Identifier: MPL-2.0
"""OCRmyPDF engine implementation using bundled local PaddleOCR models."""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from ocrmypdf.models.ocr_element import BoundingBox, OcrClass, OcrElement
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

from ocrmypdf_paddleocr.lang_map import SUPPORTED_LANGUAGES, tesseract_to_paddle

if TYPE_CHECKING:
    from ocrmypdf._options import OcrOptions

log = logging.getLogger(__name__)

# Configure logger for the engine to output to stdout, immediately flushed
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(name)s [%(levelname)s]: %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.INFO)

_paddle_engine = None
_paddle_lang = None


def _create_paddle_engine(lang: str):
    """Create a new PaddleX General OCR engine instance using local PP-OCRv6 models."""
    # Tesseract's plugin sets OMP_THREAD_LIMIT=1 which cripples PaddlePaddle.
    saved = os.environ.pop("OMP_THREAD_LIMIT", None)

    from paddlex_helpers import _get_paddlex_model

    try:
        return _get_paddlex_model()
    finally:
        if saved is not None:
            os.environ["OMP_THREAD_LIMIT"] = saved


def _get_paddle_engine(options: OcrOptions):
    """Get or create a cached PaddleOCR engine instance."""
    global _paddle_engine, _paddle_lang

    lang = tesseract_to_paddle(options.languages[0]) if options.languages else "en"

    if _paddle_engine is not None and _paddle_lang == lang:
        return _paddle_engine

    _paddle_engine = _create_paddle_engine(lang)
    _paddle_lang = lang

    return _paddle_engine


def _reset_paddle_engine() -> None:
    """Force recreation of the engine on next call."""
    global _paddle_engine, _paddle_lang
    _paddle_engine = None
    _paddle_lang = None


def _quad_to_bbox(quad) -> BoundingBox | None:
    """Convert a 4-point quad ((x1,y1),(x2,y2),(x3,y3),(x4,y4)) to a box."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    left, right = float(min(xs)), float(max(xs))
    top, bottom = float(min(ys)), float(max(ys))
    if right <= left or bottom <= top:
        return None
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


class PaddleOcrEngine(OcrEngine):
    """OCR engine using PaddleOCR."""

    @staticmethod
    def version() -> str:
        import paddlex

        return getattr(paddlex, "__version__", "unknown")

    @staticmethod
    def creator_tag(options: OcrOptions) -> str:
        return f"PaddleX {PaddleOcrEngine.version()}"

    def __str__(self) -> str:
        return f"PaddleX {self.version()}"

    @staticmethod
    def languages(options: OcrOptions) -> set[str]:
        return SUPPORTED_LANGUAGES

    @staticmethod
    def get_orientation(input_file: Path, options: OcrOptions) -> OrientationConfidence:
        # Avoid loading PaddleOCR's separate document-orientation model.  The
        # worker does not request OCRmyPDF page rotation, and returning a neutral
        # orientation keeps offline operation deterministic if OCRmyPDF asks.
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options: OcrOptions) -> float:
        engine = _get_paddle_engine(options)
        result = engine.predict(str(input_file))
        if not result or not result[0]:
            return 0.0

        dt_polys = result[0].get("dt_polys", [])
        if not dt_polys:
            return 0.0

        angles = []
        for poly in dt_polys:
            if len(poly) < 2:
                continue
            dx = float(poly[1][0] - poly[0][0])
            dy = float(poly[1][1] - poly[0][1])
            if abs(dx) < 1:
                continue
            angles.append(math.degrees(math.atan2(dy, dx)))

        if not angles:
            return 0.0

        angles.sort()
        mid = len(angles) // 2
        if len(angles) % 2 == 0:
            return (angles[mid - 1] + angles[mid]) / 2.0
        return angles[mid]

    @staticmethod
    def supports_generate_ocr() -> bool:
        return True

    @staticmethod
    def generate_ocr(
        input_file: Path,
        options: OcrOptions,
        page_number: int = 0,
    ) -> tuple[OcrElement, str]:
        """Run PaddleOCR and return an OcrElement tree."""
        engine = _get_paddle_engine(options)

        with Image.open(input_file) as img:
            img_width, img_height = img.size
            dpi_info = img.info.get("dpi", (300, 300))
            dpi = float(dpi_info[0] if isinstance(dpi_info, tuple) else dpi_info)

        page = OcrElement(
            ocr_class=OcrClass.PAGE,
            bbox=BoundingBox(left=0, top=0, right=img_width, bottom=img_height),
            dpi=dpi,
            page_number=page_number,
        )

        try:
            result = engine.predict(str(input_file), return_word_box=True)
        except KeyError:
            # PaddleOCR can raise on blank images with return_word_box=True.
            log.warning(
                f"PaddleOCR KeyError on page {page_number + 1} — using basic predict()"
            )
            result = engine.predict(str(input_file))
        except RuntimeError:
            # PaddlePaddle's C++ predictor can become stale across
            # ThreadPoolExecutor lifecycles. Destroy and recreate fully.
            log.debug("PaddlePaddle inference failed, recreating engine")
            _reset_paddle_engine()
            # Also destroy the singleton model in paddlex_helpers to avoid
            # a stale predictor from the shared pipeline
            try:
                from paddlex_helpers import destroy_paddlex_model

                destroy_paddlex_model()
            except Exception:
                log.exception("Failed to destroy paddlex model during recovery")
            engine = _get_paddle_engine(options)
            result = engine.predict(str(input_file), return_word_box=True)

        if not result or not result[0]:
            log.warning(f"No OCR result for page {page_number + 1} ({input_file})")
            return page, ""

        ocr_data = result[0]
        rec_texts = ocr_data.get("rec_texts", [])
        rec_scores = ocr_data.get("rec_scores", [])
        rec_boxes = ocr_data.get("rec_boxes", [])
        text_words = ocr_data.get("text_word", [])
        text_word_regions = ocr_data.get("text_word_region", [])

        if not rec_texts:
            log.warning(f"Page {page_number + 1}: no text lines detected (blank page)")
            return page, ""

        has_word_boxes = bool(text_words and text_word_regions)
        text_parts = []

        for idx, (text, score, box) in enumerate(zip(rec_texts, rec_scores, rec_boxes)):
            if not str(text).strip():
                continue

            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            if x2 <= x1 or y2 <= y1:
                continue

            line_bbox = BoundingBox(left=x1, top=y1, right=x2, bottom=y2)
            line = OcrElement(ocr_class=OcrClass.LINE, bbox=line_bbox)

            if has_word_boxes and idx < len(text_words) and text_words[idx]:
                for token, quad in zip(text_words[idx], text_word_regions[idx]):
                    token = str(token)
                    if not token.strip():
                        continue
                    word_bbox = _quad_to_bbox(quad)
                    if word_bbox is None:
                        continue
                    line.children.append(
                        OcrElement(
                            ocr_class=OcrClass.WORD,
                            bbox=word_bbox,
                            text=token,
                            confidence=float(score),
                        )
                    )
            else:
                line.children.append(
                    OcrElement(
                        ocr_class=OcrClass.WORD,
                        bbox=line_bbox,
                        text=str(text),
                        confidence=float(score),
                    )
                )

            if line.children:
                page.children.append(line)
                text_parts.append(str(text))

        full_text = "\n".join(text_parts)
        log.info(
            f"Page {page_number + 1}: {len(text_parts)} lines, {len(full_text)} chars"
        )
        return page, full_text

    @staticmethod
    def generate_hocr(input_file, output_hocr, output_text, options):
        raise NotImplementedError("Use generate_ocr()")

    @staticmethod
    def generate_pdf(input_file, output_pdf, output_text, options):
        raise NotImplementedError("Use generate_ocr()")
