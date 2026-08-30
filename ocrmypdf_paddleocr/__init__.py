# SPDX-License-Identifier: MPL-2.0
"""Local OCRmyPDF plugin backed by the bundled PaddleOCR PP-OCRv6 models."""

from __future__ import annotations

import logging

from ocrmypdf import hookimpl

log = logging.getLogger(__name__)


@hookimpl
def initialize(plugin_manager):
    """Check PaddleOCR is importable, then apply the zero-DPI graft shim.

    The shim patches ``OcrGrafter.graft_page`` in-process. ocrmypdf fires
    ``initialize`` once per interpreter (it caches the plugin manager) and
    ``apply_zero_dpi_graft_workaround`` is idempotent, so one call covers every
    later in-process OCR run. A shim failure must NOT break plugin load: it is
    fail-soft, and if it doesn't apply, the worker's non-retryable
    ZeroDivisionError backstop still routes zero-DPI files to ERROR/.
    """
    try:
        import paddlex  # noqa: F401
    except ImportError:
        from ocrmypdf.exceptions import MissingDependencyError

        raise MissingDependencyError(
            "PaddleX is required but not installed. Install with: pip install paddlex paddlepaddle"
        )

    from ocrmypdf_paddleocr.compat import apply_zero_dpi_graft_workaround

    apply_zero_dpi_graft_workaround()


@hookimpl
def check_options(options):
    """Limit concurrency -- PaddlePaddle's inference crashes with multiple workers."""
    if options.jobs != 1:
        log.info("PaddleOCR: forcing jobs=1 (PaddlePaddle is not multi-process safe)")
        options.jobs = 1


@hookimpl
def get_ocr_engine():
    """Return PaddleOcrEngine."""
    from ocrmypdf_paddleocr.engine import PaddleOcrEngine

    return PaddleOcrEngine()
