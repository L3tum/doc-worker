# SPDX-License-Identifier: MPL-2.0
"""Runtime compatibility shims for pinned OCRmyPDF versions.

No ocrmypdf imports at module level: this file must stay loadable (by direct
file path) in environments without ocrmypdf so the pure decision helper is
unit-testable in dev. All ocrmypdf imports are absolute and in-function.
"""

from __future__ import annotations

import functools
import inspect
import logging
import math
from typing import Any

log = logging.getLogger(__name__)

# ocrmypdf's vector-page fallback dpi. Kept as a literal so compat.py stays
# importable without ocrmypdf; the value is drift-guarded in CI (M9).
VECTOR_PAGE_DPI = 400.0

# Supported ocrmypdf major for the graft-dpi shim (pin: >=17.4.1,<18).
_SUPPORTED_OCRMYPDF_MAJOR = 17

# M2: cap full-traceback failure logging to stop log amplification on
# adversarial multi-page PDFs (a bad pdfinfo raising on every page).
_MAX_FAIL_LOGS = 3

_fail_log_count = 0  # M2


def _safe_page_dpi(dpi: Any) -> float | None:
    """Pure decision helper — no ocrmypdf dependency.

    Accepts a Resolution-like object or (x, y) pair. Returns VECTOR_PAGE_DPI
    when the dpi is unusable (any zero, negative, or non-finite axis), else None
    (meaning: leave the page dpi as-is).
    """
    try:
        x, y = float(dpi[0]), float(dpi[1])
    except Exception:  # a malformed dpi should fall back, not crash
        return VECTOR_PAGE_DPI
    if x <= 0.0 or y <= 0.0 or not (math.isfinite(x) and math.isfinite(y)):
        return VECTOR_PAGE_DPI
    return None


def _shape_problem(_graft: Any, Resolution: Any) -> str | None:
    """D3/H1: return a human-readable problem if the private ocrmypdf shape the
    shim depends on has changed, else None.

    Only checks names already imported (``_graft.OcrGrafter.graft_page`` and
    ``Resolution``), so this adds no fragile new import paths. If a 17.x patch
    moves these, the shim logs at ERROR and leaves ``graft_page`` unpatched
    (worker backstop then routes zero-DPI files to ERROR/).
    """
    # (1) Resolution must be indexable AND yield finite floats on both axes.
    try:
        res = Resolution(1.0, 1.0)
        x, y = float(res[0]), float(res[1])
        assert x == 1.0 and y == 1.0
    except Exception as exc:
        return f"Resolution not indexable: {exc!r}"
    # (2) graft_page must accept `pageno` as a keyword-only param (true in 17.x).
    try:
        pageno = inspect.signature(_graft.OcrGrafter.graft_page).parameters.get(
            "pageno"
        )
        if pageno is None or pageno.kind is not inspect.Parameter.KEYWORD_ONLY:
            return f"graft_page `pageno` is not keyword-only (got {pageno!r})"
    except (ValueError, TypeError) as exc:
        return f"could not introspect graft_page signature: {exc!r}"
    return None


def apply_zero_dpi_graft_workaround() -> None:
    """Idempotently patch OcrGrafter.graft_page to survive zero-DPI pages.

    OCRmyPDF 17.x stores the graft scale from ``self.pdfinfo[pageno].dpi``
    (a scalar) with no vector-page fallback. Imageless/vector/text-only pages
    report ``Resolution(0, 0)`` whose scalar dpi is 0.0; the fpdf2 renderer then
    divides by it — ``page_width_px * 72.0 / dpi`` in
    ``ocrmypdf/fpdf_renderer/renderer.py`` (``CoordinateTransform``) — raising
    ``ZeroDivisionError``. ``Resolution.to_scalar()`` returns 0 (it does NOT
    raise), so the crash is in the renderer, not in ``to_scalar()``. The wrapper
    substitutes ``_dpi = Resolution(VECTOR_PAGE_DPI, VECTOR_PAGE_DPI)`` for
    unusable page dpi so the stored scalar is a sane 400 before the render.

    Guarded and fail-soft: the version gate and a private-shape check run first
    (any mismatch logs at ERROR and leaves ``graft_page`` unpatched). The whole
    post-import application body is wrapped so an unexpected internal change can
    never break plugin load — it degrades to the worker backstop.
    """
    try:
        import ocrmypdf
        from ocrmypdf import _graft
        from ocrmypdf.helpers import Resolution
    except ImportError:
        log.warning("zero-dpi graft workaround skipped: ocrmypdf not importable")
        return

    try:
        try:
            major = int(str(getattr(ocrmypdf, "__version__", "")).split(".")[0])
        except ValueError:
            major = None
        if major != _SUPPORTED_OCRMYPDF_MAJOR:
            log.error(
                "zero-dpi graft workaround SKIPPED: ocrmypdf major %r (built for %s). "
                "Zero-DPI pages will hit the worker backstop (ERROR/). Verify internals "
                "and update _SUPPORTED_OCRMYPDF_MAJOR.",
                major,
                _SUPPORTED_OCRMYPDF_MAJOR,
            )
            return

        problem = _shape_problem(_graft, Resolution)
        if problem is not None:
            log.error(
                "zero-dpi graft workaround SKIPPED: ocrmypdf internals changed (%s). "
                "Zero-DPI pages will hit the worker backstop (ERROR/). Update the shim.",
                problem,
            )
            return

        original = _graft.OcrGrafter.graft_page
        if getattr(original, "_doc_worker_patched", False):
            return

        @functools.wraps(original)
        def patched_graft_page(self: Any, *args: Any, **kwargs: Any) -> Any:
            global _fail_log_count
            try:
                pageno = kwargs.get("pageno")  # M7: pageno is keyword-only in 17.x
                if pageno is not None:
                    fallback = _safe_page_dpi(self.pdfinfo[pageno].dpi)
                    if fallback is not None:
                        log.info(
                            "zero-DPI page (0-based index %s): using fallback dpi %s for graft",
                            pageno,
                            fallback,
                        )
                        # M3: mutates ocrmypdf's own PageInfo. The 400 dpi value
                        # persists for the rest of the run (also read by width/height_px)
                        # and matches ocrmypdf's own vector-page convention.
                        self.pdfinfo[pageno]._dpi = Resolution(fallback, fallback)
            except Exception:  # M2: a bad page must never break the real graft
                if _fail_log_count < _MAX_FAIL_LOGS:
                    log.exception(
                        "zero-dpi graft workaround failed; using original behavior"
                    )
                    _fail_log_count += 1
                else:
                    log.debug("zero-dpi graft workaround failed (log suppressed)")
            return original(self, *args, **kwargs)

        patched_graft_page._doc_worker_patched = True  # type: ignore[attr-defined]
        _graft.OcrGrafter.graft_page = patched_graft_page  # type: ignore[method-assign]
        log.info(
            "zero-dpi graft workaround applied (ocrmypdf %s)", ocrmypdf.__version__
        )
    except Exception:  # M1: shim application must never break plugin load
        log.exception(
            "zero-dpi graft workaround failed to apply; relying on the worker backstop"
        )
