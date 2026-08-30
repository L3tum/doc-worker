"""Tests for the ocrmypdf zero-DPI graft shim (requires ocrmypdf installed).

These run in Docker/CI where ocrmypdf is present and are skipped in the dev
environment via ``pytest.importorskip("ocrmypdf")``.
"""

from __future__ import annotations

import shutil
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def shim_ctx(monkeypatch):
    """Yield (compat, _graft) with a clean slate for (re)applying the shim.

    Ensures ``_graft.OcrGrafter.graft_page`` is the real original before
    each test, and reverts on teardown.
    """
    pytest.importorskip("ocrmypdf")
    from ocrmypdf import _graft

    from ocrmypdf_paddleocr import compat

    real_original = _graft.OcrGrafter.graft_page
    monkeypatch.setattr(_graft.OcrGrafter, "graft_page", real_original)
    yield compat, _graft


class _FakePageInfo:
    """Mimics ocrmypdf's PageInfo: dpi property backed by a _dpi cache."""

    def __init__(self, dpi: Any = None) -> None:
        self._dpi = dpi

    @property
    def dpi(self) -> Any:
        if self._dpi is None:
            from ocrmypdf.helpers import Resolution

            return Resolution(0.0, 0.0)
        return self._dpi


def test_apply_is_idempotent(shim_ctx) -> None:
    compat, _graft = shim_ctx
    compat.apply_zero_dpi_graft_workaround()
    first = _graft.OcrGrafter.graft_page
    assert getattr(first, "_doc_worker_patched", False) is True
    compat.apply_zero_dpi_graft_workaround()
    second = _graft.OcrGrafter.graft_page
    # Second application must not re-wrap: same wrapper object.
    assert first is second


def test_zero_dpi_page_gets_fallback_dpi(shim_ctx, monkeypatch) -> None:
    from ocrmypdf.helpers import Resolution

    compat, _graft = shim_ctx

    calls: list[tuple[int, dict]] = []

    def stub_graft_page(self: Any, *, pageno: Any, **kwargs: Any) -> str:
        calls.append((pageno, kwargs))
        return "ok"

    # Install the stub BEFORE apply so the wrapper captures it as "original".
    monkeypatch.setattr(_graft.OcrGrafter, "graft_page", stub_graft_page)
    compat.apply_zero_dpi_graft_workaround()

    page = _FakePageInfo()  # _dpi is None -> dpi property returns Resolution(0,0)
    fake_self = SimpleNamespace(pdfinfo={1: page})

    result = _graft.OcrGrafter.graft_page(fake_self, pageno=1)

    assert result == "ok"
    assert calls == [(1, {})]
    assert page._dpi is not None
    assert page._dpi.to_scalar() == 400.0
    # Sanity: the substitute is a usable non-zero Resolution.
    assert page._dpi == Resolution(400.0, 400.0)


def test_usable_dpi_untouched(shim_ctx, monkeypatch) -> None:
    from ocrmypdf.helpers import Resolution

    compat, _graft = shim_ctx

    def stub_graft_page(self: Any, *, pageno: Any, **kwargs: Any) -> str:
        return "ok"

    monkeypatch.setattr(_graft.OcrGrafter, "graft_page", stub_graft_page)
    compat.apply_zero_dpi_graft_workaround()

    initial = Resolution(72.0, 72.0)
    page = _FakePageInfo(initial)
    fake_self = SimpleNamespace(pdfinfo={1: page})

    _graft.OcrGrafter.graft_page(fake_self, pageno=1)

    # A usable dpi is left alone (same object, not replaced).
    assert page._dpi is initial


def test_fail_soft_when_pdfinfo_raises(shim_ctx, monkeypatch) -> None:
    compat, _graft = shim_ctx

    def stub_graft_page(self: Any, *, pageno: Any, **kwargs: Any) -> str:
        return "ok"

    monkeypatch.setattr(_graft.OcrGrafter, "graft_page", stub_graft_page)
    compat.apply_zero_dpi_graft_workaround()

    class _BadPdfInfo:
        def __getitem__(self, i: Any) -> Any:
            raise KeyError(i)

    fake_self = SimpleNamespace(pdfinfo=_BadPdfInfo())

    # Must not raise; must still delegate to the original (stub).
    assert _graft.OcrGrafter.graft_page(fake_self, pageno=1) == "ok"


def test_initialize_applies_shim(shim_ctx) -> None:
    """D1=A: plugin initialize() wraps graft_page (shim applied at plugin load)."""
    pytest.importorskip("ocrmypdf")
    pytest.importorskip("paddlex")
    from ocrmypdf import _graft

    from ocrmypdf_paddleocr import initialize

    initialize(None)  # our hook does not use plugin_manager
    assert getattr(_graft.OcrGrafter.graft_page, "_doc_worker_patched", False) is True


def test_zero_dpi_scalar_is_zero_and_substitute_is_usable() -> None:
    """H2 (floor): a real Resolution(0,0) yields scalar dpi 0 (the value the fpdf2
    renderer divides by -> ZDE); the shim's substitute yields a usable scalar.
    Uses only public ocrmypdf.helpers.Resolution — no private ctor, so it is not flaky.
    """
    pytest.importorskip("ocrmypdf")
    from ocrmypdf.helpers import Resolution

    zero = Resolution(0.0, 0.0)
    assert zero.to_scalar() == 0  # to_scalar does NOT raise; the renderer does
    with pytest.raises(ZeroDivisionError):
        _ = 1600 * 72.0 / zero.to_scalar()  # exact CoordinateTransform arithmetic
    sub = Resolution(400.0, 400.0)
    assert sub.to_scalar() == 400.0
    assert 1600 * 72.0 / sub.to_scalar() == 1600 * 72.0 / 400.0


def test_coordinate_transform_zero_dpi_raises() -> None:
    """H2 (direct): drive ocrmypdf's real fpdf renderer at dpi=0.

    Exact shape (ocrmypdf/fpdf_renderer/renderer.py):
    CoordinateTransform(dpi, page_width_px, page_height_px).page_width_pt =
    page_width_px * 72.0 / dpi. Skips (not fails) if the shape moved, so the
    suite stays green; the floor test above carries the guarantee.
    """
    pytest.importorskip("ocrmypdf")
    try:
        from ocrmypdf.fpdf_renderer import renderer
    except ImportError:
        pytest.skip("fpdf_renderer not importable at expected path")
    CT = getattr(renderer, "CoordinateTransform", None)
    if CT is None:
        pytest.skip(
            "CoordinateTransform not at expected path; update for pinned ocrmypdf"
        )
    try:
        with pytest.raises(ZeroDivisionError):
            _ = CT(dpi=0, page_width_px=1600, page_height_px=1200).page_width_pt
        assert CT(dpi=400, page_width_px=1600, page_height_px=1200).page_width_pt == (
            1600 * 72.0 / 400.0
        )
    except (
        TypeError,
        AttributeError,
    ) as exc:  # ctor/attr shape changed in pinned ocrmypdf
        pytest.skip(
            f"CoordinateTransform shape changed; the floor test carries the guarantee: {exc!r}"
        )


@pytest.mark.parametrize("version", ["18.0.0", "16.5.0", "99.9.9"])
def test_not_patched_for_unsupported_major(shim_ctx, monkeypatch, version) -> None:
    import ocrmypdf

    compat, _graft = shim_ctx
    monkeypatch.setattr(ocrmypdf, "__version__", version, raising=False)
    compat.apply_zero_dpi_graft_workaround()
    assert (
        getattr(_graft.OcrGrafter.graft_page, "_doc_worker_patched", False) is not True
    )


@pytest.mark.parametrize("version", ["", "dev", "abc"])
def test_not_patched_for_unparseable_version(shim_ctx, monkeypatch, version) -> None:
    import ocrmypdf

    compat, _graft = shim_ctx
    monkeypatch.setattr(ocrmypdf, "__version__", version, raising=False)
    compat.apply_zero_dpi_graft_workaround()
    assert (
        getattr(_graft.OcrGrafter.graft_page, "_doc_worker_patched", False) is not True
    )


def test_vector_page_dpi_matches_upstream() -> None:
    """M9: our hardcoded constant must equal ocrmypdf's real value (drift guard)."""
    pytest.importorskip("ocrmypdf")
    from ocrmypdf_paddleocr import compat

    try:
        from ocrmypdf import _pipeline

        upstream = float(_pipeline.VECTOR_PAGE_DPI)
    except (ImportError, AttributeError) as exc:
        pytest.fail(
            f"ocrmypdf._pipeline.VECTOR_PAGE_DPI not at expected path ({exc!r}). "
            "The pinned ocrmypdf range moved a private constant — update "
            "compat.VECTOR_PAGE_DPI and this drift guard instead of silently skipping."
        )
    assert compat.VECTOR_PAGE_DPI == upstream, (
        f"compat.VECTOR_PAGE_DPI is {compat.VECTOR_PAGE_DPI}, "
        f"upstream is {upstream}. Update compat.py to match."
    )


def test_max_fail_logs_cap_respected(shim_ctx, monkeypatch, caplog) -> None:
    """After _MAX_FAIL_LOGS (3) ERROR logs, subsequent failures log at DEBUG only."""
    import logging

    compat, _graft = shim_ctx

    # Reset the module-level counter for order-independence
    monkeypatch.setattr(compat, "_fail_log_count", 0)

    # Install a stub that passes the shape gate
    def stub_graft_page(self: Any, *, pageno: Any, **kwargs: Any) -> str:
        return "ok"

    monkeypatch.setattr(_graft.OcrGrafter, "graft_page", stub_graft_page)
    compat.apply_zero_dpi_graft_workaround()

    # Make pdfinfo raise so the wrapper's except path fires on every call
    class _BadPdfInfo:
        def __getitem__(self, i: Any) -> Any:
            raise KeyError(i)

    fake_self = SimpleNamespace(pdfinfo=_BadPdfInfo())

    with caplog.at_level(logging.DEBUG, logger="ocrmypdf_paddleocr.compat"):
        for _ in range(5):
            assert _graft.OcrGrafter.graft_page(fake_self, pageno=1) == "ok"

    full = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "using original behavior" in r.getMessage()
    ]
    suppressed = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and "log suppressed" in r.getMessage()
    ]
    assert len(full) == compat._MAX_FAIL_LOGS  # 3
    assert len(suppressed) == 2  # calls 4 and 5


def test_real_pageinfo_dpi_is_indexable_resolution(tmp_path) -> None:
    """A real PageInfo.dpi must return an indexable Resolution (not a scalar).

    Guards against a future ocrmypdf change that would make _safe_page_dpi
    silently fall back to 400.0 for every page.
    """
    pytest.importorskip("ocrmypdf")
    pytest.importorskip("img2pdf")
    import img2pdf
    from ocrmypdf._pipeline import get_pdfinfo
    from PIL import Image as PILImage

    img = tmp_path / "img.png"
    PILImage.new("RGB", (300, 200), "white").save(img, dpi=(300, 300))
    pdf = tmp_path / "img.pdf"
    pdf.write_bytes(img2pdf.convert(str(img)))

    dpi = get_pdfinfo(str(pdf))[0].dpi
    assert dpi[0] == pytest.approx(300.0)  # indexable → Resolution-shaped
    assert dpi[1] == pytest.approx(300.0)
    assert dpi.to_scalar() == pytest.approx(300.0)


@pytest.mark.skipif(
    shutil.which("gs") is None,
    reason="ghostscript required for ocrmypdf rasterisation",
)
def test_e2e_vector_page_runs_without_zero_division(tmp_path, monkeypatch) -> None:
    """A vector-only PDF (dpi=(0,0)) must OCR without ZeroDivisionError.

    This is the integration smoke test that proves the shim's core claim:
    mutating PageInfo._dpi prevents the fpdf2 renderer's division-by-zero.
    All other shim tests use hand-written fakes.
    """
    pytest.importorskip("ocrmypdf")
    pytest.importorskip("paddlex")
    pytest.importorskip("pikepdf")
    import ocrmypdf
    import pikepdf
    from ocrmypdf import _graft

    from ocrmypdf_paddleocr import compat

    # 1. Create a 1-page vector-only PDF (no images → PageInfo._dpi is None → dpi (0,0))
    src = tmp_path / "vector_only.pdf"
    with pikepdf.Pdf.new() as pdf:
        page = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=[0, 0, 200, 100],
            )
        )
        page.Contents = pikepdf.Stream(
            pdf, b"1 0 0 1 0 0 cm\n0 0 0 RG\n20 20 160 60 re S\n"
        )
        pdf.pages.append(page)
        pdf.save(str(src))

    # 2. Fake OCR pipeline (inline — NOT the conftest fixture, whose
    #    predict.side_effect rejects the return_word_box= kwarg used by engine)
    pipeline = MagicMock()
    pipeline.predict.side_effect = lambda path, **kw: iter(
        [
            {
                "rec_texts": ["hello"],
                "rec_scores": [0.9],
                "rec_boxes": [[10, 10, 80, 30]],
                "text_word": [],
                "text_word_region": [],
            }
        ]
    )
    monkeypatch.setattr("paddlex_helpers._get_paddlex_model", lambda: pipeline)

    # 3. Order-independence: an earlier shim_ctx test may have restored the
    #    unpatched real graft_page while ocrmypdf's cached plugin manager
    #    already ran initialize. Explicitly re-apply the shim.
    orig_graft = _graft.OcrGrafter.graft_page
    if getattr(orig_graft, "_doc_worker_patched", False):
        orig_graft = orig_graft.__wrapped__
    _graft.OcrGrafter.graft_page = orig_graft
    compat.apply_zero_dpi_graft_workaround()

    # 4. Run ocrmypdf end-to-end
    out = tmp_path / "out.pdf"
    ocrmypdf.ocr(
        str(src),
        str(out),
        plugins=["ocrmypdf_paddleocr"],
        language="deu",
        force_ocr=True,
    )

    # 5. Assert success — the ZDE is gone
    assert out.exists()
    assert out.stat().st_size > 0

    # Cleanup: restore the unpatched real graft_page
    _graft.OcrGrafter.graft_page = orig_graft
