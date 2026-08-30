from __future__ import annotations

import importlib
import os
from typing import Any

import pytest


def test_ocrmypdf_engine_uses_shared_local_paddleocr_factory(monkeypatch):
    pytest.importorskip("ocrmypdf")
    pytest.importorskip("PIL")

    engine = importlib.import_module("ocrmypdf_paddleocr.engine")
    fake_model = object()
    calls: list[dict[str, Any]] = []

    def fake_create_paddlex_ocr_pipeline(
        *, use_textline_orientation: bool = True
    ) -> object:
        calls.append({"use_textline_orientation": use_textline_orientation})
        return fake_model

    monkeypatch.setattr(
        "paddlex_helpers._create_paddlex_ocr_pipeline", fake_create_paddlex_ocr_pipeline
    )
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")

    assert engine._create_paddle_engine("german") is fake_model
    assert calls == [{"use_textline_orientation": True}]
    assert os.environ["OMP_THREAD_LIMIT"] == "1"


def test_ocrmypdf_language_map_accepts_readme_codes():
    pytest.importorskip("ocrmypdf")

    from ocrmypdf_paddleocr.lang_map import SUPPORTED_LANGUAGES, tesseract_to_paddle

    assert tesseract_to_paddle("deu") == "german"
    assert tesseract_to_paddle("eng") == "en"
    assert tesseract_to_paddle("ch_sim") == "ch"
    assert tesseract_to_paddle("ch_tra") == "chinese_cht"
    assert {"deu", "eng", "ch_sim", "ch_tra"}.issubset(SUPPORTED_LANGUAGES)


# --- generate_ocr DPI hygiene (zero/NaN tag -> VECTOR_PAGE_DPI) --------------


def _make_fake_image(dpi: object) -> type:
    """Return a fake PIL.Image class whose .open() yields an img with *dpi* tag."""

    class _Img:
        size = (800, 600)

        def __init__(self_inner) -> None:
            self_inner.info = {"dpi": dpi}

        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *exc):
            return False

    class _Image:
        @staticmethod
        def open(_path):
            return _Img()

    return _Image


@pytest.mark.parametrize(
    ("tag_dpi", "expected_dpi"),
    [
        ((0.0, 0.0), 400.0),  # zero -> 400
        ((float("nan"), float("nan")), 400.0),  # NaN -> 400
        ((float("inf"), float("inf")), 400.0),  # inf -> 400: the D2=B isfinite fix
        ((300.0, 300.0), 300.0),  # usable -> unchanged
        ((-300.0, -300.0), 400.0),  # negative DPI -> fallback to VECTOR_PAGE_DPI
        (
            0.0,
            400.0,
        ),  # scalar (non-tuple) tag: engine reads dpi_info[0] if tuple, else dpi_info
    ],
)
def test_generate_ocr_dpi_hygiene(
    monkeypatch, mock_paddlex_ocr_pipeline, tag_dpi, expected_dpi
):
    """A zero-DPI PNG tag must fall back to VECTOR_PAGE_DPI (400); a usable
    tag is left unchanged. Exercises the exact blank-page shape that previously
    produced page.dpi == 0.0."""
    pytest.importorskip("ocrmypdf")
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    # Mock paddlex so the engine's `import paddlex` (inside version())
    # succeeds without triggering paddlex's heavy __init__ side effects.
    monkeypatch.setitem(sys.modules, "paddlex", MagicMock())

    engine = importlib.import_module("ocrmypdf_paddleocr.engine")

    # Clear the engine cache so _get_paddle_engine uses the mocked factory.
    monkeypatch.setattr(engine, "_paddle_engine", None)
    monkeypatch.setattr(engine, "_paddle_lang", None)
    monkeypatch.setattr(engine, "Image", _make_fake_image(tag_dpi))

    # Blank page: predict yields no text lines -> early return with dpi set.
    mock_paddlex_ocr_pipeline.predict.side_effect = lambda _path, **kw: iter(
        [{"rec_texts": []}]
    )

    options = SimpleNamespace(languages=["deu"])
    page, text = engine.PaddleOcrEngine.generate_ocr(Path("x.png"), options, 0)
    assert text == ""
    assert page.dpi == expected_dpi


def test_initialize_fail_soft_when_shim_application_breaks(monkeypatch):
    """Even if the shim's internal gate blows up, initialize() must not raise.

    The fail-soft boundary is compat.apply_zero_dpi_graft_workaround's own
    except Exception (not the __init__.py wrapper, which was removed).
    """
    pytest.importorskip("ocrmypdf")
    import sys
    from unittest.mock import MagicMock

    # Mock paddlex so initialize()'s `import paddlex` succeeds without
    # triggering paddlex's heavy __init__ side effects (repo_manager).
    monkeypatch.setitem(sys.modules, "paddlex", MagicMock())

    from ocrmypdf import _graft

    from ocrmypdf_paddleocr import compat, initialize

    # Break the shim's internal gate so compat.py's broad M1 except must fire
    monkeypatch.setattr(compat, "_shape_problem", lambda g, R: 1 / 0)
    initialize(None)  # must NOT raise

    # The graft function was NOT patched (the shim failed before wrapping)
    assert (
        getattr(_graft.OcrGrafter.graft_page, "_doc_worker_patched", False) is not True
    )
