"""Pure decision-helper tests for ocrmypdf_paddleocr/compat.py.

These run in the dev environment WITHOUT ocrmypdf installed. compat.py is loaded
by direct file path (not via the package import) so that its ocrmypdf-requiring
package ``__init__`` is never executed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

COMPAT_PATH = Path(__file__).resolve().parents[2] / "ocrmypdf_paddleocr" / "compat.py"

_spec = importlib.util.spec_from_file_location("compat_under_test", COMPAT_PATH)
compat = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(compat)


class TestSafePageDpi:
    def test_zero_dpi_falls_back(self) -> None:
        assert compat._safe_page_dpi((0.0, 0.0)) == 400.0

    def test_single_axis_zero_falls_back(self) -> None:
        assert compat._safe_page_dpi((0.0, 72.0)) == 400.0
        assert compat._safe_page_dpi((72.0, 0.0)) == 400.0

    def test_negative_falls_back(self) -> None:
        assert compat._safe_page_dpi((-1.0, 72.0)) == 400.0
        assert compat._safe_page_dpi((72.0, -1.0)) == 400.0

    def test_non_finite_falls_back(self) -> None:
        assert compat._safe_page_dpi((float("nan"), 72.0)) == 400.0
        assert compat._safe_page_dpi((72.0, float("nan"))) == 400.0
        assert compat._safe_page_dpi((float("inf"), 72.0)) == 400.0
        assert compat._safe_page_dpi((72.0, float("inf"))) == 400.0

    def test_malformed_input_falls_back_without_raising(self) -> None:
        # A scalar is not indexable with two positions; must fall back, not crash.
        assert compat._safe_page_dpi(0) == 400.0
        assert compat._safe_page_dpi(None) == 400.0
        # Only one axis present.
        assert compat._safe_page_dpi((72.0,)) == 400.0

    def test_usable_dpi_is_untouched(self) -> None:
        assert compat._safe_page_dpi((300.0, 150.0)) is None
        assert compat._safe_page_dpi((72.0, 72.0)) is None
        # Integer dpi (as Resolution may hold).
        assert compat._safe_page_dpi((400, 400)) is None

    def test_works_on_indexable_namedtuple_like(self) -> None:
        # Resolution is an indexable object; mimic it with a tiny sequence.
        class FakeResolution:
            def __init__(self, x: float, y: float) -> None:
                self._vals = (x, y)

            def __getitem__(self, idx: int) -> float:
                return self._vals[idx]

        assert compat._safe_page_dpi(FakeResolution(0.0, 0.0)) == 400.0
        assert compat._safe_page_dpi(FakeResolution(300.0, 300.0)) is None


# --- Shape problem tests (dev-runnable, no ocrmypdf needed) ---


class _IndexableResolution:
    """Minimal stand-in for ocrmypdf's Resolution (indexable, float-convertible)."""

    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __getitem__(self, idx):
        return (self._x, self._y)[idx]


def _fake_graft(graft_page):
    """Return a namespace mimicking the ocrmypdf._graft module structure."""
    return SimpleNamespace(OcrGrafter=SimpleNamespace(graft_page=graft_page))


class TestShapeProblem:
    """Tests for _shape_problem (pure function, no ocrmypdf dependency)."""

    def test_shape_ok_returns_none(self):
        """A keyword-only pageno + indexable Resolution → no problem."""

        def graft_page(
            self,
            *,
            pageno,
            image=None,
            ocr_output=None,
            ocr_tree=None,
            autorotate_correction=0,
        ):
            pass

        result = compat._shape_problem(_fake_graft(graft_page), _IndexableResolution)
        assert result is None

    def test_non_indexable_resolution_returns_problem(self):
        """A Resolution without __getitem__ → problem string."""

        class BadResolution:
            def __init__(self, x, y):
                pass  # no __getitem__

        def graft_page(
            self,
            *,
            pageno,
            image=None,
            ocr_output=None,
            ocr_tree=None,
            autorotate_correction=0,
        ):
            pass

        result = compat._shape_problem(_fake_graft(graft_page), BadResolution)
        assert result is not None
        assert result.startswith("Resolution not indexable")

    def test_positional_pageno_returns_problem(self):
        """A POSITIONAL_OR_KEYWORD pageno → problem string mentioning keyword-only."""

        def graft_page(
            self,
            pageno,
            image=None,
            ocr_output=None,
            ocr_tree=None,
            autorotate_correction=0,
        ):
            pass

        result = compat._shape_problem(_fake_graft(graft_page), _IndexableResolution)
        assert result is not None
        assert "keyword-only" in result

    def test_missing_pageno_returns_problem(self):
        """No pageno parameter at all → problem string mentioning 'got None'."""

        def graft_page(
            self, *, image=None, ocr_output=None, ocr_tree=None, autorotate_correction=0
        ):
            pass

        result = compat._shape_problem(_fake_graft(graft_page), _IndexableResolution)
        assert result is not None
        assert "got None" in result


class TestApplyFailSoft:
    """Fail-soft behavior of apply_zero_dpi_graft_workaround."""

    def test_apply_without_ocrmypdf_logs_warning_and_returns(self, monkeypatch, caplog):
        """When ocrmypdf is not importable, apply() logs WARNING and returns cleanly."""
        import logging
        import sys

        # sys.modules["ocrmypdf"] = None makes `import ocrmypdf` raise ImportError
        monkeypatch.setitem(sys.modules, "ocrmypdf", None)

        with caplog.at_level(logging.WARNING, logger=compat.__name__):
            compat.apply_zero_dpi_graft_workaround()  # must not raise

        assert any(
            "ocrmypdf not importable" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )
