"""
Doc-Worker — Tests for the Docling-API sidecar path (worker.call_docling_convert).

Closes the gap from the themed review: the sidecar contract "sidecar
output remains keyed by the original filename" was only asserted for the
native backend — every process_file test patches worker.call_docling_convert
to a stub, so a regression of the API path (e.g. the sidecar getting keyed
by the AcroForm-flattened intermediate's name) would have gone unnoticed.

The real `requests` package is not installed locally (it is in CI via
requirements.txt). The worker imports it lazily inside
call_docling_convert(), so these tests install a stub module into
sys.modules and control its `post` mock.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import worker


class _RequestException(Exception):
    """Stub of requests.RequestException carried by the fake module."""


class _FakeResponse:
    """Minimal stand-in for the requests.Response parts the worker uses."""

    def __init__(self, payload: object) -> None:
        self._payload: object = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> object:
        return self._payload


@pytest.fixture()
def fake_requests(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a stub `requests` module so the lazy import finds it.

    Exposes `post` (a controllable MagicMock) and `RequestException`
    (a real Exception subclass); monkeypatch.setitem removes the stub
    again after each test.
    """
    fake = types.ModuleType("requests")
    fake.post = MagicMock(name="requests.post")  # type: ignore[attr-defined]
    fake.RequestException = _RequestException  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


class TestCallDoclingConvert:
    """worker.call_docling_convert() against a stubbed Docling API.

    The fake response returns a single conversion result and the stubbed
    `requests.post` is the only network touchpoint, so the tests stay
    hermetic and assert the sidecar naming contract directly.
    """

    def _point_worker_at(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Point DOCLING_BASE_URL and DOCLING_OUT at test-local values.

        Returns the monkeypatched DOCLING_OUT for later assertions.
        """
        monkeypatch.setattr(worker, "DOCLING_BASE_URL", "http://docling-test:12000")
        docling_out = tmp_path / "docling_out"
        monkeypatch.setattr(worker, "DOCLING_OUT", docling_out)
        return docling_out

    def test_sidecar_keyed_by_original_filename(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_requests: types.ModuleType,
    ) -> None:
        """The review gap: the sidecar must stay keyed by the original filename.

        The worker uploads the AcroForm-flattened intermediate but passes
        sidecar_name=<original filename>; both the upload name and the
        DOCLING_OUT key must use the original, so the intermediate name
        never leaks into the sidecar output.
        """
        docling_out = self._point_worker_at(tmp_path, monkeypatch)

        # the dummy plays the role of the flattened intermediate the worker
        # would actually upload.
        intermediate = tmp_path / "form_flattened.pdf"
        intermediate.write_bytes(b"%PDF-1.4 fake")

        fake_requests.post.return_value = _FakeResponse([{"md": "# doc", "json": "{}"}])

        result = worker.call_docling_convert(intermediate, sidecar_name="form.pdf")
        assert result is True

        fake_requests.post.assert_called_once()
        call = fake_requests.post.call_args
        # the url must be the (monkeypatched) base url plus the convert path.
        assert call.args == ("http://docling-test:12000/v1alpha/convert",)
        # the uploaded filename must be the original, not the intermediate's.
        assert call.kwargs["files"]["files"][0] == "form.pdf"
        assert call.kwargs["files"]["files"][2] == "application/pdf"
        assert call.kwargs["data"] == {"output_formats": ["md", "json"]}
        assert call.kwargs["timeout"] == 900

        md_path = docling_out / "form" / "form.md"
        json_path = docling_out / "form" / "form.json"
        assert md_path.exists()
        assert json_path.exists()
        assert md_path.read_text(encoding="utf-8") == "# doc"
        assert json_path.read_text(encoding="utf-8") == "{}"
        # the intermediate name must not appear anywhere under DOCLING_OUT.
        written = [p.name for p in docling_out.rglob("*")]
        assert not any("form_flattened" in name for name in written)

    def test_keyed_by_path_name_without_sidecar_name(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_requests: types.ModuleType,
    ) -> None:
        """Without sidecar_name, upload and output key fall back to pdf_path.name.

        A plain file (no intermediate) must produce sidecars under
        DOCLING_OUT/<its stem>/ named after itself.
        """
        docling_out = self._point_worker_at(tmp_path, monkeypatch)

        report = tmp_path / "report.pdf"
        report.write_bytes(b"%PDF-1.4 fake")

        fake_requests.post.return_value = _FakeResponse(
            [{"md": "# report", "json": "{}"}]
        )

        assert worker.call_docling_convert(report) is True

        fake_requests.post.assert_called_once()
        call = fake_requests.post.call_args
        # the uploaded filename must be the plain path's own name.
        assert call.kwargs["files"]["files"][0] == "report.pdf"

        md_path = docling_out / "report" / "report.md"
        json_path = docling_out / "report" / "report.json"
        assert md_path.exists()
        assert json_path.exists()
        assert md_path.read_text(encoding="utf-8") == "# report"
        assert json_path.read_text(encoding="utf-8") == "{}"

    def test_request_failure_returns_false(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_requests: types.ModuleType,
    ) -> None:
        """A requests.RequestException is caught, logged, and reported as failure.

        No sidecar files may be written when the request itself fails.
        """
        docling_out = self._point_worker_at(tmp_path, monkeypatch)

        report = tmp_path / "report.pdf"
        report.write_bytes(b"%PDF-1.4 fake")

        fake_requests.post.side_effect = _RequestException("connection refused")

        assert worker.call_docling_convert(report) is False
        # nothing may be written under DOCLING_OUT on a request failure.
        assert not docling_out.exists()
