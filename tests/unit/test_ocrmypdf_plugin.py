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
