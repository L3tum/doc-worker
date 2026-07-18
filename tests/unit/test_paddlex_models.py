from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

# Import from paddlex_helpers directly (paddleocr_helpers is a shim)
from paddlex_helpers import (
    create_paddleocr_model,
    run_paddleocr,
    validate_paddlex_models,
    TEXT_DETECTION_MODEL,
    TEXT_RECOGNITION_MODEL,
    TEXTLINE_ORIENTATION_MODEL,
    LAYOUT_DETECTION_MODEL,
    PADDLEX_MODEL_DIRS,
    _model_dir,
)

REQUIRED_MODELS = (
    TEXT_DETECTION_MODEL,
    TEXT_RECOGNITION_MODEL,
    TEXTLINE_ORIENTATION_MODEL,
    LAYOUT_DETECTION_MODEL,
)


def _write_model(
    root: Path,
    logical_model_name: str,
    *,
    yml_model_name: str | None = None,
    model_dir_name: str | None = None,
    indent: str = "  ",
    include_json: bool = True,
) -> Path:
    """Create a minimal PaddleX inference model directory."""
    from paddlex_helpers import PADDLEX_MODEL_DIRS

    model_dir = root / (model_dir_name or PADDLEX_MODEL_DIRS[logical_model_name])
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "inference.pdiparams").write_text("params", encoding="utf-8")
    if include_json:
        (model_dir / "inference.json").write_text("{}", encoding="utf-8")
    (model_dir / "inference.yml").write_text(
        "Global:\n"
        f"{indent}model_name: {yml_model_name or logical_model_name}\n"
        "PreProcess: {}\n",
        encoding="utf-8",
    )
    return model_dir


def _write_all_models(root: Path, **kwargs: Any) -> None:
    for logical_model_name in REQUIRED_MODELS:
        _write_model(root, logical_model_name, **kwargs)


def test_constants_use_logical_model_names_not_infer_directory_names(
    tmp_path, monkeypatch
):
    import paddlex_helpers

    monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))

    assert TEXT_DETECTION_MODEL == "PP-OCRv6_medium_det"
    assert TEXT_RECOGNITION_MODEL == "PP-OCRv6_medium_rec"
    assert TEXTLINE_ORIENTATION_MODEL == "PP-LCNet_x1_0_textline_ori"
    assert LAYOUT_DETECTION_MODEL == "PP-DocLayout-L"
    assert all(not model_name.endswith("_infer") for model_name in REQUIRED_MODELS)

    assert _model_dir(TEXT_DETECTION_MODEL) == (  # noqa: SLF001
        tmp_path / "PP-OCRv6_medium_det_infer"
    )
    assert _model_dir(TEXT_RECOGNITION_MODEL) == (  # noqa: SLF001
        tmp_path / "PP-OCRv6_medium_rec_infer"
    )
    assert _model_dir(TEXTLINE_ORIENTATION_MODEL) == (  # noqa: SLF001
        tmp_path / "PP-LCNet_x1_0_textline_ori_infer"
    )
    assert _model_dir(LAYOUT_DETECTION_MODEL) == (  # noqa: SLF001
        tmp_path / "PP-DocLayout-L_infer"
    )


def test_validate_accepts_official_infer_dirs_with_logical_model_names(
    tmp_path, monkeypatch
):
    import paddlex_helpers

    monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_all_models(tmp_path)

    validate_paddlex_models()


def test_validate_accepts_variable_indentation_in_inference_yml(tmp_path, monkeypatch):
    import paddlex_helpers

    monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_all_models(tmp_path, indent="\t")

    validate_paddlex_models()


def test_validate_rejects_stale_patch_that_uses_infer_dir_as_model_name(
    tmp_path, monkeypatch
):
    import paddlex_helpers

    monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_all_models(tmp_path)

    det_dir_name = PADDLEX_MODEL_DIRS[TEXT_DETECTION_MODEL]
    (tmp_path / det_dir_name / "inference.yml").write_text(
        f"Global:\n  model_name: {det_dir_name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PP-OCRv6_medium_det_infer"):
        validate_paddlex_models()


def test_validate_rejects_mixed_ppocr_generation_model_names(tmp_path, monkeypatch):
    import paddlex_helpers

    monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_all_models(tmp_path)

    det_dir_name = PADDLEX_MODEL_DIRS[TEXT_DETECTION_MODEL]
    (tmp_path / det_dir_name / "inference.yml").write_text(
        "Global:\n  model_name: PP-OCRv5_server_det\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PP-OCRv5_server_det"):
        validate_paddlex_models()


def test_validate_reports_missing_required_model_files(tmp_path, monkeypatch):
    import paddlex_helpers

    monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_all_models(tmp_path)

    rec_dir_name = PADDLEX_MODEL_DIRS[TEXT_RECOGNITION_MODEL]
    (tmp_path / rec_dir_name / "inference.json").unlink()

    with pytest.raises(FileNotFoundError, match="PP-OCRv6_medium_rec"):
        validate_paddlex_models()


def test_validate_renames_legacy_orientation_directory(tmp_path, monkeypatch):
    """Test that validation renames the legacy orientation directory (lowercase lcnet)."""
    import paddlex_helpers

    monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_model(tmp_path, TEXT_DETECTION_MODEL)
    _write_model(tmp_path, TEXT_RECOGNITION_MODEL)
    # Create the legacy directory name (lowercase lcnet)
    legacy_dir = tmp_path / "PP-OCRv6_lcnet_x1_0_textline_ori_infer"
    legacy_dir.mkdir()
    (legacy_dir / "inference.pdiparams").write_text("params")
    (legacy_dir / "inference.yml").write_text(
        "Global:\n  model_name: PP-LCNet_x1_0_textline_ori\n"
    )
    (legacy_dir / "inference.json").write_text("{}")
    _write_model(tmp_path, LAYOUT_DETECTION_MODEL)

    # Before validation: legacy dir exists, expected dir doesn't
    expected_dir = tmp_path / "PP-LCNet_x1_0_textline_ori_infer"
    assert legacy_dir.exists()
    assert not expected_dir.exists()

    validate_paddlex_models()

    # After validation, the rename should have happened
    assert expected_dir.exists()
    assert not legacy_dir.exists()
    # After rename, expected_dir should exist (even though we mocked, the actual
    # dir exists because rename was called)
    # Note: we can't assert expected_dir.exists() since rename was mocked,
    # but the rename call proves the logic works.


def test_create_paddleocr_model_pins_logical_names_and_local_dirs(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
    _write_all_models(tmp_path)

    calls: list[dict[str, Any]] = []

    def fake_create_pipeline(name: str, **kwargs: Any) -> dict:
        calls.append(kwargs)
        return {}  # dummy pipeline

    fake_logger = types.SimpleNamespace(setLevel=lambda _level: None)
    # Mock paddlex module
    pdx_module = types.ModuleType("paddlex")
    pdx_module.__path__ = []  # type: ignore[attr-defined]
    pdx_module.create_pipeline = fake_create_pipeline
    pdx_module._utils = types.ModuleType("paddlex._utils")
    pdx_module._utils.__path__ = []  # type: ignore[attr-defined]
    pdx_module._utils.logging = types.ModuleType("paddlex._utils.logging")
    pdx_module._utils.logging.logger = fake_logger
    monkeypatch.setitem(sys.modules, "paddlex", pdx_module)
    monkeypatch.setitem(sys.modules, "paddlex._utils", pdx_module._utils)
    monkeypatch.setitem(
        sys.modules, "paddlex._utils.logging", pdx_module._utils.logging
    )

    create_paddleocr_model(use_textline_orientation=True)

    assert len(calls) == 1
    kwargs = calls[0]
    # The pipeline name is "ocr"
    assert "text_detection_model_name" in kwargs
    assert kwargs["text_detection_model_name"] == "PP-OCRv6_medium_det"
    assert kwargs["text_recognition_model_name"] == "PP-OCRv6_medium_rec"
    assert kwargs["textline_orientation_model_name"] == "PP-LCNet_x1_0_textline_ori"
    assert kwargs["text_detection_model_dir"].endswith("PP-OCRv6_medium_det_infer")
    assert kwargs["text_recognition_model_dir"].endswith("PP-OCRv6_medium_rec_infer")
    assert kwargs["textline_orientation_model_dir"].endswith(
        "PP-LCNet_x1_0_textline_ori_infer"
    )
    assert kwargs["use_textline_orientation"] is True
    # Language should be present
    assert "lang" in kwargs
    assert kwargs["lang"] == "german"  # default OCR_LANG=deu


class _ArrayLike:
    def __init__(self, value: list[int]) -> None:
        self.value = value

    def tolist(self) -> list[int]:
        return self.value


class _FakePredictModel:
    def __init__(self, result: list[dict[str, Any]]) -> None:
        self.result = result

    def predict(self, file_path: str) -> list[dict[str, Any]]:
        assert file_path == "document.pdf"
        return self.result


def test_run_paddleocr_converts_paddlex_predict_result_to_sidecar_pages(
    monkeypatch,
):
    monkeypatch.setattr(
        "paddlex_helpers._get_paddlex_model",
        lambda: _FakePredictModel(
            [
                {
                    "rec_texts": ["Hello", "", "World"],
                    "rec_scores": [0.98765, 0.5, 0.87654],
                    "rec_boxes": [
                        _ArrayLike([1, 2, 3, 4]),
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                    ],
                }
            ]
        ),
    )

    pages = run_paddleocr("document.pdf")

    assert pages == [
        {
            "page": 1,
            "text": "Hello\nWorld",
            "blocks": [
                {"text": "Hello", "bbox": [1, 2, 3, 4], "confidence": 0.9877},
                {"text": "World", "bbox": [9, 10, 11, 12], "confidence": 0.8765},
            ],
        }
    ]


def test_run_paddleocr_falls_back_to_polygons_when_boxes_are_missing(monkeypatch):
    monkeypatch.setattr(
        "paddlex_helpers._get_paddlex_model",
        lambda: _FakePredictModel(
            [
                {
                    "rec_texts": ["Only polygon"],
                    "rec_scores": [0.91],
                    "rec_boxes": [],
                    "rec_polys": [_ArrayLike([[1, 1], [2, 1], [2, 2], [1, 2]])],
                }
            ]
        ),
    )

    pages = run_paddleocr("document.pdf")

    assert pages[0]["blocks"] == [
        {
            "text": "Only polygon",
            "bbox": [[1, 1], [2, 1], [2, 2], [1, 2]],
            "confidence": 0.91,
        }
    ]
