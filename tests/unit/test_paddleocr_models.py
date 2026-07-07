from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

import paddleocr_helpers as helpers


REQUIRED_MODELS = (
    helpers.TEXT_DETECTION_MODEL,
    helpers.TEXT_RECOGNITION_MODEL,
    helpers.TEXTLINE_ORIENTATION_MODEL,
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
    """Create a minimal PaddleOCR/PaddleX inference model directory."""
    model_dir = root / (
        model_dir_name or helpers.PADDLEOCR_MODEL_DIRS[logical_model_name]
    )
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


def test_constants_use_logical_model_names_not_infer_directory_names(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "PADDLEOCR_MODELS", str(tmp_path))

    assert helpers.TEXT_DETECTION_MODEL == "PP-OCRv6_medium_det"
    assert helpers.TEXT_RECOGNITION_MODEL == "PP-OCRv6_medium_rec"
    assert helpers.TEXTLINE_ORIENTATION_MODEL == "PP-LCNet_x1_0_textline_ori"
    assert all(not model_name.endswith("_infer") for model_name in REQUIRED_MODELS)

    assert helpers._model_dir(helpers.TEXT_DETECTION_MODEL) == (  # noqa: SLF001
        tmp_path / "PP-OCRv6_medium_det_infer"
    )
    assert helpers._model_dir(helpers.TEXT_RECOGNITION_MODEL) == (  # noqa: SLF001
        tmp_path / "PP-OCRv6_medium_rec_infer"
    )
    assert helpers._model_dir(helpers.TEXTLINE_ORIENTATION_MODEL) == (  # noqa: SLF001
        tmp_path / "PP-LCNet_x1_0_textline_ori_infer"
    )


def test_validate_accepts_official_infer_dirs_with_logical_model_names(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_all_models(tmp_path)

    helpers.validate_paddleocr_models()


def test_validate_accepts_variable_indentation_in_inference_yml(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_all_models(tmp_path, indent="\t")

    helpers.validate_paddleocr_models()


def test_validate_rejects_stale_patch_that_uses_infer_dir_as_model_name(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_all_models(tmp_path)

    det_dir_name = helpers.PADDLEOCR_MODEL_DIRS[helpers.TEXT_DETECTION_MODEL]
    (tmp_path / det_dir_name / "inference.yml").write_text(
        f"Global:\n  model_name: {det_dir_name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PP-OCRv6_medium_det_infer"):
        helpers.validate_paddleocr_models()


def test_validate_rejects_mixed_ppocr_generation_model_names(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_all_models(tmp_path)

    det_dir_name = helpers.PADDLEOCR_MODEL_DIRS[helpers.TEXT_DETECTION_MODEL]
    (tmp_path / det_dir_name / "inference.yml").write_text(
        "Global:\n  model_name: PP-OCRv5_server_det\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PP-OCRv5_server_det"):
        helpers.validate_paddleocr_models()


def test_validate_reports_missing_required_model_files(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_all_models(tmp_path)

    rec_dir_name = helpers.PADDLEOCR_MODEL_DIRS[helpers.TEXT_RECOGNITION_MODEL]
    (tmp_path / rec_dir_name / "inference.json").unlink()

    with pytest.raises(FileNotFoundError, match="PP-OCRv6_medium_rec"):
        helpers.validate_paddleocr_models()


def test_validate_renames_legacy_orientation_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_model(tmp_path, helpers.TEXT_DETECTION_MODEL)
    _write_model(tmp_path, helpers.TEXT_RECOGNITION_MODEL)
    legacy_dir = _write_model(
        tmp_path,
        helpers.TEXTLINE_ORIENTATION_MODEL,
        model_dir_name="PP-OCRv6_lcnet_x1_0_textline_ori_infer",
    )

    helpers.validate_paddleocr_models()

    expected_dir = tmp_path / helpers.PADDLEOCR_MODEL_DIRS[
        helpers.TEXTLINE_ORIENTATION_MODEL
    ]
    assert expected_dir.is_dir()
    assert not legacy_dir.exists()


def test_create_paddleocr_model_pins_logical_names_and_local_dirs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(helpers, "PADDLEOCR_MODELS", str(tmp_path))
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
    _write_all_models(tmp_path)

    calls: list[dict[str, Any]] = []

    class FakePaddleOCR:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    fake_logger = types.SimpleNamespace(setLevel=lambda _level: None)
    paddleocr_module = types.ModuleType("paddleocr")
    paddleocr_module.__path__ = []  # type: ignore[attr-defined]
    paddleocr_module.PaddleOCR = FakePaddleOCR  # type: ignore[attr-defined]
    utils_module = types.ModuleType("paddleocr._utils")
    utils_module.__path__ = []  # type: ignore[attr-defined]
    logging_module = types.ModuleType("paddleocr._utils.logging")
    logging_module.logger = fake_logger  # type: ignore[attr-defined]
    paddleocr_module._utils = utils_module  # type: ignore[attr-defined]
    utils_module.logging = logging_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr_module)
    monkeypatch.setitem(sys.modules, "paddleocr._utils", utils_module)
    monkeypatch.setitem(sys.modules, "paddleocr._utils.logging", logging_module)

    model = helpers.create_paddleocr_model(use_textline_orientation=True)

    assert isinstance(model, FakePaddleOCR)
    assert len(calls) == 1
    kwargs = calls[0]
    assert "lang" not in kwargs
    assert kwargs["use_doc_orientation_classify"] is False
    assert kwargs["use_doc_unwarping"] is False
    assert kwargs["use_textline_orientation"] is True
    assert kwargs["text_detection_model_name"] == "PP-OCRv6_medium_det"
    assert kwargs["text_recognition_model_name"] == "PP-OCRv6_medium_rec"
    assert kwargs["textline_orientation_model_name"] == "PP-LCNet_x1_0_textline_ori"
    assert kwargs["text_detection_model_dir"].endswith("PP-OCRv6_medium_det_infer")
    assert kwargs["text_recognition_model_dir"].endswith("PP-OCRv6_medium_rec_infer")
    assert kwargs["textline_orientation_model_dir"].endswith(
        "PP-LCNet_x1_0_textline_ori_infer"
    )


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


def test_run_paddleocr_converts_paddleocr3_predict_result_to_sidecar_pages(
    monkeypatch,
):
    monkeypatch.setattr(
        helpers,
        "_get_paddleocr_model",
        lambda: _FakePredictModel(
            [
                {
                    "rec_texts": ["Hello", "", "World"],
                    "rec_scores": [0.98765, 0.5, 0.87654],
                    "rec_boxes": [_ArrayLike([1, 2, 3, 4]), [5, 6, 7, 8], [9, 10, 11, 12]],
                }
            ]
        ),
    )

    pages = helpers.run_paddleocr("document.pdf")

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
        helpers,
        "_get_paddleocr_model",
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

    pages = helpers.run_paddleocr("document.pdf")

    assert pages[0]["blocks"] == [
        {
            "text": "Only polygon",
            "bbox": [[1, 1], [2, 1], [2, 2], [1, 2]],
            "confidence": 0.91,
        }
    ]
