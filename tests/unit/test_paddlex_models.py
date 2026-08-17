from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

# Import from paddlex_helpers directly (paddleocr_helpers is a shim)
from paddlex_helpers import (
    DOC_ORIENTATION_MODEL,
    LAYOUT_DETECTION_MODEL,
    PADDLEX_MODEL_DIRS,
    TEXT_DETECTION_MODEL,
    TEXT_RECOGNITION_MODEL,
    TEXTLINE_ORIENTATION_MODEL,
    _model_dir,
    create_paddleocr_model,
    run_paddleocr,
    validate_paddlex_models,
)


# ── Fixture: reset singleton state between tests ──────────────────────────
@pytest.fixture(autouse=True)
def _reset_paddlex_singleton():
    """Clear cached PaddleX model state and submodules between tests.

    When the real PaddleX is installed (e.g. CI), importing paddlex_helpers
    at module level triggers _patch_paddlex_official_models() which caches
    references to real PaddleX submodules in sys.modules.  Subsequent tests
    that monkeypatch sys.modules["paddlex"] with a fake module can still
    reach the real submodules through Python's import fallback logic.

    This fixture removes all paddlex.* entries from sys.modules so that
    monkeypatches take full effect.
    """
    import paddlex_helpers

    # Clear any cached model/exception state
    for obj in (
        paddlex_helpers._get_paddlex_model,
        paddlex_helpers._get_paddlex_structure_v3_model,
    ):
        for attr in ("_model", "_init_exception"):
            if hasattr(obj, attr):
                delattr(obj, attr)

    # Reset the patch flag so patch tests can apply cleanly
    original_patched = paddlex_helpers._PADDLEX_PATCHED
    paddlex_helpers._PADDLEX_PATCHED = False

    # Remove cached PaddleX submodules so test monkeypatches take effect.
    # Keep the top-level "paddlex" key — tests may or may not replace it.
    _saved_paddlex_modules: dict[str, Any] = {}
    for _key in list(sys.modules):
        if _key == "paddlex":
            continue
        if _key.startswith("paddlex."):
            _saved_paddlex_modules[_key] = sys.modules[_key]
            del sys.modules[_key]

    yield

    # Restore saved submodules
    sys.modules.update(_saved_paddlex_modules)

    # Restore the patch flag after test (in case of leaks)
    paddlex_helpers._PADDLEX_PATCHED = original_patched

    # Also clear after test in case of leaks
    for obj in (
        paddlex_helpers._get_paddlex_model,
        paddlex_helpers._get_paddlex_structure_v3_model,
    ):
        for attr in ("_model", "_init_exception"):
            if hasattr(obj, attr):
                delattr(obj, attr)


REQUIRED_MODEL_NAMES = (
    TEXT_DETECTION_MODEL,
    TEXT_RECOGNITION_MODEL,
    TEXTLINE_ORIENTATION_MODEL,
    DOC_ORIENTATION_MODEL,
    LAYOUT_DETECTION_MODEL,
)


class _MockOfficialModels:
    """Shared mock for PaddleX's official_models object in tests."""

    def __getitem__(self, name: str) -> str:
        return f"original-{name}"

    def get(self, name: str, default=None):
        return f"original-{name}"

    def __contains__(self, name: object) -> bool:
        return False


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
    for logical_model_name in REQUIRED_MODEL_NAMES:
        _write_model(root, logical_model_name, **kwargs)


def _ocr_config_fixture() -> dict[str, Any]:
    """Mirror of PaddleX's default OCR.yaml (nested DocPreprocessor sub-pipeline)."""
    return {
        "pipeline_name": "OCR",
        "text_type": "general",
        "use_doc_preprocessor": True,
        "use_textline_orientation": True,
        "SubPipelines": {
            "DocPreprocessor": {
                "pipeline_name": "doc_preprocessor",
                "use_doc_orientation_classify": True,
                "use_doc_unwarping": True,
                "SubModules": {
                    "DocOrientationClassify": {
                        "module_name": "doc_text_orientation",
                        "model_name": DOC_ORIENTATION_MODEL,
                        "model_dir": None,
                    },
                    "DocUnwarping": {
                        "module_name": "image_unwarping",
                        "model_name": "UVDoc",
                        "model_dir": None,
                    },
                },
            }
        },
        "SubModules": {
            "TextDetection": {
                "module_name": "text_detection",
                "model_name": TEXT_DETECTION_MODEL,
                "model_dir": None,
            },
            "TextLineOrientation": {
                "module_name": "textline_orientation",
                "model_name": TEXTLINE_ORIENTATION_MODEL,
                "model_dir": None,
            },
            "TextRecognition": {
                "module_name": "text_recognition",
                "model_name": TEXT_RECOGNITION_MODEL,
                "model_dir": None,
            },
        },
    }


def _layout_parsing_config_fixture() -> dict[str, Any]:
    """Mirror of PaddleX's default layout_parsing.yaml (heavier, non-bundled defaults)."""
    return {
        "pipeline_name": "layout_parsing",
        "use_doc_preprocessor": True,
        "use_seal_recognition": True,
        "use_table_recognition": True,
        "use_formula_recognition": False,
        "SubModules": {
            "LayoutDetection": {
                "module_name": "layout_detection",
                "model_name": "RT-DETR-H_layout_17cls",
                "model_dir": None,
            },
        },
        "SubPipelines": {
            "DocPreprocessor": {
                "pipeline_name": "doc_preprocessor",
                "use_doc_orientation_classify": True,
                "use_doc_unwarping": True,
                "SubModules": {
                    "DocOrientationClassify": {
                        "module_name": "doc_text_orientation",
                        "model_name": DOC_ORIENTATION_MODEL,
                        "model_dir": None,
                    },
                    "DocUnwarping": {
                        "module_name": "image_unwarping",
                        "model_name": "UVDoc",
                        "model_dir": None,
                    },
                },
            },
            "GeneralOCR": {
                "pipeline_name": "OCR",
                "use_doc_preprocessor": False,
                "SubModules": {
                    "TextDetection": {
                        "module_name": "text_detection",
                        "model_name": "PP-OCRv4_server_det",
                        "model_dir": None,
                    },
                    "TextRecognition": {
                        "module_name": "text_recognition",
                        "model_name": "PP-OCRv4_server_rec",
                        "model_dir": None,
                    },
                },
            },
        },
    }


def _install_fake_paddlex(
    monkeypatch,
    config_by_name: dict[str, dict],
    create_pipeline: Any = None,
) -> dict:
    """Install a fake ``paddlex`` + ``paddlex.inference.pipelines`` into sys.modules.

    The fake ``load_pipeline_config`` returns the config fixture for the requested
    pipeline name (and records the call). ``create_pipeline`` defaults to a capture
    stub; pass a custom callable to control counting / flaky / permanent behaviour
    (it will be invoked as ``create_pipeline(config=cfg)``). Returns a captured dict
    with keys ``pipeline`` / ``config`` (last call) and ``load_calls`` (a list).
    """
    captured: dict[str, Any] = {"load_calls": []}

    def _default_create_pipeline(pipeline=None, *, config=None, **kwargs: Any) -> dict:
        captured["pipeline"] = pipeline
        captured["config"] = config
        return {}  # dummy pipeline

    def fake_load_pipeline_config(pipeline: str) -> dict:
        captured["load_calls"].append(pipeline)
        return config_by_name[pipeline]

    pdx_module = types.ModuleType("paddlex")
    pdx_module.__path__ = []  # type: ignore[attr-defined]
    pdx_module.create_pipeline = (
        create_pipeline if create_pipeline is not None else _default_create_pipeline
    )

    inference_mod = types.ModuleType("paddlex.inference")
    inference_mod.__path__ = []  # type: ignore[attr-defined]

    pipelines_mod = types.ModuleType("paddlex.inference.pipelines")
    pipelines_mod.load_pipeline_config = fake_load_pipeline_config

    inference_mod.pipelines = pipelines_mod  # type: ignore[attr-defined]
    pdx_module.inference = inference_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "paddlex", pdx_module)
    monkeypatch.setitem(sys.modules, "paddlex.inference", inference_mod)
    monkeypatch.setitem(sys.modules, "paddlex.inference.pipelines", pipelines_mod)

    return captured


def test_constants_use_logical_model_names_not_infer_directory_names(
    tmp_path, monkeypatch
):
    import paddlex_helpers

    monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))

    assert TEXT_DETECTION_MODEL == "PP-OCRv6_medium_det"
    assert TEXT_RECOGNITION_MODEL == "PP-OCRv6_medium_rec"
    assert TEXTLINE_ORIENTATION_MODEL == "PP-LCNet_x1_0_textline_ori"
    assert paddlex_helpers.DOC_ORIENTATION_MODEL == "PP-LCNet_x1_0_doc_ori"
    assert LAYOUT_DETECTION_MODEL == "PP-DocLayout-L"
    assert all(not model_name.endswith("_infer") for model_name in REQUIRED_MODEL_NAMES)

    assert _model_dir(TEXT_DETECTION_MODEL) == (tmp_path / "PP-OCRv6_medium_det_infer")
    assert _model_dir(TEXT_RECOGNITION_MODEL) == (
        tmp_path / "PP-OCRv6_medium_rec_infer"
    )
    assert _model_dir(TEXTLINE_ORIENTATION_MODEL) == (
        tmp_path / "PP-LCNet_x1_0_textline_ori_infer"
    )
    assert _model_dir(paddlex_helpers.DOC_ORIENTATION_MODEL) == (
        tmp_path / "PP-LCNet_x1_0_doc_ori_infer"
    )
    assert _model_dir(LAYOUT_DETECTION_MODEL) == (tmp_path / "PP-DocLayout-L_infer")


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


def test_migrate_renames_legacy_orientation_directory(tmp_path, monkeypatch):
    """Test that migrate_legacy_model_dirs() renames the legacy orientation directory."""
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

    # Before migration: legacy dir exists, expected dir doesn't
    expected_dir = tmp_path / "PP-LCNet_x1_0_textline_ori_infer"
    assert legacy_dir.exists()
    assert not expected_dir.exists()

    paddlex_helpers.migrate_legacy_model_dirs()

    # After migration
    assert expected_dir.exists()
    assert not legacy_dir.exists()


def test_validate_does_not_rename_legacy_directory(tmp_path, monkeypatch):
    """validate_paddlex_models() should be read-only — no os.rename()."""
    import paddlex_helpers

    monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
    _write_model(tmp_path, TEXT_DETECTION_MODEL)
    _write_model(tmp_path, TEXT_RECOGNITION_MODEL)
    # Create the legacy directory
    legacy_dir = tmp_path / "PP-OCRv6_lcnet_x1_0_textline_ori_infer"
    legacy_dir.mkdir()
    _write_model(tmp_path, LAYOUT_DETECTION_MODEL)

    # validate should fail (missing the expected dir) and NOT rename
    with pytest.raises(FileNotFoundError):
        paddlex_helpers.validate_paddlex_models()

    # Legacy dir should still exist (no rename happened)
    assert legacy_dir.exists()


def test_create_paddleocr_model_pins_logical_names_and_local_dirs(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
    _write_all_models(tmp_path)

    captured = _install_fake_paddlex(monkeypatch, {"OCR": _ocr_config_fixture()})

    create_paddleocr_model(use_textline_orientation=True)

    # create_pipeline is now called with the built config, not dropped kwargs.
    assert captured["pipeline"] is None
    assert "config" in captured
    cfg = captured["config"]
    assert cfg["pipeline_name"] == "OCR"

    # Top-level model dirs point at the bundled local weights.
    assert cfg["SubModules"]["TextDetection"]["model_dir"].endswith(
        "PP-OCRv6_medium_det_infer"
    )
    assert cfg["SubModules"]["TextRecognition"]["model_dir"].endswith(
        "PP-OCRv6_medium_rec_infer"
    )
    assert cfg["SubModules"]["TextLineOrientation"]["model_dir"].endswith(
        "PP-LCNet_x1_0_textline_ori_infer"
    )

    # The nested doc-preprocessor orientation model dir points at bundled doc_ori.
    doc_pre = cfg["SubPipelines"]["DocPreprocessor"]
    assert doc_pre["SubModules"]["DocOrientationClassify"]["model_dir"].endswith(
        "PP-LCNet_x1_0_doc_ori_infer"
    )

    # The (un-bundled) doc-unwarping model is disabled so it is never downloaded.
    assert doc_pre["use_doc_unwarping"] is False

    assert cfg["use_textline_orientation"] is True
    assert cfg["lang"] == "german"  # legacy no-op key, kept for parity


def test_structure_v3_uses_bundled_models_and_disables_unbundled(tmp_path, monkeypatch):
    """layout_parsing must resolve to the bundled set and never need UVDoc/tables/seal."""
    import paddlex_helpers

    monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
    _write_all_models(tmp_path)

    captured = _install_fake_paddlex(
        monkeypatch, {"layout_parsing": _layout_parsing_config_fixture()}
    )

    paddlex_helpers._create_structure_v3_pipeline()

    assert captured["pipeline"] is None
    cfg = captured["config"]
    assert cfg["pipeline_name"] == "layout_parsing"

    # Layout detection is overridden to the bundled PP-DocLayout-L (default RT-DETR-H).
    layout = cfg["SubModules"]["LayoutDetection"]
    assert layout["model_name"] == "PP-DocLayout-L"
    assert layout["model_dir"].endswith("PP-DocLayout-L_infer")

    # GeneralOCR sub-pipeline is overridden to the bundled PP-OCRv6_medium_* set.
    general_ocr = cfg["SubPipelines"]["GeneralOCR"]["SubModules"]
    assert general_ocr["TextDetection"]["model_name"] == "PP-OCRv6_medium_det"
    assert general_ocr["TextDetection"]["model_dir"].endswith(
        "PP-OCRv6_medium_det_infer"
    )
    assert general_ocr["TextRecognition"]["model_dir"].endswith(
        "PP-OCRv6_medium_rec_infer"
    )

    # Nested doc-preprocessor orientation model is local and unwarping is disabled.
    doc_pre = cfg["SubPipelines"]["DocPreprocessor"]
    assert doc_pre["SubModules"]["DocOrientationClassify"]["model_dir"].endswith(
        "PP-LCNet_x1_0_doc_ori_infer"
    )
    assert doc_pre["use_doc_unwarping"] is False

    # Optional sub-pipelines whose models are not bundled are disabled.
    assert cfg["use_table_recognition"] is False
    assert cfg["use_seal_recognition"] is False
    assert cfg["use_formula_recognition"] is False


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


# ── Model lifecycle: destroy + recreate ──────────────────────────────────


def test_destroy_and_recreate_paddlex_model(tmp_path, monkeypatch):
    """Verify that destroy_paddlex_model() + _get_paddlex_model() recreates the pipeline."""
    import paddlex_helpers

    monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
    _write_all_models(tmp_path)

    create_count = 0

    def counting_create_pipeline(*args: Any, **kwargs: Any) -> dict:
        nonlocal create_count
        create_count += 1
        return {}

    _install_fake_paddlex(
        monkeypatch,
        {
            "OCR": _ocr_config_fixture(),
            "layout_parsing": _layout_parsing_config_fixture(),
        },
        create_pipeline=counting_create_pipeline,
    )

    # 1st creation
    paddlex_helpers._get_paddlex_model()
    assert create_count == 1

    # Destroy
    paddlex_helpers.destroy_paddlex_model()

    # Re-creation
    paddlex_helpers._get_paddlex_model()
    assert create_count == 2  # pipeline was recreated, not reused


# ── Retry logic: transient error recovery ────────────────────────────────


def test_retry_on_transient_error(tmp_path, monkeypatch):
    """Transient errors (not 'already been initialized') trigger retries."""
    import paddlex_helpers

    monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
    _write_all_models(tmp_path)

    attempts = 0

    def flaky_create_pipeline(*args: Any, **kwargs: Any) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("network timeout")  # transient
        return {}  # succeeds on 2nd attempt

    _install_fake_paddlex(
        monkeypatch,
        {"OCR": _ocr_config_fixture()},
        create_pipeline=flaky_create_pipeline,
    )

    # Should succeed after retry
    model = paddlex_helpers._get_paddlex_model()
    assert attempts == 2  # exactly 2 attempts (1 transient + 1 success)
    assert model == {}


# ── Retry logic: permanent error caching ─────────────────────────────────


def test_cache_permanent_error(tmp_path, monkeypatch):
    """Permanent errors ('already been initialized') are cached and don't retry."""
    import paddlex_helpers

    monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
    _write_all_models(tmp_path)

    attempts = 0
    original_error = RuntimeError("PDX has already been initialized")

    def permanent_failure_create_pipeline(*args: Any, **kwargs: Any) -> dict:
        nonlocal attempts
        attempts += 1
        raise original_error

    _install_fake_paddlex(
        monkeypatch,
        {"OCR": _ocr_config_fixture()},
        create_pipeline=permanent_failure_create_pipeline,
    )

    # First call: should raise
    with pytest.raises(RuntimeError, match="already been initialized"):
        paddlex_helpers._get_paddlex_model()
    assert attempts == 1  # only 1 attempt, no retries for permanent errors

    # Second call: should re-raise the cached exception (not retry)
    with pytest.raises(RuntimeError, match="already been initialized"):
        paddlex_helpers._get_paddlex_model()
    assert attempts == 1  # still 1 — the cached exception was re-raised, no new attempt


# ── Pipeline name correctness regression test ────────────────────────────


def test_ocr_pipeline_uses_uppercase_ocr_name(tmp_path, monkeypatch):
    """Regression: ensure 'OCR' (uppercase) is used, not 'ocr' (lowercase).

    PaddleX 3.x pipeline registry is case-sensitive. The name 'ocr' (lowercase)
    was accepted by test mocks but rejected by the real PaddleX API.
    """
    import paddlex_helpers

    monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
    _write_all_models(tmp_path)

    captured = _install_fake_paddlex(monkeypatch, {"OCR": _ocr_config_fixture()})

    # Clear any cached model from previous tests
    if hasattr(paddlex_helpers._get_paddlex_model, "_model"):
        del paddlex_helpers._get_paddlex_model._model

    paddlex_helpers._get_paddlex_model()
    # The pipeline name is passed to load_pipeline_config (case-sensitive in PaddleX).
    # This assertion would fail if the code used lowercase "ocr".
    assert captured["load_calls"] == ["OCR"], (
        f"Pipeline name must be 'OCR' (uppercase), got {captured['load_calls']!r}"
    )


# ── Monkey-patch for air-gapped operation ────────────────────────────────


class TestPaddlexOfficialModelsPatch:
    """Tests for _patch_paddlex_official_models() — air-gapped compatibility."""

    def test_patch_returns_local_dir(self, tmp_path, monkeypatch):
        """Patched official_models resolves known model names to local dirs."""
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)

        mock_obj = _MockOfficialModels()

        # Set up BOTH the parent module AND the submodule so the import works
        mock_utils = types.ModuleType("paddlex.inference.utils")
        mock_om_module = types.ModuleType("paddlex.inference.utils.official_models")
        mock_om_module.official_models = mock_obj
        mock_utils.official_models = mock_om_module

        monkeypatch.setitem(sys.modules, "paddlex.inference.utils", mock_utils)
        monkeypatch.setitem(
            sys.modules, "paddlex.inference.utils.official_models", mock_om_module
        )

        # Manually apply the patch now that mocks are in place
        paddlex_helpers._PADDLEX_PATCHED = False
        paddlex_helpers._patch_paddlex_official_models()

        # The module-level variable should now be our _LocalModelResolver
        result = mock_om_module.official_models[TEXT_DETECTION_MODEL]
        assert result == str(tmp_path / "PP-OCRv6_medium_det_infer")

    def test_patch_raises_for_unknown_model(self, tmp_path, monkeypatch):
        """Unknown model names raise RuntimeError instead of falling back to PaddleX."""
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)

        mock_obj = _MockOfficialModels()

        mock_utils = types.ModuleType("paddlex.inference.utils")
        mock_om_module = types.ModuleType("paddlex.inference.utils.official_models")
        mock_om_module.official_models = mock_obj
        mock_utils.official_models = mock_om_module

        monkeypatch.setitem(sys.modules, "paddlex.inference.utils", mock_utils)
        monkeypatch.setitem(
            sys.modules, "paddlex.inference.utils.official_models", mock_om_module
        )

        paddlex_helpers._PADDLEX_PATCHED = False
        paddlex_helpers._patch_paddlex_official_models()

        # Unknown model name should raise RuntimeError, not delegate to original
        resolver = mock_om_module.official_models
        with pytest.raises(RuntimeError, match="not available locally"):
            _ = resolver["UNKNOWN_MODEL_XYZ"]

    def test_patch_is_idempotent(self, tmp_path, monkeypatch):
        """Calling the patch multiple times doesn't break anything."""
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)

        mock_obj = _MockOfficialModels()

        mock_utils = types.ModuleType("paddlex.inference.utils")
        mock_om_module = types.ModuleType("paddlex.inference.utils.official_models")
        mock_om_module.official_models = mock_obj
        mock_utils.official_models = mock_om_module

        monkeypatch.setitem(sys.modules, "paddlex.inference.utils", mock_utils)
        monkeypatch.setitem(
            sys.modules, "paddlex.inference.utils.official_models", mock_om_module
        )

        paddlex_helpers._PADDLEX_PATCHED = False

        # Apply patch multiple times
        paddlex_helpers._patch_paddlex_official_models()
        paddlex_helpers._patch_paddlex_official_models()
        paddlex_helpers._patch_paddlex_official_models()

        # Should still work
        result = mock_om_module.official_models[TEXT_DETECTION_MODEL]
        assert result == str(tmp_path / "PP-OCRv6_medium_det_infer")

    def test_patch_handles_import_error(self, monkeypatch):
        """Patch gracefully handles ImportError when PaddleX is not installed."""
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "_PADDLEX_PATCHED", False)

        # Temporarily remove the module so import fails
        saved = sys.modules.get("paddlex.inference.utils.official_models")
        saved_utils = sys.modules.get("paddlex.inference.utils")
        try:
            if "paddlex.inference.utils.official_models" in sys.modules:
                del sys.modules["paddlex.inference.utils.official_models"]
            if "paddlex.inference.utils" in sys.modules:
                del sys.modules["paddlex.inference.utils"]

            # Patch should not raise when import fails
            paddlex_helpers._patch_paddlex_official_models()
        finally:
            if saved is not None:
                sys.modules["paddlex.inference.utils.official_models"] = saved
            else:
                sys.modules.pop("paddlex.inference.utils.official_models", None)
            if saved_utils is not None:
                sys.modules["paddlex.inference.utils"] = saved_utils
            else:
                sys.modules.pop("paddlex.inference.utils", None)


# ── Retry logic: permanent error detection ───────────────────────────────


class TestGetPaddlexModelPermanentError:
    """Tests for permanent error detection in _get_paddlex_model()."""

    def test_permanent_error_no_retry(self, tmp_path, monkeypatch):
        """ "No available model hosting platforms" fails immediately without retry."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
        _write_all_models(tmp_path)

        attempts = 0

        def permanent_failure_create_pipeline(*args: Any, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            raise RuntimeError(
                "No available model hosting platforms detected. Please check "
                "your network connection."
            )

        _install_fake_paddlex(
            monkeypatch,
            {"OCR": _ocr_config_fixture()},
            create_pipeline=permanent_failure_create_pipeline,
        )

        # First call: should raise immediately (no retries)
        with pytest.raises(RuntimeError, match="No available model hosting platforms"):
            paddlex_helpers._get_paddlex_model()
        assert attempts == 1  # only 1 attempt, no retries

    def test_permanent_error_cached(self, tmp_path, monkeypatch):
        """Second call re-raises the cached exception without a new attempt."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
        _write_all_models(tmp_path)

        attempts = 0

        def permanent_failure_create_pipeline(*args: Any, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("No available model hosting platforms detected.")

        _install_fake_paddlex(
            monkeypatch,
            {"OCR": _ocr_config_fixture()},
            create_pipeline=permanent_failure_create_pipeline,
        )

        # First call: should raise
        with pytest.raises(RuntimeError, match="No available model hosting platforms"):
            paddlex_helpers._get_paddlex_model()
        assert attempts == 1

        # Second call: should re-raise the cached exception (not retry)
        with pytest.raises(RuntimeError, match="No available model hosting platforms"):
            paddlex_helpers._get_paddlex_model()
        assert attempts == 1  # still 1 — cached exception re-raised

    def test_permanent_error_clear_message(self, tmp_path, monkeypatch):
        """Exception from permanent error is preserved with original message."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
        _write_all_models(tmp_path)

        original_error = RuntimeError(
            "No available model hosting platforms detected. Please check "
            "your network connection."
        )

        def permanent_failure_create_pipeline(*args: Any, **kwargs: Any) -> dict:
            raise original_error

        _install_fake_paddlex(
            monkeypatch,
            {"OCR": _ocr_config_fixture()},
            create_pipeline=permanent_failure_create_pipeline,
        )

        with pytest.raises(RuntimeError, match="No available model hosting"):
            paddlex_helpers._get_paddlex_model()

        # Verify the cached exception preserves the original message
        cached = paddlex_helpers.get_paddlex_init_exception()
        assert cached is not None
        assert "No available model hosting" in str(cached)


# ── Structure V3 retry logic ─────────────────────────────────────────────


class TestStructureV3RetryLogic:
    """Tests for retry logic in _get_paddlex_structure_v3_model()."""

    def test_structure_v3_transient_retry(self, tmp_path, monkeypatch):
        """Transient errors in Structure V3 retry up to 3 times."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
        _write_all_models(tmp_path)

        attempts = 0

        def flaky_create_pipeline(*args: Any, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("network timeout")  # transient
            return {}  # succeeds on 3rd attempt

        _install_fake_paddlex(
            monkeypatch,
            {"layout_parsing": _layout_parsing_config_fixture()},
            create_pipeline=flaky_create_pipeline,
        )

        model = paddlex_helpers._get_paddlex_structure_v3_model()
        assert attempts == 3  # 2 failures + 1 success
        assert model == {}

    def test_structure_v3_permanent_no_retry(self, tmp_path, monkeypatch):
        """Permanent errors in Structure V3 fail immediately without retry."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
        _write_all_models(tmp_path)

        attempts = 0

        def permanent_failure_create_pipeline(*args: Any, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("No available model hosting platforms detected.")

        _install_fake_paddlex(
            monkeypatch,
            {"layout_parsing": _layout_parsing_config_fixture()},
            create_pipeline=permanent_failure_create_pipeline,
        )

        with pytest.raises(RuntimeError, match="No available model hosting platforms"):
            paddlex_helpers._get_paddlex_structure_v3_model()
        assert attempts == 1  # only 1 attempt

    def test_structure_v3_permanent_cached(self, tmp_path, monkeypatch):
        """Second Structure V3 call re-raises cached permanent exception."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
        _write_all_models(tmp_path)

        attempts = 0

        def permanent_failure_create_pipeline(*args: Any, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("No available model hosting platforms detected.")

        _install_fake_paddlex(
            monkeypatch,
            {"layout_parsing": _layout_parsing_config_fixture()},
            create_pipeline=permanent_failure_create_pipeline,
        )

        # First call
        with pytest.raises(RuntimeError, match="No available model hosting platforms"):
            paddlex_helpers._get_paddlex_structure_v3_model()
        assert attempts == 1

        # Second call — should re-raise cached exception
        with pytest.raises(RuntimeError, match="No available model hosting platforms"):
            paddlex_helpers._get_paddlex_structure_v3_model()
        assert attempts == 1  # still 1

    def test_structure_v3_exhausts_retries(self, tmp_path, monkeypatch):
        """Structure V3 exhausts all 3 retry attempts on persistent transient error."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
        _write_all_models(tmp_path)

        attempts = 0

        def always_fails(*args: Any, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("persistent network error")

        _install_fake_paddlex(
            monkeypatch,
            {"layout_parsing": _layout_parsing_config_fixture()},
            create_pipeline=always_fails,
        )

        with pytest.raises(RuntimeError, match="persistent network error"):
            paddlex_helpers._get_paddlex_structure_v3_model()
        assert attempts == 3  # exhausted all retries


# ── Structure V3 destroy and recreate ────────────────────────────────────


class TestStructureV3DestroyAndRecreate:
    """Tests for Structure V3 model lifecycle."""

    def test_destroy_and_recreate_structure_v3(self, tmp_path, monkeypatch):
        """Verify that destroy_paddlex_model() + _get_paddlex_structure_v3_model()
        recreates the pipeline."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
        _write_all_models(tmp_path)

        create_count = 0

        def counting_create_pipeline(*args: Any, **kwargs: Any) -> dict:
            nonlocal create_count
            create_count += 1
            return {}

        _install_fake_paddlex(
            monkeypatch,
            {"layout_parsing": _layout_parsing_config_fixture()},
            create_pipeline=counting_create_pipeline,
        )

        # 1st creation
        paddlex_helpers._get_paddlex_structure_v3_model()
        assert create_count == 1

        # Destroy
        paddlex_helpers.destroy_paddlex_model()

        # Re-creation
        paddlex_helpers._get_paddlex_structure_v3_model()
        assert create_count == 2  # pipeline was recreated, not reused


# ── _LocalModelResolver tests ────────────────────────────────────────────


class TestLocalModelResolver:
    """Tests for _LocalModelResolver composition-based wrapper."""

    def test_raises_attribute_error_for_underscore_names(self, tmp_path, monkeypatch):
        """__getattr__ raises AttributeError for ALL underscore-prefixed names."""
        import paddlex_helpers

        resolver = paddlex_helpers._LocalModelResolver(_MockOfficialModels())

        with pytest.raises(AttributeError):
            _ = resolver._nonexistent
        with pytest.raises(AttributeError):
            _ = resolver.__custom_dunder__

    def test_raises_for_unknown_model_via_getitem(self, tmp_path, monkeypatch):
        """__getitem__ raises RuntimeError for unknown models instead of falling back."""
        import paddlex_helpers

        resolver = paddlex_helpers._LocalModelResolver(_MockOfficialModels())
        with pytest.raises(RuntimeError, match="not available locally"):
            _ = resolver["UNKNOWN_MODEL_XYZ"]

    def test_get_raises_when_no_default(self, tmp_path, monkeypatch):
        """get() with no default raises RuntimeError for unknown models."""
        import paddlex_helpers

        resolver = paddlex_helpers._LocalModelResolver(_MockOfficialModels())
        with pytest.raises(RuntimeError, match="not available locally"):
            resolver.get("unknown")

    def test_get_returns_explicit_default(self, tmp_path, monkeypatch):
        """get() with an explicit default returns that default instead of raising."""
        import paddlex_helpers

        resolver = paddlex_helpers._LocalModelResolver(_MockOfficialModels())
        assert resolver.get("unknown", "my-fallback") == "my-fallback"
        assert resolver.get("unknown", None) is None

    def test_contains_only_checks_local(self, tmp_path, monkeypatch):
        """__contains__ only checks local models, never the original (prevents network probes)."""
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)

        class MockWithContains(_MockOfficialModels):
            def __contains__(self, name):
                return name == "known_model"

        resolver = paddlex_helpers._LocalModelResolver(MockWithContains())
        # Local models are found
        assert "PP-OCRv6_medium_det" in resolver
        # Unknown models are NOT found (no fallback to original)
        assert "known_model" not in resolver
        assert "unknown_model" not in resolver

    def test_items_delegates_to_original(self, tmp_path, monkeypatch):
        """items() delegates to the original object."""
        import paddlex_helpers

        class MockWithItems(_MockOfficialModels):
            def items(self):
                return [("a", 1), ("b", 2)]

        resolver = paddlex_helpers._LocalModelResolver(MockWithItems())
        assert list(resolver.items()) == [("a", 1), ("b", 2)]

    def test_pop_delegates_to_original(self, tmp_path, monkeypatch):
        """pop() delegates to the original object."""
        import paddlex_helpers

        class MockWithPop(_MockOfficialModels):
            def pop(self, key, default=None):
                return f"popped-{key}"

        resolver = paddlex_helpers._LocalModelResolver(MockWithPop())
        assert resolver.pop("key") == "popped-key"

    def test_setdefault_delegates_to_original(self, tmp_path, monkeypatch):
        """setdefault() delegates to the original object."""
        import paddlex_helpers

        class MockWithSetdefault(_MockOfficialModels):
            def setdefault(self, key, default=None):
                return f"default-{key}"

        resolver = paddlex_helpers._LocalModelResolver(MockWithSetdefault())
        assert resolver.setdefault("key") == "default-key"

    def test_update_delegates_to_original(self, tmp_path, monkeypatch):
        """update() delegates to the original object."""
        import paddlex_helpers

        class MockWithUpdate(_MockOfficialModels):
            def update(self, other=None, **kwargs):
                self._updated = True

        mock = MockWithUpdate()
        resolver = paddlex_helpers._LocalModelResolver(mock)
        resolver.update({})
        assert mock._updated is True


# ── _build_model_dir_status and _build_enriched_permanent_error tests ────


class TestModelDirStatusHelper:
    """Tests for _build_model_dir_status() and _build_enriched_permanent_error()."""

    def test_model_dir_status_shows_checkmarks(self, tmp_path, monkeypatch):
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)

        status = paddlex_helpers._build_model_dir_status()
        assert "PP-OCRv6_medium_det: ✓" in status
        assert "PP-OCRv6_medium_rec: ✓" in status
        assert "PP-LCNet_x1_0_textline_ori: ✓" in status
        assert "PP-LCNet_x1_0_doc_ori: ✓" in status
        assert "PP-DocLayout-L: ✓" in status

    def test_model_dir_status_shows_x_for_missing(self, tmp_path, monkeypatch):
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
        _write_model(tmp_path, TEXT_DETECTION_MODEL)
        _write_model(tmp_path, TEXT_RECOGNITION_MODEL)

        status = paddlex_helpers._build_model_dir_status()
        assert "PP-OCRv6_medium_det: ✓" in status
        assert "PP-DocLayout-L: ✗" in status

    def test_enriched_error_contains_diagnostics(self, tmp_path, monkeypatch):
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
        original_error = RuntimeError("No available model hosting platforms detected")

        enriched = paddlex_helpers._build_enriched_permanent_error(original_error)
        assert "PaddleX model initialization failed (permanent)" in str(enriched)
        assert "No available model hosting platforms" in str(enriched)
        assert "Local model directories:" in str(enriched)
        assert "Patch applied:" in str(enriched)
        assert "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=" in str(enriched)
        assert "Models directory:" in str(enriched)
        assert "PADDLEOCR_MODELS=" not in str(enriched)  # full path redacted

    def test_enriched_error_preserves_cause_chain(self, tmp_path, monkeypatch):
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
        original_error = RuntimeError("original cause")

        enriched = paddlex_helpers._build_enriched_permanent_error(original_error)
        assert enriched.__cause__ is original_error


# ── warmup_paddlex_models tests ──────────────────────────────────────────


class TestWarmup:
    """Tests for warmup_paddlex_models()."""

    def test_warmup_succeeds_for_both_pipelines(self, tmp_path, monkeypatch):
        """Warmup completes successfully when both pipelines initialize."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
        _write_all_models(tmp_path)

        create_count = 0

        def counting_create_pipeline(*args: Any, **kwargs: Any) -> dict:
            nonlocal create_count
            create_count += 1
            return {}

        _install_fake_paddlex(
            monkeypatch,
            {
                "OCR": _ocr_config_fixture(),
                "layout_parsing": _layout_parsing_config_fixture(),
            },
            create_pipeline=counting_create_pipeline,
        )

        paddlex_helpers.warmup_paddlex_models()
        assert create_count == 2  # OCR + Structure V3

    def test_warmup_clears_cached_exception_on_failure(
        self, tmp_path, monkeypatch, caplog
    ):
        """Warmup clears _init_exception so first real use gets a fresh attempt."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
        _write_all_models(tmp_path)

        attempts = 0

        def failing_create_pipeline(*args: Any, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            # Always fail with transient error — warmup will exhaust retries
            raise RuntimeError("warmup fails initially")

        _install_fake_paddlex(
            monkeypatch,
            {"OCR": _ocr_config_fixture()},
            create_pipeline=failing_create_pipeline,
        )

        with caplog.at_level("WARNING"):
            paddlex_helpers.warmup_paddlex_models()

        # Warmup logged warning about failure (outer warning, not retry warnings)
        assert "warm-up failed" in caplog.text

    def test_warmup_propagates_system_exit(self, tmp_path, monkeypatch):
        """SystemExit and KeyboardInterrupt should not be swallowed by warmup."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
        _write_all_models(tmp_path)

        def system_exit_pipeline(*args: Any, **kwargs: Any) -> dict:
            raise SystemExit("forced shutdown")

        _install_fake_paddlex(
            monkeypatch,
            {"OCR": _ocr_config_fixture()},
            create_pipeline=system_exit_pipeline,
        )

        with pytest.raises(SystemExit):
            paddlex_helpers.warmup_paddlex_models()


# ── _is_model_error tests ────────────────────────────────────────────────


class TestIsModelError:
    """Tests for _is_model_error() classification."""

    def test_paddle_error_is_model_error(self):
        from paddlex_helpers import _is_model_error

        assert _is_model_error(RuntimeError("PaddleX failed to initialize"))
        assert _is_model_error(RuntimeError("CUDA out of memory"))
        assert _is_model_error(RuntimeError("Predictor inference failed"))

    def test_docling_error_is_not_model_error(self):
        from paddlex_helpers import _is_model_error

        assert not _is_model_error(RuntimeError("Docling API timeout"))
        assert not _is_model_error(RuntimeError("HTTP 503"))
        assert not _is_model_error(FileNotFoundError("file.pdf not found"))

    def test_ocrmypdf_error_is_not_model_error(self):
        from paddlex_helpers import _is_model_error

        assert not _is_model_error(RuntimeError("ocrmypdf exit code 1"))
        assert not _is_model_error(RuntimeError("paperless push failed"))


# ── _model_dir security tests ────────────────────────────────────────────


class TestModelDirSecurity:
    """Tests for _model_dir() path traversal protection."""

    def test_model_dir_raises_for_unknown_name(self, tmp_path, monkeypatch):
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))

        with pytest.raises(ValueError, match="Unknown model name"):
            paddlex_helpers._model_dir("../../etc/passwd")

    def test_model_dir_raises_for_traversal_attempt(self, tmp_path, monkeypatch):
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))

        with pytest.raises(ValueError, match="Unknown model name"):
            paddlex_helpers._model_dir("PP-OCRv6_medium_det/../../etc")

    def test_model_dir_returns_valid_path_for_known_model(self, tmp_path, monkeypatch):
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))

        result = paddlex_helpers._model_dir(TEXT_DETECTION_MODEL)
        assert result == tmp_path / "PP-OCRv6_medium_det_infer"
