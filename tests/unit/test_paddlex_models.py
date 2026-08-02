from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

# Import from paddlex_helpers directly (paddleocr_helpers is a shim)
from paddlex_helpers import (
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
    """Clear cached PaddleX model state before each test."""
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

    yield

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

    assert _model_dir(TEXT_DETECTION_MODEL) == (tmp_path / "PP-OCRv6_medium_det_infer")
    assert _model_dir(TEXT_RECOGNITION_MODEL) == (
        tmp_path / "PP-OCRv6_medium_rec_infer"
    )
    assert _model_dir(TEXTLINE_ORIENTATION_MODEL) == (
        tmp_path / "PP-LCNet_x1_0_textline_ori_infer"
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

    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_create_pipeline(name: str, **kwargs: Any) -> dict:
        calls.append((name, kwargs))
        return {}  # dummy pipeline

    # Mock paddlex module
    pdx_module = types.ModuleType("paddlex")
    pdx_module.__path__ = []  # type: ignore[attr-defined]
    pdx_module.create_pipeline = fake_create_pipeline
    monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

    create_paddleocr_model(use_textline_orientation=True)

    assert len(calls) == 1
    pipeline_name, kwargs = calls[0]
    # PaddleX 3.x pipeline registry is case-sensitive — "OCR" (uppercase) is correct
    assert pipeline_name == "OCR"
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


# ── Model lifecycle: destroy + recreate ──────────────────────────────────


def test_destroy_and_recreate_paddlex_model(tmp_path, monkeypatch):
    """Verify that destroy_paddlex_model() + _get_paddlex_model() recreates the pipeline."""
    import paddlex_helpers

    monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
    _write_all_models(tmp_path)

    create_count = 0

    def counting_create_pipeline(name: str, **kwargs: Any) -> dict:
        nonlocal create_count
        create_count += 1
        return {}

    pdx_module = types.ModuleType("paddlex")
    pdx_module.__path__ = []  # type: ignore[attr-defined]
    pdx_module.create_pipeline = counting_create_pipeline
    monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

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

    def flaky_create_pipeline(name: str, **kwargs: Any) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("network timeout")  # transient
        return {}  # succeeds on 2nd attempt

    pdx_module = types.ModuleType("paddlex")
    pdx_module.__path__ = []  # type: ignore[attr-defined]
    pdx_module.create_pipeline = flaky_create_pipeline
    monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

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

    def permanent_failure_create_pipeline(name: str, **kwargs: Any) -> dict:
        nonlocal attempts
        attempts += 1
        raise original_error

    pdx_module = types.ModuleType("paddlex")
    pdx_module.__path__ = []  # type: ignore[attr-defined]
    pdx_module.create_pipeline = permanent_failure_create_pipeline
    monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

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

    captured_name: list[str] = []

    def capturing_create_pipeline(name: str, **kwargs: Any) -> dict:
        captured_name.append(name)
        return {}

    pdx_module = types.ModuleType("paddlex")
    pdx_module.__path__ = []  # type: ignore[attr-defined]
    pdx_module.create_pipeline = capturing_create_pipeline
    monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

    # Clear any cached model from previous tests
    if hasattr(paddlex_helpers._get_paddlex_model, "_model"):
        del paddlex_helpers._get_paddlex_model._model

    paddlex_helpers._get_paddlex_model()
    assert len(captured_name) == 1
    # This assertion would fail if the code used lowercase "ocr"
    assert captured_name[0] == "OCR", (
        f"Pipeline name must be 'OCR' (uppercase), got '{captured_name[0]}'"
    )


# ── Monkey-patch for air-gapped operation ────────────────────────────────


class TestPaddlexOfficialModelsPatch:
    """Tests for _patch_paddlex_official_models() — air-gapped compatibility."""

    def test_patch_returns_local_dir(self, tmp_path, monkeypatch):
        """Patched __getitem__ returns our local model path for known model names."""
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)

        # Create a mock official_models object
        mock_class = type("MockOfficialModels", (), {})
        mock_obj = mock_class()
        mock_obj.__class__.__getitem__ = lambda self, name: f"original-{name}"

        # Set up the full module hierarchy so `from ... import` works
        mock_utils = types.ModuleType("paddlex.inference.utils")
        mock_om_module = types.ModuleType("paddlex.inference.utils.official_models")
        mock_om_module.official_models = mock_obj
        mock_utils.official_models = mock_om_module

        monkeypatch.setitem(sys.modules, "paddlex.inference.utils", mock_utils)
        monkeypatch.setitem(
            sys.modules, "paddlex.inference.utils.official_models", mock_om_module
        )

        # Reset patch flag so it gets applied
        paddlex_helpers._PADDLEX_PATCHED = False
        paddlex_helpers._patch_paddlex_official_models()

        # Known model name should resolve to our local dir
        result = mock_obj[TEXT_DETECTION_MODEL]
        assert result == str(tmp_path / "PP-OCRv6_medium_det_infer")

    def test_patch_falls_back_to_original(self, tmp_path, monkeypatch):
        """Unknown model names fall back to original behavior."""
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)

        mock_class = type("MockOfficialModels", (), {})
        mock_obj = mock_class()
        original_getitem = lambda self, name: f"original-{name}"
        mock_obj.__class__.__getitem__ = original_getitem

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

        # Unknown model name should use original behavior
        result = mock_obj["UNKNOWN_MODEL_XYZ"]
        assert result == "original-UNKNOWN_MODEL_XYZ"

    def test_patch_is_idempotent(self, tmp_path, monkeypatch):
        """Calling the patch multiple times doesn't break anything."""
        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)

        mock_class = type("MockOfficialModels", (), {})
        mock_obj = mock_class()
        mock_obj.__class__.__getitem__ = lambda self, name: f"original-{name}"

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
        result = mock_obj[TEXT_DETECTION_MODEL]
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

        def permanent_failure_create_pipeline(name: str, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            raise RuntimeError(
                "No available model hosting platforms detected. Please check "
                "your network connection."
            )

        pdx_module = types.ModuleType("paddlex")
        pdx_module.__path__ = []  # type: ignore[attr-defined]
        pdx_module.create_pipeline = permanent_failure_create_pipeline
        monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

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

        def permanent_failure_create_pipeline(name: str, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("No available model hosting platforms detected.")

        pdx_module = types.ModuleType("paddlex")
        pdx_module.__path__ = []  # type: ignore[attr-defined]
        pdx_module.create_pipeline = permanent_failure_create_pipeline
        monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

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

        def permanent_failure_create_pipeline(name: str, **kwargs: Any) -> dict:
            raise original_error

        pdx_module = types.ModuleType("paddlex")
        pdx_module.__path__ = []  # type: ignore[attr-defined]
        pdx_module.create_pipeline = permanent_failure_create_pipeline
        monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

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

        def flaky_create_pipeline(name: str, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("network timeout")  # transient
            return {}  # succeeds on 3rd attempt

        pdx_module = types.ModuleType("paddlex")
        pdx_module.__path__ = []  # type: ignore[attr-defined]
        pdx_module.create_pipeline = flaky_create_pipeline
        monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

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

        def permanent_failure_create_pipeline(name: str, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("No available model hosting platforms detected.")

        pdx_module = types.ModuleType("paddlex")
        pdx_module.__path__ = []  # type: ignore[attr-defined]
        pdx_module.create_pipeline = permanent_failure_create_pipeline
        monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

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

        def permanent_failure_create_pipeline(name: str, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("No available model hosting platforms detected.")

        pdx_module = types.ModuleType("paddlex")
        pdx_module.__path__ = []  # type: ignore[attr-defined]
        pdx_module.create_pipeline = permanent_failure_create_pipeline
        monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

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

        def always_fails(name: str, **kwargs: Any) -> dict:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("persistent network error")

        pdx_module = types.ModuleType("paddlex")
        pdx_module.__path__ = []  # type: ignore[attr-defined]
        pdx_module.create_pipeline = always_fails
        monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

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

        def counting_create_pipeline(name: str, **kwargs: Any) -> dict:
            nonlocal create_count
            create_count += 1
            return {}

        pdx_module = types.ModuleType("paddlex")
        pdx_module.__path__ = []  # type: ignore[attr-defined]
        pdx_module.create_pipeline = counting_create_pipeline
        monkeypatch.setitem(sys.modules, "paddlex", pdx_module)

        # 1st creation
        paddlex_helpers._get_paddlex_structure_v3_model()
        assert create_count == 1

        # Destroy
        paddlex_helpers.destroy_paddlex_model()

        # Re-creation
        paddlex_helpers._get_paddlex_structure_v3_model()
        assert create_count == 2  # pipeline was recreated, not reused
