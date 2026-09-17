from __future__ import annotations

import logging
import sys
import time
import types
from collections.abc import Callable
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
    ModelIdleTracker,
    _model_dir,
    create_paddleocr_model,
    run_paddleocr,
    run_paddlex_structure_v3,
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

    # Reset the in-flight guard counter so a leaked context cannot
    # suppress idle-unload decisions in later tests.
    paddlex_helpers._model_in_flight = 0

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

    paddlex_helpers._model_in_flight = 0


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
    (it will be invoked as ``create_pipeline(config=cfg, device=...)``). Returns a
    captured dict with keys ``pipeline`` / ``config`` / ``kwargs`` (last call; all
    keyword args recorded, e.g. ``device``) and ``load_calls`` (a list).
    """
    captured: dict[str, Any] = {"load_calls": []}

    def _default_create_pipeline(pipeline=None, *, config=None, **kwargs: Any) -> dict:
        captured["pipeline"] = pipeline
        captured["config"] = config
        captured["kwargs"] = dict(kwargs)
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


def _install_fake_paddle(
    monkeypatch,
    *,
    compiled_with_cuda: bool = True,
    device_count: int = 1,
    allocated_mb: tuple[float, ...] = (),
    reserved_mb: tuple[float, ...] = (),
) -> dict[str, int]:
    """Install fake ``paddle`` / ``paddle.device`` / ``paddle.device.cuda`` modules.

    ``allocated_mb`` / ``reserved_mb`` are the MB values returned by
    ``memory_allocated()`` / ``memory_reserved()`` in call order (the last
    value repeats once the sequence is exhausted; 0 when empty). This lets a
    test script the before/after stats of one destroy call. Returns per-API
    call counters.
    """
    calls = {"empty_cache": 0, "allocated": 0, "reserved": 0}

    def _seq_mb(values: tuple[float, ...]) -> Callable[[], int]:
        state = {"i": 0, "last": 0.0}

        def fn() -> int:
            if values:
                state["last"] = values[min(state["i"], len(values) - 1)]
                state["i"] += 1
            return int(state["last"] * 1024 * 1024)

        return fn

    alloc_fn = _seq_mb(allocated_mb)
    resv_fn = _seq_mb(reserved_mb)

    paddle_mod = types.ModuleType("paddle")
    device_mod = types.ModuleType("paddle.device")
    cuda_mod = types.ModuleType("paddle.device.cuda")

    def empty_cache() -> None:
        calls["empty_cache"] += 1

    def is_compiled_with_cuda() -> bool:
        return compiled_with_cuda

    def count_devices() -> int:
        return device_count

    def memory_allocated() -> int:
        calls["allocated"] += 1
        return alloc_fn()

    def memory_reserved() -> int:
        calls["reserved"] += 1
        return resv_fn()

    cuda_mod.empty_cache = empty_cache  # type: ignore[attr-defined]
    cuda_mod.device_count = count_devices  # type: ignore[attr-defined]
    cuda_mod.memory_allocated = memory_allocated  # type: ignore[attr-defined]
    cuda_mod.memory_reserved = memory_reserved  # type: ignore[attr-defined]
    device_mod.cuda = cuda_mod  # type: ignore[attr-defined]
    device_mod.is_compiled_with_cuda = is_compiled_with_cuda  # type: ignore[attr-defined]
    paddle_mod.device = device_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "paddle", paddle_mod)
    monkeypatch.setitem(sys.modules, "paddle.device", device_mod)
    monkeypatch.setitem(sys.modules, "paddle.device.cuda", cuda_mod)
    return calls


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


class _GeneratorPredictModel:
    """Mimic PaddleX >=3.0: ``predict`` returns a *generator* (one result per sample).

    Real PaddleX 3.x ``pipeline.predict()`` is a generator; this fake *yields* dicts
    (instead of returning a list) so the tests exercise the materialization path that
    the production bug broke. No real model is loaded or downloaded.
    """

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages

    def predict(self, file_path: str):
        yield from self._pages


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


# ── Generator-consumption regression (PaddleX >=3.0 predict() yields a generator) ──
# The last "fix model loading" commit made predict() reachable; these tests pin the
# consumption contract so a regression (treating the generator as a list/dict) fails in
# CI WITHOUT downloading any real models (the model factory is monkeypatched directly).


def test_run_paddleocr_consumes_generator_predict_result(monkeypatch):
    """run_paddleocr must handle predict() returning a generator (PaddleX >=3.0)."""
    monkeypatch.setattr(
        "paddlex_helpers._get_paddlex_model",
        lambda: _GeneratorPredictModel(
            [
                {
                    "rec_texts": ["Hello", "", "World"],
                    "rec_scores": [0.98765, 0.5, 0.87654],
                    "rec_boxes": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
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


def test_run_paddlex_structure_v3_consumes_generator_predict_result(
    monkeypatch, mock_pdf_to_images
):
    """run_paddlex_structure_v3 must handle predict() yielding a generator.

    Reproduces the production error "'generator' object has no attribute 'get'": a real
    PDF path is converted to images (mocked, so no pdftoppm and no model download) and
    each page result is produced by a generator.
    """
    monkeypatch.setattr(
        "paddlex_helpers._get_paddlex_structure_v3_model",
        lambda: _GeneratorPredictModel(
            [
                {
                    "overall_ocr_res": {
                        "rec_texts": ["Title", "Body", "Sub"],
                        "rec_scores": [0.99, 0.95, 0.92],
                        "rec_boxes": [
                            [50, 50, 200, 70],
                            [50, 90, 300, 130],
                            [50, 150, 180, 170],
                        ],
                        "rec_polys": [],
                    },
                    "parsing_res_list": [
                        {
                            "block_label": "title",
                            "block_content": "Title",
                            "block_bbox": [50, 50, 200, 70],
                        },
                        {
                            "block_label": "text",
                            "block_content": "Body",
                            "block_bbox": [50, 90, 300, 130],
                        },
                        {
                            "block_label": "paragraph_title",
                            "block_content": "Sub",
                            "block_bbox": [50, 150, 180, 170],
                        },
                    ],
                    "layout_det_res": {
                        "boxes": [
                            {"label": "title", "score": 0.99},
                            {"label": "text", "score": 0.97},
                            {"label": "paragraph_title", "score": 0.96},
                        ],
                    },
                }
            ]
        ),
    )

    pages = run_paddlex_structure_v3("document.pdf")

    assert pages == [
        {
            "page": 1,
            "text": "Title\nBody\nSub",
            "blocks": [
                {"text": "Title", "bbox": [50, 50, 200, 70], "confidence": 0.99},
                {"text": "Body", "bbox": [50, 90, 300, 130], "confidence": 0.95},
                {"text": "Sub", "bbox": [50, 150, 180, 170], "confidence": 0.92},
            ],
            "structured_blocks": [
                {
                    "type": "title",
                    "bbox": [50, 50, 200, 70],
                    "text": "Title",
                    "confidence": 0.99,
                },
                {
                    "type": "text",
                    "bbox": [50, 90, 300, 130],
                    "text": "Body",
                    "confidence": 0.97,
                },
                {
                    "type": "paragraph_title",
                    "bbox": [50, 150, 180, 170],
                    "text": "Sub",
                    "confidence": 0.96,
                },
            ],
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


def test_paddlex_model_is_loaded_reflects_resident_pipelines(tmp_path, monkeypatch):
    """paddlex_model_is_loaded() tracks whether a warmed-up model is resident.

    Regression test for the VRAM-leak bug: the idle-unload thread only destroys
    a model once the process has marked it as used, and startup warmup is the
    path that loads the pipelines without any request. If a warmed-up model is
    not reported as loaded, the process never starts the idle clock and the
    GPU memory is never reclaimed.
    """
    import paddlex_helpers

    monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "paddlex-cache"))
    _write_all_models(tmp_path)

    def counting_create_pipeline(*args: Any, **kwargs: Any) -> dict:
        return {}

    _install_fake_paddlex(
        monkeypatch,
        {
            "OCR": _ocr_config_fixture(),
            "layout_parsing": _layout_parsing_config_fixture(),
        },
        create_pipeline=counting_create_pipeline,
    )

    # Nothing resident yet (singleton cleared by the autouse fixture)
    assert paddlex_helpers.paddlex_model_is_loaded() is False

    # Startup warmup loads the pipelines -> resident
    paddlex_helpers.warmup_paddlex_models()
    assert paddlex_helpers.paddlex_model_is_loaded() is True

    # Destruction reclaims them -> not resident
    paddlex_helpers.destroy_paddlex_model()
    assert paddlex_helpers.paddlex_model_is_loaded() is False


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


# ── empty_cache gate: runtime CUDA availability, not OCR_USE_GPU ─────────


class TestDestroyEmptyCacheGate:
    """destroy_paddlex_model() must flush Paddle's CUDA allocator pool based on
    RUNTIME CUDA availability — not the OCR_USE_GPU env var. The old gate
    (OCR_USE_GPU) let a cuda deployment with OCR_USE_GPU=false leak its pool
    forever, because device selection was left to PaddleX auto-detection."""

    def test_flushes_when_cuda_available_even_if_ocr_use_gpu_false(
        self, tmp_path, monkeypatch
    ):
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)
        _install_fake_paddlex(monkeypatch, {"OCR": _ocr_config_fixture()})
        calls = _install_fake_paddle(monkeypatch)  # CUDA available
        monkeypatch.setattr(paddlex_helpers, "OCR_USE_GPU", False)

        paddlex_helpers._get_paddlex_model()
        paddlex_helpers.destroy_paddlex_model()

        assert calls["empty_cache"] == 1

    def test_no_cuda_api_when_cuda_unavailable(self, tmp_path, monkeypatch):
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)
        _install_fake_paddlex(monkeypatch, {"OCR": _ocr_config_fixture()})
        calls = _install_fake_paddle(monkeypatch, compiled_with_cuda=False)

        paddlex_helpers._get_paddlex_model()
        paddlex_helpers.destroy_paddlex_model()

        assert calls["empty_cache"] == 0
        assert calls["allocated"] == 0
        assert calls["reserved"] == 0


# ── explicit device= kwarg on create_pipeline ────────────────────────────


class TestCreatePipelineDeviceKwarg:
    """create_pipeline() must receive an explicit device= for both pipelines.

    No PaddleX auto-detection: it silently ignored OCR_USE_GPU, which is how
    the 'OCR_USE_GPU=false but running on GPU' deployments happened.
    """

    def test_ocr_pipeline_device_cpu_default(self, tmp_path, monkeypatch):
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)
        monkeypatch.setattr(paddlex_helpers, "OCR_USE_GPU", False)
        monkeypatch.setattr(paddlex_helpers, "_cuda_available", lambda: False)
        captured = _install_fake_paddlex(monkeypatch, {"OCR": _ocr_config_fixture()})

        paddlex_helpers._get_paddlex_model()
        assert captured["kwargs"]["device"] == "cpu"

    def test_ocr_pipeline_device_gpu_when_requested_and_available(
        self, tmp_path, monkeypatch
    ):
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)
        monkeypatch.setattr(paddlex_helpers, "OCR_USE_GPU", True)
        monkeypatch.setattr(paddlex_helpers, "_cuda_available", lambda: True)
        captured = _install_fake_paddlex(monkeypatch, {"OCR": _ocr_config_fixture()})

        paddlex_helpers._get_paddlex_model()
        assert captured["kwargs"]["device"] == "gpu:0"

    def test_layout_parsing_pipeline_device_gpu(self, tmp_path, monkeypatch):
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)
        monkeypatch.setattr(paddlex_helpers, "OCR_USE_GPU", True)
        monkeypatch.setattr(paddlex_helpers, "_cuda_available", lambda: True)
        captured = _install_fake_paddlex(
            monkeypatch, {"layout_parsing": _layout_parsing_config_fixture()}
        )

        paddlex_helpers._get_paddlex_structure_v3_model()
        assert captured["kwargs"]["device"] == "gpu:0"

    def test_gpu_requested_but_cuda_unavailable_falls_back_to_cpu(
        self, tmp_path, monkeypatch
    ):
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)
        monkeypatch.setattr(paddlex_helpers, "OCR_USE_GPU", True)
        monkeypatch.setattr(paddlex_helpers, "_cuda_available", lambda: False)
        captured = _install_fake_paddlex(monkeypatch, {"OCR": _ocr_config_fixture()})

        paddlex_helpers._get_paddlex_model()
        assert captured["kwargs"]["device"] == "cpu"


# ── destroy_paddlex_model(verify_reclaim=True) reclamation logging ───────


class TestDestroyReclaimVerification:
    """verify_reclaim=True (idle path) must measure before/after GPU stats and
    log a distinct warning when the VRAM was NOT actually returned."""

    def test_logs_reclaimed_when_stats_drop(self, tmp_path, monkeypatch, caplog):
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)
        _install_fake_paddlex(monkeypatch, {"OCR": _ocr_config_fixture()})
        calls = _install_fake_paddle(
            monkeypatch, allocated_mb=(500, 10), reserved_mb=(600, 20)
        )

        paddlex_helpers._get_paddlex_model()
        with caplog.at_level(logging.INFO, logger="doc-worker.paddlex_helpers"):
            paddlex_helpers.destroy_paddlex_model(verify_reclaim=True)

        assert calls["allocated"] == 2  # before + after
        assert calls["reserved"] == 2
        assert calls["empty_cache"] == 1
        assert "VRAM reclaimed" in caplog.text
        assert "VRAM NOT reclaimed" not in caplog.text
        # before/after values are logged (500 -> 10 MB allocated)
        assert "500" in caplog.text and "10" in caplog.text

    def test_warns_when_allocated_stays_high(self, tmp_path, monkeypatch, caplog):
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)
        _install_fake_paddlex(monkeypatch, {"OCR": _ocr_config_fixture()})
        _install_fake_paddle(
            monkeypatch, allocated_mb=(500, 450), reserved_mb=(600, 580)
        )

        paddlex_helpers._get_paddlex_model()
        with caplog.at_level(logging.WARNING, logger="doc-worker.paddlex_helpers"):
            paddlex_helpers.destroy_paddlex_model(verify_reclaim=True)

        assert "VRAM NOT reclaimed" in caplog.text
        assert "still referenced" in caplog.text

    def test_no_stats_query_with_default_verify_reclaim(self, tmp_path, monkeypatch):
        """Non-idle paths (shutdown/retry/recovery) must never query stats."""
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)
        _install_fake_paddlex(monkeypatch, {"OCR": _ocr_config_fixture()})
        calls = _install_fake_paddle(
            monkeypatch, allocated_mb=(500,), reserved_mb=(600,)
        )

        paddlex_helpers._get_paddlex_model()
        paddlex_helpers.destroy_paddlex_model()

        assert calls["allocated"] == 0
        assert calls["reserved"] == 0

    def test_cpu_build_logs_without_reclamation_claim(
        self, tmp_path, monkeypatch, caplog
    ):
        import paddlex_helpers

        monkeypatch.setenv("PADDLEOCR_MODELS", str(tmp_path))
        _write_all_models(tmp_path)
        _install_fake_paddlex(monkeypatch, {"OCR": _ocr_config_fixture()})
        calls = _install_fake_paddle(monkeypatch, compiled_with_cuda=False)

        paddlex_helpers._get_paddlex_model()
        with caplog.at_level(logging.INFO, logger="doc-worker.paddlex_helpers"):
            paddlex_helpers.destroy_paddlex_model(verify_reclaim=True)

        assert calls["allocated"] == 0
        assert calls["reserved"] == 0
        assert calls["empty_cache"] == 0
        assert "memory reclaimed" not in caplog.text
        assert "PaddleX models destroyed" in caplog.text


# ── shared idle-unload decision (ModelIdleTracker) ──────────────────────


class TestModelIdleTracker:
    """The single shared idle-unload decision used by server and worker."""

    def _check(
        self,
        monkeypatch,
        stamp: float,
        loaded: bool = True,
        in_flight: bool = False,
        timeout: float = 30,
    ) -> tuple[list[dict], ModelIdleTracker]:
        import paddlex_helpers

        tracker = paddlex_helpers.ModelIdleTracker()
        tracker._last_used = stamp
        if loaded:
            paddlex_helpers._get_paddlex_model._model = object()
        destroys: list[dict] = []
        monkeypatch.setattr(
            paddlex_helpers,
            "destroy_paddlex_model",
            lambda **kw: destroys.append(kw),
        )

        if in_flight:
            with paddlex_helpers.model_in_flight():
                tracker.check_once(timeout)
        else:
            tracker.check_once(timeout)
        return destroys, tracker

    def test_mark_used_sets_stamp(self):
        import paddlex_helpers

        tracker = paddlex_helpers.ModelIdleTracker()
        assert tracker._last_used == 0.0
        before = time.time()
        tracker.mark_used()
        after = time.time()
        assert before <= tracker._last_used <= after

    def test_noop_when_stamp_zero(self, monkeypatch):
        destroys, tracker = self._check(monkeypatch, stamp=0.0)
        assert destroys == []
        assert tracker._last_used == 0.0

    def test_noop_when_timeout_not_elapsed(self, monkeypatch):
        orig = time.time() - 10
        destroys, tracker = self._check(monkeypatch, stamp=orig)
        assert destroys == []
        assert tracker._last_used is orig  # stamp untouched

    def test_destroys_when_idle_loaded_and_not_in_flight(self, monkeypatch):
        destroys, tracker = self._check(monkeypatch, stamp=time.time() - 100)
        assert destroys == [{"verify_reclaim": True}]
        assert tracker._last_used == 0.0

    def test_resets_stamp_without_destroy_when_not_loaded(self, monkeypatch):
        destroys, tracker = self._check(
            monkeypatch, stamp=time.time() - 100, loaded=False
        )
        assert destroys == []  # no spurious destroy
        assert tracker._last_used == 0.0  # but the stamp was reset

    def test_skips_destroy_while_job_in_flight(self, monkeypatch):
        destroys, tracker = self._check(
            monkeypatch, stamp=time.time() - 100, in_flight=True
        )
        assert destroys == []
        assert tracker._last_used != 0.0  # stamp preserved — retry on next poll

    def test_negative_timeout_is_clamped_not_inverted(self, monkeypatch):
        """A misconfigured negative timeout must be clamped (>= 1 s), not let
        invert the ``elapsed <= timeout`` comparison — otherwise a just-used
        model (elapsed ≈ 0) would be destroyed on the first poll after every
        use."""
        destroys, tracker = self._check(monkeypatch, stamp=time.time(), timeout=-5)
        assert destroys == []
        assert tracker._last_used != 0.0  # preserved — not destroyed


# ── in-flight model guard ────────────────────────────────────────────────


class TestModelInFlightGuard:
    """model_in_flight() counter: exception safety is the load-bearing part
    (a wedged counter would permanently block idle unloading)."""

    def test_counter_tracks_nesting(self):
        import paddlex_helpers

        assert paddlex_helpers.model_in_flight_count() == 0
        with paddlex_helpers.model_in_flight():
            assert paddlex_helpers.model_in_flight_count() == 1
            with paddlex_helpers.model_in_flight():
                assert paddlex_helpers.model_in_flight_count() == 2
        assert paddlex_helpers.model_in_flight_count() == 0

    def test_counter_returns_to_zero_after_exception(self):
        import paddlex_helpers

        assert paddlex_helpers.model_in_flight_count() == 0
        with pytest.raises(RuntimeError, match="boom"):
            with paddlex_helpers.model_in_flight():
                raise RuntimeError("boom")
        assert paddlex_helpers.model_in_flight_count() == 0

    def test_run_paddleocr_holds_guard_during_predict(self, monkeypatch):
        """The guard must be held across model fetch + predict so the idle
        thread never sees a loaded model as idle mid-job."""
        import paddlex_helpers

        seen: dict[str, int] = {}

        class _SpyModel:
            def predict(self, _path: str):
                seen["count"] = paddlex_helpers.model_in_flight_count()
                yield {"rec_texts": [], "rec_scores": [], "rec_boxes": []}

        monkeypatch.setattr(paddlex_helpers, "_get_paddlex_model", _SpyModel)

        paddlex_helpers.run_paddleocr("document.pdf")
        assert seen["count"] == 1
        assert paddlex_helpers.model_in_flight_count() == 0

    def test_guard_releases_after_predict_exception(self, monkeypatch):
        import paddlex_helpers

        class _BoomModel:
            def predict(self, _path: str):
                raise RuntimeError("predictor exploded")
                yield  # pragma: no cover — makes this a generator

        monkeypatch.setattr(paddlex_helpers, "_get_paddlex_model", _BoomModel)

        with pytest.raises(RuntimeError, match="predictor exploded"):
            paddlex_helpers.run_paddleocr("document.pdf")
        assert paddlex_helpers.model_in_flight_count() == 0

    def test_run_paddlex_structure_v3_holds_guard_across_convert_and_predict(
        self, monkeypatch
    ):
        """run_paddlex_structure_v3 is the primary pipeline for the worker
        sidecar and the server /layout-parsing + /extract-text endpoints: the
        guard must be held across model fetch, the PDF→image conversion, and
        every per-page predict. Dropping the wrap on this path would let the
        idle thread destroy the pipeline mid-job with no failing test."""
        import paddlex_helpers

        seen: dict[str, int] = {}

        class _SpyModel:
            def predict(self, _path: str):
                seen["predict"] = paddlex_helpers.model_in_flight_count()
                yield {
                    "overall_ocr_res": {
                        "rec_texts": [],
                        "rec_scores": [],
                        "rec_boxes": [],
                    }
                }

        def fake_pdf_to_images(_pdf_path: str, _tmp_dir: str) -> list[str]:
            seen["convert"] = paddlex_helpers.model_in_flight_count()
            return ["page-1.png"]

        monkeypatch.setattr(
            paddlex_helpers, "_get_paddlex_structure_v3_model", _SpyModel
        )
        monkeypatch.setattr(paddlex_helpers, "_pdf_to_images", fake_pdf_to_images)

        paddlex_helpers.run_paddlex_structure_v3("document.pdf")

        assert seen["convert"] == 1  # held across the pdftoppm conversion
        assert seen["predict"] == 1  # held during per-page predict
        assert paddlex_helpers.model_in_flight_count() == 0  # released after

    def test_run_paddlex_structure_v3_releases_guard_on_exception(self, monkeypatch):
        import paddlex_helpers

        class _BoomModel:
            def predict(self, _path: str):
                raise RuntimeError("structure predictor exploded")
                yield  # pragma: no cover — makes this a generator

        monkeypatch.setattr(
            paddlex_helpers, "_get_paddlex_structure_v3_model", _BoomModel
        )
        monkeypatch.setattr(
            paddlex_helpers, "_pdf_to_images", lambda _p, _d: ["page-1.png"]
        )

        with pytest.raises(RuntimeError, match="structure predictor exploded"):
            paddlex_helpers.run_paddlex_structure_v3("document.pdf")
        assert paddlex_helpers.model_in_flight_count() == 0
