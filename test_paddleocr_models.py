"""
Tests for PaddleOCR model validation — catches the "model name mismatch" bug
before it reaches production.
"""

import re

import pytest

from paddleocr_helpers import (
    TEXT_DETECTION_MODEL,
    TEXT_RECOGNITION_MODEL,
    TEXTLINE_ORIENTATION_MODEL,
    _model_dir,
    validate_paddleocr_models,
    _get_paddleocr_model,
)

# ── Test 1: inference.yml model_name matches directory name ─────────────


@pytest.mark.parametrize(
    "model_name",
    [
        TEXT_DETECTION_MODEL,
        TEXT_RECOGNITION_MODEL,
        TEXTLINE_ORIENTATION_MODEL,
    ],
)
def test_model_name_matches_directory(model_name: str) -> None:
    """Each model's inference.yml must declare a model_name equal to its directory.

    This test catches the exact bug that caused 'model name mismatch' errors
    in production when the server-side tarball updated the model name inside
    inference.yml but the directory remained the same (or vice versa).
    """
    model_dir = _model_dir(model_name)
    assert model_dir.is_dir(), f"Model directory missing: {model_dir}"

    yml_path = model_dir / "inference.yml"
    assert yml_path.is_file(), f"Missing inference.yml in {model_dir}"

    yml_content = yml_path.read_text(encoding="utf-8")
    match = re.search(r"^  model_name:\s*(.+)$", yml_content, re.MULTILINE)
    assert match, f"Could not find 'model_name' line in {yml_path}"

    yml_model_name = match.group(1).strip()
    assert (
        yml_model_name == model_name
    ), f"Model name mismatch: inference.yml declares '{yml_model_name}' but directory is '{model_name}'"


# ── Test 2: full validation passes ────────────────────────────────────


def test_validate_paddleocr_models_passes() -> None:
    """validate_paddleocr_models() should not raise for the bundled models.

    If this fails, the worker will crash on startup, so we want to know.
    """
    validate_paddleocr_models()  # should not raise


# ── Test 3: PaddleOCR singleton loads without "model name mismatch" ───


def test_paddleocr_model_loads() -> None:
    """_get_paddleocr_model() should return without error.

    This catches the full cascade: validation passes, but PaddleOCR itself
    still rejects the models because of a subtle difference.
    """
    model = _get_paddleocr_model()
    assert model is not None
