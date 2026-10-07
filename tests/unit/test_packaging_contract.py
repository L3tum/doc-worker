from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_requirements_use_paddlex_and_paddlepaddle():
    """Verify requirements.txt uses PaddleX instead of standalone paddleocr.

    The PaddleX pin must stay exactly in sync with the paddlex_helpers
    get_word_info patch (which targets the PaddleX 3.2.0-3.7.2 source):
    bumping the pin requires re-running the e2e-ocr gate.
    """
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "paddlex[base]==3.7.2" in requirements
    assert "paddlepaddle==3.3.0" in requirements
    assert "paddleocr" not in requirements.lower()


def test_dockerfile_validates_paddlex_model_names():
    """Verify Dockerfile validates PaddleX model names (including PP-StructureV3)."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "sed -i" not in dockerfile
    assert "PP-OCRv6_medium_det_infer PP-OCRv6_medium_det" in dockerfile
    assert "PP-OCRv6_medium_rec_infer PP-OCRv6_medium_rec" in dockerfile
    assert "PP-LCNet_x1_0_textline_ori_infer PP-LCNet_x1_0_textline_ori" in dockerfile
    assert "PP-DocLayout-L_infer PP-DocLayout-L" in dockerfile
