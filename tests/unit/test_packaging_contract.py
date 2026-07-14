from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_requirements_use_paddleocr_version_that_supports_ppocrv6():
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "ocrmypdf-paddleocr" not in requirements
    assert "paddleocr>=3.7.0,<3.8.0" in requirements


def test_dockerfile_does_not_patch_inference_yml_to_infer_directory_names():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "sed -i" not in dockerfile
    assert "PP-OCRv6_medium_det_infer PP-OCRv6_medium_det" in dockerfile
    assert "PP-OCRv6_medium_rec_infer PP-OCRv6_medium_rec" in dockerfile
    assert "PP-LCNet_x1_0_textline_ori_infer PP-LCNet_x1_0_textline_ori" in dockerfile
