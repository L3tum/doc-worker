"""
Doc-Worker — Unit tests for paddlex_helpers.

Tests blocks_to_markdown, run_paddlex_ocr (with mocks), and model validation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from paddlex_helpers import (
    blocks_to_markdown,
    paddleocr_lang_code,
    run_paddleocr,
    run_paddlex_structure_v3,
    validate_paddlex_models,
)


# ── Language mapping tests ────────────────────────────────────────────────
class TestLanguageMapping:
    def test_german(self):
        with patch("paddlex_helpers.OCR_LANG", "deu"):
            assert paddleocr_lang_code() == "german"

    def test_english(self):
        with patch("paddlex_helpers.OCR_LANG", "eng"):
            assert paddleocr_lang_code() == "en"

    def test_french(self):
        with patch("paddlex_helpers.OCR_LANG", "fra"):
            assert paddleocr_lang_code() == "french"

    def test_chinese(self):
        with patch("paddlex_helpers.OCR_LANG", "chs"):
            assert paddleocr_lang_code() == "ch"

    def test_fallback(self):
        with patch("paddlex_helpers.OCR_LANG", "xyz"):
            assert paddleocr_lang_code() == "en"


# ── Blocks to markdown tests ──────────────────────────────────────────────
class TestBlocksToMarkdown:
    def test_title_block(self):
        blocks = [
            {
                "type": "title",
                "text": "Main Title",
                "bbox": [0, 0, 0, 0],
                "confidence": 0.99,
            }
        ]
        result = blocks_to_markdown(blocks)
        assert "# Main Title" in result

    def test_paragraph_title_block(self):
        blocks = [
            {
                "type": "paragraph_title",
                "text": "Section 1",
                "bbox": [0, 0, 0, 0],
                "confidence": 0.95,
            }
        ]
        result = blocks_to_markdown(blocks)
        assert "## Section 1" in result

    def test_text_block(self):
        blocks = [
            {
                "type": "text",
                "text": "Some paragraph text here.",
                "bbox": [0, 0, 0, 0],
                "confidence": 0.92,
            }
        ]
        result = blocks_to_markdown(blocks)
        assert "Some paragraph text here." in result

    def test_figure_title_block(self):
        """A figure_title on its own (without a following image) is rendered as italic text."""
        blocks = [
            {
                "type": "figure_title",
                "text": "Fig. 1: Diagram",
                "bbox": [0, 0, 0, 0],
                "confidence": 0.90,
            }
        ]
        result = blocks_to_markdown(blocks)
        # figure_title alone is now rendered as italic caption
        assert "*Fig. 1: Diagram*" in result

    def test_image_block_after_figure_title(self):
        blocks = [
            {
                "type": "figure_title",
                "text": "Figure caption",
                "bbox": [0, 0, 0, 0],
                "confidence": 0.95,
            },
            {"type": "image", "text": "", "bbox": [0, 0, 0, 0], "confidence": 0.99},
        ]
        result = blocks_to_markdown(blocks)
        # Only the image placeholder with alt text should appear (no duplicate italic)
        assert "![Figure caption]" in result
        assert "*Figure caption*" not in result

    def test_image_block_without_figure_title(self):
        blocks = [
            {"type": "image", "text": "", "bbox": [0, 0, 0, 0], "confidence": 0.99},
        ]
        result = blocks_to_markdown(blocks)
        assert "[Image]" in result

    def test_table_block_html(self):
        blocks = [
            {
                "type": "table",
                "text": "<html><body><table><tr><td>Cell1</td><td>Cell2</td></tr><tr><td>Cell3</td><td>Cell4</td></tr></table></body></html>",
                "bbox": [0, 0, 0, 0],
                "confidence": 0.95,
            }
        ]
        result = blocks_to_markdown(blocks)
        # The converter should produce a markdown table
        assert "|" in result
        assert "Cell1" in result
        assert "Cell2" in result
        assert "Cell3" in result
        assert "Cell4" in result

    def test_table_block_html_with_colspan(self):
        blocks = [
            {
                "type": "table",
                "text": '<html><body><table><tr><td colspan="2">Header</td></tr><tr><td>A</td><td>B</td></tr></table></body></html>',
                "bbox": [0, 0, 0, 0],
                "confidence": 0.95,
            }
        ]
        result = blocks_to_markdown(blocks)
        assert "Header" in result
        assert "A" in result
        assert "B" in result

    def test_table_block_plain_text(self):
        """If table text is not HTML, it's rendered as plain text."""
        blocks = [
            {
                "type": "table",
                "text": "Name\tAge\nAlice\t30",
                "bbox": [0, 0, 0, 0],
                "confidence": 0.95,
            }
        ]
        result = blocks_to_markdown(blocks)
        assert "Name" in result
        assert "Alice" in result

    def test_reading_order_sort(self):
        blocks = [
            {
                "type": "text",
                "text": "Right block",
                "bbox": [300, 50, 400, 100],
                "confidence": 0.9,
            },
            {
                "type": "title",
                "text": "Title",
                "bbox": [50, 50, 200, 70],
                "confidence": 0.99,
            },
            {
                "type": "text",
                "text": "Left block",
                "bbox": [50, 100, 200, 150],
                "confidence": 0.95,
            },
        ]
        result = blocks_to_markdown(blocks)
        # Title (y=50, x=50) should come first, then Right block (y=50, x=300), then Left block (y=100, x=50)
        assert result.index("Title") < result.index("Right block")
        assert result.index("Right block") < result.index("Left block")

    def test_reading_order_multi_column(self):
        """Test multi-column layout: left column items before right column at same height."""
        blocks = [
            {
                "type": "text",
                "text": "Right col 1",
                "bbox": [350, 20, 500, 100],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "Left col 1",
                "bbox": [20, 20, 200, 100],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "Left col 2",
                "bbox": [20, 110, 200, 200],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "Right col 2",
                "bbox": [350, 110, 500, 200],
                "confidence": 0.9,
            },
        ]
        result = blocks_to_markdown(blocks)
        # Left col 1 should come before Right col 1, and Left col 2 before Right col 2
        assert result.index("Left col 1") < result.index("Right col 1")
        assert result.index("Left col 2") < result.index("Right col 2")
        # Columns interleave: Left1, Right1, Left2, Right2 (same y ranges)
        assert result.index("Right col 1") < result.index("Left col 2")

    def test_empty_blocks(self):
        result = blocks_to_markdown([])
        assert result == ""

    def test_mixed_blocks(self):
        blocks = [
            {
                "type": "title",
                "text": "Title",
                "bbox": [0, 0, 0, 0],
                "confidence": 0.99,
            },
            {
                "type": "text",
                "text": "Paragraph",
                "bbox": [0, 20, 0, 0],
                "confidence": 0.95,
            },
            {
                "type": "paragraph_title",
                "text": "Subtitle",
                "bbox": [0, 40, 0, 0],
                "confidence": 0.92,
            },
        ]
        result = blocks_to_markdown(blocks)
        assert "# Title" in result
        assert "Paragraph" in result
        assert "## Subtitle" in result


# ── OCR run function tests (with mocks) ───────────────────────────────────
class TestRunPaddlexOcr:
    @pytest.mark.usefixtures("mock_paddlex_ocr_pipeline", "mock_create_pipeline")
    def test_run_paddleocr_basic(self):
        """Test that run_paddleocr returns expected format."""
        pages = run_paddleocr("/tmp/test.pdf")
        assert len(pages) == 1
        assert pages[0]["page"] == 1
        assert pages[0]["text"] == "Hello World\nThis is a test"
        assert len(pages[0]["blocks"]) == 2  # two non-empty text blocks


# ── Structure V3 run function tests (with mocks) ──────────────────────────
class TestRunPaddlexStructureV3:
    @pytest.mark.usefixtures(
        "mock_paddlex_structure_v3_pipeline",
        "mock_pdf_to_images",
    )
    def test_run_structure_v3(self):
        """Test that run_paddlex_structure_v3 returns structured blocks."""
        pages = run_paddlex_structure_v3("/tmp/test.pdf")
        assert len(pages) == 1
        assert "Title of Document" in pages[0]["text"]
        assert len(pages[0]["structured_blocks"]) == 3
        assert pages[0]["structured_blocks"][0]["type"] == "title"
        assert pages[0]["structured_blocks"][1]["type"] == "text"
        assert pages[0]["structured_blocks"][2]["type"] == "paragraph_title"

    @pytest.mark.usefixtures(
        "mock_paddlex_structure_v3_pipeline",
        "mock_pdf_to_images",
    )
    def test_run_structure_v3_pdf(self):
        """Test that PP-StructureV3 handles PDFs by converting pages to images."""
        pages = run_paddlex_structure_v3("/tmp/test.pdf")
        # The mock PDF has 1 page
        assert len(pages) == 1
        assert pages[0]["page"] == 1

    @pytest.mark.usefixtures(
        "mock_paddlex_structure_v3_pipeline",
        "mock_create_pipeline",
    )
    def test_run_structure_v3_image_only(self):
        """Test that PP-StructureV3 handles single image files directly (no PDF conversion)."""
        pages = run_paddlex_structure_v3("/tmp/test.png")
        assert len(pages) == 1
        assert pages[0]["page"] == 1
        assert "Title of Document" in pages[0]["text"]
        assert len(pages[0]["structured_blocks"]) == 3


# ── Model validation tests ────────────────────────────────────────────────
class TestValidatePaddlexModels:
    def test_valid_models(self):
        """Test validation passes when models are correctly set up."""
        import tempfile
        from pathlib import Path

        tmp_base = Path(tempfile.mkdtemp()) / "models"
        for model_name, dir_name in {
            "PP-OCRv6_medium_det": "PP-OCRv6_medium_det_infer",
            "PP-OCRv6_medium_rec": "PP-OCRv6_medium_rec_infer",
            "PP-LCNet_x1_0_textline_ori": "PP-LCNet_x1_0_textline_ori_infer",
            "PP-DocLayout-L": "PP-DocLayout-L_infer",
        }.items():
            model_dir = tmp_base / dir_name
            model_dir.mkdir(parents=True)
            (model_dir / "inference.pdiparams").write_text("")
            (model_dir / "inference.yml").write_text(f"model_name: {model_name}\n")
            (model_dir / "inference.json").write_text("{}")

        with patch("paddlex_helpers.PADDLEOCR_MODELS", str(tmp_base)):
            validate_paddlex_models()

    def test_missing_model(self):
        """Test validation fails when a model directory is missing."""
        import tempfile
        from pathlib import Path

        tmp_base = Path(tempfile.mkdtemp()) / "models"
        tmp_base.mkdir()
        # Only create one model
        model_dir = tmp_base / "PP-OCRv6_medium_det_infer"
        model_dir.mkdir()
        (model_dir / "inference.pdiparams").write_text("")
        (model_dir / "inference.yml").write_text("model_name: PP-OCRv6_medium_det\n")
        (model_dir / "inference.json").write_text("{}")

        with patch("paddlex_helpers.PADDLEOCR_MODELS", str(tmp_base)):
            with pytest.raises(FileNotFoundError) as exc_info:
                validate_paddlex_models()
            assert "missing" in str(exc_info.value).lower() or "Missing" in str(
                exc_info.value
            )

    def test_inference_yml_mismatch(self):
        """Test validation catches model name mismatches in inference.yml."""
        import tempfile
        from pathlib import Path

        tmp_base = Path(tempfile.mkdtemp()) / "models"
        model_dir = tmp_base / "PP-OCRv6_medium_det_infer"
        model_dir.mkdir(parents=True)
        (model_dir / "inference.pdiparams").write_text("")
        (model_dir / "inference.yml").write_text(
            "model_name: WRONG_NAME\n"
        )  # Mismatch!
        (model_dir / "inference.json").write_text("{}")

        # Add other models with correct names
        for model_name, dir_name in {
            "PP-OCRv6_medium_rec": "PP-OCRv6_medium_rec_infer",
            "PP-LCNet_x1_0_textline_ori": "PP-LCNet_x1_0_textline_ori_infer",
            "PP-DocLayout-L": "PP-DocLayout-L_infer",
        }.items():
            md = tmp_base / dir_name
            md.mkdir(parents=True)
            (md / "inference.pdiparams").write_text("")
            (md / "inference.yml").write_text(f"model_name: {model_name}\n")
            (md / "inference.json").write_text("{}")

        with patch("paddlex_helpers.PADDLEOCR_MODELS", str(tmp_base)):
            with pytest.raises(ValueError) as exc_info:
                validate_paddlex_models()
            assert "mismatch" in str(exc_info.value).lower()
