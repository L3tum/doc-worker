"""
Doc-Worker — Unit tests for paddlex_helpers.

Tests blocks_to_markdown, run_paddlex_ocr (with mocks), and model validation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from paddlex_helpers import (
    DEFAULT_PDFTOPPM_TIMEOUT,
    LEGACY_DPI,
    RENDER_DPI,
    _sort_blocks_by_reading_order,
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

    def test_out_of_order_input_preserved(self):
        """No-sort contract: input order is taken as-is (NO sorting).

        A text block lower on the page comes FIRST in the input; a title
        block higher on the page comes second. A (re-)introduced sort
        would emit the heading first; the pure serializer keeps the given
        order (paragraph line before the heading line).
        """
        blocks = [
            {
                "type": "text",
                "text": "Lower paragraph first.",
                "bbox": [10, 500, 400, 540],
                "confidence": 0.9,
            },
            {
                "type": "title",
                "text": "Upper title second",
                "bbox": [10, 50, 300, 80],
                "confidence": 0.95,
            },
        ]
        result = blocks_to_markdown(blocks)
        assert result.index("Lower paragraph first.") < result.index(
            "# Upper title second"
        )

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
        order = [b["text"] for b in _sort_blocks_by_reading_order(blocks)]
        # No span (all widths ≤ 70% of page width 400) and two column
        # clusters (left edges 50 / 300, gap > 50px) → the whole page is
        # one band read column-first: Title and Left block (left column,
        # top→bottom) before Right block (right column). The old (y, col)
        # key produced Title, Right, Left (top-to-bottom interleaving).
        assert order == ["Title", "Left block", "Right block"]

    def test_reading_order_multi_column(self):
        """Test multi-column layout: the entire left column reads before the entire right column."""
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
        # Two column clusters (left edges 20 / 350) and no spans → one
        # band read column-first: Left col 1, Left col 2, then Right col
        # 1, Right col 2 (top→bottom within each column). The old (y, col)
        # key interleaved L1, R1, L2, R2.
        assert [b["text"] for b in _sort_blocks_by_reading_order(blocks)] == [
            "Left col 1",
            "Left col 2",
            "Right col 1",
            "Right col 2",
        ]

    def test_reading_order_spanning_title_first(self):
        """A full-width (span) title above two columns stays first."""
        blocks = [
            {
                "type": "title",
                "text": "Report Title",
                "bbox": [50, 20, 550, 60],  # width 500 > 70% of 550 → span
                "confidence": 0.99,
            },
            {
                "type": "text",
                "text": "Left one",
                "bbox": [20, 100, 200, 180],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "Right one",
                "bbox": [350, 100, 530, 180],
                "confidence": 0.9,
            },
        ]
        # Span band-splitting: title first, then band 1 column-first.
        assert [b["text"] for b in _sort_blocks_by_reading_order(blocks)] == [
            "Report Title",
            "Left one",
            "Right one",
        ]

    def test_reading_order_span_between_bands(self):
        """A full-width table span between two column bands stays in place."""
        table_html = (
            "<html><body><table><tr><td>MIDTABLE</td></tr></table></body></html>"
        )
        blocks = [
            {
                "type": "text",
                "text": "Left one",
                "bbox": [20, 20, 200, 100],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "Right one",
                "bbox": [350, 20, 530, 100],
                "confidence": 0.9,
            },
            {
                "type": "table",
                "text": table_html,
                "bbox": [0, 150, 600, 180],  # width 600 > 70% of 600 → span
                "confidence": 0.95,
            },
            {
                "type": "text",
                "text": "Left two",
                "bbox": [20, 200, 200, 280],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "Right two",
                "bbox": [350, 200, 530, 280],
                "confidence": 0.9,
            },
        ]
        # Band 0 (L1, R1) → span → band 1 (L2, R2): the table is emitted
        # between the bands, not dragged into a column by its left edge.
        assert [b["text"] for b in _sort_blocks_by_reading_order(blocks)] == [
            "Left one",
            "Right one",
            table_html,
            "Left two",
            "Right two",
        ]

    def test_reading_order_full_width_footer_stays(self):
        """A full-width footer below two columns is emitted after both columns."""
        blocks = [
            {
                "type": "text",
                "text": "Left one",
                "bbox": [20, 20, 200, 100],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "Right one",
                "bbox": [350, 20, 530, 100],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "Footer text",
                "bbox": [0, 300, 600, 320],  # width 600 > 70% of 600 → span
                "confidence": 0.9,
            },
        ]
        assert [b["text"] for b in _sort_blocks_by_reading_order(blocks)] == [
            "Left one",
            "Right one",
            "Footer text",
        ]

    def test_reading_order_narrow_cells_column_blocks(self):
        """Word-style LEFT|RIGHT split: full-width header + two ~0.45-width cells.

        The cells are non-spans (width < 70% of page) and read as column
        blocks: header first, then the full left cell, then the full right
        cell (not interleaved row-by-row).
        """
        blocks = [
            {
                "type": "text",
                "text": "WORDHEADER",
                "bbox": [0, 10, 600, 40],  # width 600 > 70% of 600 → span
                "confidence": 0.99,
            },
            {
                "type": "text",
                "text": "WORDLEFT",
                "bbox": [20, 60, 300, 140],  # width 280 ≈ 0.47 of 600 → non-span
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "WORDRIGHT",
                "bbox": [320, 60, 600, 140],  # width 280 → non-span
                "confidence": 0.9,
            },
        ]
        assert [b["text"] for b in _sort_blocks_by_reading_order(blocks)] == [
            "WORDHEADER",
            "WORDLEFT",
            "WORDRIGHT",
        ]

    def test_reading_order_single_column_unchanged(self):
        """A single-column document stays in plain top-to-bottom order."""
        blocks = [
            {
                "type": "text",
                "text": "Para two",
                "bbox": [55, 200, 165, 260],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "Para one",
                "bbox": [50, 20, 160, 80],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "Para three",
                "bbox": [52, 300, 162, 360],
                "confidence": 0.9,
            },
        ]
        # Left edges 50/55/52 within the default (DPI-scaled) threshold →
        # one column cluster → plain (y, x) sort, even though input is
        # shuffled.
        assert [b["text"] for b in _sort_blocks_by_reading_order(blocks)] == [
            "Para one",
            "Para two",
            "Para three",
        ]

    # ── H2: DPI-aware default column threshold ─────────────────────────
    def test_reading_order_hanging_indent_default_threshold(self):
        """Hanging-indent regression (H2): an 80 px indent must not look like a column.

        Single-column page at its TRUE width (page_width=2480, A4 @ 300
        dpi), three narrow blocks (~400 px wide, well under the 70% span
        limit): A is a paragraph line, C is A's hanging-indent
        continuation (80 px ≈ 6.8 mm at 300 dpi), B is the next paragraph
        line.

        Default threshold = 50 px * RENDER_DPI / 150 = 100 px at 300 dpi
        (≈ 8.5 mm — the physical scale of the old 50 px @ 150 dpi literal):
        the 80 px indent stays inside one column cluster → plain (y, x) →
        [A, C, B].

        Guard: with the old 50 px literal the 80 px indent splits a phantom
        second column and the column-first sort displaces the continuation
        line after the next paragraph: [A, B, C].
        """
        blocks = [
            {
                "type": "text",
                "text": "A",
                "bbox": [100, 100, 500, 140],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "C",
                "bbox": [180, 150, 580, 190],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "B",
                "bbox": [100, 200, 500, 240],
                "confidence": 0.9,
            },
        ]
        assert [
            b["text"] for b in _sort_blocks_by_reading_order(blocks, page_width=2480)
        ] == ["A", "C", "B"]
        # Old 50 px literal (≈ 4.2 mm at 300 dpi) mis-splits the indent.
        assert [
            b["text"]
            for b in _sort_blocks_by_reading_order(
                blocks, column_threshold=50.0, page_width=2480
            )
        ] == ["A", "B", "C"]

    # ── H4: span heuristic must use the true page width ─────────────────
    def test_reading_order_span_boundary_true_width(self):
        """Span limit must use the TRUE page width, not the content extent.

        page_width=1000 → span limit 700 px: A (710 px = 71%) is a span,
        W (690 px = 69%) is NOT. Expected: A first, then band 1 read
        column-first (left column L1, W, L2 top→bottom, then right column
        R1).

        Pins the fix: the old content-extent heuristic measures 950 px here
        (R1's right edge) → limit 665 px → W also classifies as a span and
        splits the bands, yielding [A, L1, R1, W, L2] (W drags L2 after
        itself).
        """
        blocks = [
            {"type": "text", "text": "A", "bbox": [0, 0, 710, 40], "confidence": 0.9},
            {
                "type": "text",
                "text": "W",
                "bbox": [0, 200, 690, 240],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "L1",
                "bbox": [0, 100, 400, 140],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "L2",
                "bbox": [0, 300, 400, 340],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "R1",
                "bbox": [900, 100, 950, 140],
                "confidence": 0.9,
            },
        ]
        assert [
            b["text"] for b in _sort_blocks_by_reading_order(blocks, page_width=1000)
        ] == ["A", "L1", "W", "L2", "R1"]

    def test_reading_order_span_boundary_pair_69_vs_71(self):
        """At page_width=1000, a width-710 block classifies as a span, width-690 does not.

        Minimal layout where BOTH classifications are observable:
        - if T71 (width 710 = 71%) were NOT a span, the whole page would
          be one column-first band and T71 (left column, topmost) would be
          emitted BEFORE the top-right narrow block;
        - if N69 (width 690 = 69%) WERE a span, it would split a band and
          the right-column block below it (RMid) would be emitted BEFORE
          it.
        Expected: RTop first (band 0, above the span), T71 (span), then
        band 1 column-first (N69 in the left column before RMid on the
        right).
        """
        blocks = [
            {
                "type": "text",
                "text": "RTop",
                "bbox": [900, 20, 950, 60],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "T71",
                "bbox": [0, 100, 710, 140],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "N69",
                "bbox": [0, 250, 690, 290],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "RMid",
                "bbox": [900, 180, 950, 220],
                "confidence": 0.9,
            },
        ]
        assert [
            b["text"] for b in _sort_blocks_by_reading_order(blocks, page_width=1000)
        ] == ["RTop", "T71", "N69", "RMid"]

    def test_reading_order_fallback_content_extent(self):
        """page_width=None reproduces the pre-true-width content-extent heuristic.

        Same layout as test_reading_order_span_boundary_true_width but with
        no true width: the extent is R1's right edge (950) → limit 665 px →
        W (690 px) ALSO classifies as a span and splits the bands:
        [A, L1, R1, W, L2] — vs [A, L1, W, L2, R1] with page_width=1000.
        """
        blocks = [
            {"type": "text", "text": "A", "bbox": [0, 0, 710, 40], "confidence": 0.9},
            {
                "type": "text",
                "text": "W",
                "bbox": [0, 200, 690, 240],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "L1",
                "bbox": [0, 100, 400, 140],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "L2",
                "bbox": [0, 300, 400, 340],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "R1",
                "bbox": [900, 100, 950, 140],
                "confidence": 0.9,
            },
        ]
        assert [b["text"] for b in _sort_blocks_by_reading_order(blocks)] == [
            "A",
            "L1",
            "R1",
            "W",
            "L2",
        ]

    # ── Review T2: branch coverage ──────────────────────────────────────
    def test_reading_order_single_cluster_band_fallback(self):
        """Two column clusters overall, but one band whose blocks all lie within
        the column threshold → that band sorts plain (y, x).

        The page has two column clusters overall (left edges 0 / 600 in
        band 0). Below the span, R1 (x=100) and R2 (x=170) are 70 px apart
        (≤ the default 100 px threshold at 300 dpi): they share one
        cluster, so their band is emitted plain (y, x). R2 is listed first
        in the input to prove the band is sorted, not passed through.
        """
        blocks = [
            {"type": "text", "text": "Q1", "bbox": [0, 20, 150, 60], "confidence": 0.9},
            {
                "type": "text",
                "text": "Q2",
                "bbox": [600, 30, 750, 70],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "T",
                "bbox": [0, 100, 710, 140],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "R2",
                "bbox": [170, 210, 320, 250],
                "confidence": 0.9,
            },
            {
                "type": "text",
                "text": "R1",
                "bbox": [100, 200, 250, 240],
                "confidence": 0.9,
            },
        ]
        assert [
            b["text"] for b in _sort_blocks_by_reading_order(blocks, page_width=1000)
        ] == ["Q1", "Q2", "T", "R1", "R2"]

    def test_reading_order_degenerate_overlap_joins_band_below(self):
        """A block whose top edge falls INSIDE a span's y-range (overlapping model
        output) joins the band BELOW that span → emitted after it.

        Span S occupies y=0..50; B's top edge y=25 is inside that range.
        Deterministic degenerate case: B joins band 1 (below S), so the
        order is [S, B, C], not [B, S, C].
        """
        blocks = [
            {"type": "text", "text": "S", "bbox": [0, 0, 710, 50], "confidence": 0.9},
            {"type": "text", "text": "B", "bbox": [0, 25, 300, 70], "confidence": 0.9},
            {"type": "text", "text": "C", "bbox": [0, 60, 300, 100], "confidence": 0.9},
        ]
        assert [
            b["text"] for b in _sort_blocks_by_reading_order(blocks, page_width=1000)
        ] == ["S", "B", "C"]

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


# ── DPI-scaled defaults ─────────────────────────────────────────────────────
class TestRenderDpiDefaults:
    """The pixel-scaled defaults derive from one LEGACY_DPI base."""

    def test_pdftoppm_timeout_scales_from_legacy_dpi(self):
        """DEFAULT_PDFTOPPM_TIMEOUT scales off LEGACY_DPI, not a hardcoded 150."""
        assert DEFAULT_PDFTOPPM_TIMEOUT == round(120 * (RENDER_DPI / LEGACY_DPI) ** 2)
        # True at RENDER_DPI=300, LEGACY_DPI=150.
        assert DEFAULT_PDFTOPPM_TIMEOUT == 480

    def test_render_dpi_env_override_drives_pdftoppm(self, tmp_path, monkeypatch):
        """RENDER_DPI (env-overridable at import) is what pdftoppm renders
        at — the `-r` argument must carry the current value."""
        import subprocess as sp

        import paddlex_helpers

        monkeypatch.setattr(paddlex_helpers, "RENDER_DPI", 200)
        (tmp_path / "page-1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        captured: dict = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return sp.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(paddlex_helpers.subprocess, "run", fake_run)
        images = paddlex_helpers._pdf_to_images(str(tmp_path / "in.pdf"), str(tmp_path))

        assert images == [str(tmp_path / "page-1.png")]
        assert "200" in captured["cmd"]


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

    def test_null_fields_produce_empty_page(self, tmp_path):
        """F4: null OCR fields (not just missing keys) must not crash run_paddleocr."""
        img = tmp_path / "page.png"
        img.write_bytes(b"\x89PNG")
        page_result = {
            "rec_texts": None,
            "rec_scores": None,
            "rec_boxes": None,
            "rec_polys": None,
        }

        class _FakeModel:
            def predict(self, _path):
                return iter([page_result])

        with patch("paddlex_helpers._get_paddlex_model", return_value=_FakeModel()):
            pages = run_paddleocr(str(img))

        assert pages[0]["page"] == 1
        assert pages[0]["text"] == ""
        assert pages[0]["blocks"] == []


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

    def test_page_width_glue_true_image_end_to_end(
        self, mock_paddlex_structure_v3_pipeline, tmp_path
    ):
        """Real page image → _image_width_px → span sort + page_width_px.

        The sibling mocked tests use a nonexistent PNG path, so
        _image_width_px fails soft to None and a regression that always
        returns None would pass them all. Here the "rendered" page is a
        REAL 2480 px-wide PNG (A4 @ 300 dpi) and the mocked model emits a
        two-column layout where the TRUE width (2480 → span limit 1736 px)
        changes the reading order vs the content-extent fallback:

        - H (width 1900 px) is a span at true width; W (1600 px) is not.
          Content-extent fallback: extent = 2100 (R1's right edge) → span
          limit 1470 px → W is a span too, dragging R1 up before W:
          [H, L1, R1, W, L2].
        - True width: H first (wide span/header), then the remaining band
          read column-first — left column top→bottom (L1, W, L2), then the
          right column (R1): [H, L1, W, L2, R1].
        """
        pytest.importorskip("PIL")
        from PIL import Image

        png_path = tmp_path / "page-1.png"
        Image.new("RGB", (2480, 3508), "white").save(str(png_path))

        result_dict = {
            "overall_ocr_res": {
                "rec_texts": [
                    "Right col line",
                    "Wide header",
                    "Left col line 2",
                    "Wide body line",
                    "Left col line 1",
                ],
                "rec_scores": [0.9] * 5,
                "rec_boxes": [[0, 0, 100, 20]] * 5,
                "rec_polys": [],
            },
            # Deliberately out of reading order (raw model output order):
            # the sort inside _process_structure_v3_pages must reorder it.
            "parsing_res_list": [
                {
                    "block_label": "text",
                    "block_content": "Right col line",
                    "block_bbox": [2000, 100, 2100, 140],
                },
                {
                    "block_label": "text",
                    "block_content": "Wide header",
                    "block_bbox": [0, 0, 1900, 40],
                },
                {
                    "block_label": "text",
                    "block_content": "Left col line 2",
                    "block_bbox": [0, 300, 400, 340],
                },
                {
                    "block_label": "text",
                    "block_content": "Wide body line",
                    "block_bbox": [0, 200, 1600, 240],
                },
                {
                    "block_label": "text",
                    "block_content": "Left col line 1",
                    "block_bbox": [0, 100, 400, 140],
                },
            ],
            "layout_det_res": {"boxes": []},
        }

        # Same PaddleX-mocking pattern as the sibling tests, but with this
        # test's own model output and a REAL rendered page image.
        mock_paddlex_structure_v3_pipeline.predict.side_effect = lambda _path: iter(
            [result_dict]
        )
        with patch("paddlex_helpers._pdf_to_images", return_value=[str(png_path)]):
            pages = run_paddlex_structure_v3("/tmp/test.pdf")

        assert len(pages) == 1
        # (The page's true width is still consumed internally by the sort;
        # page_width_px itself was an unused key in the page dict and is no
        # longer exposed. A real PNG above is what makes _image_width_px
        # succeed here — a missing path fails soft to the fallback.)
        # True width (2480) drives the span heuristic: H is the only span,
        # the remaining band is read column-first.
        assert [b["text"] for b in pages[0]["structured_blocks"]] == [
            "Wide header",
            "Left col line 1",
            "Wide body line",
            "Left col line 2",
            "Right col line",
        ]

    @pytest.mark.usefixtures(
        "mock_paddlex_structure_v3_pipeline",
        "mock_pdf_to_images",
    )
    @pytest.mark.parametrize(
        "degenerate_bbox",
        [[], [1.0], [1, None, 3, 4], "garbage", None],
        ids=["empty", "short", "none-element", "string", "none"],
    )
    def test_degenerate_block_bbox_fails_soft_to_zero_box(
        self, mock_paddlex_structure_v3_pipeline, degenerate_bbox
    ):
        """A degenerate model block_bbox must not 500 the whole request.

        The reading-order span heuristic indexes bbox[0]/bbox[2] (and takes
        the max right edge); an empty/short box, a None or non-numeric
        element, or a bare non-sequence used to raise IndexError/TypeError
        in _sort_blocks_by_reading_order. Extraction now normalizes such
        boxes to the zero box at the source, keeping the block's text.
        """
        import json

        result_dict = {
            "overall_ocr_res": {
                "rec_texts": ["Good", "Bad"],
                "rec_scores": [0.9, 0.9],
                "rec_boxes": [[0, 0, 100, 20], [0, 40, 100, 60]],
                "rec_polys": [],
            },
            # One well-formed box (passthrough assertion) plus the
            # parameterized degenerate one.
            "parsing_res_list": [
                {
                    "block_label": "text",
                    "block_content": "Good",
                    "block_bbox": [10.0, 20.0, 300.0, 60.0],
                },
                {
                    "block_label": "text",
                    "block_content": "Bad",
                    "block_bbox": degenerate_bbox,
                },
            ],
            "layout_det_res": {"boxes": []},
        }

        # Same PaddleX-mocking pattern as the sibling end-to-end tests,
        # with this test's own model output.
        mock_paddlex_structure_v3_pipeline.predict.side_effect = lambda _path: iter(
            [result_dict]
        )
        pages = run_paddlex_structure_v3("/tmp/test.pdf")

        assert len(pages) == 1
        by_text = {b["text"]: b for b in pages[0]["structured_blocks"]}
        # Both blocks survive extraction + the reading-order sort.
        assert set(by_text) == {"Good", "Bad"}
        # The well-formed box passes through unchanged (byte-identical).
        assert by_text["Good"]["bbox"] == [10.0, 20.0, 300.0, 60.0]
        # The degenerate box is replaced by the zero box at extraction.
        assert by_text["Bad"]["bbox"] == [0, 0, 0, 0]
        # The full page dict must serialize with NO default= (500 mode).
        json.dumps(pages)


# ── Numpy-typed structured_blocks serializability ─────────────────────────
class TestJsonable:
    """_jsonable must recursively scrub numpy types to plain Python values."""

    def test_ndarray(self):
        import numpy as np

        from paddlex_helpers import _jsonable

        out = _jsonable(np.array([1.5, 2.5], dtype=np.float32))
        assert out == [1.5, 2.5]
        assert all(type(v) is float for v in out)

    def test_list_of_numpy_scalars(self):
        """The production-incident shape: block_bbox as a list of np.float32."""
        import numpy as np

        from paddlex_helpers import _jsonable

        out = _jsonable([np.float32(1.5), np.float32(2.5)])
        assert out == [1.5, 2.5]
        assert all(type(v) is float for v in out)

    def test_nested_mixed(self):
        import numpy as np

        from paddlex_helpers import _jsonable

        out = _jsonable([np.float32(1.0), [np.int32(2)], {"k": np.float64(3.0)}])
        assert out == [1.0, [2], {"k": 3.0}]
        assert type(out[1][0]) is int
        assert type(out[2]["k"]) is float

    def test_numpy_scalar(self):
        import numpy as np

        from paddlex_helpers import _jsonable

        out = _jsonable(np.float32(0.5))
        assert out == 0.5
        assert type(out) is float

    def test_plain_types_pass_through(self):
        from paddlex_helpers import _jsonable

        assert _jsonable("s") == "s"
        assert _jsonable(1) == 1
        assert _jsonable(1.5) == 1.5
        assert _jsonable(True) is True
        assert _jsonable(None) is None

    def test_tuple_and_set(self):
        import numpy as np

        from paddlex_helpers import _jsonable

        assert _jsonable((np.float32(1.0), "x")) == [1.0, "x"]
        # Single-element set: deterministic order
        assert _jsonable({np.int32(2)}) == [2]

    def test_item_fallback(self):
        from paddlex_helpers import _jsonable

        class _ItemOnly:
            """numpy-like scalar exposing .item() but not .tolist()."""

            def __init__(self, value: float) -> None:
                self._value = value

            def item(self) -> float:
                return self._value

        assert _jsonable(_ItemOnly(3.5)) == 3.5


# ── bbox normalization ──────────────────────────────────────────────────────
class TestNormalizeBbox:
    """_normalize_bbox coerces any degenerate model box to the zero box."""

    def test_wellformed_list_passthrough(self):
        from paddlex_helpers import _normalize_bbox

        out = _normalize_bbox([1.0, 2.0, 3.0, 4.0])
        assert out == [1.0, 2.0, 3.0, 4.0]
        assert all(type(v) is float for v in out)

    def test_ndarray_passthrough(self):
        """The 4-element ndarray shape PaddleX emits for normal blocks."""
        import numpy as np

        from paddlex_helpers import _normalize_bbox

        out = _normalize_bbox(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert out == [1.0, 2.0, 3.0, 4.0]
        assert all(type(v) is float for v in out)

    def test_list_of_numpy_scalars_passthrough(self):
        """The production-incident shape: block_bbox as a list of np.float32."""
        import numpy as np

        from paddlex_helpers import _normalize_bbox

        out = _normalize_bbox(
            [np.float32(1.0), np.float32(2.0), np.float32(3.0), np.float32(4.0)]
        )
        assert out == [1.0, 2.0, 3.0, 4.0]
        assert all(type(v) is float for v in out)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, [0.0, 0.0, 0.0, 0.0]),
            ([], [0.0, 0.0, 0.0, 0.0]),
            ([1.0], [0.0, 0.0, 0.0, 0.0]),
            ([1, None, 3, 4], [0.0, 0.0, 0.0, 0.0]),
            (["a", 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]),
            ("garbage", [0.0, 0.0, 0.0, 0.0]),
            ("1234", [0.0, 0.0, 0.0, 0.0]),
            (5, [0.0, 0.0, 0.0, 0.0]),
            ([float("nan"), 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]),
            ([1.0, float("inf"), 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]),
            ({"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0}, [0.0, 0.0, 0.0, 0.0]),
        ],
        ids=[
            "none",
            "empty",
            "short",
            "none-element",
            "non-numeric-element",
            "string",
            "numeric-string",
            "bare-scalar",
            "nan-element",
            "inf-element",
            "dict",
        ],
    )
    def test_degenerate_inputs_fail_soft_to_zero_box(self, raw, expected):
        """Anything but exactly four finite numerics degrades to the zero box.

        NaN/Inf are valid floats but would 500 the response later (the
        server serializes with json.dumps(allow_nan=False)); a 4-key dict
        would otherwise yield its *keys* as the box.
        """
        from paddlex_helpers import _normalize_bbox

        out = _normalize_bbox(raw)
        assert out == expected
        assert all(type(v) is float for v in out)


class TestStructuredBlocksNumpySerializability:
    """Regression: PaddleX returns numpy float32 scores + np.array bboxes.

    The structured_blocks path in _process_structure_v3_pages must convert
    these to plain Python types so the sidecar JSON (worker.py) and the
    server.py /layout-parsing endpoint can serialize without default=.
    """

    def test_structured_blocks_numpy_types_json_serializable(self):
        import json

        import numpy as np

        from paddlex_helpers import _process_structure_v3_pages

        result_dict = {
            "overall_ocr_res": {
                "rec_texts": ["Hello"],
                "rec_scores": [np.float32(0.95)],
                "rec_boxes": [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)],
                "rec_polys": [],
            },
            "layout_det_res": {
                "boxes": [{"label": "text", "score": np.float32(0.90)}],
            },
            "parsing_res_list": [
                {
                    "block_label": "text",
                    "block_content": "Hello",
                    "block_bbox": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                }
            ],
        }

        class _FakeModel:
            def predict(self, _path):
                yield result_dict

        pages = _process_structure_v3_pages(_FakeModel(), ["/tmp/dummy.png"])
        assert len(pages) == 1
        sb = pages[0]["structured_blocks"]
        assert len(sb) == 1

        # confidence must be a Python float (not np.float32)
        assert type(sb[0]["confidence"]) is float
        # bbox must be a plain list (not an ndarray)
        assert isinstance(sb[0]["bbox"], list)
        assert all(type(v) is float for v in sb[0]["bbox"])

        # The full page dict must serialize with json.dumps and NO default=.
        # This is the exact failure mode from the production incident.
        json.dumps(pages)  # raises TypeError if any numpy type leaked

    def test_structured_blocks_list_of_numpy_scalars_json_serializable(self):
        """Production-incident shape: block_bbox as a list of np.float32.

        The ndarray shape is covered above; PaddleX also hands back
        block_bbox as a plain list of np.float32 scalars, which the old
        top-level-only _jsonable passed through unconverted and which
        crashed JSONResponse with "Object of type float32 is not JSON
        serializable".
        """
        import json

        import numpy as np

        from paddlex_helpers import _process_structure_v3_pages

        result_dict = {
            "overall_ocr_res": {
                "rec_texts": ["Hello", "World"],
                "rec_scores": [np.float32(0.95), np.float32(0.90)],
                "rec_boxes": [
                    np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                    [
                        np.float32(5.0),
                        np.float32(6.0),
                        np.float32(7.0),
                        np.float32(8.0),
                    ],
                ],
                "rec_polys": [],
            },
            "layout_det_res": {
                "boxes": [
                    {"label": "title", "score": np.float32(0.90)},
                    {"label": "text", "score": np.float32(0.85)},
                ],
            },
            "parsing_res_list": [
                {
                    "block_label": "title",
                    "block_content": "Hello",
                    "block_bbox": [
                        np.float32(1.0),
                        np.float32(2.0),
                        np.float32(3.0),
                        np.float32(4.0),
                    ],
                },
                {
                    "block_label": "text",
                    "block_content": "World",
                    "block_bbox": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
                },
            ],
        }

        class _FakeModel:
            def predict(self, _path):
                yield result_dict

        pages = _process_structure_v3_pages(_FakeModel(), ["/tmp/dummy.png"])
        assert len(pages[0]["structured_blocks"]) == 2
        assert len(pages[0]["blocks"]) == 2
        # Raw OCR bboxes must also be plain Python lists (ndarray and
        # list-of-numpy-scalar shapes alike)
        for block in pages[0]["blocks"]:
            assert isinstance(block["bbox"], list)
            assert all(type(v) is float for v in block["bbox"])
        # The full page dict must serialize with NO default= (incident mode)
        json.dumps(pages)


# ── Null-field tolerance (PaddleX emits null, not just missing keys) ─────
class TestNullTolerance:
    """PaddleX 3.x emits null for absent result sections.

    The `or {}` / `or []` guards keep the pipeline alive for that; a
    refactor back to `.get(..., default)` would silently regress these.
    """

    def test_null_page_fields_yield_empty_page(self):
        """F1: parsing_res_list=null used to 500 with 'NoneType' not iterable."""
        import json

        from paddlex_helpers import _process_structure_v3_pages

        result_dict = {
            "overall_ocr_res": None,
            "layout_det_res": None,
            "parsing_res_list": None,
        }

        class _FakeModel:
            def predict(self, _path):
                yield result_dict

        pages = _process_structure_v3_pages(_FakeModel(), ["/tmp/dummy.png"])
        assert pages[0]["text"] == ""
        assert pages[0]["blocks"] == []
        assert pages[0]["structured_blocks"] == []
        json.dumps(pages)

    def test_null_layout_score_defaults_to_zero(self):
        """F3: explicit-null layout score must not hit float(None)."""
        import json

        import numpy as np

        from paddlex_helpers import _process_structure_v3_pages

        result_dict = {
            "overall_ocr_res": {
                "rec_texts": ["Hello"],
                "rec_scores": [np.float32(0.9)],
                "rec_boxes": [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)],
                "rec_polys": [],
            },
            "layout_det_res": {
                "boxes": [
                    {"label": "title", "score": None},
                    {"label": "text"},  # missing score
                ],
            },
            "parsing_res_list": [
                {
                    "block_label": "title",
                    "block_content": "Hello",
                    "block_bbox": [1.0, 2.0, 3.0, 4.0],
                }
            ],
        }

        class _FakeModel:
            def predict(self, _path):
                yield result_dict

        pages = _process_structure_v3_pages(_FakeModel(), ["/tmp/dummy.png"])
        assert pages[0]["structured_blocks"][0]["confidence"] == 0.0
        json.dumps(pages)

    def test_non_finite_scores_degrade_to_finite(self):
        """NaN/Inf model scores must not poison JSON (allow_nan=False)."""
        import json

        import numpy as np

        from paddlex_helpers import _process_structure_v3_pages

        result_dict = {
            "overall_ocr_res": {
                "rec_texts": ["Hello"],
                "rec_scores": [np.float32("nan")],
                "rec_boxes": [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)],
                "rec_polys": [],
            },
            "layout_det_res": {
                "boxes": [{"label": "title", "score": np.float32("inf")}]
            },
            "parsing_res_list": [
                {
                    "block_label": "title",
                    "block_content": "Hello",
                    "block_bbox": [1.0, 2.0, 3.0, 4.0],
                }
            ],
        }

        class _FakeModel:
            def predict(self, _path):
                yield result_dict

        pages = _process_structure_v3_pages(_FakeModel(), ["/tmp/dummy.png"])
        assert pages[0]["blocks"][0]["confidence"] == 0.0
        assert pages[0]["structured_blocks"][0]["confidence"] == 0.0
        # Must survive strict JSON serialization (SafeJSONResponse settings)
        json.dumps(pages, allow_nan=False)


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
            "PP-LCNet_x1_0_doc_ori": "PP-LCNet_x1_0_doc_ori_infer",
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
            "PP-LCNet_x1_0_doc_ori": "PP-LCNet_x1_0_doc_ori_infer",
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


# ── Permanent error detection ─────────────────────────────────────────────
class TestPermanentErrorDetection:
    """Tests for _is_permanent_model_init_error()."""

    def test_detects_no_available_hosting(self):
        """'No available model hosting platforms' is permanent."""
        from paddlex_helpers import _is_permanent_model_init_error

        assert _is_permanent_model_init_error(
            RuntimeError(
                "No available model hosting platforms detected. Please check "
                "your network connection."
            )
        )

    def test_detects_already_initialized(self):
        """'already been initialized' is permanent."""
        from paddlex_helpers import _is_permanent_model_init_error

        assert _is_permanent_model_init_error(
            RuntimeError("PDX has already been initialized")
        )

    def test_network_timeout_is_transient(self):
        """Generic network errors are NOT permanent."""
        from paddlex_helpers import _is_permanent_model_init_error

        assert not _is_permanent_model_init_error(RuntimeError("network timeout"))

    def test_file_not_found_is_transient(self):
        """File errors are NOT permanent (could be temporary)."""
        from paddlex_helpers import _is_permanent_model_init_error

        assert not _is_permanent_model_init_error(
            FileNotFoundError("Model file not found")
        )

    def test_generic_exception_is_transient(self):
        """Unknown exceptions default to transient (retryable)."""
        from paddlex_helpers import _is_permanent_model_init_error

        assert not _is_permanent_model_init_error(
            RuntimeError("something unexpected happened")
        )


# ── _ensure_paddlex_offline_compat ImportError branch ─────────────────────
class TestEnsurePaddlexOfflineCompatImportError:
    """The ImportError branch of _ensure_paddlex_offline_compat (PaddleX not
    installed / broken install) is deterministic: log the cause and degrade
    (patch disabled) — no crash. The model factory calls it on every init
    attempt, so repeated calls must stay soft."""

    def test_import_error_is_logged_and_soft(self, caplog):
        import importlib
        import logging

        import paddlex_helpers

        if importlib.util.find_spec("paddlex") is not None:
            pytest.skip("paddlex installed — ImportError branch unreachable")

        with caplog.at_level(logging.WARNING, logger="doc-worker.paddlex_helpers"):
            paddlex_helpers._ensure_paddlex_offline_compat()
            paddlex_helpers._ensure_paddlex_offline_compat()  # repeated call: still soft

        assert any(
            "PaddleX import failed" in record.message for record in caplog.records
        )
