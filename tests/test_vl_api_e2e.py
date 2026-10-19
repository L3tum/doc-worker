"""
Doc-Worker — E2E tests for the PaddleOCR-VL compatible API (Open-WebUI).

Exercises the full request path end-to-end:

    HTTP request → server endpoint → real paddlex_helpers pipeline code
    → JSON response

Only the model boundary (``pipeline.predict``) and the PDF→image conversion
(``_pdf_to_images``) are mocked. The mock payloads are shaped like real
PaddleX output — numpy arrays and float32 scalars, including ``block_bbox``
as a *list* of np.float32. That is the exact payload class that caused the
production incident ("Object of type float32 is not JSON serializable");
the plain-Python mock fixtures used elsewhere in the suite cannot catch
that kind of regression.

Endpoints covered:
    POST /layout-parsing  — Open-WebUI PaddleOCR-VL contract
    POST /extract         — direct upload API
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import tempfile
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import paddlex_helpers
import server
from server import SafeJSONResponse, app


# ── Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture
def client() -> TestClient:
    """TestClient for the FastAPI app (no lifespan → no background threads)."""
    return TestClient(app)


def _pdf_with_pages(
    pages: list[tuple[tuple | None, tuple | None]],
    pages_media: tuple[float, float, float, float] | None = None,
    count: int | None = None,
) -> bytes:
    """Build an N-page minimal PDF with a REAL xref table — openable by
    BOTH pypdfium2 and pypdf (the old trailer-only fake was pypdfium2-
    tolerant but pypdf rejects it, and server._enforce_pdf_caps measures
    the per-page pixel cap with pypdf — the effective-box measurement
    pdftoppm renders). Each page spec is (mediabox 4-tuple or None = no
    /MediaBox entry, cropbox 4-tuple or None); ``pages_media``
    optionally puts a /MediaBox on the /Pages node (the page-tree-
    inheritance shape); ``count`` optionally lies about the page count
    in /Count (spec-optional, attacker-writable — the lying-/Count
    regression shape). Mirrors test_server.py::TestEnforcePdfCaps.
    _pdf_with_pages (mirrored by hand — the unit variant additionally
    accepts a RAW /CropBox entry string for the malformed-shape tests;
    dedup is a tracked follow-up)."""
    n = len(pages)
    pm = (
        f" /MediaBox [{pages_media[0]} {pages_media[1]} "
        f"{pages_media[2]} {pages_media[3]}]"
        if pages_media
        else ""
    )
    objs: dict[int, bytes] = {1: b"<< /Type /Catalog /Pages 2 0 R >>"}
    kids = " ".join(f"{3 + i} 0 R" for i in range(n))
    objs[2] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {count if count is not None else n}{pm} >>".encode()
    )
    for i, (mb, cb) in enumerate(pages):
        media = f" /MediaBox [{mb[0]} {mb[1]} {mb[2]} {mb[3]}]" if mb else ""
        crop = f" /CropBox [{cb[0]} {cb[1]} {cb[2]} {cb[3]}]" if cb else ""
        objs[3 + i] = f"<< /Type /Page{media}{crop} /Parent 2 0 R >>".encode()
    objs[3 + n] = b"<< /Length 20 >> stream\n01234567890123456789\nendstream"
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, len(objs) + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return bytes(out)


def _minimal_pdf(
    width: float | None,
    height: float | None,
    cropbox: tuple[float, float] | None = None,
    pages_media: tuple[float, float, float, float] | None = None,
    mediabox: tuple[float, float, float, float] | None = None,
    count: int | None = None,
) -> bytes:
    """Single-page minimal PDF (see _pdf_with_pages) with MediaBox
    ``mediabox`` (explicit 4-tuple, e.g. inverted corners) or the default
    ``[0 0 width height]`` (width/height None = no /MediaBox entry at
    all), an optional CropBox, an optional /Pages-node MediaBox, and an
    optional lying /Count (``count``). Valid enough for BOTH pypdfium2
    (page count) and pypdf (box measurement) to open — the request goes
    through the OCR path as binary."""
    if mediabox is not None:
        mb = mediabox
    elif width is not None:
        mb = (0.0, 0.0, float(width), float(height))
    else:
        mb = None
    cb = (0.0, 0.0, float(cropbox[0]), float(cropbox[1])) if cropbox else None
    return _pdf_with_pages([(mb, cb)], pages_media, count)


# Minimal 612x792 one-page PDF (xref-complete, see _pdf_with_pages) —
# passes _is_text_content() as binary, so the request goes through the
# OCR path instead of the text passthrough.
_FAKE_PDF = _minimal_pdf(612, 792)

_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _tiny_png(width: int, height: int) -> bytes:
    """Minimal valid PNG: 8-byte signature + IHDR (the DECLARED canvas)
    + IEND, no pixel data. Image.open() reads the size from the IHDR
    without decoding — exactly what the header-only image cap reads —
    and the pipeline is mocked in the cap tests, so the missing pixels
    are never touched."""
    import struct
    import zlib

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IEND", b"")


# ~50-byte PNG declaring a 100000x100000 px canvas (≈20 GB at decode):
# passes MAX_REQUEST_SIZE (tiny input) yet blows the default
# MAX_RENDERED_PIXELS envelope — the IMAGE equivalent of the
# huge-MediaBox PDFs below. Above PIL's decompression-bomb ERROR
# threshold (2 × MAX_IMAGE_PIXELS), so it exercises the cap's bomb
# branch; _tiny_png(13000, 13000) (below that threshold) exercises the
# direct w*h branch.
_HUGE_PNG = _tiny_png(100000, 100000)

# Real 2-page PDF fixture (verified: pypdfium2 counts 2 pages, by path and
# by raw bytes) for the page-cap tests that must count pages from the
# in-memory request bytes — the real count is the thing under test. (Same
# directory convention as test_server.py's fixtures/.) _FAKE_PDF above
# stays for the 1-page tests.
_TWOPAGE_PDF = (Path(__file__).parent / "fixtures" / "twopage.pdf").read_bytes()

# Three pages that LIE about their count: /Count declares 1, but the page
# tree has three real 612x792 kids. /Count is spec-optional and
# attacker-writable, and both parsers' count APIs are /Count-gated
# (pypdfium2's CountPages returns the raw /Count; pypdf does the same for
# encrypted documents) — the pre-fix page cap would let this ~600-byte PDF
# through at MAX_OCR_PAGES=1 (every /Count-gated consumer — poppler's
# render loop included, which is clamped to the raw /Count — sees only the
# first /Count pages; the concrete DoS residual is the page-size
# dimension, closed by the pixel cap). The cap must count the page tree
# (which pypdf walks), so the true count (3) is what 413s.
_LYING_COUNT_PDF = _pdf_with_pages([((0.0, 0.0, 612.0, 792.0), None)] * 3, count=1)

# The PARSER-DISAGREEMENT shape: a spec-legal PDF with a valid xref
# table + trailer but NO startxref section. pypdf's constructor raises
# PdfReadError("startxref not found") while pypdfium2/poppler read the
# xref table/trailer directly and open the same bytes. Pre-fix this
# failed SOFT in _enforce_pdf_caps (page cap on the /Count-gated count
# only, pixel cap skipped entirely) — page 1's 100000x100000 pt box
# would have rendered ~521 GB at 300 dpi from this ~300-byte file. The
# cap must 413 it fail-CLOSED ("Unparseable PDF") before slot +
# staging. (Unit-level siblings with self-verifying parser-disagreement
# checks: test_server.py::TestEnforcePdfCaps::
# test_pypdf_constructor_failure_413_unparseable / ::
# test_pypdf_constructor_failure_real_missing_startxref_413.)
_NO_STARTXREF_PDF = re.sub(
    rb"startxref\n\d+\n",
    b"",
    _pdf_with_pages(
        [
            ((0.0, 0.0, 100000.0, 100000.0), None),
            ((0.0, 0.0, 612.0, 792.0), None),
            ((0.0, 0.0, 612.0, 792.0), None),
        ]
    ),
)


# One page, MediaBox [0 0 100000 100000]: at 300 dpi that renders to
# ~1.7e11 px (≈521 GB of RGB) from a ~290-byte file — passes both
# MAX_REQUEST_SIZE and MAX_OCR_PAGES, so only the rendered-pixel cap
# rejects it. (612x792 = _FAKE_PDF's box, for comparison.)
_HUGE_PDF = _minimal_pdf(100000, 100000)

# The CROPBOX BYPASS variant of _HUGE_PDF: same huge MediaBox, but with a
# normal CropBox [0 0 612 792] set. pypdfium2's get_width/get_height
# report the CropBox (612x792 — a perfectly legal page), while pdftoppm
# (no -cropbox flag) renders the MediaBox (~1.7e11 px at 300 dpi). This is
# the shape that used to pass every cap: the regression tests below pin
# that the cap now measures the larger box.
_HUGE_CROP_PDF = _minimal_pdf(100000, 100000, cropbox=(612, 792))

# The PAGE-TREE BYPASS variant: no page-level MediaBox at all, a huge
# MediaBox on the /Pages NODE, and a normal CropBox on the page.
# pypdfium2's page-level box APIs do not walk /Parent (PDFium crbug 1786)
# — get_width()/get_mediabox() both report the 612x792 CropBox — while
# pdftoppm inherits the parent's MediaBox and renders ~1.7e11 px at 300
# dpi. This is the ~290-byte shape that passed every cap until the
# cap started measuring the effective box with pypdf (which resolves the
# same inheritance poppler does).
_HUGE_PARENT_MEDIA_PDF = _minimal_pdf(
    None, None, cropbox=(612, 792), pages_media=(0, 0, 100000, 100000)
)

# The INVERTED-CORNERS BYPASS variant: a page-level MediaBox with
# inverted corners [0 0 -100000 -100000] (spec-legal per PDF 32000-1)
# behind a normal CropBox. The old measurement took RAW corner deltas
# (negative here) while get_width() reported the CropBox; pdftoppm
# normalizes the corners and renders ~1.7e11 px at 300 dpi.
_INVERTED_MEDIA_PDF = _minimal_pdf(
    None, None, cropbox=(612, 792), mediabox=(0, 0, -100000, -100000)
)

# ── Degenerate page-tree shapes: pypdf's OWN guards refuse to measure
# them (depth > 100 / array elements > 100,000 / cycle — verified
# against pinned pypdf 6.21.1, which the cap's fail-CLOSED 413 depends
# on; see tests/test_server.py::TestEnforcePdfCaps._degenerate_tree_pdf,
# mirrored by hand — the docstring, one comment, and the `import io`
# placement differ, the builder body matches). The pre-fix fail-soft fallback (pdfium's
# /Count-gated count + pixel-cap skip) passed ALL of these; the huge
# first leaf is what made the pass a ~521 GB render. ──


def _degenerate_tree_pdf(shape: str, encrypted: bool = False) -> bytes:
    """Build a degenerate page-tree fixture (see the section comment
    above). ``encrypted`` = owner-only encryption (empty USER password
    — the renderable shape), pinning the ENCRYPTED branch (decrypt("")
    → _flatten(True)) to the same 413."""
    import io

    pytest.importorskip("pypdf")
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DictionaryObject,
        NameObject,
        NumberObject,
    )

    def _dict(*pairs: tuple[str, object]) -> DictionaryObject:
        d = DictionaryObject()
        for key, value in pairs:
            d[NameObject(key)] = value
        return d

    writer = PdfWriter()
    catalog = writer.root_object
    if shape == "deep":
        # 101 intermediate /Pages nodes in a chain, then one huge leaf
        # (pypdf's default depth limit is 100).
        node_refs = [
            writer._add_object(
                _dict(
                    ("/Type", NameObject("/Pages")),
                    ("/Count", NumberObject(1)),
                )
            )
            for _ in range(101)
        ]
        leaf = writer._add_object(
            _dict(
                ("/Type", NameObject("/Page")),
                ("/Parent", node_refs[-1]),
                (
                    "/MediaBox",
                    ArrayObject(
                        [
                            NumberObject(0),
                            NumberObject(0),
                            NumberObject(100000),
                            NumberObject(100000),
                        ]
                    ),
                ),
            )
        )
        for i, ref in enumerate(node_refs):
            nxt = node_refs[i + 1] if i + 1 < len(node_refs) else leaf
            ref.get_object()[NameObject("/Kids")] = ArrayObject([nxt])
        catalog[NameObject("/Pages")] = node_refs[0]
    elif shape == "wide":
        # One /Pages node with 100,001 kids (pypdf's array/element
        # limit is 100,000), first leaf huge, lying /Count 1.
        pages = writer._add_object(
            _dict(
                ("/Type", NameObject("/Pages")),
                ("/Count", NumberObject(1)),
            )
        )
        leaf_refs: list = []
        for i in range(100001):
            leaf_pairs = [
                ("/Type", NameObject("/Page")),
                ("/Parent", pages),
            ]
            if i == 0:
                leaf_pairs.append(
                    (
                        "/MediaBox",
                        ArrayObject(
                            [
                                NumberObject(0),
                                NumberObject(0),
                                NumberObject(100000),
                                NumberObject(100000),
                            ]
                        ),
                    )
                )
            leaf_refs.append(writer._add_object(_dict(*leaf_pairs)))
        pages.get_object()[NameObject("/Kids")] = ArrayObject(leaf_refs)
        catalog[NameObject("/Pages")] = pages
    elif shape == "cyclic":
        # /Pages cycle: A→B→A (no reachable leaves at all).
        d2 = writer._add_object(_dict(("/Type", NameObject("/Pages"))))
        d3 = writer._add_object(_dict(("/Type", NameObject("/Pages"))))
        d2.get_object()[NameObject("/Kids")] = ArrayObject([d3])
        d3.get_object()[NameObject("/Kids")] = ArrayObject([d2])
        d2.get_object()[NameObject("/Parent")] = d3
        d3.get_object()[NameObject("/Parent")] = d2
        catalog[NameObject("/Pages")] = d2
    else:
        raise ValueError(f"unknown shape: {shape!r}")
    if encrypted:
        # OWNER-only password (empty user password — the renderable
        # shape; decrypt("") succeeds).
        writer.encrypt(user_password="", owner_password="test-owner")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


_DEEP_TREE_PDF = _degenerate_tree_pdf("deep")
_WIDE_TREE_PDF = _degenerate_tree_pdf("wide")
_CYCLIC_TREE_PDF = _degenerate_tree_pdf("cyclic")
_DEEP_TREE_PDF_ENCRYPTED = _degenerate_tree_pdf("deep", encrypted=True)


def _recording_slot(
    slot_calls: list[int],
) -> Callable[[], AbstractAsyncContextManager[None]]:
    """Stand-in for server._ocr_job_slot: records every entry, never
    blocks. A job rejected BEFORE the slot (413 caps) must leave the
    list empty — the callers patch it in and assert that."""

    @asynccontextmanager
    async def _slot():
        slot_calls.append(1)
        yield

    return _slot


def _per_loop_parse_semaphores(
    capacity: int,
) -> tuple[
    Callable[[], asyncio.Semaphore], dict[asyncio.AbstractEventLoop, asyncio.Semaphore]
]:
    """Per-loop parse-slot semaphores (mirrors the production per-event-
    loop pattern — the TestClient may run one request under a fresh
    loop, and an asyncio.Semaphore binds to the loop that first uses it:
    a single shared instance across requests would raise on the second
    loop). Returns (factory for patching server._pdf_parse_semaphore,
    registry) — the registry lets the test inspect the admission state
    AFTER the request (e.g. that a 413 rejection left its loop's slot
    free; locked() is loop-independent, so it is readable from the test
    thread)."""
    registry: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}

    def _semaphore() -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = registry.get(loop)
        if sem is None:
            sem = asyncio.Semaphore(capacity)
            registry[loop] = sem
        return sem

    return _semaphore, registry


# ── Realistic PaddleX payloads (numpy-flavored, like production) ─────────
def _numpy_page_result(page_no: int) -> dict:
    """Build one PP-StructureV3 page result with real PaddleX data types.

    Mirrors production output: rec scores/layout scores as np.float32,
    rec boxes as ndarrays, and ``block_bbox`` as a *list* of np.float32
    scalars (the shape that broke JSON serialization in the incident).
    """
    return {
        "overall_ocr_res": {
            "rec_texts": [f"Title {page_no}", f"Body text {page_no}."],
            "rec_scores": [np.float32(0.99), np.float32(0.95)],
            "rec_boxes": [
                np.array([50.0, 50.0, 200.0, 70.0], dtype=np.float32),
                np.array([50.0, 90.0, 300.0, 130.0], dtype=np.float32),
            ],
            "rec_polys": [
                np.array([[50, 50], [200, 50], [200, 70], [50, 70]], dtype=np.int32),
                np.array([[50, 90], [300, 90], [300, 130], [50, 130]], dtype=np.int32),
            ],
        },
        "parsing_res_list": [
            {
                "block_label": "title",
                "block_content": f"Title {page_no}",
                "block_bbox": [
                    np.float32(50.0),
                    np.float32(50.0),
                    np.float32(200.0),
                    np.float32(70.0),
                ],
            },
            {
                "block_label": "text",
                "block_content": f"Body text {page_no}.",
                "block_bbox": np.array([50.0, 90.0, 300.0, 130.0], dtype=np.float32),
            },
        ],
        "layout_det_res": {
            "boxes": [
                {
                    "label": "title",
                    "score": np.float32(0.99),
                    "coordinate": np.array([50, 50, 200, 70], dtype=np.int32),
                },
                {
                    "label": "text",
                    "score": np.float32(0.97),
                    "coordinate": np.array([50, 90, 300, 130], dtype=np.int32),
                },
            ]
        },
    }


def _numpy_ocr_result(page_no: int) -> dict:
    """Build one General OCR result with numpy types (USE_STRUCTURE_V3=false)."""
    return {
        "rec_texts": [f"Plain OCR line {page_no}"],
        "rec_scores": [np.float32(0.97)],
        "rec_boxes": [np.array([10.0, 10.0, 100.0, 30.0], dtype=np.float32)],
        "rec_polys": [],
    }


class _FakeV3Model:
    """PP-StructureV3 pipeline stand-in: one predict() result per image."""

    def __init__(self, results: list[dict]) -> None:
        self._results = list(results)

    def predict(self, _path: str):
        if not self._results:
            return iter(())
        yield self._results.pop(0)


class _FakeOcrModel:
    """General OCR pipeline stand-in: all pages from one predict() call."""

    def __init__(self, results: list[dict]) -> None:
        self._results = list(results)

    def predict(self, _path: str):
        return iter(self._results)


# ── POST /layout-parsing — Open-WebUI PaddleOCR-VL contract ──────────────
class TestLayoutParsingE2E:
    def test_pdf_single_page_contract(self, client: TestClient) -> None:
        """1-page PDF → 200 with the Open-WebUI response contract.

        Regression for the production incident: a numpy-flavored PaddleX
        payload must produce valid JSON, not
        "TypeError: Object of type float32 is not JSON serializable".
        """
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={
                    "file": file_b64,
                    "fileType": 0,
                    "useDocOrientationClassify": True,
                    "useDocUnwarping": True,
                    "useChartRecognition": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        results = data["result"]["layoutParsingResults"]
        assert len(results) == 1
        # Open-WebUI reads markdown.text
        assert "# Title 1" in results[0]["markdown"]["text"]
        assert "Body text 1." in results[0]["markdown"]["text"]
        # structuredBlocks must carry plain Python numerics (JSON round-trip)
        blocks = results[0]["structuredBlocks"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "title"
        assert all(isinstance(v, (int, float)) for v in blocks[0]["bbox"])
        assert isinstance(blocks[0]["confidence"], float)

    def test_pdf_two_pages(self, client: TestClient) -> None:
        """A 2-page PDF yields two layoutParsingResults, one per page."""
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model(
                    [_numpy_page_result(1), _numpy_page_result(2)]
                ),
            ),
            patch.object(
                paddlex_helpers,
                "_pdf_to_images",
                return_value=["/tmp/page-1.png", "/tmp/page-2.png"],
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert len(results) == 2
        assert "Title 1" in results[0]["markdown"]["text"]
        assert "Title 2" in results[1]["markdown"]["text"]

    def test_image_file_type(self, client: TestClient) -> None:
        """fileType=1 (image) skips PDF conversion and goes straight to the model."""
        file_b64 = base64.b64encode(_FAKE_PNG).decode()
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 1},
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert len(results) == 1
        assert "Title 1" in results[0]["markdown"]["text"]

    def test_null_page_fields_no_500(self, client: TestClient) -> None:
        """PaddleX null page sections (F1) must yield empty results, not a 500.

        Regression-locks the `or {}`/`or []` guards through the full HTTP path.
        """
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        null_page = {
            "overall_ocr_res": None,
            "layout_det_res": None,
            "parsing_res_list": None,
        }
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([null_page]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert len(results) == 1
        assert results[0]["structuredBlocks"] == []

    def test_plain_ocr_fallback(self, client: TestClient) -> None:
        """USE_STRUCTURE_V3=false → plain OCR path, structuredBlocks=None."""
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_model",
                return_value=_FakeOcrModel([_numpy_ocr_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.dict(os.environ, {"USE_STRUCTURE_V3": "false"}),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert len(results) == 1
        assert "Plain OCR line 1" in results[0]["markdown"]["text"]
        assert results[0]["structuredBlocks"] is None

    def test_auth_wrong_token(self, client: TestClient) -> None:
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        with patch.object(server, "PADDLEOCR_VL_TOKEN", "secret"):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer wrong-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 403

    def test_auth_missing_token(self, client: TestClient) -> None:
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        with patch.object(server, "PADDLEOCR_VL_TOKEN", "secret"):
            response = client.post(
                "/layout-parsing", json={"file": file_b64, "fileType": 0}
            )
        assert response.status_code == 401

    def test_invalid_base64(self, client: TestClient) -> None:
        with patch.object(server, "PADDLEOCR_VL_TOKEN", ""):
            response = client.post(
                "/layout-parsing", json={"file": "a==b", "fileType": 0}
            )
        assert response.status_code == 400
        assert "base64" in response.json()["detail"].lower()

    def test_oversized_request(self, client: TestClient) -> None:
        with patch.object(server, "MAX_REQUEST_SIZE", 64):
            response = client.post(
                "/layout-parsing",
                json={"file": base64.b64encode(_FAKE_PDF).decode(), "fileType": 0},
            )
        assert response.status_code == 413

    def test_page_limit_413_real_two_page_pdf(self, client: TestClient) -> None:
        """A REAL 2-page PDF is 413 before any temp-file staging or model
        work when MAX_OCR_PAGES=1 — the guard counts pages from the
        in-memory request bytes and the job takes no slot."""
        file_b64 = base64.b64encode(_TWOPAGE_PDF).decode()
        slot_calls: list[int] = []
        with (
            patch.object(server, "MAX_OCR_PAGES", 1),
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 413
        assert response.json()["detail"] == "Too many pages: 2 (max 1)"
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_page_limit_413_lying_count(self, client: TestClient) -> None:
        """A LYING /Count bypass: a 3-page PDF declaring /Count 1 is 413
        before any temp-file staging or model work when MAX_OCR_PAGES=1.
        The /Count-gated count (pypdfium2's CountPages; pypdf's for
        encrypted docs) reads 1, so the pre-fix gate passed this
        ~600-byte PDF (every /Count-gated consumer — poppler's render
        loop included, which is clamped to the raw /Count — sees only
        the first /Count pages; the concrete DoS residual is the
        page-size dimension, closed by the pixel cap). The cap counts
        the page tree (pypdf's get_num_pages → _flatten), so the true
        count (3) is what 413s. The job takes no slot."""
        file_b64 = base64.b64encode(_LYING_COUNT_PDF).decode()
        slot_calls: list[int] = []
        with (
            patch.object(server, "MAX_OCR_PAGES", 1),
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 413
        assert response.json()["detail"] == "Too many pages: 3 (max 1)"
        mock_tmp.assert_not_called()
        assert slot_calls == []

    @pytest.mark.parametrize(
        ("label", "pdf"),
        [
            ("deep", _DEEP_TREE_PDF),
            ("wide", _WIDE_TREE_PDF),
            ("cyclic", _CYCLIC_TREE_PDF),
            ("deep-encrypted", _DEEP_TREE_PDF_ENCRYPTED),
        ],
    )
    def test_page_limit_413_page_tree_guard(
        self, client: TestClient, label: str, pdf: bytes
    ) -> None:
        """Fail-CLOSED 413 when pypdf's OWN page-tree guard trips
        (depth > 100 / array elements > 100,000 / cycle — shapes in
        _degenerate_tree_pdf). The pre-fix fail-soft fallback (pdfium's
        /Count-gated count = 1 + pixel-cap skip) passed all of these;
        the huge first leaf made that pass a ~521 GB render. The 413
        fires before any temp-file staging or model work, and the job
        takes no slot. Covers both the plaintext branch (page-tree
        count) and the encrypted branch (decrypt("") → _flatten)."""
        file_b64 = base64.b64encode(pdf).decode()
        slot_calls: list[int] = []
        with (
            patch.object(server, "MAX_OCR_PAGES", 1),
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 413
        assert "Page tree unmeasurable" in response.json()["detail"]
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_page_limit_413_unparseable_pdfium_only(self, client: TestClient) -> None:
        """Fail-closed PARSER-DISAGREEMENT: a spec-legal PDF whose
        xref table + trailer are valid but which has NO startxref
        section (_NO_STARTXREF_PDF) — pypdf's constructor rejects it,
        pypdfium2/poppler open it. Pre-fix, _enforce_pdf_caps failed
        SOFT here (page cap on the /Count-gated count only, pixel cap
        skipped ENTIRELY) — page 1's 100000x100000 pt box would have
        rendered ~521 GB at 300 dpi from this ~300-byte file, per
        unauthenticated request. Now: 413 "Unparseable PDF" before any
        temp-file staging or model work, and the job takes no slot.
        Self-verifies both sides of the parser disagreement so a
        pypdf/pypdfium2 behavior change fails LOUD here instead of
        silently re-opening the hole (the unit-level TestEnforcePdfCaps
        pins cover the same branch at the cap-function level)."""
        pytest.importorskip("pypdf")
        import io

        import pypdf
        import pypdfium2

        with pytest.raises(pypdf.errors.PdfReadError):
            pypdf.PdfReader(io.BytesIO(_NO_STARTXREF_PDF), strict=False)
        with pypdfium2.PdfDocument(_NO_STARTXREF_PDF) as doc:
            assert len(doc) == 3

        file_b64 = base64.b64encode(_NO_STARTXREF_PDF).decode()
        slot_calls: list[int] = []
        with (
            patch.object(server, "MAX_OCR_PAGES", 1),
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 413
        assert response.json()["detail"] == "Unparseable PDF"
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_page_limit_boundary_real_two_page_pdf(self, client: TestClient) -> None:
        """count == MAX_OCR_PAGES must PASS (the cap rejects page_count >
        MAX, not >=): a real 2-page PDF with MAX_OCR_PAGES=2 runs the full
        pipeline to 200 with two results."""
        file_b64 = base64.b64encode(_TWOPAGE_PDF).decode()
        with (
            patch.object(server, "MAX_OCR_PAGES", 2),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model(
                    [_numpy_page_result(1), _numpy_page_result(2)]
                ),
            ),
            patch.object(
                paddlex_helpers,
                "_pdf_to_images",
                return_value=["/tmp/page-1.png", "/tmp/page-2.png"],
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert len(results) == 2

    def test_page_limit_parse_slot_saturated_503(self, client: TestClient) -> None:
        """Pre-slot parse fan-out bound: with every parse slot held, the
        endpoint 503s BEFORE staging, job slot, or pipeline (consistent
        with the job slot's saturation 503). The parse runs BEFORE the
        job slot, so its fan-out is bounded by MAX_CONCURRENT_PDF_PARSSES
        — without the slot, up to the thread-pool size of concurrent
        100-MB parses could be in flight at once under saturation."""
        file_b64 = base64.b64encode(_TWOPAGE_PDF).decode()
        slot_calls: list[int] = []
        with (
            # Saturated parse gate: a fully-depleted semaphore (locked()
            # is True) — the real _pdf_parse_slot does the 503.
            patch.object(server, "_pdf_parse_semaphore", lambda: asyncio.Semaphore(0)),
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 503
        assert "parse slots" in response.json()["detail"]
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_page_limit_413_releases_parse_slot(self, client: TestClient) -> None:
        """A 413 cap rejection RELEASES its parse slot: at capacity 1,
        the rejected request must leave its event loop's slot free (a
        leaked slot would 503 every later PDF request until restart).
        The per-loop registry mirrors the production per-event-loop
        pattern so the slot state is inspectable after the request."""
        file_b64 = base64.b64encode(_TWOPAGE_PDF).decode()
        sem_factory, registry = _per_loop_parse_semaphores(1)
        with (
            patch.object(server, "MAX_OCR_PAGES", 1),
            patch.object(server, "_pdf_parse_semaphore", sem_factory),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 413
        assert len(registry) == 1, "the request never reached the parse slot"
        assert not next(iter(registry.values())).locked(), (
            "the 413 rejection leaked its parse slot"
        )

    # ── Content-based routing: the bytes decide the pipeline, the hint is a hint ──
    def test_image_content_with_pdf_hint(self, client: TestClient) -> None:
        """fileType=0 (PDF hint) but the bytes are an actual PNG → image pipeline.

        The old code trusted the hint and sent the image down the PDF path
        (pdftoppm on a PNG → 500).
        """
        file_b64 = base64.b64encode(_FAKE_PNG).decode()
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert "Title 1" in results[0]["markdown"]["text"]

    def test_pdf_content_with_image_hint(self, client: TestClient) -> None:
        """fileType=1 (image hint) but the bytes are an actual PDF → PDF pipeline.

        The old code trusted the hint and skipped the PDF pre-render
        (PaddleX chokes on a raw PDF path).
        """
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 1},
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert "Title 1" in results[0]["markdown"]["text"]

    def test_text_content_with_image_hint_passthrough(self, client: TestClient) -> None:
        """fileType=1 but the bytes are plain text → text passthrough (was 500)."""
        file_b64 = base64.b64encode(b"Hello, just text.\n").decode()
        with patch.object(server, "PADDLEOCR_VL_TOKEN", ""):
            response = client.post(
                "/layout-parsing", json={"file": file_b64, "fileType": 1}
            )

        assert response.status_code == 200
        results = response.json()["result"]["layoutParsingResults"]
        assert results[0]["markdown"]["text"] == "Hello, just text.\n"
        assert results[0]["structuredBlocks"] == []

    def test_unknown_binary_415(self, client: TestClient) -> None:
        """Unrecognized binary content → 415 (was a 500 from deep in the pipeline)."""
        garbage = b"\x80\x81\x82" * 64  # invalid UTF-8, no known signature
        with patch.object(server, "PADDLEOCR_VL_TOKEN", ""):
            response = client.post(
                "/layout-parsing",
                json={"file": base64.b64encode(garbage).decode(), "fileType": 0},
            )
        assert response.status_code == 415


# ── POST /extract — direct upload API ────────────────────────────────────
class TestExtractE2E:
    def test_extract_page_limit_413_real_two_page_pdf(self, client: TestClient) -> None:
        """/extract side: a REAL 2-page PDF (multipart upload) is 413
        before staging when MAX_OCR_PAGES=1, taking no job slot."""
        slot_calls: list[int] = []
        with (
            patch.object(server, "MAX_OCR_PAGES", 1),
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _TWOPAGE_PDF, "application/pdf")},
            )
        assert response.status_code == 413
        assert response.json()["detail"] == "Too many pages: 2 (max 1)"
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_extract_page_limit_413_lying_count(self, client: TestClient) -> None:
        """Lying-/Count bypass on /extract (shared cap): a 3-page PDF
        declaring /Count 1 is 413 before any temp-file staging or model
        work when MAX_OCR_PAGES=1 — the page-tree count (3) is
        authoritative, not the /Count-gated 1. The job takes no slot."""
        slot_calls: list[int] = []
        with (
            patch.object(server, "MAX_OCR_PAGES", 1),
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _LYING_COUNT_PDF, "application/pdf")},
            )
        assert response.status_code == 413
        assert response.json()["detail"] == "Too many pages: 3 (max 1)"
        mock_tmp.assert_not_called()
        assert slot_calls == []

    @pytest.mark.parametrize(
        ("label", "pdf"),
        [
            ("deep", _DEEP_TREE_PDF),
            ("wide", _WIDE_TREE_PDF),
            ("cyclic", _CYCLIC_TREE_PDF),
            ("deep-encrypted", _DEEP_TREE_PDF_ENCRYPTED),
        ],
    )
    def test_extract_page_limit_413_page_tree_guard(
        self, client: TestClient, label: str, pdf: bytes
    ) -> None:
        """Multipart wiring for the same fail-closed 413: the /extract
        endpoint (multipart upload, no fileType) must hit the identical
        pre-slot gate on pypdf's page-tree guard trip — no staging, no
        slot, no pipeline."""
        slot_calls: list[int] = []
        with (
            patch.object(server, "MAX_OCR_PAGES", 1),
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": (f"guard-{label}.pdf", pdf, "application/pdf")},
            )
        assert response.status_code == 413
        assert "Page tree unmeasurable" in response.json()["detail"]
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_extract_page_limit_413_unparseable_pdfium_only(
        self, client: TestClient
    ) -> None:
        """Multipart wiring for the same fail-closed 413: the /extract
        endpoint (multipart upload, no fileType) must hit the identical
        pre-slot gate on the parser-disagreement shape (pypdf's
        constructor rejects _NO_STARTXREF_PDF, pypdfium2 opens it) —
        no staging, no slot, no pipeline."""
        slot_calls: list[int] = []
        with (
            patch.object(server, "MAX_OCR_PAGES", 1),
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _NO_STARTXREF_PDF, "application/pdf")},
            )
        assert response.status_code == 413
        assert response.json()["detail"] == "Unparseable PDF"
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_extract_page_limit_boundary_real_two_page_pdf(
        self, client: TestClient
    ) -> None:
        """/extract boundary: count == MAX_OCR_PAGES passes (200, not 413)."""
        with (
            patch.object(server, "MAX_OCR_PAGES", 2),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model(
                    [_numpy_page_result(1), _numpy_page_result(2)]
                ),
            ),
            patch.object(
                paddlex_helpers,
                "_pdf_to_images",
                return_value=["/tmp/page-1.png", "/tmp/page-2.png"],
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _TWOPAGE_PDF, "application/pdf")},
            )

        assert response.status_code == 200
        assert len(response.json()["pages"]) == 2

    def test_extract_parse_slot_saturated_503(self, client: TestClient) -> None:
        """Multipart wiring for the pre-slot parse fan-out bound: a
        saturated parse gate 503s BEFORE staging, job slot, or pipeline
        (shared _run_ocr_pipeline — pinned on this endpoint for the
        same reason the 413 caps are pinned on both)."""
        slot_calls: list[int] = []
        with (
            patch.object(server, "_pdf_parse_semaphore", lambda: asyncio.Semaphore(0)),
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _TWOPAGE_PDF, "application/pdf")},
            )
        assert response.status_code == 503
        assert "parse slots" in response.json()["detail"]
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_extract_job_slot_saturated_503(self, client: TestClient) -> None:
        """JOB SLOT SATURATION on /extract (the parse-slot sibling above
        pins the PRE-slot bound; this pins the JOB gate on this
        endpoint): with the job gate fully depleted, the endpoint 503s
        BEFORE staging or the pipeline. Same pattern as the parse-slot
        test — patch the SEMAPHORE factory, keep the real
        _ocr_job_slot (its 503)."""
        with (
            patch.object(server, "_ocr_job_semaphore", lambda: asyncio.Semaphore(0)),
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _TWOPAGE_PDF, "application/pdf")},
            )
        assert response.status_code == 503
        assert "job slots" in response.json()["detail"]
        mock_tmp.assert_not_called()

    def test_extract_pdf_full_flow(self, client: TestClient) -> None:
        """Multipart upload → structured pages + blocks, numpy payload safe."""
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _FAKE_PDF, "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "doc.pdf"
        assert len(data["pages"]) == 1
        page = data["pages"][0]
        assert page["page"] == 1
        assert "# Title 1" in page["markdown"]
        assert "Body text 1." in page["text"]
        assert "Body text 1." in data["full_text"]
        blocks = page["structured_blocks"]
        assert blocks[0]["type"] == "title"
        assert all(isinstance(v, (int, float)) for v in blocks[0]["bbox"])
        # Raw OCR blocks must also be JSON-safe (numpy bboxes scrubbed)
        assert all(isinstance(v, (int, float)) for v in page["blocks"][0]["bbox"])

    def test_extract_image_full_flow(self, client: TestClient) -> None:
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("scan.png", _FAKE_PNG, "image/png")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "scan.png"
        assert "Title 1" in data["pages"][0]["markdown"]

    def test_extract_plain_ocr_fallback(self, client: TestClient) -> None:
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_model",
                return_value=_FakeOcrModel([_numpy_ocr_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.dict(os.environ, {"USE_STRUCTURE_V3": "false"}),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _FAKE_PDF, "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "Plain OCR line 1" in data["full_text"]
        assert data["pages"][0]["structured_blocks"] is None

    # ── Content-based routing: the bytes decide the pipeline, the name is a hint ──
    def test_extract_pdf_content_image_name(self, client: TestClient) -> None:
        """A file named *.png whose bytes are a PDF → PDF pipeline.

        The old code routed on the extension (image path on raw PDF bytes
        → 500 from deep in PaddleX).
        """
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.png", _FAKE_PDF, "image/png")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "doc.png"
        assert "Title 1" in data["pages"][0]["markdown"]

    def test_extract_image_content_pdf_name(self, client: TestClient) -> None:
        """A file named *.pdf whose bytes are a PNG → image pipeline."""
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _FAKE_PNG, "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "doc.pdf"
        assert "Title 1" in data["pages"][0]["markdown"]

    def test_extract_unknown_binary_415(self, client: TestClient) -> None:
        """Unrecognized binary content → 415 (was a 500 from deep in the pipeline)."""
        garbage = b"\x80\x81\x82" * 64  # invalid UTF-8, no known signature
        with patch.object(server, "PADDLEOCR_VL_TOKEN", ""):
            response = client.post(
                "/extract",
                files={"file": ("blob.bin", garbage, "application/octet-stream")},
            )
        assert response.status_code == 415


# ── Rendered-pixel cap: 413 before slot + temp-file staging (both endpoints) ──
class TestRenderPixelLimitE2E:
    """The rendered-pixel cap (MAX_RENDERED_PIXELS) end-to-end: a page that
    would render above the cap is 413 BEFORE the job slot and BEFORE any
    temp-file staging — measured from the in-memory request bytes."""

    def test_layout_parsing_pixel_limit_413_huge_page(self, client: TestClient) -> None:
        """A ~290-byte 1-page PDF with a 100k-pt MediaBox renders to
        ~1.7e11 px at 300 dpi → 413 under the default cap, and nothing ran
        (no pipeline, no temp file, no job slot)."""
        file_b64 = base64.b64encode(_HUGE_PDF).decode()
        slot_calls: list[int] = []
        with (
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert "exceeds" in detail
        assert str(server.MAX_RENDERED_PIXELS) in detail  # the cap is named
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_layout_parsing_pixel_limit_413_huge_mediabox_normal_cropbox(
        self, client: TestClient
    ) -> None:
        """CROPBOX BYPASS regression (both endpoints' shared guard, pinned
        here end-to-end): pypdfium2's get_width/get_height report the
        CropBox when one is set, but pdftoppm (no -cropbox flag) renders
        the MediaBox. A normal-CropBox/huge-MediaBox page must still 413
        before slot + staging — a CropBox-only measurement would let this
        ~250-byte PDF stage a ~521 GB render."""
        file_b64 = base64.b64encode(_HUGE_CROP_PDF).decode()
        slot_calls: list[int] = []
        with (
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert "exceeds" in detail
        assert str(server.MAX_RENDERED_PIXELS) in detail  # the cap is named
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_layout_parsing_pixel_limit_413_page_tree_inherited_mediabox(
        self, client: TestClient
    ) -> None:
        """PAGE-TREE BYPASS regression, pinned end-to-end: a huge MediaBox
        on the /Pages node with a normal page CropBox is invisible to
        pypdfium2's page-level box APIs (FPDFPage_GetMediaBox does not
        walk /Parent — PDFium crbug 1786), but pdftoppm inherits it and
        renders ~521 GB at 300 dpi. The shared guard must 413 this
        ~300-byte PDF before slot + staging (the cap now measures the
        effective box with pypdf, which resolves the same inheritance
        poppler does)."""
        file_b64 = base64.b64encode(_HUGE_PARENT_MEDIA_PDF).decode()
        slot_calls: list[int] = []
        with (
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert "exceeds" in detail
        assert str(server.MAX_RENDERED_PIXELS) in detail  # the cap is named
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_layout_parsing_pixel_limit_413_inverted_mediabox_corners(
        self, client: TestClient
    ) -> None:
        """INVERTED-CORNERS BYPASS regression, pinned end-to-end: a
        page-level MediaBox with inverted corners (spec-legal per PDF
        32000-1) behind a normal CropBox — the old raw-delta measurement
        saw a negative MediaBox delta and a 612x792 CropBox and passed,
        while pdftoppm normalizes the corners and renders ~521 GB at 300
        dpi. The shared guard must 413 before slot + staging."""
        file_b64 = base64.b64encode(_INVERTED_MEDIA_PDF).decode()
        slot_calls: list[int] = []
        with (
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert "exceeds" in detail
        assert str(server.MAX_RENDERED_PIXELS) in detail  # the cap is named
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_layout_parsing_pixel_limit_413_normal_page_tight_cap(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the cap below a normal page's rendered size, even the
        ordinary 612x792 _FAKE_PDF (2550x3300 px = 8,415,000 px at the
        default 300 dpi) is 413 — the check is real, not just a giant-box
        guard. The tight cap is DERIVED from the rendered size at the
        CURRENT server.RENDER_DPI (a documented RENDER_DPI override must
        not break the test — it used to hard-code 300-dpi numbers)."""
        page_px = (612 * server.RENDER_DPI / 72.0) * (792 * server.RENDER_DPI / 72.0)
        tight_cap = int(page_px) - 1  # one below the rendered size → guaranteed 413
        assert tight_cap >= 1
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", tight_cap)
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        slot_calls: list[int] = []
        with (
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 0},
            )
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert "exceeds" in detail
        assert str(tight_cap) in detail  # the (derived) cap is named
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_extract_pixel_limit_413_huge_page(self, client: TestClient) -> None:
        """Same guard on the /extract multipart path: huge MediaBox → 413,
        nothing ran."""
        slot_calls: list[int] = []
        with (
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("huge.pdf", _HUGE_PDF, "application/pdf")},
            )
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert "exceeds" in detail
        assert str(server.MAX_RENDERED_PIXELS) in detail
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_extract_pixel_limit_413_normal_page_tight_cap(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """/extract side of the tight-cap case: the ordinary PDF 413s when
        the cap sits below its rendered size. The tight cap is DERIVED from
        the rendered size at the CURRENT server.RENDER_DPI (a documented
        RENDER_DPI override must not break the test)."""
        page_px = (612 * server.RENDER_DPI / 72.0) * (792 * server.RENDER_DPI / 72.0)
        tight_cap = int(page_px) - 1  # one below the rendered size → guaranteed 413
        assert tight_cap >= 1
        monkeypatch.setattr(server, "MAX_RENDERED_PIXELS", tight_cap)
        slot_calls: list[int] = []
        with (
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("doc.pdf", _FAKE_PDF, "application/pdf")},
            )
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert "exceeds" in detail
        assert str(tight_cap) in detail  # the (derived) cap is named
        mock_tmp.assert_not_called()
        assert slot_calls == []

    # ── Image canvas cap (same pre-slot gate, image kind) ──
    @pytest.mark.parametrize(
        ("png", "label"),
        [
            (_HUGE_PNG, "bomb-branch"),
            (_tiny_png(13000, 13000), "direct-branch"),
        ],
    )
    def test_image_pixel_limit_413_huge_canvas(
        self, client: TestClient, png: bytes, label: str
    ) -> None:
        """IMAGE CANVAS cap, both branches (both endpoints' shared
        guard): a ~50-byte PNG declaring a canvas whose decode would
        blow MAX_RENDERED_PIXELS passes MAX_REQUEST_SIZE (tiny input)
        yet must 413 before the job slot and staging — the IMAGE
        equivalent of the huge-MediaBox PDFs above. The 'bomb-branch'
        canvas is above PIL's decompression-bomb ERROR threshold (PIL
        raises before the size is readable); the 'direct-branch' canvas
        is between the warning and error thresholds (the w*h check
        fires on the readable size)."""
        file_b64 = base64.b64encode(png).decode()
        slot_calls: list[int] = []
        with (
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 1},
            )
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert "exceeds" in detail
        assert str(server.MAX_RENDERED_PIXELS) in detail  # the cap is named
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_extract_image_pixel_limit_413_huge_canvas(
        self, client: TestClient
    ) -> None:
        """Multipart wiring for the image canvas cap (shared guard —
        pinned on this endpoint for the same reason the PDF caps are)."""
        slot_calls: list[int] = []
        with (
            patch.object(tempfile, "NamedTemporaryFile") as mock_tmp,
            patch.object(server, "_ocr_job_slot", _recording_slot(slot_calls)),
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                side_effect=AssertionError("pipeline must not run"),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/extract",
                headers={"Authorization": "Bearer test-token"},
                files={"file": ("huge.png", _HUGE_PNG, "image/png")},
            )
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert "exceeds" in detail
        mock_tmp.assert_not_called()
        assert slot_calls == []

    def test_image_pixel_limit_413_releases_parse_slot(
        self, client: TestClient
    ) -> None:
        """The image-cap 413 RELEASES its parse slot (same leak class
        the PDF 413 test above pins — a leaked slot would 503 every
        later parse until restart)."""
        file_b64 = base64.b64encode(_HUGE_PNG).decode()
        sem_factory, registry = _per_loop_parse_semaphores(1)
        with (
            patch.object(server, "_pdf_parse_semaphore", sem_factory),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 1},
            )
        assert response.status_code == 413
        assert len(registry) == 1, "the request never reached the parse slot"
        assert not next(iter(registry.values())).locked(), (
            "the 413 rejection leaked its parse slot"
        )

    def test_image_pixel_limit_small_canvas_passes(self, client: TestClient) -> None:
        """A normal small-canvas PNG (612x792 px — the 300-dpi A4
        equivalent) passes the header cap and runs the pipeline (no
        false positive from the new image gate)."""
        file_b64 = base64.b64encode(_tiny_png(612, 792)).decode()
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", "test-token"),
        ):
            response = client.post(
                "/layout-parsing",
                headers={"Authorization": "Bearer test-token"},
                json={"file": file_b64, "fileType": 1},
            )
        assert response.status_code == 200


# ── Helper layer (covers the worker.py sidecar path too) ─────────────────
class TestHelperLayerE2E:
    def test_structure_v3_pages_are_json_serializable(self, tmp_path) -> None:
        """run_paddlex_structure_v3 output must survive plain json.dumps.

        This is the exact failure mode from the production incident — the
        worker's sidecar JSON and both server endpoints serialize these
        pages without a default= handler at the call site.
        """
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(_FAKE_PDF)
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
        ):
            pages = paddlex_helpers.run_paddlex_structure_v3(str(pdf))

        json.dumps(pages)  # no default= — must not raise TypeError
        assert pages[0]["structured_blocks"][0]["bbox"] == [50.0, 50.0, 200.0, 70.0]
        assert all(type(v) is float for v in pages[0]["structured_blocks"][0]["bbox"])

    def test_ocr_pages_are_json_serializable(self, tmp_path) -> None:
        img = tmp_path / "page.png"
        img.write_bytes(_FAKE_PNG)
        with patch.object(
            paddlex_helpers,
            "_get_paddlex_model",
            return_value=_FakeOcrModel([_numpy_ocr_result(1)]),
        ):
            pages = paddlex_helpers.run_paddleocr(str(img))

        json.dumps(pages)  # no default= — must not raise TypeError
        assert all(type(v) is float for v in pages[0]["blocks"][0]["bbox"])


# ── SafeJSONResponse backstop ────────────────────────────────────────────
class TestSafeJSONResponse:
    def test_numpy_leak_serialized_not_crash(self) -> None:
        """A numpy type that slips past the source-level fix must still serialize.

        This is the defense-in-depth backstop (serialization.json_default):
        the endpoint must never return a 500 because of a non-JSON type.
        """
        response = SafeJSONResponse(
            content={
                "scalar_list": [np.float32(1.5)],
                "ndarray": np.array([1, 2], dtype=np.int32),
            }
        )
        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["scalar_list"] == [1.5]
        assert body["ndarray"] == [1, 2]

    def test_unknown_type_still_raises(self) -> None:
        """The backstop must not silently string-ify unknown objects."""
        with pytest.raises(TypeError):
            SafeJSONResponse(content={"obj": object()})


# ── Endpoint-level concurrency: job slots + predict lock under real HTTP ──
class TestEndpointConcurrency:
    """The OCR job-slot cap (at most MAX_CONCURRENT_OCR_JOBS in flight, 503
    when saturated) and the process-wide predict lock under REAL HTTP
    concurrency.

    TestClient cannot overlap requests (its portal is synchronous per
    call), so this drives the ASGI app through httpx's ASGITransport with
    concurrent tasks — the strongest proof that the slot semaphore (per
    event loop: each asyncio.run gets a fresh loop and thus a fresh
    semaphore, so no cross-test state) and the predict lock both hold
    under concurrent load. The 503 scenario is barrier-deterministic (see
    test_three_concurrent_jobs_two_200_one_503), not sleep-based.
    """

    def test_failed_job_releases_slot(self) -> None:
        """A job that fails mid-pipeline must release its job slot —
        otherwise every later request would 503 forever (the semaphore
        permit would leak with the dead job)."""

        class _FailingV3Model:
            """predict() raises on first iteration — the failure the
            endpoint must convert into a generic 500."""

            def predict(self, _path: str):
                raise RuntimeError("boom")
                yield  # unreachable: makes predict() a generator

        slot_entries: list[str] = []

        @asynccontextmanager
        async def _counting_slot():
            slot_entries.append("enter")
            try:
                yield
            finally:
                slot_entries.append("exit")

        client = TestClient(server.app)
        file_b64 = base64.b64encode(_FAKE_PDF).decode()
        payload = {"file": file_b64, "fileType": 0}

        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FailingV3Model(),
            ),
            patch.object(
                paddlex_helpers,
                "_pdf_to_images",
                return_value=["/tmp/page-1.png"],
            ),
            patch.object(server, "_ocr_job_slot", _counting_slot),
            patch.object(server, "PADDLEOCR_VL_TOKEN", ""),
        ):
            response = client.post("/layout-parsing", json=payload)

        # (1) Generic 500, exception text not leaked into the response.
        assert response.status_code == 500
        assert response.json()["detail"] == "Layout parsing failed"
        assert "boom" not in response.text
        # (2) The slot was entered exactly once and exited exactly once —
        # released on failure, not leaked.
        assert slot_entries == ["enter", "exit"]

        # (3) A follow-up job with a healthy model is admitted (no phantom
        # 503) and completes.
        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_FakeV3Model([_numpy_page_result(1)]),
            ),
            patch.object(
                paddlex_helpers,
                "_pdf_to_images",
                return_value=["/tmp/page-1.png"],
            ),
            patch.object(server, "PADDLEOCR_VL_TOKEN", ""),
        ):
            response = client.post("/layout-parsing", json=payload)

        assert response.status_code == 200
        assert len(response.json()["result"]["layoutParsingResults"]) == 1

    def test_three_concurrent_jobs_two_200_one_503(self) -> None:
        """Deterministic (no sleep margins): jobs 1 and 2 are fired and
        provably hold BOTH job slots (a threading.Event barrier set by a
        wrapper around the REAL _ocr_job_slot) before job 3 is fired, so
        job 3's 503 is guaranteed, not a race. The admitted jobs' predicts
        then block on a release event (non-vacuity: AssertionError on
        timeout) so the slots stay held until job 3 has been rejected; the
        process-wide predict lock still serializes the two admitted jobs
        (max_in_flight == 1)."""
        import asyncio
        import threading

        import httpx

        two_slots_held = threading.Event()
        release = threading.Event()
        held_lock = threading.Lock()
        held = 0
        state = {"in_flight": 0, "max_in_flight": 0}
        state_lock = threading.Lock()

        class _BarrierV3Model:
            """predict() waits until both job slots are provably held (the
            barrier), then until the driver releases it — keeping the slots
            saturated for the whole window job 3 is fired. On timeout it
            raises AssertionError instead of hanging (non-vacuity guard).
            Keeps the existing in_flight counter."""

            def predict(self, _path: str):
                with state_lock:
                    state["in_flight"] += 1
                    state["max_in_flight"] = max(
                        state["max_in_flight"], state["in_flight"]
                    )
                try:
                    if not two_slots_held.wait(timeout=10):
                        raise AssertionError(
                            "two_slots_held never set — slots were never both held"
                        )
                    if not release.wait(timeout=10):
                        raise AssertionError("release never set — driver stuck")
                    yield _numpy_page_result(1)
                finally:
                    with state_lock:
                        state["in_flight"] -= 1

        real_slot = server._ocr_job_slot

        @asynccontextmanager
        async def _tracking_slot():
            # Wrapper around the REAL slot: increments the thread-safe held
            # counter, sets the barrier event when both slots are held,
            # decrements on exit. Runs on the event loop; the counter is
            # locked so the final assertion can read it from any thread.
            nonlocal held
            with held_lock:
                held += 1
                reached_two = held == 2
            if reached_two:
                two_slots_held.set()
            try:
                async with real_slot():
                    yield
            finally:
                with held_lock:
                    held -= 1

        async def scenario() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
            transport = httpx.ASGITransport(app=server.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                payload = {
                    "file": base64.b64encode(_FAKE_PDF).decode(),
                    "fileType": 0,
                }
                # Fire jobs 1 and 2 WITHOUT awaiting their responses: the
                # tasks progress concurrently on the loop (stream=True, so
                # send resolves only when the handler completes — after the
                # pipeline — which is exactly when we want to collect them).
                job1 = asyncio.create_task(
                    client.send(
                        client.build_request("POST", "/layout-parsing", json=payload),
                        stream=True,
                    )
                )
                job2 = asyncio.create_task(
                    client.send(
                        client.build_request("POST", "/layout-parsing", json=payload),
                        stream=True,
                    )
                )

                # Barrier: block (on a worker thread, so the event loop keeps
                # driving jobs 1 and 2) until both job slots are provably
                # held. A blocking wait on the loop thread would deadlock.
                if not await asyncio.to_thread(two_slots_held.wait, 10):
                    raise AssertionError(
                        "two_slots_held never set — slots were never both held"
                    )

                # Job 3 now deterministically sees a saturated semaphore.
                resp3 = await client.post("/layout-parsing", json=payload)

                # Release the admitted jobs and collect their responses.
                release.set()
                resp1 = await job1
                resp2 = await job2
                await resp1.aread()
                await resp2.aread()
                await resp1.aclose()
                await resp2.aclose()
                return resp1, resp2, resp3

        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_BarrierV3Model(),
            ),
            patch.object(
                paddlex_helpers,
                "_pdf_to_images",
                return_value=["/tmp/page-1.png"],
            ),
            patch.object(server, "_ocr_job_slot", _tracking_slot),
            patch.object(server, "PADDLEOCR_VL_TOKEN", ""),
        ):
            resp1, resp2, resp3 = asyncio.run(scenario())

        # Two admitted jobs got 200; the third was rejected 503 while both
        # slots were provably held.
        codes = sorted(r.status_code for r in (resp1, resp2, resp3))
        assert codes == [200, 200, 503]
        assert resp3.status_code == 503
        assert "busy" in resp3.json()["detail"]
        # The two admitted jobs serialized inside the predict lock even
        # though the requests overlapped at the HTTP layer.
        assert state["max_in_flight"] == 1
        for ok in (resp1, resp2):
            assert ok.status_code == 200
            assert len(ok.json()["result"]["layoutParsingResults"]) == 1
        # Both slots released again (the wrapper decrements on the loop
        # thread before the handler returns, so this holds once both
        # responses are collected).
        with held_lock:
            assert held == 0

    def test_slot_cap_driven_by_max_concurrent_ocr_jobs(self) -> None:
        """Wiring (review): the /layout-parsing slot is sized by the
        import-time MAX_CONCURRENT_OCR_JOBS constant. The env → constant
        link is pinned in TestJobSlotsEnvContract (fresh interpreter);
        this test pins the constant → semaphore link: with the constant
        patched to 3, a FRESH event loop (each asyncio.run) must admit
        exactly 3 concurrent jobs and reject the 4th with 503. This
        distinguishes every plausible regression: ignoring the constant
        (default 2 → codes [200, 200, 503, 503]), a hardcoded 1 (two
        503s), or a hardcoded 3+ (fourth job admitted). The parse cap is
        raised to 10 so the pre-slot parse gate cannot shadow the job
        cap under test (with its default of 2, the third job's 503 would
        come from the PARSE gate and the three-job barrier would never
        complete — the parse gate's own behavior is pinned in
        TestPdfParseSlot + the parse-slot e2e pair). Barrier-
        deterministic like the sibling test: the three admitted jobs'
        predicts block until all three slots are provably held, then
        until the driver releases them (AssertionError on timeout —
        non-vacuity guard)."""
        import asyncio
        import threading

        import httpx

        three_slots_held = threading.Event()
        release = threading.Event()
        held_lock = threading.Lock()
        held = 0

        class _BarrierV3Model:
            """predict() waits until all three job slots are provably held
            (the barrier), then until the driver releases it — keeping
            the slots saturated for the whole window job 4 is fired."""

            def predict(self, _path: str):
                if not three_slots_held.wait(timeout=10):
                    raise AssertionError(
                        "three_slots_held never set — slots were never all held"
                    )
                if not release.wait(timeout=10):
                    raise AssertionError("release never set — driver stuck")
                yield _numpy_page_result(1)

        real_slot = server._ocr_job_slot

        @asynccontextmanager
        async def _tracking_slot():
            # Wrapper around the REAL slot (which reads the patched
            # MAX_CONCURRENT_OCR_JOBS when its per-loop semaphore is
            # created): count held slots, set the barrier at three.
            nonlocal held
            with held_lock:
                held += 1
                reached_three = held == 3
            if reached_three:
                three_slots_held.set()
            try:
                async with real_slot():
                    yield
            finally:
                with held_lock:
                    held -= 1

        async def scenario() -> list[httpx.Response]:
            transport = httpx.ASGITransport(app=server.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                payload = {
                    "file": base64.b64encode(_FAKE_PDF).decode(),
                    "fileType": 0,
                }
                # Fire jobs 1-3 WITHOUT awaiting their responses (same
                # barrier pattern as the sibling test).
                jobs = [
                    asyncio.create_task(
                        client.send(
                            client.build_request(
                                "POST", "/layout-parsing", json=payload
                            ),
                            stream=True,
                        )
                    )
                    for _ in range(3)
                ]

                # Barrier: block (on a worker thread, so the event loop
                # keeps driving jobs 1-3) until all three slots are
                # provably held.
                if not await asyncio.to_thread(three_slots_held.wait, 10):
                    raise AssertionError(
                        "three_slots_held never set — slots were never all held"
                    )

                # Job 4 now deterministically sees a saturated semaphore.
                resp4 = await client.post("/layout-parsing", json=payload)

                # Release the admitted jobs and collect their responses.
                release.set()
                resps: list[httpx.Response] = [await job for job in jobs]
                for r in resps:
                    await r.aread()
                    await r.aclose()
                resps.append(resp4)
                return resps

        with (
            patch.object(
                paddlex_helpers,
                "_get_paddlex_structure_v3_model",
                return_value=_BarrierV3Model(),
            ),
            patch.object(
                paddlex_helpers, "_pdf_to_images", return_value=["/tmp/page-1.png"]
            ),
            patch.object(server, "_ocr_job_slot", _tracking_slot),
            patch.object(server, "MAX_CONCURRENT_OCR_JOBS", 3),
            # Keep the parse gate out of the way (see docstring): the
            # per-loop parse semaphore reads the constant when created
            # inside the loop, so patching before asyncio.run suffices.
            patch.object(server, "MAX_CONCURRENT_PDF_PARSSES", 10),
            patch.object(server, "PADDLEOCR_VL_TOKEN", ""),
        ):
            resps = asyncio.run(scenario())

        # Exactly the patched cap admitted: three 200s, the fourth 503 —
        # the slot honored the constant (default is 2, so this is not
        # vacuously the default behavior).
        codes = sorted(r.status_code for r in resps)
        assert codes == [200, 200, 200, 503]
        assert resps[3].status_code == 503
        assert "busy" in resps[3].json()["detail"]
        # All slots released again (the wrapper decrements on the loop
        # thread before each handler returns).
        with held_lock:
            assert held == 0
