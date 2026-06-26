"""
Doc-Worker — PaddleOCR-VL compatible server for Open-WebUI
==========================================================

Implements the ``POST /layout-parsing`` endpoint that Open-WebUI calls when
PaddleOCR-VL is selected as the Content Extraction Engine in
Admin Settings > Documents.

Open-WebUI API contract
-----------------------
POST {base_url}/layout-parsing
  Headers:  Authorization: Bearer <PADDLEOCR_VL_TOKEN>  # optional
  Body:     JSON {
              "file": "<base64>",
              "fileType": 0|1,           // 0=PDF, 1=image
              "useDocOrientationClassify": bool,
              "useDocUnwarping": bool,
              "useChartRecognition": bool
            }
  Response: JSON {
              "result": {
                "layoutParsingResults": [
                  { "markdown": { "text": "<page 1 text>" } },
                  { "markdown": { "text": "<page 2 text>" } },
                ]
              }
            }

Additional endpoints (not used by Open-WebUI, for direct API access):
  GET  /health          — Health check
  POST /extract         — Upload PDF → returns extracted text
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

from typing import Awaitable, Callable

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.responses import Response
from pydantic import BaseModel

from paddleocr_helpers import paddleocr_lang_code, run_paddleocr

# ── Config ────────────────────────────────────────────────────────────
OCR_LANG = os.getenv("OCR_LANG", "deu")
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() in ("true", "1", "yes")
PADDLEOCR_VL_TOKEN = os.getenv("PADDLEOCR_VL_TOKEN", "")
PADDLEOCR_MODELS = os.getenv("PADDLEOCR_MODELS", "/app/models")

# Max request body: 100 MB (base64-encoded PDFs can be ~33% larger than raw)
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "104857600"))  # 100 MB default

# Image extensions that Open-WebUI sends as fileType=1
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

app = FastAPI(title="Doc-Worker (PaddleOCR-VL)", version="1.0.0")
logger = logging.getLogger("doc-worker.api")


@app.middleware("http")
async def limit_request_size(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Reject requests that exceed MAX_REQUEST_SIZE to prevent disk exhaustion."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "detail": f"Request too large (max {MAX_REQUEST_SIZE // 1024 // 1024} MB)"
            },
        )
    return await call_next(request)


# ── Helpers ───────────────────────────────────────────────────────────
def _check_token(authorization: str | None) -> None:
    """Validate the Authorization: Bearer <value> header."""
    if not PADDLEOCR_VL_TOKEN:
        return  # No token configured, skip auth
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    if authorization != f"Bearer {PADDLEOCR_VL_TOKEN}":
        raise HTTPException(403, "Invalid token")


# ── Open-WebUI endpoint ──────────────────────────────────────────────
class LayoutParsingRequest(BaseModel):
    """Request body for Open-WebUI's /layout-parsing endpoint."""

    file: str  # base64-encoded file bytes
    fileType: int = 0  # 0=PDF, 1=image
    useDocOrientationClassify: bool = False
    useDocUnwarping: bool = False
    useChartRecognition: bool = False


@app.post("/layout-parsing")
async def layout_parsing(
    body: LayoutParsingRequest,
    authorization: str | None = Header(None, alias="Authorization"),
) -> JSONResponse:
    """Open-WebUI PaddleOCR-VL compatible endpoint.

    Open-WebUI calls this when PaddleOCR-VL is selected as the
    Content Extraction Engine.
    """
    _check_token(authorization)

    file_b64 = body.file
    if not file_b64:
        raise HTTPException(400, "Missing 'file' field")

    file_type = body.fileType
    is_image = file_type == 1

    try:
        file_bytes = base64.b64decode(file_b64)
    except Exception as exc:
        raise HTTPException(400, f"Invalid base64: {exc}")

    # Determine temp file extension
    suffix = ".png" if is_image else ".pdf"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        pages = run_paddleocr(tmp_path)

        # Build Open-WebUI response format
        layout_results = [{"markdown": {"text": page["text"]}} for page in pages]

        return JSONResponse(
            content={
                "result": {
                    "layoutParsingResults": layout_results,
                },
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Layout parsing failed")
        raise HTTPException(500, f"Layout parsing failed: {exc}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── Direct API endpoints (not used by Open-WebUI) ────────────────────
@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "ocr_lang": OCR_LANG,
        "paddleocr_lang": paddleocr_lang_code(),
        "gpu": OCR_USE_GPU,
        "endpoints": ["/layout-parsing", "/extract", "/health"],
    }


@app.post("/extract")
async def extract_text(
    file: UploadFile,
    authorization: str | None = Header(None, alias="Authorization"),
) -> JSONResponse:
    """Direct API: upload PDF/image, get extracted text + blocks.

    Accepts: multipart/form-data with field 'file'
    Returns: JSON with pages, blocks, full_text
    """
    _check_token(authorization)
    if not file.filename:
        raise HTTPException(400, "Missing file")

    ext = Path(file.filename).suffix.lower()
    is_image = ext in IMAGE_EXTS

    tmp_path = None
    try:
        file_bytes = await file.read()
        suffix = ext if ext else (".png" if is_image else ".pdf")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        pages = run_paddleocr(tmp_path)

        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        return JSONResponse(
            content={
                "filename": file.filename,
                "pages": [
                    {"page": p["page"], "text": p["text"], "blocks": p["blocks"]}
                    for p in pages
                ],
                "full_text": full_text,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(500, f"Extraction failed: {exc}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── Entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
