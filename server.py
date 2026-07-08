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

from paddleocr_helpers import (
    get_paddleocr_init_exception,
    paddleocr_lang_code,
    run_paddleocr,
    validate_paddleocr_models,
)

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


def _print_gpu_info() -> str:
    """Detect PaddleOCR GPU support at runtime and return a human-readable string."""
    try:
        import paddle
        has_cuda = paddle.device.is_compiled_with_cuda()
        if has_cuda and OCR_USE_GPU:
            # Try to get device count for extra info
            try:
                count = paddle.device.cuda.device_count()
                return f"GPU (CUDA, {count} device(s), OCR_USE_GPU=true)"
            except Exception:
                return "GPU (CUDA, device count unknown)"
        elif has_cuda:
            return "GPU (CUDA available, OCR_USE_GPU=false — running on CPU)"
        else:
            return "CPU (paddlepaddle CPU-only)"
    except Exception:
        return "GPU detection unavailable"


_PADDLEOCR_MODELS_LIST = [
    ("PP-OCRv6_medium_det", "PP-OCRv6_medium_det_infer"),
    ("PP-OCRv6_medium_rec", "PP-OCRv6_medium_rec_infer"),
    ("PP-LCNet_x1_0_textline_ori", "PP-LCNet_x1_0_textline_ori_infer"),
]


def _model_status(model_name: str, model_dir_name: str) -> str:
    """Return '✓' or '✗ <detail>' for a single model."""
    model_path = Path(PADDLEOCR_MODELS) / model_dir_name
    if not model_path.is_dir():
        return f"✗ {model_path} not found"
    # Check required files
    for fname in ("inference.pdiparams", "inference.yml"):
        if not (model_path / fname).is_file():
            return f"✗ {fname} missing"
    if not any(
        (model_path / f).is_file()
        for f in ("inference.json", "inference.pdmodel")
    ):
        return "✗ no model definition file"
    return "✓"


@app.on_event("startup")
async def _startup_status() -> None:
    """Print a human-readable status overview when the server starts."""
    separator = "=" * 50
    print(separator, flush=True)
    print("Doc-Worker starting up", flush=True)
    print(separator, flush=True)

    # GPU / engine info
    gpu_info = _print_gpu_info()
    print(f"  PaddleOCR engine ....... {gpu_info}", flush=True)
    print(f"  OCR language ........... {OCR_LANG} (PaddleOCR: {paddleocr_lang_code()})", flush=True)
    print(f"  Models dir ............. {PADDLEOCR_MODELS}", flush=True)

    # Model status
    print("  PaddleOCR models:", flush=True)
    any_missing = False
    for model_name, model_dir in _PADDLEOCR_MODELS_LIST:
        status = _model_status(model_name, model_dir)
        marker = "  ✓" if status == "✓" else "  ✗"
        pad = " " * (max(len(n) for n, _ in _PADDLEOCR_MODELS_LIST) - len(model_name))
        print(f"    {model_name}{pad} {marker} {status}", flush=True)
        if status != "✓":
            any_missing = True

    # Docling sidecar
    print("  Docling sidecar:", flush=True)
    docling_url = os.getenv("DOCLING_BASE_URL", "")
    if docling_url:
        try:
            import requests
            health = requests.head(f"{docling_url}/health", timeout=5)
            status = "✓ online" if health.status_code == 200 else f"✗ HTTP {health.status_code}"
        except Exception:
            status = "✗ offline (worker will wait up to 900s)"
        print(f"    DOCLING_BASE_URL .... {docling_url}  {status}", flush=True)
    else:
        print("    DOCLING_BASE_URL .... (not configured)", flush=True)

    # Paperless
    print("  Paperless-ngx:", flush=True)
    paperless = os.getenv("PAPERLESS_CONSUME", "")
    if paperless:
        print(f"    PAPERLESS_CONSUME ... {paperless}", flush=True)
    else:
        print("    PAPERLESS_CONSUME ... (not configured)", flush=True)

    # API endpoints
    print("  API endpoints:", flush=True)
    print("    /layout-parsing (Open-WebUI)", flush=True)
    print("    /extract (direct upload)", flush=True)
    print("    /health", flush=True)

    print(separator, flush=True)

    # If critical models are missing, abort
    if any_missing:
        validate_paddleocr_models()  # Will raise; caught by startup event handler


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
    init_exception = get_paddleocr_init_exception()
    if init_exception is not None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "component": "paddleocr",
                "error": str(init_exception),
            },
        )

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
