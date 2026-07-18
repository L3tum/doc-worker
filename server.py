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
from contextlib import asynccontextmanager
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

from typing import AsyncIterator, Awaitable, Callable

import http.server
import json

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.responses import Response
from pydantic import BaseModel

from paddlex_helpers import (
    blocks_to_markdown,
    destroy_paddlex_model,
    get_paddlex_init_exception,
    paddleocr_lang_code,
    run_paddleocr,
    run_paddlex_structure_v3,
    validate_paddlex_models,
)

# ── Config ────────────────────────────────────────────────────────────
OCR_LANG = os.getenv("OCR_LANG", "deu")
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() in ("true", "1", "yes")
PADDLEOCR_VL_TOKEN = os.getenv("PADDLEOCR_VL_TOKEN", "")
PADDLEOCR_MODELS = os.getenv("PADDLEOCR_MODELS", "/app/models")

# Max request body: 100 MB (base64-encoded PDFs can be ~33% larger than raw)
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "104857600"))  # 100 MB default

# Model idle timeout — destroy PaddleOCR model after N seconds of inactivity
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", "30"))

# Worker PID file path (for health check)
WORKER_PID_FILE = os.getenv("WORKER_PID_FILE", "/tmp/doc-worker.pid")
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8001"))

# Image extensions that Open-WebUI sends as fileType=1
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

logger = logging.getLogger("doc-worker.api")

# Track when the PaddleOCR model was last used (for idle-timeout destruction)
_model_last_used = 0.0
_model_last_used_lock = threading.Lock()


def _mark_model_used() -> None:
    """Update the model last-use timestamp (thread-safe)."""
    with _model_last_used_lock:
        _model_last_used = time.time()


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
    ("PP-DocLayout-L", "PP-DocLayout-L_infer"),
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
        (model_path / f).is_file() for f in ("inference.json", "inference.pdmodel")
    ):
        return "✗ no model definition file"
    return "✓"


def _idle_timeout_checker() -> None:
    """Background daemon thread: destroy the model after MODEL_IDLE_TIMEOUT seconds of inactivity.

    Wakes every 5 seconds, checks if the model is loaded and the idle timeout
    has elapsed, then calls `destroy_paddlex_model()`.
    """
    global _model_last_used
    logger.info(
        f"Model idle timeout thread started (timeout={MODEL_IDLE_TIMEOUT}s, poll=5s)"
    )
    while True:
        try:
            time.sleep(5)
            with _model_last_used_lock:
                if (
                    _model_last_used > 0
                    and time.time() - _model_last_used > MODEL_IDLE_TIMEOUT
                ):
                    destroy_paddlex_model()
                    _model_last_used = 0  # reset so we don't re-destroy on next wake
        except Exception:
            logger.exception("Idle timeout thread encountered an error, continuing")


def _start_idle_timeout_thread() -> None:
    """Start the idle-timeout checker as a daemon thread."""
    thread = threading.Thread(
        target=_idle_timeout_checker, daemon=True, name="model-idle-timeout"
    )
    thread.start()
    logger.info("Model idle-timeout background thread started")


# ── Lightweight health check HTTP server (port 8001) ─────────────────
def _worker_is_alive(pid: int) -> bool:
    """Check if the worker process is still running."""
    try:
        os.kill(pid, 0)  # signal 0 doesn't actually kill — just checks existence
        return True
    except (OSError, ProcessLookupError):
        return False


class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler for health checks on port 8001."""

    def do_GET(self) -> None:
        if self.path == "/health":
            # Check if the worker process is alive
            worker_alive = True
            if os.path.exists(WORKER_PID_FILE):
                try:
                    with open(WORKER_PID_FILE, "r") as f:
                        pid = int(f.read().strip())
                    worker_alive = _worker_is_alive(pid)
                except (ValueError, IOError):
                    worker_alive = False

            if worker_alive:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode())
            else:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"status": "unhealthy", "detail": "worker process not running"}
                    ).encode()
                )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: str) -> None:
        """Suppress noisy request logs."""
        pass


def _start_health_check_server() -> None:
    """Start a lightweight health check server on port 8001 in a background thread."""
    try:
        server = http.server.HTTPServer(
            ("0.0.0.0", HEALTH_CHECK_PORT), HealthCheckHandler
        )
        logger.info(f"Health check server starting on port {HEALTH_CHECK_PORT}")
        server.serve_forever()
    except Exception as exc:
        logger.error(f"Health check server failed to start: {exc}")


def _launch_health_check_thread() -> None:
    """Launch the health check server as a daemon thread."""
    thread = threading.Thread(
        target=_start_health_check_server, daemon=True, name="health-check"
    )
    thread.start()


# ── Shutdown helpers ─────────────────────────────────────────────────────
def _stop_idle_timeout_thread() -> None:
    """Stop the idle timeout checker thread.

    Currently no explicit stop mechanism — we rely on the thread being
    daemon so it terminates when the main process exits. If a more graceful
    shutdown is needed, a stop event can be added.
    """
    logger.info(
        "Idle timeout checker: stopping (daemon thread, will exit on process exit)"
    )


def _stop_health_check_thread() -> None:
    """Stop the health check server thread.

    Again, this is a daemon thread that will terminate with the process.
    For a graceful shutdown, we could keep a reference to the server and call
    server.shutdown(), but for this lightweight use case it's not necessary.
    """
    logger.info("Health check server: stopping (daemon thread)")


def _stop_worker_threads() -> None:
    """Stop any long-running worker threads that may be executing async tasks.

    This function is a placeholder — if the app later introduces explicit
    worker threads (e.g., for task queues), they should be added here.
    """
    # Currently no explicit worker threads beyond the background threads above
    pass


# ── Lifespan context manager (replaces deprecated on_event) ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan handler for Doc-Worker startup and shutdown."""
    # ── Startup ──
    separator = "=" * 50
    print(separator, flush=True)
    print("Doc-Worker starting up", flush=True)
    print(separator, flush=True)

    gpu_info = _print_gpu_info()
    print(f"  PaddleX engine ......... {gpu_info}", flush=True)
    print(
        f"  OCR language ........... {OCR_LANG} (PaddleX: {paddleocr_lang_code()})",
        flush=True,
    )
    print(f"  Models dir ............. {PADDLEOCR_MODELS}", flush=True)
    print(
        f"  Use Structure V3 ..... {os.getenv('USE_STRUCTURE_V3', 'true')}", flush=True
    )

    # Model status
    print("  PaddleX models:", flush=True)
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
            status = (
                "✓ online"
                if health.status_code == 200
                else f"✗ HTTP {health.status_code}"
            )
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

    # Start background threads
    _start_idle_timeout_thread()
    _launch_health_check_thread()
    print(f"  Health check server on port {HEALTH_CHECK_PORT}", flush=True)

    # Validate models — abort if critical ones are missing
    if any_missing:
        validate_paddlex_models()  # Raises FileNotFoundError or ValueError

    yield  # app runs here

    # ── Shutdown ──
    print("\nDoc-Worker shutting down...", flush=True)
    # Stop background threads
    _stop_idle_timeout_thread()
    _stop_health_check_thread()
    # Destroy PaddleX models and reclaim memory
    try:
        destroy_paddlex_model()
        print("  PaddleX models destroyed", flush=True)
    except Exception:
        logging.exception("Error during model destruction")
    # Stop any running worker threads (for async tasks)
    _stop_worker_threads()
    print("Shutdown complete", flush=True)


# ── FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(title="Doc-Worker (PaddleOCR-VL)", version="1.0.0", lifespan=lifespan)


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


def _is_text_content(raw_bytes: bytes) -> bool:
    """Check if the raw bytes look like a valid text file (not a PDF or other binary).

    Strategy:
    1. Reject known binary file signatures (PDF, PNG, JPEG, GIF, BMP, TIFF, WebP,
       ZIP-based formats like ODF, etc.).
    2. Check for excessive high-byte or control characters — if more than 5% of the
       first 8 KB are non-printable (outside ASCII printable range), treat as binary.
       This catches most binary files without requiring UTF-8 decoding.
    """
    # Step 1: Known binary signatures (must reject these early)
    binary_signatures = [
        b"%PDF",  # PDF
        b"\x89PNG\r\n\x1a\n",  # PNG
        b"\xff\xd8\xff",  # JPEG (any valid JPEG)
        b"GIF8",  # GIF (87a or 89a)
        b"BM",  # BMP
        b"RIFF",  # WebP, AVI, etc. (RIFF container)
        b"PK\x03\x04",  # ZIP-based: ODF, DOCX, XLSX, etc.
        b"\x1f\x8b",  # GZIP
        b"BZ",  # bzip2
        b"x\xda\x03",  # LZIP
    ]
    for sig in binary_signatures:
        if raw_bytes.startswith(sig):
            return False

    # Step 2: Heuristic check — count bytes that are not in the ASCII printable range
    # We only look at the first 8 KB to avoid full-file scanning.
    sample = raw_bytes[:8192]
    if not sample:
        return False  # Empty file — treat as text (empty text is still text)

    non_printable = sum(
        1
        for byte in sample
        if byte > 127 or (byte < 32 and byte not in (9, 10, 13))  # 9=tab, 10=LF, 13=CR
    )
    # If more than 5% of the sample is non-printable, likely binary
    return non_printable / max(len(sample), 1) <= 0.05


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
    Content Extraction Engine. Returns structured markdown with
    headings, paragraphs, tables, and image placeholders.

    If the file is a text file (not PDF), returns it as-is in the response.
    If PDF processing fails (e.g., pdfium error), falls back to returning
    raw content as text.
    """
    _check_token(authorization)

    # Early health guard: fail fast if model failed to initialize
    init_exc = get_paddlex_init_exception()
    if init_exc is not None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "component": "paddlex",
                "error": str(init_exc),
            },
        )

    file_b64 = body.file
    if not file_b64:
        raise HTTPException(400, "Missing 'file' field")

    file_type = body.fileType
    is_image = file_type == 1

    try:
        file_bytes = base64.b64decode(file_b64)
    except Exception as exc:
        raise HTTPException(400, f"Invalid base64: {exc}")

    # ── Text file detection: if not a PDF, return as-is ──
    if not is_image and _is_text_content(file_bytes):
        # Determine file extension from filename hint or default to .txt
        # (Open-WebUI doesn't send a filename, so we just return the text)
        content = file_bytes.decode("utf-8", errors="replace")
        return JSONResponse(
            content={
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": content}, "structuredBlocks": []}
                    ],
                }
            }
        )

    # ── PDF / image processing ──
    # Determine temp file extension
    suffix = ".png" if is_image else ".pdf"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        _mark_model_used()

        # Use PP-StructureV3 if enabled, fallback to plain OCR
        if os.getenv("USE_STRUCTURE_V3", "true").lower() in ("true", "1"):
            pages = run_paddlex_structure_v3(tmp_path)
            # Build structured markdown per page
            layout_results = []
            for page in pages:
                markdown_text = blocks_to_markdown(page.get("structured_blocks", []))
                layout_results.append(
                    {
                        "markdown": {"text": markdown_text},
                        "structuredBlocks": page.get("structured_blocks", []),
                    }
                )
        else:
            pages = run_paddleocr(tmp_path)
            layout_results = [
                {"markdown": {"text": page["text"]}, "structuredBlocks": None}
                for page in pages
            ]

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
        # ── Error fallback: PDF processing failed (e.g., pdfium error)
        # Return the file content as text if possible ──
        logger.warning(f"PDF processing failed ({exc}), returning raw content as text")
        try:
            content = file_bytes.decode("utf-8", errors="replace")
            return JSONResponse(
                content={
                    "result": {
                        "layoutParsingResults": [
                            {"markdown": {"text": content}, "structuredBlocks": []}
                        ],
                    }
                }
            )
        except Exception:
            raise HTTPException(500, f"Layout parsing failed: {exc}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── Direct API endpoints (not used by Open-WebUI) ────────────────────
@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    init_exception = get_paddlex_init_exception()
    if init_exception is not None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "component": "paddlex",
                "error": str(init_exception),
            },
        )

    return {
        "status": "ok",
        "ocr_lang": OCR_LANG,
        "paddlex_lang": paddleocr_lang_code(),
        "gpu": OCR_USE_GPU,
        "use_structure_v3": os.getenv("USE_STRUCTURE_V3", "true").lower()
        in ("true", "1"),
        "endpoints": ["/layout-parsing", "/extract", "/health"],
    }


@app.post("/extract")
async def extract_text(
    file: UploadFile,
    authorization: str | None = Header(None, alias="Authorization"),
) -> JSONResponse:
    """Direct API: upload PDF/image, get structured text + blocks.

    Accepts: multipart/form-data with field 'file'
    Returns: JSON with pages (structured markdown, structuredBlocks, raw blocks)
    """
    _check_token(authorization)

    # Early health guard: fail fast if model failed to initialize
    init_exc = get_paddlex_init_exception()
    if init_exc is not None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "component": "paddlex",
                "error": str(init_exc),
            },
        )

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

        _mark_model_used()

        # Use PP-StructureV3 if enabled, fallback to plain OCR
        if os.getenv("USE_STRUCTURE_V3", "true").lower() in ("true", "1"):
            pages = run_paddlex_structure_v3(tmp_path)
            # Build structured markdown per page
            result_pages = []
            for page in pages:
                markdown_text = blocks_to_markdown(page.get("structured_blocks", []))
                result_pages.append(
                    {
                        "page": page["page"],
                        "text": page["text"],  # raw text
                        "markdown": markdown_text,
                        "blocks": page["blocks"],  # raw OCR blocks
                        "structured_blocks": page.get("structured_blocks", []),
                    }
                )
        else:
            pages = run_paddleocr(tmp_path)
            result_pages = [
                {
                    "page": p["page"],
                    "text": p["text"],
                    "markdown": p["text"],
                    "blocks": p["blocks"],
                    "structured_blocks": None,
                }
                for p in pages
            ]

        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        return JSONResponse(
            content={
                "filename": file.filename,
                "pages": result_pages,
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
