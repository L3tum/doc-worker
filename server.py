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

import asyncio
import base64
import hmac
import http.server
import io
import json
import logging
import os
import re
import struct
import tempfile
import threading
import time
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import Response

from paddlex_helpers import (
    RENDER_DPI,
    ModelIdleTracker,
    _gpu_memory_stats_mb,
    _paddlex_device,
    blocks_to_markdown,
    destroy_paddlex_model,
    get_paddlex_init_exception,
    migrate_legacy_model_dirs,
    paddleocr_lang_code,
    paddlex_model_is_loaded,
    run_paddleocr,
    run_paddlex_structure_v3,
    validate_paddlex_models,
    warmup_paddlex_models,
)
from serialization import json_default

# ── Config ────────────────────────────────────────────────────────────
OCR_LANG = os.getenv("OCR_LANG", "deu")
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() in ("true", "1", "yes")
PADDLEOCR_VL_TOKEN = os.getenv("PADDLEOCR_VL_TOKEN", "")
PADDLEOCR_MODELS = os.getenv("PADDLEOCR_MODELS", "/app/models")

# Max request body: 100 MB (base64-encoded PDFs can be ~33% larger than raw)
# Clamped to >= 1: 0 would silently 413 every request, no import-time signal.
MAX_REQUEST_SIZE = max(1, int(os.getenv("MAX_REQUEST_SIZE", "104857600")))

# Hard page cap per request. At RENDER_DPI=300 an unbounded PDF is an
# unbounded temp-disk + memory + wall-time allocation on the OCR endpoints
# (unauthenticated when PADDLEOCR_VL_TOKEN is unset — DoS). The count is
# the PAGE TREE's (walked with pypdf; the /Count entry is spec-optional
# and attacker-writable — see _enforce_pdf_caps). Override via
# MAX_OCR_PAGES.
# Guard interaction: _enforce_pdf_caps 413s fail-closed when pypdf's OWN
# page-tree guards trip (depth > 100, entry count > 100,000, cycle —
# see _is_pypdf_page_tree_guard_error) — so raising this cap above
# 100,000 without ALSO raising pypdf's Configuration.page_tree_maximum_
# entries would 413 documents the operator expects to pass.
# Clamped to >= 1: 0 would silently 413 every request, no import-time signal.
MAX_OCR_PAGES = max(1, int(os.getenv("MAX_OCR_PAGES", "500")))

# Per-page rendered-pixel cap (width_px * height_px at RENDER_DPI, of the
# LARGER of each page's EFFECTIVE MediaBox and CropBox — poppler-faithful:
# inherited from the page tree when the page sets none, corners normalized —
# pdftoppm renders the effective MediaBox even when a CropBox is set). A
# tiny PDF with a huge MediaBox (e.g. [0 0 100000 100000], page-level or on
# the /Pages node) passes MAX_REQUEST_SIZE (tiny input) and MAX_OCR_PAGES
# (one page) yet renders to hundreds of GB of RGB at 300 dpi — unbounded
# temp-disk + memory DoS on the unauthenticated API.
# Also applies to IMAGE uploads: the cap reads the header-declared canvas
# (PIL Image.open().size — metadata only, no pixel decode) and rejects a
# few-KB file that declares a canvas whose decode would be tens of GB
# (see _enforce_image_pixel_cap).
# Override via MAX_RENDERED_PIXELS.
# Clamped to >= 1: 0 would silently 413 every request, no import-time signal.
MAX_RENDERED_PIXELS = max(1, int(os.getenv("MAX_RENDERED_PIXELS", "100000000")))

# Cap on in-flight OCR jobs (temp-file staging + pipeline). The to_thread
# offload removed the event loop's implicit one-job-at-a-time
# serialization, and the process-wide predict lock only serializes the
# model phase — so every extra concurrent request (the API is
# unauthenticated unless PADDLEOCR_VL_TOKEN is set) holds decoded bytes +
# a temp file while queued. Saturation returns 503 instead of queueing
# unbounded work. Override via MAX_CONCURRENT_OCR_JOBS.
# Clamped to >= 1 (same pattern as RENDER_DPI / PAPDFTOPPM_TIMEOUT): 0
# would admit no jobs at all (every OCR request 503s "busy" forever, with
# no import-time signal) and a negative value makes asyncio.Semaphore
# raise ValueError at the FIRST request — an opaque generic 500. Clamping
# makes the operator error uniform with the sibling knobs. (A non-numeric
# value is an operator error and still fails loudly at import.)
MAX_CONCURRENT_OCR_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_OCR_JOBS", "2")))

# Cap on concurrent in-flight cap-check PARSES (PDF: page tree +
# per-page boxes via _enforce_pdf_caps; image: header-declared canvas
# via _enforce_image_pixel_cap). They run BEFORE the job slot (a
# rejected job must not take a slot or stage a temp file), so the
# job-slot knob cannot bound their fan-out: without this semaphore, up
# to the default thread-pool size of concurrent parses — each holding up
# to MAX_REQUEST_SIZE of decoded bytes plus pypdf's working set — could
# be in flight at once under saturation.
# asyncio-side on purpose: the slot is acquired in the event loop and
# released before the job slot, so a worker thread is never blocked on
# it (blocking inside the to_thread hop would let parse waiters occupy
# pool threads ahead of pipeline jobs) — and there is no nesting between
# the two semaphores, so they cannot deadlock. Saturated requests get
# 503 instead of queueing, consistent with the job slot. Override via
# MAX_CONCURRENT_PDF_PARSSES.
# Clamped to >= 1 (same pattern as MAX_CONCURRENT_OCR_JOBS): 0 would
# 503 every PDF request "busy" forever with no import-time signal, and
# a negative value makes asyncio.Semaphore raise ValueError at the FIRST
# request — an opaque generic 500. (A non-numeric value is an operator
# error and still fails loudly at import.)
MAX_CONCURRENT_PDF_PARSSES = max(1, int(os.getenv("MAX_CONCURRENT_PDF_PARSSES", "2")))

# Model idle timeout — destroy PaddleOCR model after N seconds of inactivity
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", "30"))

# Worker PID file path (for health check)
WORKER_PID_FILE = os.getenv("WORKER_PID_FILE", "/tmp/doc-worker.pid")
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8001"))

# Image extensions that Open-WebUI sends as fileType=1
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# Shared 415 detail for /layout-parsing and /extract (single source of truth).
UNSUPPORTED_FILE_TYPE_DETAIL = (
    "Unsupported file type: content is not a recognizable PDF, "
    "image (PNG/JPEG/GIF/BMP/WebP/TIFF) or text file."
)

# Shared 503 detail for PaddleX model-init failures — the /health endpoint
# and the OCR endpoints' _require_model_ready guard (single source of
# truth). Generic on purpose: the raw exception text can leak model/temp
# paths (and enriched-download diagnostics) to unauthenticated callers;
# the full failure is logged with traceback at capture time in
# paddlex_helpers.
MODEL_INIT_FAILURE_DETAIL = {
    "status": "unhealthy",
    "component": "paddlex",
    "error": "PaddleX model initialization failed; see server logs",
}

logger = logging.getLogger("doc-worker.api")

# Track when the PaddleOCR model was last used (for idle-timeout destruction)
_model_idle_tracker = ModelIdleTracker()


def _mark_model_used() -> None:
    """Update the model last-use timestamp (thread-safe)."""
    _model_idle_tracker.mark_used()


def _check_idle_once() -> None:
    """One idle-unload decision; the shared logic lives in paddlex_helpers."""
    _model_idle_tracker.check_once(MODEL_IDLE_TIMEOUT)


def _print_gpu_info() -> str:
    """Report the PaddleX device that will actually be used at startup.

    Mirrors paddlex_helpers._paddlex_device() — the same decision that is
    passed to create_pipeline() — so the banner can never claim the model
    runs on a different device than the one it actually does.
    """
    device = _paddlex_device()
    try:
        import paddle

        device_count = (
            paddle.device.cuda.device_count()
            if paddle.device.is_compiled_with_cuda()
            else 0
        )
    except Exception:
        device_count = None

    if device == "gpu:0":
        if device_count is not None:
            return (
                f"GPU (device=gpu:0, {device_count} CUDA device(s), OCR_USE_GPU=true)"
            )
        return "GPU (device=gpu:0, CUDA available, OCR_USE_GPU=true)"
    if OCR_USE_GPU:
        # Explicit operator-requested-GPU fallback: truth-telling line so the
        # banner never hides that the flag had no effect.
        return (
            "CPU (device=cpu — OCR_USE_GPU=true but no CUDA device available; "
            "falling back to CPU)"
        )
    return "CPU (device=cpu, OCR_USE_GPU=false)"


_PADDLEOCR_MODELS_LIST = [
    ("PP-OCRv6_medium_det", "PP-OCRv6_medium_det_infer"),
    ("PP-OCRv6_medium_rec", "PP-OCRv6_medium_rec_infer"),
    ("PP-LCNet_x1_0_textline_ori", "PP-LCNet_x1_0_textline_ori_infer"),
    ("PP-LCNet_x1_0_doc_ori", "PP-LCNet_x1_0_doc_ori_infer"),
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

    Wakes every 5 seconds and delegates the decision to
    paddlex_helpers.ModelIdleTracker.check_once() (shared with the worker),
    which also skips destroys while a job is in flight.
    """
    logger.info(
        f"Model idle timeout thread started (timeout={MODEL_IDLE_TIMEOUT}s, poll=5s)"
    )
    while True:
        try:
            time.sleep(5)
            _check_idle_once()
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
                except (OSError, ValueError):
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
    for model_name, model_dir in _PADDLEOCR_MODELS_LIST:
        status = _model_status(model_name, model_dir)
        marker = "  ✓" if status == "✓" else "  ✗"
        pad = " " * (max(len(n) for n, _ in _PADDLEOCR_MODELS_LIST) - len(model_name))
        print(f"    {model_name}{pad} {marker} {status}", flush=True)

    # Validate models — always run, not just when the status table finds gaps
    migrate_legacy_model_dirs()
    validate_paddlex_models()  # Raises FileNotFoundError or ValueError

    # Warm up model pipelines — surfaces errors before first request arrives
    try:
        warmup_paddlex_models()
        print("  PaddleX models warmed up successfully.", flush=True)
        # Log the loaded-state VRAM so the loaded-vs-idle delta is visible in
        # the container log (the idle-unload log reports the after state).
        gpu_stats = _gpu_memory_stats_mb()
        if gpu_stats is not None:
            print(
                f"  GPU memory after warmup: allocated={gpu_stats[0]:.0f} MB, "
                f"reserved={gpu_stats[1]:.0f} MB",
                flush=True,
            )
        # Start the idle-unload clock from warmup time. Without this, a
        # warmed-up-but-unused model keeps the idle tracker's stamp at 0 and
        # the idle thread's `stamp > 0` guard would never destroy it — leaking
        # VRAM when no requests ever arrive.
        if paddlex_model_is_loaded():
            _mark_model_used()
    except Exception as exc:
        print(
            f"  PaddleX model warm-up failed: {exc}",
            flush=True,
        )
        print(
            "  Continuing anyway — models will be initialized lazily on first use.",
            flush=True,
        )

    # Docling sidecar
    print("  Docling sidecar:", flush=True)
    docling_url = os.getenv("DOCLING_BASE_URL", "")
    if docling_url:
        try:
            import requests

            health = requests.head(f"{docling_url}/health", timeout=5)  # noqa: ASYNC210 — startup check, not request handler
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
        logger.exception("Error during model destruction")
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
    """Bound every request body (see _body_size_policy) to prevent memory
    and temp-disk exhaustion."""
    violation = _body_size_policy(
        request.method,
        request.headers.get("content-length"),
        request.headers.get("transfer-encoding"),
    )
    if violation is not None:
        status, detail = violation
        return JSONResponse(status_code=status, content={"detail": detail})
    return await call_next(request)


def _body_size_policy(
    method: str, content_length: str | None, transfer_encoding: str | None
) -> tuple[int, str] | None:
    """(status_code, detail) for a rejected request body, or None if it may
    proceed.

    - POST with any Transfer-Encoding header: 413 — the body is fully
      materialized (memory + temp file) before processing, and per
      RFC 7230 §3.3.3 the transport honors Transfer-Encoding and IGNORES
      Content-Length when both are present — a lying small Content-Length
      (e.g. ``Content-Length: 10`` + ``Transfer-Encoding: chunked``) would
      otherwise sail past the size policy while the app materializes the
      full chunked body (unbounded memory / temp-disk DoS on the
      unauthenticated-by-default API). Checked BEFORE Content-Length: the
      length is unbounded no matter what it claims.
    - POST without Content-Length (chunked): 413 — same reasoning: without
      a declared length the body is unbounded.
    - POST with a non-digit Content-Length (RFC 7230: 1*DIGIT), or a
      digit string too long for int() to parse (>4300 digits on
      CPython 3.11+): 400 — malformed header (the HTTP layer usually
      rejects it first, but the app must not 500).
    - POST with Content-Length > MAX_REQUEST_SIZE: 413.
    - Non-POST: no body to bound.
    """
    if method != "POST":
        return None
    if transfer_encoding is not None:
        return (
            413,
            (
                "Transfer-Encoding is not supported; send Content-Length only "
                "(a Transfer-Encoding makes Content-Length ignored and the "
                "body unbounded)"
            ),
        )
    if content_length is None:
        return (
            413,
            "Transfer-Encoding: chunked uploads are not supported; send Content-Length",
        )
    if not (content_length.isascii() and content_length.isdigit()):
        return 400, "Invalid Content-Length header"
    try:
        length = int(content_length)
    except ValueError:
        # CPython 3.11+ raises ValueError for digit strings longer than
        # 4300 characters (int-str digit limit) — a ~4 KB all-digit
        # header is malformed for our purposes: 400, never a 500.
        return 400, "Invalid Content-Length header"
    if length > MAX_REQUEST_SIZE:
        return 413, f"Request too large (max {MAX_REQUEST_SIZE // 1024 // 1024} MB)"
    return None


# ── Helpers ───────────────────────────────────────────────────────────
class SafeJSONResponse(JSONResponse):
    """JSONResponse that survives numpy scalars/arrays leaking into payloads."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            default=json_default,
        ).encode("utf-8")


_BEARER_PREFIX = "Bearer "


def _check_token(authorization: str | None) -> None:
    """Validate the Authorization: Bearer <value> header.

    The presented token is compared with hmac.compare_digest (constant-time)
    against PADDLEOCR_VL_TOKEN.
    """
    if not PADDLEOCR_VL_TOKEN:
        return  # No token configured, skip auth
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    if authorization[: len(_BEARER_PREFIX)].lower() != _BEARER_PREFIX.lower():
        # RFC 6750: the auth scheme is case-insensitive (RFC 7235) —
        # clients may send "bearer"/"BEARER"; the token after the
        # scheme is NOT case-folded (the compare below stays
        # constant-time).
        raise HTTPException(403, "Invalid token")
    token = authorization[len(_BEARER_PREFIX) :]
    if not hmac.compare_digest(
        token.encode("utf-8"), PADDLEOCR_VL_TOKEN.encode("utf-8")
    ):
        raise HTTPException(403, "Invalid token")


# ── Magic-byte signatures (canonical table) ───────────────────────────
# ONE canonical table of (name, predicate) entries drives BOTH
# _is_text_content's step-1 binary rejection and detect_file_type's
# pdf/image classification. Signatures are anchored at the start of the
# content after skipping leading whitespace/control bytes (see
# _leading_noise_offset).
#
# The reject-only entries (zip, gzip, bzip2, lzip) are binary containers we
# do not accept: _is_text_content rejects them, and detect_file_type reports
# them as "unknown" (never "image").
#
# WebP is the only RIFF content accepted: WAV/AVI share the RIFF container,
# so a bare "RIFF" prefix must not classify as image. BMP requires a
# plausible file-size DWORD (offset 2) plus a known DIB header-size DWORD
# (offset 14), which is what excludes plain text starting with "BMW…".

_BMP_DIB_SIZES = frozenset({12, 40, 64, 108, 124})


def _sig_pdf(content: bytes, offset: int) -> bool:
    return content.startswith(b"%PDF", offset)


def _sig_png(content: bytes, offset: int) -> bool:
    return content.startswith(b"\x89PNG\r\n\x1a\n", offset)


def _sig_jpeg(content: bytes, offset: int) -> bool:
    return content.startswith(b"\xff\xd8\xff", offset)


def _sig_gif(content: bytes, offset: int) -> bool:
    return content.startswith(b"GIF8", offset)  # GIF87a or GIF89a


def _sig_bmp(content: bytes, offset: int) -> bool:
    # "BM" + file-size DWORD (anchor+2; 0 = unspecified, else >= 14) +
    # DIB header-size DWORD (anchor+14) in _BMP_DIB_SIZES. 18 bytes from
    # the anchor needed to read both DWORDs.
    if len(content) - offset < 18 or content[offset : offset + 2] != b"BM":
        return False
    file_size = struct.unpack_from("<I", content, offset + 2)[0]
    dib_size = struct.unpack_from("<I", content, offset + 14)[0]
    return (file_size == 0 or file_size >= 14) and dib_size in _BMP_DIB_SIZES


def _sig_webp(content: bytes, offset: int) -> bool:
    # RIFF-anchored: "RIFF" magic at the anchor plus the "WEBP" fourcc at
    # anchor+8. The anchor is required so a non-RIFF "WEBP" at anchor+8
    # (e.g. plain ASCII text) does not classify as image; WAV/AVI share
    # the RIFF container and stay excluded by the fourcc. The length
    # guard is implicit — slicing shorter bytes yields a shorter result.
    return (
        content[offset : offset + 4] == b"RIFF"
        and content[offset + 8 : offset + 12] == b"WEBP"
    )


def _sig_tiff_le(content: bytes, offset: int) -> bool:
    return content.startswith(b"II\x2a\x00", offset)


def _sig_tiff_be(content: bytes, offset: int) -> bool:
    return content.startswith(b"MM\x00\x2a", offset)


def _sig_zip(content: bytes, offset: int) -> bool:
    return content.startswith(b"PK\x03\x04", offset)  # ZIP-based: ODF, DOCX, XLSX, etc.


def _sig_gzip(content: bytes, offset: int) -> bool:
    return content.startswith(b"\x1f\x8b", offset)


def _sig_bzip2(content: bytes, offset: int) -> bool:
    return content.startswith(b"BZ", offset)


def _sig_lzip(content: bytes, offset: int) -> bool:
    return content.startswith(b"x\xda\x03", offset)


_FILE_SIGNATURES: tuple[tuple[str, Callable[[bytes, int], bool]], ...] = (
    ("pdf", _sig_pdf),
    ("png", _sig_png),
    ("jpeg", _sig_jpeg),
    ("gif", _sig_gif),
    ("bmp", _sig_bmp),
    ("webp", _sig_webp),
    ("tiff_le", _sig_tiff_le),
    ("tiff_be", _sig_tiff_be),
    # Reject-only: binary containers, unsupported input.
    ("zip", _sig_zip),
    ("gzip", _sig_gzip),
    ("bzip2", _sig_bzip2),
    ("lzip", _sig_lzip),
)

# _FILE_SIGNATURES names that detect_file_type classifies as "image".
_IMAGE_SIGNATURE_NAMES = frozenset(
    {"png", "jpeg", "gif", "bmp", "webp", "tiff_le", "tiff_be"}
)

# Leading bytes skipped before the magic-byte anchors: controls 0x00–0x1F
# plus space. Some producers pad the file header, and an offset-0-strict
# "%PDF" anchor would misroute such files: leading binary header bytes made
# them 415 (they OCR'd fine before), and leading whitespace made them pass
# through as raw text. Matching the anchored offset inspects only the short
# leading run at the front of the buffer (no full-body copy or scan), so
# it stays cheap on 100 MB uploads.
_LEADING_NOISE_RE = re.compile(rb"[\x00-\x20]*")


def _leading_noise_offset(content: bytes) -> int:
    """Offset where leading whitespace/control bytes (0x00–0x20) end.

    The anchor for the magic-byte signatures (see _FILE_SIGNATURES):
    predicates run at this offset, so a %PDF header padded with leading
    LF/CR/NUL still classifies as PDF. The probe is a pure function of the
    input — same input, same result — whichever predicate consumes it.
    """
    # ``[\x00-\x20]*`` always matches (the empty string at minimum), so
    # the match is never None at runtime; the explicit fallback (not an
    # assert, which -O strips) is the same value 0 the regex would report
    # anyway, so a regex change can never misclassify a file.
    m = _LEADING_NOISE_RE.match(content)
    return m.end() if m is not None else 0


def _is_text_content(raw_bytes: bytes) -> bool:
    """Check if raw bytes look like valid text (not a PDF or other binary).

    Strategy:
    1. Reject known binary file signatures (the canonical
       _FILE_SIGNATURES table, anchored after leading whitespace/control
       bytes — see _leading_noise_offset).
    2. Strict UTF-8 decode of the first 8 KB: UnicodeDecodeError means
       binary. NUL bytes in the sample are rejected explicitly — NUL is
       *valid* UTF-8, so the decode alone cannot catch 7-bit-clean
       binaries (e.g. a near-silent small WAV).
    3. For files > 10 KB, also decode the last 8 KB to catch binary formats
       with printable headers (e.g., uncompressed PDFs with binary stream
       data later in the file); NULs in the tail are rejected the same way.
    """
    if not raw_bytes:
        return False

    # Step 1: Known binary signatures (anchored after leading
    # whitespace/control bytes — see _leading_noise_offset)
    offset = _leading_noise_offset(raw_bytes)
    for _name, predicate in _FILE_SIGNATURES:
        if predicate(raw_bytes, offset):
            return False

    # Step 2: Strict UTF-8 decode of first 8 KB — binary data almost always
    # contains invalid UTF-8 byte sequences. This correctly handles multi-byte
    # UTF-8 text (German umlauts, CJK, emoji) which the old 5% high-byte ratio
    # heuristic misclassified as binary.
    sample = raw_bytes[:8192]
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False

    # Step 2b: NUL is valid UTF-8, so the decode above cannot catch
    # 7-bit-clean binaries (every byte < 0x80) — without this check a
    # near-silent small WAV would pass as "text" and be returned verbatim
    # instead of a 415. Real text files do not contain NUL bytes.
    if b"\x00" in sample:
        return False

    # Step 3: For large files, also check the tail (last 8 KB) to catch binary
    # formats with printable headers (e.g., uncompressed PDFs whose first 8 KB
    # is valid UTF-8 dictionary text but whose stream data is binary).
    if len(raw_bytes) > 10240:
        tail = raw_bytes[-8192:]
        try:
            tail.decode("utf-8")
        except UnicodeDecodeError:
            return False
        if b"\x00" in tail:
            return False

    return True


def detect_file_type(content: bytes) -> Literal["pdf", "image", "text", "unknown"]:
    """Classify decoded file content by magic bytes, not caller hints.

    Order: %PDF → known image signatures → UTF-8 text check → unknown.
    Reject-only binaries (ZIP/GZIP/BZ2/LZIP) never classify as "image" —
    they fail the text check and therefore yield "unknown".
    The result drives pipeline routing (PDF → OCR / image → OCR / text
    passthrough) and the temp-file suffix: a wrong or missing fileType
    hint or filename extension must not be able to send bytes down the
    wrong pipeline.
    """
    if not content:
        return "unknown"
    # One leading-noise scan for the whole signature pass: the offset is a
    # pure function of the input, so every predicate sees the same anchor
    # regardless of order — same input, same result.
    offset = _leading_noise_offset(content)
    for name, predicate in _FILE_SIGNATURES:
        if predicate(content, offset):
            if name == "pdf":
                return "pdf"
            if name in _IMAGE_SIGNATURE_NAMES:
                return "image"
            return "unknown"  # reject-only binary (zip/gzip/bzip2/lzip)
    if _is_text_content(content):
        return "text"
    return "unknown"


# ── Shared request plumbing (used by /layout-parsing and /extract) ──────
def _require_model_ready() -> None:
    """Early health guard for the OCR endpoints: fail fast with the same
    503 shape as /health if model initialization failed at startup."""
    if get_paddlex_init_exception() is not None:
        raise HTTPException(status_code=503, detail=MODEL_INIT_FAILURE_DETAIL)


def _route_upload(file_bytes: bytes, hint_kind: str, endpoint: str) -> str:
    """Content-based routing shared by /layout-parsing and /extract: the
    BYTES decide the pipeline, not the caller's fileType hint (which
    Open-WebUI can get wrong or omit) or the upload's extension.

    Returns the detected kind ("pdf", "image", or "text"); raises 415 for
    unrecognized non-text content.
    """
    kind = detect_file_type(file_bytes)
    if kind == "unknown":
        raise HTTPException(415, UNSUPPORTED_FILE_TYPE_DETAIL)
    if kind != hint_kind:
        # A pdf/image hint on plain text is the normal Open-WebUI/.txt
        # passthrough case — debug, not a warning.
        log_fn = logger.debug if kind == "text" else logger.warning
        log_fn(
            "%s: hint %r disagrees with content detection %s — routing by content",
            endpoint,
            hint_kind,
            kind,
        )
    return kind


def _is_pypdf_page_tree_guard_error(exc: BaseException) -> bool:
    """True if *exc* is one of pypdf's OWN page-tree safety guards rather
    than a parse failure — the distinction that keeps the fail-soft
    fallback safe.

    pypdf refuses to measure page trees it cannot bound: depth >
    Configuration.page_tree_maximum_depth (default 100), array/entry
    counts > its 100,000-element limits, or cycles — verified against
    pinned pypdf 6.21.1 to raise ``pypdf.errors.LimitReachedError``
    (depth/entries) or ``pypdf.errors.PdfReadError`` (cycles). The
    ``PdfReadError`` clause is deliberately broader than cycles: ANY
    ``PdfReadError`` from the guarded walk fails closed, so a
    hypothetical legitimate document whose tree walk raises for a
    non-cycle reason is also rejected — the safe direction (a
    malformed page tree does not render in poppler either, and the
    only alternative count is still the attacker-writable /Count).
    The classification applies to exceptions from the GUARDED TREE WALK
    (get_num_pages → _flatten / _flatten(True) in _enforce_pdf_caps),
    not to the whole pypdf pass: a guard trip there means the tree is
    provably degenerate (unmeasurable by pypdf), so the only remaining
    page count — pypdfium2's /Count-gated count — is exactly the
    attacker-writable value the gate exists to distrust: falling back
    to it would silently re-open the lying-/Count DoS, so the guarded
    branch fails CLOSED (413). A CONSTRUCTOR-level parse failure (pypdf
    chokes opening the document — e.g. a spec-legal PDF missing the
    startxref section — while pypdfium2 opens the same bytes) is not a
    guard trip either, but it fails CLOSED in _enforce_pdf_caps (the
    whole per-page pixel cap would be skipped for a renderer-openable
    document; see the pypdf pin in requirements.txt). Only a post-open
    mid-pass failure — or a document NEITHER parser opens — keeps the
    documented fail-soft. Returns False when pypdf is not importable
    (nothing matches; the caller keeps the fail-soft fallback).
    """
    try:
        from pypdf import errors as pypdf_errors
    except ImportError:
        return False
    return isinstance(exc, (pypdf_errors.LimitReachedError, pypdf_errors.PdfReadError))


def _enforce_pdf_caps(file_bytes: bytes, kind: str) -> None:
    """Enforce BOTH PDF request caps over the in-memory request bytes,
    BEFORE any render or model work — a rejected job never stages its
    temp file. Images/text are single-page by construction; images are
    additionally bounded by the header-declared-canvas pixel cap in
    _enforce_image_pixel_cap (both caps run before the job slot).

    Concurrency: this function is stateless, but its caller bounds how
    many cap checks run at once — _run_ocr_pipeline wraps the call in
    the parse slot (_pdf_parse_slot, MAX_CONCURRENT_PDF_PARSSES),
    because the check runs BEFORE the job slot and the job-slot knob
    cannot bound work that precedes it.

    Page cap (MAX_OCR_PAGES, checked FIRST, before any per-page work):
    an unbounded page count is unbounded 300-dpi temp-disk staging. The
    count must be the PAGE TREE's count: the /Count entry of the /Pages
    node is spec-optional and attacker-writable, and pypdfium2's
    CountPages() returns the raw /Count whenever 0 < /Count < 0xFFFFF
    without cross-checking /Kids (verified against pinned pypdfium2
    5.14.0: /Count=1 + 3 real kids → len() == 1; indexed page access is
    /Count-gated the same way, so PDFium never sees the extra pages). A
    spec-legal few-MB PDF with a small /Count and N real leaf pages
    passes such a count gate — and every other consumer of the same
    /Count-gated count misses the tree too: a per-page defense that
    walks the /Count-gated page sequence measures only the first /Count
    pages, and poppler's render loop is bounded by the same raw /Count
    (verified against poppler 22.09/22.12/24.02 source: Catalog::
    getNumPages() returns the raw /Count — the lying-small vector is a
    count-gate + measurement-skip, NOT a render-all-N vector; the page-
    SIZE dimension is the concrete residual: a huge box on any of the
    first /Count pages renders hundreds of GB from a few-KB file). Un-
    bounded temp-disk + CPU on the API, which is unauthenticated by
    default. An OVERSTATED /Count is the mirror image: pypdfium2's
    count inflates and a valid-but-malformed document would 413
    spuriously. The authoritative count is therefore pypdf's — it walks
    the page tree (get_num_pages → _flatten, with pypdf's own
    cycle/depth guards) — and the pypdfium2 count is only the fallback
    when pypdf cannot parse the document. For ENCRYPTED documents
    pypdf's page count is /Count-gated instead (get_num_pages returns
    the raw /Count when is_encrypted — no tree walk — and the page
    sequence is truncated to the same /Count), so the cap decrypts with
    the empty user password and forces pypdf's guarded tree walk
    (_flatten) and counts + measures its page list; a failed decryption
    means the document requires a user password, which pdftoppm cannot
    satisfy either — the pipeline reports its own error (fail-soft).

    Per-page pixel cap (MAX_RENDERED_PIXELS): a tiny one-page PDF with a
    huge MediaBox (e.g. [0 0 100000 100000]) passes MAX_REQUEST_SIZE
    (tiny input) and MAX_OCR_PAGES (one page) yet stages hundreds of GB
    of RGB at 300 dpi. Measured with pypdf, because the renderer
    (pdftoppm, no -cropbox flag) rasterizes the EFFECTIVE MediaBox —
    inherited from the page tree when the page sets none, inverted
    corners normalized — and pypdfium2's page-level box APIs see
    neither: FPDFPage_GetMediaBox does not walk /Parent (PDFium crbug
    1786) and returns the corners as written. Spec-legal ~300-byte PDFs
    exploit exactly this: a 100000x100000 MediaBox on the /Pages node
    with a normal CropBox on the page measures 612x792 to pypdfium2 yet
    renders 416667x416667 px at 300 dpi (~521 GB); a page-level
    MediaBox with swapped corners ([100000 0 0 792] behind a CropBox)
    is likewise invisible. pypdf resolves the page-tree inheritance and
    poppler does the same, so the measurement matches the render. Each
    page is measured as the LARGER of its effective MediaBox and
    effective CropBox per axis (abs() of the corner deltas — pypdf
    keeps inverted corners as written, poppler swaps them) because the
    pypdfium2 rasterizer ocrmypdf uses renders the CropBox while
    pdftoppm renders the MediaBox; a page whose w_px * h_px at
    RENDER_DPI exceeds the cap 413s.

    FAIL CLOSED when pypdf's OWN page-tree guards trip DURING the guarded
    tree walk (depth > 100, array/entry counts > 100,000, cycle —
    LimitReachedError / PdfReadError, verified against pinned pypdf
    6.21.1, see _is_pypdf_page_tree_guard_error): the tree is then
    provably unmeasurable, and the only remaining count (pypdfium2's)
    is /Count-gated — the attacker-writable value this gate distrusts;
    a degenerate tree with a lying small /Count and a huge-box leaf is
    exactly the shape that would otherwise sail through the fallback
    (413, logged with traceback). ALSO FAIL CLOSED when pypdf's
    CONSTRUCTOR rejects the bytes while pypdfium2 opens the same ones
    (the verified parser-disagreement shape: a spec-legal PDF with a
    valid xref table + trailer but no startxref section — pypdf raises
    PdfReadError("startxref not found"), pypdfium2/poppler read the
    xref table/trailer directly): failing soft there would skip the
    per-page pixel cap ENTIRELY for a renderer-openable document, and a
    ~51 KB 500-page × 6000 pt file would then render ≈1.88 GB of RGB
    per page (~19× the default MAX_RENDERED_PIXELS envelope) for the
    full render timeout, per unauthenticated request (413
    "Unparseable PDF", logged with traceback). Fail-soft otherwise,
    and only where a spurious 413 would be worse: bytes NEITHER parser
    can open return immediately (BOTH caps skipped — the pipeline
    reports its own error instead of a spurious 413), and a document
    pypdf OPENED but that then fails a mid-pass step (decrypt /
    enumeration — the page cap already applied when the tree count
    succeeded) is gated on the page cap by the pypdfium2 count where
    that is the only count available — a documented residual, logged
    with traceback; see the pypdf pin in requirements.txt. A page whose
    box pypdf cannot resolve (missing everywhere, malformed, cyclic
    /Parent, or the inline-/Kids-leaf shape — an inline /Page dict whose
    own entries the list-only flatten drops: on the unencrypted path
    because the read-only reader's get_num_pages runs the list-only
    _flatten, on the encrypted path via _flatten(True)) is skipped
    individually: poppler renders such a page at the 612x792 spec
    default, or rejects it outright and renders ≈1 px (the
    inline-/Kids-leaf shape, verified against the base-image poppler
    22.09) — bounded either way.

    Deliberate over-rejection (the fail-closed direction): an
    UNDERSTATED /Count whose huge box sits on a page BEYOND the raw
    /Count 413s on the page cap (the true count) even though poppler's
    render loop is clamped to the raw /Count and would never render
    that page — the true tree count is the value to trust, and the
    /Count-gated count is the attacker-writable one this gate exists
    to distrust.
    """
    if kind != "pdf":
        return

    # pypdfium2 count: a fast C++ read of the same bytes the pipeline
    # rasterizes. It is /Count-gated (see docstring) — authoritative
    # only when pypdf cannot parse the document.
    pdfium_page_count: int | None = None
    try:
        import pypdfium2

        with pypdfium2.PdfDocument(file_bytes) as doc:
            pdfium_page_count = len(doc)
    except Exception:
        pdfium_page_count = None

    try:
        import pypdf
    except ImportError:
        # pypdf is a hard dep of the cap (requirements.txt) — an import
        # failure is an install error, not a document property: keep the
        # documented fail-soft (page cap on the /Count-gated pypdfium2
        # count only).
        logger.warning(
            "_enforce_pdf_caps: pypdf not importable — page cap on the "
            "/Count-gated pypdfium2 count only (known residual)"
        )
        if pdfium_page_count is not None and pdfium_page_count > MAX_OCR_PAGES:
            raise HTTPException(
                status_code=413,
                detail=(f"Too many pages: {pdfium_page_count} (max {MAX_OCR_PAGES})"),
            )
        return

    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes), strict=False)
    except Exception as exc:
        if pdfium_page_count is None:
            # NEITHER parser opens the bytes: nothing downstream can
            # render them — fail-soft, the pipeline reports its own
            # error (no spurious 413).
            logger.warning(
                "_enforce_pdf_caps: pypdf could not parse the document "
                "(%s) and pypdfium2 could not open it either — caps "
                "skipped",
                type(exc).__name__,
                exc_info=True,
            )
            return
        # pypdf's CONSTRUCTOR rejects the bytes while pypdfium2 opens
        # the same ones (verified shape: a spec-legal PDF with a valid
        # xref table + trailer but no startxref section — pypdf raises
        # PdfReadError("startxref not found"), pypdfium2/poppler read
        # the xref table/trailer directly). Failing soft here would
        # skip the per-page pixel cap ENTIRELY while the renderer still
        # renders: a ~51 KB 500-page × 6000 pt file would render ≈1.88
        # GB of RGB per page (~19× the default MAX_RENDERED_PIXELS
        # envelope) for the full render timeout, per unauthenticated
        # request. Fail CLOSED — consistent with the page-tree guard
        # trips (and with the repo's verified poppler behavior on
        # unmeasurable shapes).
        logger.warning(
            "_enforce_pdf_caps: pypdf could not parse the document (%s) "
            "but pypdfium2 opens it — 413 (the pixel cap cannot be "
            "applied; the /Count-gated count is attacker-writable)",
            type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(status_code=413, detail="Unparseable PDF")

    try:
        # Both branches produce a sequence of PageObjects (pypdf's list
        # or its virtual page list); the annotation is never evaluated
        # at runtime (PEP 526: local variable annotations are not
        # evaluated) so the lazy pypdf import stays import-time-free.
        pages: Sequence[pypdf.PageObject]
        if reader.is_encrypted and not reader.decrypt(""):
            # A failed empty-user-password decryption means a user
            # password is required (pdftoppm cannot open it either) —
            # fail-soft, the pipeline reports its own error. (The
            # encrypted page count/sequence is /Count-gated while the
            # renderer walks the page tree — the guarded block below
            # forces pypdf's tree walk for encrypted docs.)
            return
        try:
            # THE GUARDED TREE WALK — the only place pypdf's OWN safety
            # guards can fire: get_num_pages → _flatten (depth / entry-
            # count / cycle). A guard trip means the tree is provably
            # unmeasurable, so the page count is unobtainable except via
            # the /Count-gated fallback this gate distrusts: fail CLOSED
            # (413) instead of falling soft. Parse-level failures
            # (constructor, decrypt) sit OUTSIDE this block and keep the
            # documented fail-soft residual in the outer handler.
            if reader.is_encrypted:
                # Force pypdf's guarded tree walk (the encrypted page
                # count is /Count-gated — no tree walk).
                reader._flatten(True)
                if reader.flattened_pages is None:
                    # _flatten always sets the list (possibly empty);
                    # None would be a pypdf internal-state regression —
                    # treat as uncountable (outer handler falls back to
                    # the pypdfium2 count). No assert: stripped under -O.
                    raise ValueError("pypdf page-tree flatten returned no list")
                pages = reader.flattened_pages
                page_count = len(pages)
            else:
                # Page-tree count (get_num_pages → _flatten) —
                # authoritative, unlike the /Count-gated counts above.
                page_count = len(reader.pages)
                pages = reader.pages
        except Exception as tree_exc:
            if _is_pypdf_page_tree_guard_error(tree_exc):
                logger.warning(
                    "_enforce_pdf_caps: pypdf page-tree guard tripped "
                    "(%s) — 413 (tree unmeasurable; the /Count-gated "
                    "fallback would defeat the cap)",
                    type(tree_exc).__name__,
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=413,
                    detail=("Page tree unmeasurable (exceeds internal parse guard)"),
                )
            raise
        if page_count > MAX_OCR_PAGES:
            raise HTTPException(
                status_code=413,
                detail=f"Too many pages: {page_count} (max {MAX_OCR_PAGES})",
            )
        for i, page in enumerate(pages, start=1):
            try:
                mediabox = page.mediabox
            except Exception:
                # Unresolvable box (missing everywhere, malformed, cyclic
                # /Parent, encrypted inline-/Kids leaf): poppler either
                # renders such a page at the 612x792 spec default or
                # rejects it outright (≈1 px) — bounded either way; skip
                # it, keep capping the rest.
                logger.debug(
                    "_enforce_pdf_caps: page %s box unresolvable, "
                    "skipping pixel-cap check for that page",
                    i,
                    exc_info=True,
                )
                continue
            try:
                cropbox = page.cropbox
            except Exception:
                # CropBox malformed but MediaBox fine: the MediaBox is
                # the pdftoppm box — cap on it alone.
                logger.debug(
                    "_enforce_pdf_caps: page %s CropBox unresolvable, "
                    "capping on the MediaBox alone",
                    i,
                    exc_info=True,
                )
                cropbox = mediabox
            w_pt = max(abs(mediabox.width), abs(cropbox.width))
            h_pt = max(abs(mediabox.height), abs(cropbox.height))
            w_px = w_pt * RENDER_DPI / 72.0
            h_px = h_pt * RENDER_DPI / 72.0
            if w_px * h_px > MAX_RENDERED_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Rendered page {i} exceeds {MAX_RENDERED_PIXELS} "
                        f"pixels at {RENDER_DPI} dpi"
                    ),
                )
    except HTTPException:
        raise
    except Exception as exc:
        # pypdf OPENED the document but a LATER step failed (decrypt /
        # enumeration / a non-guarded tree walk), not a tree-guard trip
        # (those 413 inside the guarded block above) and not a
        # constructor failure (handled above, fail-closed). The page
        # cap already applied when the tree count succeeded; the
        # /Count-gated pypdfium2 count is the only count left to gate
        # on — apply the page cap on it where available. This branch
        # degrades a security invariant, so it must be visible in the
        # logs.
        logger.warning(
            "_enforce_pdf_caps: pypdf failed after opening the document "
            "(%s) — page cap on the /Count-gated pypdfium2 count where "
            "applicable (known residual)",
            type(exc).__name__,
            exc_info=True,
        )
        if pdfium_page_count is not None and pdfium_page_count > MAX_OCR_PAGES:
            raise HTTPException(
                status_code=413,
                detail=(f"Too many pages: {pdfium_page_count} (max {MAX_OCR_PAGES})"),
            )
        return


def _enforce_image_pixel_cap(file_bytes: bytes) -> None:
    """Enforce MAX_RENDERED_PIXELS on IMAGE uploads — metadata-only,
    BEFORE the job slot (a rejected job must not take a slot or stage a
    temp file).

    A highly compressible PNG/TIFF can be a few KB/MB on disk yet
    declare a 100 000 × 100 000 px canvas — tens of GB at decode — and
    still pass MAX_REQUEST_SIZE: the file is tiny, the canvas is not
    (the PDF equivalent is the huge-MediaBox shape
    _enforce_pdf_caps closes). The decode happens inside the pipeline
    (PaddleX reads the staged file), so the cap reads the DECLARED
    canvas from the header — PIL Image.open().size touches the header
    only, no pixel decode — and 413s a canvas whose decode would blow
    the envelope.

    Fail-soft on unreadable headers (PIL cannot identify the bytes):
    they are undecodable by the pipeline too (its own error applies) —
    there is no render to bound. A canvas above PIL's OWN
    decompression-bomb ERROR threshold (2 × MAX_IMAGE_PIXELS, default
    ≈179 Mpx) makes Image.open raise before the size is readable; that
    413s whenever the threshold sits at/above MAX_RENDERED_PIXELS
    (the default case: ≈179 Mpx > 100 Mpx) — an operator who raises
    the cap above the threshold keeps PIL's own bounded error from the
    pipeline instead (no allocation past the threshold).
    """
    try:
        from PIL import Image
    except ImportError:
        # Pillow is a hard dep of the pipeline (ocrmypdf) — an import
        # failure is an install error, not a document property.
        return
    try:
        # Suppress PIL's "exceeds limit ... could be a decompression
        # bomb" WARNING for canvases between the warning threshold and
        # the error threshold — the cap below makes the outcome
        # deterministic either way.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(io.BytesIO(file_bytes)) as img:
                w, h = img.size
    except Image.DecompressionBombError:
        threshold = (
            2 * Image.MAX_IMAGE_PIXELS if Image.MAX_IMAGE_PIXELS is not None else None
        )
        if threshold is not None and threshold >= MAX_RENDERED_PIXELS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Image exceeds {MAX_RENDERED_PIXELS} pixels at the "
                    "header-declared canvas size"
                ),
            )
        return
    except Exception:
        return  # undecodable header → the pipeline's own error applies
    if w * h > MAX_RENDERED_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image exceeds {MAX_RENDERED_PIXELS} pixels at the "
                f"header-declared canvas size ({w}x{h})"
            ),
        )


# Per-event-loop semaphores: asyncio primitives bind to the running
# loop on first use, and the test suite drives the app under multiple
# loops (one per TestClient request) — a single module-level semaphore
# would raise "bound to a different event loop" on the second loop.
# Invariant: the cap is PER EVENT LOOP; a single uvicorn process has one
# loop, so MAX_CONCURRENT_OCR_JOBS / MAX_CONCURRENT_PDF_PARSSES bound the
# whole process in production. Each registry holds strong refs to its
# entries, and closed-loop entries are pruned on lookup so per-test-loop
# churn does not grow the dict unboundedly.
_OCR_JOB_SEMAPHORES: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}
_PDF_PARSE_SEMAPHORES: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


def _per_loop_semaphore(
    registry: dict[asyncio.AbstractEventLoop, asyncio.Semaphore],
    capacity: int,
) -> asyncio.Semaphore:
    """Semaphore for the current event loop, created lazily on first use
    (see the registry note above). The capacity is read at CREATION
    time (first use on this loop) from the current module constant, so
    a patched constant applies to a fresh registry (the test pattern).
    """
    loop = asyncio.get_running_loop()
    # Opportunistic prune: drop entries whose loop is already closed (the
    # test suite creates one loop per test) so the strong-ref dict does
    # not accumulate one entry per test run.
    stale = [
        stale_loop
        for stale_loop in registry
        if stale_loop is not loop and stale_loop.is_closed()
    ]
    for stale_loop in stale:
        del registry[stale_loop]
    sem = registry.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(capacity)
        registry[loop] = sem
    return sem


def _ocr_job_semaphore() -> asyncio.Semaphore:
    """Semaphore for the current event loop (job slot), created lazily."""
    return _per_loop_semaphore(_OCR_JOB_SEMAPHORES, MAX_CONCURRENT_OCR_JOBS)


def _pdf_parse_semaphore() -> asyncio.Semaphore:
    """Semaphore for the current event loop (parse slot), created lazily."""
    return _per_loop_semaphore(_PDF_PARSE_SEMAPHORES, MAX_CONCURRENT_PDF_PARSSES)


@asynccontextmanager
async def _saturated_slot(sem: asyncio.Semaphore, detail: str) -> AsyncIterator[None]:
    """Admit under *sem*; 503 with *detail* when saturated — queued
    requests would otherwise each keep their decoded request bytes (up
    to MAX_REQUEST_SIZE) alive while waiting. acquire()'s fast path is
    synchronous (no yield), so the locked() check and the acquire are
    atomic on the single-threaded event loop. A rejection (e.g. 413)
    inside the slot still releases it (the finally)."""
    if sem.locked():
        raise HTTPException(503, detail)
    await sem.acquire()
    try:
        yield
    finally:
        sem.release()


@asynccontextmanager
async def _ocr_job_slot() -> AsyncIterator[None]:
    """Admit at most MAX_CONCURRENT_OCR_JOBS in-flight OCR jobs; 503 when
    saturated — queued requests would otherwise keep their decoded request
    bytes alive while waiting behind the process-wide predict lock (temp-file
    staging happens only AFTER the slot is taken, so a queued job holds no
    temp file — only the in-memory request bytes)."""
    async with _saturated_slot(
        _ocr_job_semaphore(),
        "OCR service busy (all job slots in use); try again shortly",
    ):
        yield


@asynccontextmanager
async def _pdf_parse_slot() -> AsyncIterator[None]:
    """Admit at most MAX_CONCURRENT_PDF_PARSSES in-flight cap-check
    parses (PDF: page tree + per-page boxes; image: header-declared
    canvas); 503 when saturated — queued parses would otherwise each
    keep their decoded request bytes (up to MAX_REQUEST_SIZE) alive
    while waiting. The parse runs BEFORE the job slot (a rejected job
    must not take a slot), so its fan-out is bounded HERE, not by
    MAX_CONCURRENT_OCR_JOBS. Acquired in the event loop — a worker
    thread is never blocked waiting for a slot — and released before
    the job slot is taken, so the two semaphores are never nested (no
    deadlock). A 413 rejection inside the slot still releases it (the
    finally in _saturated_slot)."""
    async with _saturated_slot(
        _pdf_parse_semaphore(),
        "OCR service busy (all parse slots in use); try again shortly",
    ):
        yield


async def _run_ocr_pipeline(file_bytes: bytes, kind: str) -> tuple[list[dict], bool]:
    """Shared OCR core for /layout-parsing and /extract: enforce the
    request caps — PDF: page count (the page tree's count, since the
    /Count entry is attacker-writable) + per-page rendered pixels;
    image: the header-declared canvas (a few-KB file can declare a
    canvas whose decode is tens of GB) — a single pass over the
    in-memory request bytes, before any temp-file staging, offloaded
    to a worker thread via asyncio.to_thread: it can parse up to 100 MB
    of file bytes and must not run on the event loop; the parse itself
    is bounded by its own pre-slot semaphore — at most
    MAX_CONCURRENT_PDF_PARSSES in-flight parses, 503 when saturated —
    because the job slot cannot bound work that runs before it — take
    a concurrency slot (at most MAX_CONCURRENT_OCR_JOBS
    staging+pipeline jobs in flight; 503 when saturated), stage the
    detected content in a temp file whose suffix matches the *detected*
    kind (run_paddlex_structure_v3 routes on the .pdf suffix —
    pdftoppm pre-render — vs image — direct), run the configured
    pipeline in a worker thread (it can take minutes; must not stall
    the event loop), and always clean up the temp file.

    Returns (pages, used_structure_v3). Job completion is stamped
    before and after so a job longer than MODEL_IDLE_TIMEOUT restarts
    the idle clock instead of being destroyed on the first poll after
    it finishes.
    """
    suffix = ".png" if kind == "image" else ".pdf"
    # Rejections happen BEFORE the job slot: a rejected job must not
    # take a slot or stage a temp file. Offloaded to a worker thread:
    # the cap check parses up to 100 MB of file bytes (pypdf for the
    # page-tree count and the per-page boxes, pypdfium2's count as
    # fallback; PIL's header for the image canvas) and must not stall
    # the event loop. Bounded by the parse slot — acquired in the
    # event loop (never a blocked worker thread) and released before
    # the job slot: the parse runs before the job slot, so
    # MAX_CONCURRENT_OCR_JOBS cannot bound its fan-out.
    # Text kind returns immediately inside _enforce_pdf_caps, so it
    # skips the executor hop entirely (it would only queue behind
    # in-flight PDF parses under executor saturation).
    if kind == "pdf":
        async with _pdf_parse_slot():
            await asyncio.to_thread(_enforce_pdf_caps, file_bytes, kind)
    elif kind == "image":
        # Metadata-only (header read), but still in a worker thread
        # under the parse slot: a pathological header (e.g. a deep TIFF
        # IFD chain in a 100 MB file) is O(file size) to parse and must
        # not stall the event loop, and the slot bounds the same
        # pre-slot fan-out class as the PDF parse.
        async with _pdf_parse_slot():
            await asyncio.to_thread(_enforce_image_pixel_cap, file_bytes)
    async with _ocr_job_slot():
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                # Assign BEFORE the write: if tmp.write raises (e.g. ENOSPC),
                # the finally block still unlinks the partial file.
                tmp_path = tmp.name
                tmp.write(file_bytes)

            _mark_model_used()

            if os.getenv("USE_STRUCTURE_V3", "true").lower() in ("true", "1"):
                # PP-StructureV3
                pages = await asyncio.to_thread(run_paddlex_structure_v3, tmp_path)
                used_v3 = True
            else:
                # Plain-OCR fallback — same offload: it blocks for minutes too.
                pages = await asyncio.to_thread(run_paddleocr, tmp_path)
                used_v3 = False

            _mark_model_used()
            return pages, used_v3
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)


# ── Open-WebUI endpoint ──────────────────────────────────────────────
class LayoutParsingRequest(BaseModel):
    """Request body for Open-WebUI's /layout-parsing endpoint."""

    file: str  # base64-encoded file bytes
    fileType: int = 0  # 0=PDF, 1=image (hint only — content detection wins)
    useDocOrientationClassify: bool = False
    useDocUnwarping: bool = False
    useChartRecognition: bool = False


@app.post("/layout-parsing")
async def layout_parsing(
    body: LayoutParsingRequest,
    authorization: str | None = Header(None, alias="Authorization"),
) -> SafeJSONResponse:
    """Open-WebUI PaddleOCR-VL compatible endpoint.

    Open-WebUI calls this when PaddleOCR-VL is selected as the
    Content Extraction Engine. Returns structured markdown with
    headings, paragraphs, tables, and image placeholders.

    If the file is a text file (not PDF), returns it as-is in the response.
    """
    _check_token(authorization)
    _require_model_ready()

    file_b64 = body.file
    if not file_b64:
        raise HTTPException(400, "Missing 'file' field")

    file_type = body.fileType
    hint_kind: Literal["pdf", "image"] = "image" if file_type == 1 else "pdf"

    try:
        file_bytes = base64.b64decode(file_b64)
    except Exception as exc:
        raise HTTPException(400, f"Invalid base64: {exc}")

    if not file_bytes:
        raise HTTPException(400, "File content is empty")

    kind = _route_upload(file_bytes, hint_kind, "layout-parsing")

    # ── Text file: return as-is ──
    if kind == "text":
        content = file_bytes.decode("utf-8", errors="replace")
        return SafeJSONResponse(
            content={
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": content}, "structuredBlocks": []}
                    ],
                }
            }
        )

    # ── PDF / image processing ──
    try:
        pages, used_v3 = await _run_ocr_pipeline(file_bytes, kind)
        if used_v3:
            # Build structured markdown per page
            layout_results = [
                {
                    "markdown": {
                        "text": blocks_to_markdown(page.get("structured_blocks", []))
                    },
                    "structuredBlocks": page.get("structured_blocks", []),
                }
                for page in pages
            ]
        else:
            layout_results = [
                {"markdown": {"text": page["text"]}, "structuredBlocks": None}
                for page in pages
            ]

        return SafeJSONResponse(
            content={
                "result": {
                    "layoutParsingResults": layout_results,
                },
            }
        )
    except HTTPException:
        raise
    except Exception:
        # Generic detail: full error text stays in the server logs only.
        logger.exception("Layout parsing failed")
        raise HTTPException(500, "Layout parsing failed")


# ── Direct API endpoints (not used by Open-WebUI) ────────────────────
@app.get("/health")
async def health() -> dict:
    """Health check endpoint. Model-init failures report the same generic
    503 detail as the OCR endpoints (no raw exception text — it can leak
    model/temp paths; the full failure is in the server logs)."""
    if get_paddlex_init_exception() is not None:
        raise HTTPException(status_code=503, detail=MODEL_INIT_FAILURE_DETAIL)

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
) -> SafeJSONResponse:
    """Direct API: upload PDF/image, get structured text + blocks.

    Accepts: multipart/form-data with field 'file'
    Returns: JSON with pages (structured markdown, structuredBlocks, raw blocks)
    """
    _check_token(authorization)
    _require_model_ready()

    if not file.filename:
        raise HTTPException(400, "Missing file")

    ext = Path(file.filename).suffix.lower()
    hint_kind: Literal["pdf", "image"] = "image" if ext in IMAGE_EXTS else "pdf"

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "File content is empty")

    kind = _route_upload(file_bytes, hint_kind, "extract")

    # ── Text file: return as-is ──
    if kind == "text":
        content = file_bytes.decode("utf-8", errors="replace")
        return SafeJSONResponse(
            content={
                "filename": file.filename,
                "pages": [
                    {
                        "page": 1,
                        "text": content,
                        "markdown": content,
                        "blocks": [],
                        "structured_blocks": [],
                    }
                ],
                "full_text": content,
            }
        )

    # ── PDF / image processing ──
    try:
        pages, used_v3 = await _run_ocr_pipeline(file_bytes, kind)
        if used_v3:
            # Build structured markdown per page
            result_pages = [
                {
                    "page": page["page"],
                    "text": page["text"],  # raw text
                    "markdown": blocks_to_markdown(page.get("structured_blocks", [])),
                    "blocks": page["blocks"],  # raw OCR blocks
                    "structured_blocks": page.get("structured_blocks", []),
                }
                for page in pages
            ]
        else:
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
        return SafeJSONResponse(
            content={
                "filename": file.filename,
                "pages": result_pages,
                "full_text": full_text,
            }
        )
    except HTTPException:
        raise
    except Exception:
        # Generic detail: full error text stays in the server logs only.
        logger.exception("Extraction failed")
        raise HTTPException(500, "Extraction failed")


# ── Entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
