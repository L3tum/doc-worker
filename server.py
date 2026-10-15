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
import json
import logging
import os
import struct
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import Response

from paddlex_helpers import (
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
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "104857600"))  # 100 MB default

# Hard page cap per request. At RENDER_DPI=300 an unbounded PDF is an
# unbounded temp-disk + memory + wall-time allocation on the OCR endpoints
# (unauthenticated when PADDLEOCR_VL_TOKEN is unset — DoS). Override via
# MAX_OCR_PAGES.
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "500"))

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
    violation = _body_size_policy(request.method, request.headers.get("content-length"))
    if violation is not None:
        return JSONResponse(status_code=413, content={"detail": violation})
    return await call_next(request)


def _body_size_policy(method: str, content_length: str | None) -> str | None:
    """413 detail for a request body, or None if it may proceed.

    - POST without Content-Length (chunked): rejected — the body is fully
      materialized (memory + temp file) before processing, so without a
      declared length it is unbounded.
    - POST with Content-Length > MAX_REQUEST_SIZE: rejected.
    - Non-POST: no body to bound.
    """
    if method != "POST":
        return None
    if content_length is None:
        return (
            "Transfer-Encoding: chunked uploads are not supported; send Content-Length"
        )
    if int(content_length) > MAX_REQUEST_SIZE:
        return f"Request too large (max {MAX_REQUEST_SIZE // 1024 // 1024} MB)"
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


def _check_token(authorization: str | None) -> None:
    """Validate the Authorization: Bearer <value> header.

    The presented token is compared with hmac.compare_digest (constant-time)
    against PADDLEOCR_VL_TOKEN.
    """
    if not PADDLEOCR_VL_TOKEN:
        return  # No token configured, skip auth
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(403, "Invalid token")
    token = authorization[len("Bearer ") :]
    if not hmac.compare_digest(
        token.encode("utf-8"), PADDLEOCR_VL_TOKEN.encode("utf-8")
    ):
        raise HTTPException(403, "Invalid token")


# ── Magic-byte signatures (canonical table) ───────────────────────────
# ONE canonical table of (name, predicate) entries drives BOTH
# _is_text_content's step-1 binary rejection and detect_file_type's
# pdf/image classification. Signatures are anchored at the start of the
# content after skipping leading whitespace/control bytes (see
# _skip_leading_noise).
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


def _sig_pdf(content: bytes) -> bool:
    return content.startswith(b"%PDF")


def _sig_png(content: bytes) -> bool:
    return content.startswith(b"\x89PNG\r\n\x1a\n")


def _sig_jpeg(content: bytes) -> bool:
    return content.startswith(b"\xff\xd8\xff")


def _sig_gif(content: bytes) -> bool:
    return content.startswith(b"GIF8")  # GIF87a or GIF89a


def _sig_bmp(content: bytes) -> bool:
    # "BM" + file-size DWORD (offset 2; 0 = unspecified, else >= 14) +
    # DIB header-size DWORD (offset 14) in _BMP_DIB_SIZES. 18 bytes
    # needed to read both DWORDs.
    if len(content) < 18 or content[0:2] != b"BM":
        return False
    file_size = struct.unpack_from("<I", content, 2)[0]
    dib_size = struct.unpack_from("<I", content, 14)[0]
    return (file_size == 0 or file_size >= 14) and dib_size in _BMP_DIB_SIZES


def _sig_webp(content: bytes) -> bool:
    # RIFF-anchored: "RIFF" magic at offset 0 plus the "WEBP" fourcc at
    # offset 8. The anchor is required so a non-RIFF "WEBP" at offset 8
    # (e.g. plain ASCII text) does not classify as image; WAV/AVI share
    # the RIFF container and stay excluded by the fourcc. The length
    # guard is implicit — slicing shorter bytes yields a shorter result.
    return content[:4] == b"RIFF" and content[8:12] == b"WEBP"


def _sig_tiff_le(content: bytes) -> bool:
    return content.startswith(b"II\x2a\x00")


def _sig_tiff_be(content: bytes) -> bool:
    return content.startswith(b"MM\x00\x2a")


def _sig_zip(content: bytes) -> bool:
    return content.startswith(b"PK\x03\x04")  # ZIP-based: ODF, DOCX, XLSX, etc.


def _sig_gzip(content: bytes) -> bool:
    return content.startswith(b"\x1f\x8b")


def _sig_bzip2(content: bytes) -> bool:
    return content.startswith(b"BZ")


def _sig_lzip(content: bytes) -> bool:
    return content.startswith(b"x\xda\x03")


_FILE_SIGNATURES: tuple[tuple[str, Callable[[bytes], bool]], ...] = (
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
# through as raw text. bytes.lstrip is a C-speed scan, so this stays cheap
# on large uploads.
_LEADING_ANCHOR_NOISE = bytes(range(0x21))


def _skip_leading_noise(content: bytes) -> bytes:
    """Return *content* with leading whitespace/control bytes removed.

    The anchor view for the magic-byte signatures (see
    _LEADING_ANCHOR_NOISE): predicates run against this, so a %PDF header
    padded with leading LF/CR/NUL still classifies as PDF.
    """
    return content.lstrip(_LEADING_ANCHOR_NOISE)


def _is_text_content(raw_bytes: bytes) -> bool:
    """Check if raw bytes look like valid text (not a PDF or other binary).

    Strategy:
    1. Reject known binary file signatures (the canonical
       _FILE_SIGNATURES table, anchored after leading whitespace/control
       bytes — see _skip_leading_noise).
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
    # whitespace/control bytes — see _skip_leading_noise)
    probe = _skip_leading_noise(raw_bytes)
    for _name, predicate in _FILE_SIGNATURES:
        if predicate(probe):
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
    for name, predicate in _FILE_SIGNATURES:
        if predicate(_skip_leading_noise(content)):
            if name == "pdf":
                return "pdf"
            if name in _IMAGE_SIGNATURE_NAMES:
                return "image"
            return "unknown"  # reject-only binary (zip/gzip/bzip2/lzip)
    if _is_text_content(content):
        return "text"
    return "unknown"


# ── Open-WebUI endpoint ──────────────────────────────────────────────
# ── Shared request plumbing (used by /layout-parsing and /extract) ──────
def _require_model_ready() -> None:
    """Early health guard for the OCR endpoints: fail fast with the same
    503 shape as /health if model initialization failed at startup."""
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


def _count_pdf_pages(file_path: str) -> int | None:
    """Page count of a PDF via pypdfium2 (already a hard dependency for the
    AcroForm pre-flatten in worker.py), or None when uncountable — the
    caller then passes the file through and the pipeline reports its own
    error if it is actually broken."""
    try:
        import pypdfium2

        with pypdfium2.PdfDocument(file_path) as doc:
            return len(doc)
    except Exception:
        return None


def _enforce_page_limit(file_path: str, kind: str) -> None:
    """Reject PDFs with more than MAX_OCR_PAGES pages BEFORE any render or
    model work (unbounded 300-dpi temp-disk staging — see MAX_OCR_PAGES).
    Images/text are single-page by construction."""
    if kind != "pdf":
        return
    page_count = _count_pdf_pages(file_path)
    if page_count is not None and page_count > MAX_OCR_PAGES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many pages: {page_count} (max {MAX_OCR_PAGES})",
        )


async def _run_ocr_pipeline(file_bytes: bytes, kind: str) -> tuple[list[dict], bool]:
    """Shared OCR core for /layout-parsing and /extract: enforce the page
    cap, stage the detected content in a temp file whose suffix matches the
    *detected* kind (run_paddlex_structure_v3 routes on the .pdf suffix —
    pdftoppm pre-render — vs image — direct), run the configured pipeline
    in a worker thread (it can take minutes; must not stall the event
    loop), and always clean up the temp file.

    Returns (pages, used_structure_v3). Job completion is stamped before
    and after so a job longer than MODEL_IDLE_TIMEOUT restarts the idle
    clock instead of being destroyed on the first poll after it finishes.
    """
    suffix = ".png" if kind == "image" else ".pdf"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        _enforce_page_limit(tmp_path, kind)
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
    If PDF processing fails (e.g., pdfium error), falls back to returning
    raw content as text.
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
    except Exception as exc:
        logger.exception("Layout parsing failed")
        raise HTTPException(500, f"Layout parsing failed: {exc}")


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
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(500, f"Extraction failed: {exc}")


# ── Entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
