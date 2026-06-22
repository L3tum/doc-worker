#!/usr/bin/env python3
"""
Doc-Worker — OCR pipeline for PDFs
===================================

1. Polls an inbox directory for new PDF files.
2. Runs OCR using OCRmyPDF with the RapidOCR ONNX plugin (via Python API).
3. Generates sidecar documents via the Docling API (Markdown + JSON).
4. Pushes the processed PDFs into a Paperless-ngx consume directory.

All paths and settings are controlled by environment variables (see defaults
below).  The worker loops indefinitely, sleeping between polls.

Crash-recovery: on startup, any leftover files in the PROCESSING directory are
moved to ERROR/ so they are not silently lost.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, cast

import ocrmypdf
import rapidocr
import requests

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------
INBOX = Path(os.getenv("INBOX", "/work/inbox"))
PROCESSING = Path(os.getenv("PROCESSING", "/work/processing"))
DONE = Path(os.getenv("DONE", "/work/done"))
ERROR = Path(os.getenv("ERROR", "/work/error"))
DOCLING_OUT = Path(os.getenv("DOCLING_DIR", "/work/docling"))
PAPERLESS_CONSUME = Path(os.getenv("PAPERLESS_CONSUME", "/paperless-consume"))

DOCLING_BASE_URL = os.getenv("DOCLING_BASE_URL", "http://docling:5001").rstrip("/")
DOCLING_MODE = os.getenv(
    "DOCLING_MODE", "best_effort"
)  # "off" | "best_effort" | "required"
OCR_LANG = os.getenv("OCR_LANG", "deu")
OCR_RUNTIME = os.getenv(
    "OCR_RUNTIME", "cpu"
).lower()  # "cpu" | "cuda" | "openvino" | "rocm"

for path in [INBOX, PROCESSING, DONE, ERROR, DOCLING_OUT, PAPERLESS_CONSUME]:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging helpers — plain print, flushed immediately
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def log_error(msg: str) -> None:
    print(f"ERROR: {msg}", flush=True, file=sys.stderr)


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------
def recover_leftover_files() -> None:
    """Move any stale files from PROCESSING into ERROR on startup."""
    leftovers = list(PROCESSING.iterdir()) if PROCESSING.exists() else []
    if not leftovers:
        return

    log(f"Crash recovery: found {len(leftovers)} file(s) in PROCESSING/")
    for f in leftovers:
        dest = ERROR / f.name
        # Avoid overwriting existing error files
        if dest.exists():
            stem = f.stem
            suffix = f.suffix
            ts = time.strftime("%Y%m%d%H%M%S")
            dest = ERROR / f"{stem}_{ts}{suffix}"
        shutil.move(str(f), str(dest))
        log(f"  Recovered: {f.name} → ERROR/")


# ---------------------------------------------------------------------------
# Docling API helpers
# ---------------------------------------------------------------------------
def call_docling_convert(pdf_path: Path) -> bool:
    """Send a PDF to the Docling API and return True on success."""
    url = f"{DOCLING_BASE_URL}/v1alpha/convert"
    try:
        with open(pdf_path, "rb") as f:
            files = {"files": (pdf_path.name, f, "application/pdf")}
            data = {"output_formats": ["md", "json"]}
            resp = requests.post(url, files=files, data=data, timeout=900)
            resp.raise_for_status()

        # Docling returns a list of conversion results
        results = resp.json()
        if isinstance(results, list) and len(results) > 0:
            filename_stem = pdf_path.stem
            out_dir = DOCLING_OUT / filename_stem
            out_dir.mkdir(parents=True, exist_ok=True)

            for fmt in ("md", "json"):
                content = results[0].get(fmt, "")
                ext = "md" if fmt == "md" else "json"
                out_file = out_dir / f"{filename_stem}.{ext}"
                with open(out_file, "w", encoding="utf-8") as wf:
                    wf.write(content)
                log(f"  Docling {ext.upper()} written: {out_file}")

        return True

    except requests.RequestException as exc:
        log_error(f"Docling API error: {exc}")
        return False
    except Exception as exc:
        log_error(f"Docling processing error: {exc}")
        return False


def handle_docling(pdf_path: Path) -> bool:
    """
    Run Docling based on DOCLING_MODE.

    Returns:
        True  — continue pipeline (Docling succeeded or is disabled).
        False — abort pipeline (Docling failed in 'required' mode).
    """
    if DOCLING_MODE == "off":
        log("  Docling: SKIPPED (mode=off)")
        return True

    success = call_docling_convert(pdf_path)

    if success:
        log("  Docling: OK")
        return True

    # Docling failed
    if DOCLING_MODE == "required":
        log_error("Docling failed and mode=required — aborting this file.")
        return False

    # best_effort: warn and continue
    log("  Docling: FAILED (mode=best_effort, continuing)")
    return True


# ---------------------------------------------------------------------------
# OCRmyPDF via Python API
# ---------------------------------------------------------------------------
def _patch_rapidocr_provider_config() -> None:
    """Experimentally prepend an ONNX Runtime provider in RapidOCR.

    RapidOCR's ONNX Runtime backend currently exposes config toggles for only
    some execution providers. OpenVINO and ROCm may still be available from the
    installed onnxruntime package, so this patch lets us pass that provider to
    InferenceSession while keeping CPU as a fallback.
    """
    try:
        from rapidocr.inference_engine.onnxruntime.provider_config import ProviderConfig
    except Exception as exc:
        log(f"WARNING: Could not patch RapidOCR provider config: {exc}")
        return

    if getattr(ProviderConfig, "_doc_worker_provider_patch", False):
        return

    original_get_ep_list = ProviderConfig.get_ep_list

    def patched_get_ep_list(self: Any) -> list[Any]:
        ep_list: list[Any] = cast(list[Any], original_get_ep_list(self))
        requested_provider = os.environ.get("RAPIDOCR_ONNXRUNTIME_PROVIDER")
        if not requested_provider or requested_provider == "CPUExecutionProvider":
            return ep_list

        if requested_provider not in self.had_providers:
            log(
                f"WARNING: Experimental provider {requested_provider} is not available. "
                f"Using RapidOCR defaults: {self.had_providers}"
            )
            return ep_list

        filtered = [ep for ep in ep_list if ep[0] != requested_provider]
        log(
            f"INFO: Prepending experimental ONNX Runtime provider: {requested_provider}"
        )
        return [(requested_provider, {})] + filtered

    ProviderConfig.get_ep_list = patched_get_ep_list
    ProviderConfig._doc_worker_provider_patch = True


def _configure_rapidocr_runtime() -> None:
    """Configure RapidOCR for the selected runtime backend.

    Supported backends (via OCR_RUNTIME env var):
      cpu       — CPUExecutionProvider (default, always available)
      cuda      — CUDAExecutionProvider (NVIDIA GPU, requires onnxruntime-gpu)
      openvino  — OpenVINOExecutionProvider (Intel GPU/CPU, requires onnxruntime-openvino)
      rocm      — ROCmExecutionProvider (AMD GPU, requires onnxruntime-rocm)

    Auto-detection: if the requested provider is not available, falls back
    to CPU with a warning.
    """
    global _rapidocr_configured, OCR_RUNTIME, _rapidocr_params
    if _rapidocr_configured:
        return
    _rapidocr_configured = True

    # Map runtime names to (provider name, pip package)
    BACKENDS = {
        "cpu": ("CPUExecutionProvider", None),
        "cuda": ("CUDAExecutionProvider", "onnxruntime-gpu"),
        "openvino": ("OpenVINOExecutionProvider", "onnxruntime-openvino"),
        "rocm": ("ROCMExecutionProvider", "onnxruntime-rocm"),
    }

    # Validate runtime
    if OCR_RUNTIME not in BACKENDS:
        log(
            f"WARNING: OCR_RUNTIME='{OCR_RUNTIME}' is invalid, falling back to 'cpu'. "
            f"Valid values: {', '.join(BACKENDS)}"
        )
        OCR_RUNTIME = "cpu"

    target_provider, package_name = BACKENDS[OCR_RUNTIME]

    # Check availability
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()

        if target_provider not in available:
            if package_name:
                log(
                    f"WARNING: OCR_RUNTIME={OCR_RUNTIME} requested but {target_provider} "
                    f"is not available. Falling back to CPU. "
                    f"Install '{package_name}' or rebuild with ONNX_RUNTIME={OCR_RUNTIME}. "
                    f"Available providers: {available}"
                )
            else:
                log(f"INFO: Using CPU runtime. Available providers: {available}")
            OCR_RUNTIME = "cpu"
            target_provider = "CPUExecutionProvider"
        else:
            log(
                f"INFO: {target_provider} found — {OCR_RUNTIME.upper()} acceleration enabled"
            )
    except Exception as exc:
        log(
            f"WARNING: Failed to check runtime availability ({exc}), falling back to CPU"
        )
        OCR_RUNTIME = "cpu"
        target_provider = "CPUExecutionProvider"

    log(f"INFO: RapidOCR runtime = {OCR_RUNTIME} ({target_provider})")

    # Build flat dot-notation params for RapidOCR 3.x.
    # RapidOCR's ParseParams.update_batch() expects keys like
    # "EngineConfig.onnxruntime.use_cuda", NOT nested dicts.
    _rapidocr_params = {}

    if OCR_RUNTIME == "cuda":
        # CUDA is directly supported by RapidOCR's ONNX Runtime provider config.
        _rapidocr_params = {
            "EngineConfig.onnxruntime.use_cuda": True,
        }
    elif OCR_RUNTIME in ("openvino", "rocm"):
        # Experimental: RapidOCR's ONNX Runtime provider config does not expose
        # OpenVINO/ROCm toggles, even when onnxruntime lists those providers.
        # Store the requested provider in an env var and patch ProviderConfig so
        # ONNX Runtime receives e.g. ["ROCMExecutionProvider", "CPUExecutionProvider"].
        os.environ["RAPIDOCR_ONNXRUNTIME_PROVIDER"] = target_provider
        _patch_rapidocr_provider_config()

    # Monkey-patch RapidOCR.__init__ to inject our runtime params.
    # The ocrmypdf_rapidocr plugin calls RapidOCR(config_path=..., params=...)
    # so our patched __init__ must accept `params` as a kwarg name.
    _orig_init = rapidocr.RapidOCR.__init__

    def _patched_init(
        self: Any,
        config_path: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        # Merge our flat dot-notation runtime params with any user-provided params.
        # Both dicts use flat dot-notation keys, so a simple dict merge works.
        merged = dict(_rapidocr_params)
        if params:
            merged.update(params)
        _orig_init(self, config_path=config_path, params=merged if merged else None)

    rapidocr.RapidOCR.__init__ = _patched_init


_rapidocr_configured = False
_rapidocr_params: dict[str, Any] = {}


def run_ocrmypdf(input_pdf: Path, output_pdf: Path) -> None:
    """Run OCRmyPDF with RapidOCR engine via Python API."""
    _configure_rapidocr_runtime()

    # input_file_or_options and output_file are positional args in ocrmypdf.ocr()
    # Everything else is keyword-only (after the * in the signature)
    # The plugin is auto-registered via the `plugins` parameter (pluggy).
    ocrmypdf.ocr(
        input_pdf,
        output_pdf,
        plugins=["ocrmypdf_rapidocr"],
        language=OCR_LANG,
        force_ocr=True,
        rapidocr_config_path=os.environ.get("RAPIDOCR_CONFIG"),
    )


# ---------------------------------------------------------------------------
# Paperless push
# ---------------------------------------------------------------------------
def push_to_paperless(ocr_pdf: Path) -> bool:
    """Upload the OCR'd PDF to Paperless-ngx consume directory.

    Uses an atomic write pattern (write to .tmp, then rename) so Paperless
    never picks up a partially-written file.
    """
    dest = PAPERLESS_CONSUME / ocr_pdf.name
    tmp_dest = PAPERLESS_CONSUME / f"{ocr_pdf.name}.tmp"
    try:
        shutil.copy2(str(ocr_pdf), str(tmp_dest))
        os.replace(str(tmp_dest), str(dest))
        log(f"  Pushed to Paperless: {dest}")
        return True
    except Exception as exc:
        log_error(f"Paperless push failed: {exc}")
        # Clean up temp file on failure
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False


# ---------------------------------------------------------------------------
# Docling health check
# ---------------------------------------------------------------------------
def wait_for_docling(timeout: int = 120) -> None:
    """Wait for Docling API to become available before processing."""
    log(f"Waiting for Docling at {DOCLING_BASE_URL}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{DOCLING_BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                log("Docling is ready.")
                return
        except Exception:
            pass
        time.sleep(2)
    log(f"WARNING: Docling did not become ready within {timeout}s — continuing anyway.")


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------
def process_file(pdf_path: Path) -> None:
    """Process a single PDF: Docling → OCR → Paperless."""
    filename = pdf_path.name
    log(f"\n{'=' * 60}")
    log(f"Processing: {filename}")
    log(f"{'=' * 60}")

    # Move to processing
    processing_path = PROCESSING / filename
    shutil.move(str(pdf_path), str(processing_path))
    log(f"  Moved to processing: {processing_path}")

    # Docling
    if not handle_docling(processing_path):
        shutil.move(str(processing_path), str(ERROR / filename))
        log("  → ERROR/ (Docling required but failed)")
        return

    # OCR
    ocr_output = PROCESSING / f"{processing_path.stem}_ocr{processing_path.suffix}"
    try:
        run_ocrmypdf(processing_path, ocr_output)
        log("  OCR: OK")
    except Exception as exc:
        log_error(f"OCR failed: {exc}")
        if ocr_output.exists():
            ocr_output.unlink()
        shutil.move(str(processing_path), str(ERROR / filename))
        log("  → ERROR/ (OCR failed)")
        return

    # Paperless
    if not push_to_paperless(ocr_output):
        if ocr_output.exists():
            ocr_output.unlink()
        shutil.move(str(processing_path), str(ERROR / filename))
        log("  → ERROR/ (Paperless push failed)")
        return

    # Done
    shutil.move(str(processing_path), str(DONE / filename))
    # Clean up intermediate OCR output file
    if ocr_output.exists():
        ocr_output.unlink()
    log(f"  → DONE/ ({filename})")


def wait_for_stable_file(pdf_path: Path, stability_timeout: int | None = None) -> bool:
    """Wait until the file size stops changing (upload finished).

    Polls the file size for up to *stability_timeout* seconds (default from
    STABILITY_TIMEOUT env-var, fallback 10 s).  If the size stays the same for
    the full period the file is considered stable.
    """
    if stability_timeout is None:
        stability_timeout = int(os.getenv("STABILITY_TIMEOUT", "10"))
    try:
        last_size = pdf_path.stat().st_size
        start = time.time()
        while time.time() - start < stability_timeout:
            time.sleep(1)
            current_size = pdf_path.stat().st_size
            if current_size != last_size:
                last_size = current_size
                start = time.time()  # reset timer on change
                continue
        # If we exited because size stayed constant → stable;
        # if timer expired → still process (warn).
        log(f"  Stable: {pdf_path.name}")
        return True
    except FileNotFoundError:
        log(f"  File disappeared during stability check: {pdf_path.name}")
        return False


def main() -> None:
    log("Doc-Worker starting...")
    log(f"  INBOX:     {INBOX}")
    log(f"  PROCESSING: {PROCESSING}")
    log(f"  DONE:      {DONE}")
    log(f"  ERROR:     {ERROR}")
    log(f"  DOCLING:   {DOCLING_OUT}")
    log(f"  PAPERLESS: {PAPERLESS_CONSUME}")
    log(f"  DOCLING_URL: {DOCLING_BASE_URL}")
    log(f"  DOCLING_MODE: {DOCLING_MODE}")
    log(f"  OCR_LANG:  {OCR_LANG}")
    log(f"  OCR_RUNTIME: {OCR_RUNTIME}")
    log(f"  STABILITY_TIMEOUT: {os.getenv('STABILITY_TIMEOUT', '10')}s")

    # Crash recovery
    recover_leftover_files()

    # Wait for Docling to become available (if configured)
    if DOCLING_BASE_URL and DOCLING_MODE != "off":
        docling_timeout = int(os.getenv("DOCLING_TIMEOUT", "900"))
        wait_for_docling(docling_timeout)

    poll_interval = int(os.getenv("POLL_INTERVAL", "5"))
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay = int(os.getenv("RETRY_DELAY", "10"))

    log(f"  POLL_INTERVAL: {poll_interval}s")
    log(f"  MAX_RETRIES:   {max_retries}")
    log(f"  RETRY_DELAY:   {retry_delay}s")
    log("Entering main loop...")

    while True:
        try:
            # Find new PDFs in inbox (match both lower- and upper-case extensions)
            pdf_files = sorted(
                {f for p in ("*.pdf", "*.PDF") for f in INBOX.glob(p)},
                key=lambda p: p.name.lower(),
            )

            for pdf in pdf_files:
                # Stability check
                if not wait_for_stable_file(pdf):
                    log(f"  Skipping (unstable): {pdf.name}")
                    continue

                # Process with retries
                retries = 0
                while retries < max_retries:
                    try:
                        # If a previous attempt already moved the file to PROCESSING,
                        # retry with that path instead of the original INBOX path.
                        current_path = (
                            PROCESSING / pdf.name
                            if (PROCESSING / pdf.name).exists()
                            else pdf
                        )
                        process_file(current_path)
                        break  # Success — move to next file
                    except Exception as exc:
                        retries += 1
                        if retries < max_retries:
                            log_error(
                                f"Retry {retries}/{max_retries} for {pdf.name}: {exc}"
                            )
                            time.sleep(retry_delay)
                        else:
                            log_error(f"Max retries reached for {pdf.name}: {exc}")
                            # Make sure the file ends up in ERROR/
                            if pdf.exists():
                                shutil.move(str(pdf), str(ERROR / pdf.name))
                            elif (PROCESSING / pdf.name).exists():
                                shutil.move(
                                    str(PROCESSING / pdf.name),
                                    str(ERROR / pdf.name),
                                )

        except Exception as exc:
            log_error(f"Main loop error: {exc}")

        # Sleep before next poll
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
