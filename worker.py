#!/usr/bin/env python3
"""
Doc-Worker — OCR pipeline for PDFs
===================================

1. Polls an inbox directory for new PDF files.
2. Runs OCR using OCRmyPDF with the PaddleOCR plugin (via Python API).
3. Generates sidecar documents (Markdown + JSON) via the Docling API or local PaddleOCR.
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

import ocrmypdf
import requests
import json

from paddleocr_helpers import run_paddleocr, validate_paddleocr_models

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------
INBOX = Path(os.getenv("INBOX", "/work/inbox"))
PROCESSING = Path(os.getenv("PROCESSING", "/work/processing"))
DONE = Path(os.getenv("DONE", "/work/done"))
ERROR = Path(os.getenv("ERROR", "/work/error"))
DOCLING_OUT = Path(os.getenv("DOCLING_DIR", "/work/docling"))
PAPERLESS_CONSUME = Path(os.getenv("PAPERLESS_CONSUME", "/paperless-consume"))

DOCLING_BASE_URL = os.getenv("DOCLING_BASE_URL", "http://docling:12000").rstrip("/")
DOCLING_MODE = os.getenv(
    "DOCLING_MODE", "best_effort"
)  # "off" | "best_effort" | "required" | "native"
OCR_LANG = os.getenv("OCR_LANG", "deu")
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() in ("true", "1", "yes")
PADDLEOCR_MODELS = os.getenv("PADDLEOCR_MODELS", "/app/models")

for path in [INBOX, PROCESSING, DONE, ERROR, DOCLING_OUT, PAPERLESS_CONSUME]:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging helpers — plain print, flushed immediately
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def log_error(msg: str) -> None:
    print(f"ERROR: {msg}", flush=True, file=sys.stderr)


def unique_destination(directory: Path, filename: str) -> Path:
    """Return a non-existing destination path inside *directory*."""
    dest = directory / filename
    if not dest.exists():
        return dest

    stem = dest.stem
    suffix = dest.suffix
    ts = time.strftime("%Y%m%d%H%M%S")
    return directory / f"{stem}_{ts}{suffix}"


def move_to_error(file_path: Path, reason: str) -> None:
    """Move *file_path* to ERROR/ without overwriting an existing file."""
    if not file_path.exists():
        log_error(f"Cannot move missing file to ERROR/: {file_path}")
        return

    dest = unique_destination(ERROR, file_path.name)
    shutil.move(str(file_path), str(dest))
    log(f"  → ERROR/ ({reason})")


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
    Run sidecar generation based on DOCLING_MODE.

    Returns:
        True  — continue pipeline (sidecar succeeded or is disabled).
        False — abort pipeline (sidecar failed in 'required' mode).
    """
    if DOCLING_MODE == "off":
        log("  Sidecar: SKIPPED (mode=off)")
        return True

    if DOCLING_MODE == "native":
        success = generate_native_sidecar(pdf_path)
    else:
        success = call_docling_convert(pdf_path)

    if success:
        mode_label = "native" if DOCLING_MODE == "native" else "Docling"
        log(f"  {mode_label}: OK")
        return True

    # Sidecar generation failed
    if DOCLING_MODE == "required":
        log_error("Sidecar generation failed and mode=required — aborting this file.")
        return False

    # best_effort or native: warn and continue
    mode_label = "native" if DOCLING_MODE == "native" else "Docling"
    log(f"  {mode_label}: FAILED (mode={DOCLING_MODE}, continuing)")
    return True


# ---------------------------------------------------------------------------
# Native PaddleOCR sidecar generation
# ---------------------------------------------------------------------------
def generate_native_sidecar(pdf_path: Path) -> bool:
    """Generate Markdown + JSON sidecar files using local PaddleOCR.

    Outputs to DOCLING_OUT/{stem}/{stem}.md and DOCLING_OUT/{stem}/{stem}.json
    using the same naming convention as the Docling API.
    """
    try:
        pages = run_paddleocr(str(pdf_path))

        filename_stem = pdf_path.stem
        out_dir = DOCLING_OUT / filename_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build JSON sidecar
        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        sidecar_json = {
            "filename": pdf_path.name,
            "pages": [
                {"page": p["page"], "text": p["text"], "blocks": p["blocks"]}
                for p in pages
            ],
            "full_text": full_text,
        }
        json_out = out_dir / f"{filename_stem}.json"
        with open(json_out, "w", encoding="utf-8") as wf:
            json.dump(sidecar_json, wf, ensure_ascii=False, indent=2)
        log(f"  Native JSON written: {json_out}")

        # Build Markdown sidecar (simple page-by-page text)
        md_parts: list[str] = []
        for p in pages:
            md_parts.append(f"## Page {p['page']}")
            if p["text"]:
                md_parts.append(p["text"])
        md_out = out_dir / f"{filename_stem}.md"
        with open(md_out, "w", encoding="utf-8") as wf:
            wf.write("\n\n".join(md_parts))
        log(f"  Native MD written: {md_out}")

        return True

    except Exception as exc:
        log_error(f"Native sidecar generation failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# OCRmyPDF via Python API — PaddleOCR engine
# ---------------------------------------------------------------------------
def run_ocrmypdf(input_pdf: Path, output_pdf: Path) -> None:
    """Run OCRmyPDF with PaddleOCR engine via Python API.

    Uses the ocrmypdf-paddleocr plugin which properly implements the
    OcrEngine interface, generating hOCR from PaddleOCR bounding boxes.
    """
    ocrmypdf.ocr(
        input_pdf,
        output_pdf,
        plugins=["ocrmypdf_paddleocr"],
        language=OCR_LANG,
        force_ocr=True,
        paddle_use_gpu=OCR_USE_GPU,
        paddle_det_model_dir=f"{PADDLEOCR_MODELS}/PP-OCRv6_medium_det_infer",
        paddle_rec_model_dir=f"{PADDLEOCR_MODELS}/PP-OCRv6_medium_rec_infer",
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
    """Wait for Docling API to become available before processing.

    Skips the health check when DOCLING_MODE is 'off' or 'native' since no
    external Docling service is needed in those cases.
    """
    if DOCLING_MODE in ("off", "native"):
        return

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
def process_file(pdf_path: Path) -> bool:
    """Process a single PDF: Docling → OCR → Paperless.

    Returns True on success. On failure, leaves the original PDF in PROCESSING/
    so the caller can retry it and decide when to move it to ERROR/.
    """
    filename = pdf_path.name
    log(f"\n{'=' * 60}")
    log(f"Processing: {filename}")
    log(f"{'=' * 60}")

    # Move to processing unless this is already a retry from PROCESSING/.
    processing_path = PROCESSING / filename
    if pdf_path != processing_path:
        shutil.move(str(pdf_path), str(processing_path))
        log(f"  Moved to processing: {processing_path}")
    else:
        log(f"  Retrying from processing: {processing_path}")

    # Docling
    if not handle_docling(processing_path):
        log_error("Sidecar generation failed and will be retried if attempts remain.")
        return False

    # OCR
    ocr_output = PROCESSING / f"{processing_path.stem}_ocr{processing_path.suffix}"
    try:
        run_ocrmypdf(processing_path, ocr_output)
        log("  OCR: OK")
    except Exception as exc:
        log_error(f"OCR failed: {exc}")
        if ocr_output.exists():
            ocr_output.unlink()
        return False

    # Paperless
    if not push_to_paperless(ocr_output):
        if ocr_output.exists():
            ocr_output.unlink()
        return False

    # Done
    shutil.move(str(processing_path), str(DONE / filename))
    # Clean up intermediate OCR output file
    if ocr_output.exists():
        ocr_output.unlink()
    log(f"  → DONE/ ({filename})")
    return True


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
    log(f"  OCR_USE_GPU: {OCR_USE_GPU}")
    log(f"  STABILITY_TIMEOUT: {os.getenv('STABILITY_TIMEOUT', '10')}s")

    # Crash recovery
    recover_leftover_files()

    # Validate PaddleOCR models — fail fast if they're missing or mismatched
    try:
        validate_paddleocr_models()
        log("PaddleOCR models validated successfully.")
    except Exception as exc:
        log_error(f"PaddleOCR model validation failed: {exc}")
        sys.exit(1)

    # Wait for Docling to become available (if configured)
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
                attempts = max(1, max_retries)
                for attempt in range(1, attempts + 1):
                    # If a previous attempt already moved the file to PROCESSING,
                    # retry with that path instead of the original INBOX path.
                    current_path = (
                        PROCESSING / pdf.name if (PROCESSING / pdf.name).exists() else pdf
                    )

                    try:
                        success = process_file(current_path)
                    except Exception as exc:
                        success = False
                        log_error(
                            f"Processing attempt {attempt}/{attempts} for {pdf.name} "
                            f"raised: {exc}"
                        )

                    if success:
                        break  # Success — move to next file

                    if attempt < attempts:
                        log_error(
                            f"Processing attempt {attempt}/{attempts} failed for "
                            f"{pdf.name}; retrying in {retry_delay}s"
                        )
                        time.sleep(retry_delay)
                        continue

                    log_error(f"Max retries reached for {pdf.name}")
                    # Make sure the file ends up in ERROR/ after the final attempt.
                    failed_path = (
                        PROCESSING / pdf.name if (PROCESSING / pdf.name).exists() else pdf
                    )
                    move_to_error(failed_path, "max retries reached")

        except Exception as exc:
            log_error(f"Main loop error: {exc}")

        # Sleep before next poll
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
