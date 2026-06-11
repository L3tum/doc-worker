#!/usr/bin/env python3
"""Doc-Worker — poll an inbox, run OCR + Docling, push to Paperless."""

import os, sys, time, shutil, subprocess, re
import logging
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INBOX = Path(os.environ.get("INBOX", "/work/inbox"))
PROCESSING = Path(os.environ.get("PROCESSING", "/work/processing"))
DONE = Path(os.environ.get("DONE", "/work/done"))
ERROR = Path(os.environ.get("ERROR", "/work/error"))
DOCLING_DIR = Path(os.environ.get("DOCLING_DIR", "/work/docling"))
PAPERLESS_CONSUME = Path(os.environ.get("PAPERLESS_CONSUME", "/paperless-consume"))

OCR_LANG = os.environ.get("OCR_LANG", "deu+eng")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))
DOCLING_BASE_URL = os.environ.get("DOCLING_BASE_URL", "http://docling:12000")
DOCLING_TIMEOUT = int(os.environ.get("DOCLING_TIMEOUT", "900"))
RAPIDOCR_CONFIG = os.environ.get("RAPIDOCR_CONFIG", "/app/rapidocr.yaml")

# Retry settings
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "10"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("doc-worker")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    """Create all staging directories if they don't exist."""
    for d in (INBOX, PROCESSING, DONE, ERROR, DOCLING_DIR, PAPERLESS_CONSUME):
        d.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    """Return a filesystem-safe version of *name*."""
    safe = re.sub(r"[^a-zA-Z0-9._\-\s]", "", name)
    safe = safe.strip()
    return safe or "unnamed"


def is_stable(path: Path, timeout: int = 30) -> bool:
    """Return True only if *path* stops growing for *timeout* seconds."""
    last_size = path.stat().st_size
    waited = 0
    while waited < timeout:
        time.sleep(1)
        waited += 1
        try:
            current_size = path.stat().st_size
        except FileNotFoundError:
            return False
        if current_size == last_size:
            waited += 5  # bonus stability
        else:
            last_size = current_size
        if waited >= timeout:
            break
    # final check
    return path.stat().st_size == last_size


def wait_for_docling(timeout: int = 120):
    """Block until the Docling health endpoint responds or *timeout* expires."""
    health = f"{DOCLING_BASE_URL}/v1health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = Request(health, method="GET")
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    log.info("Docling is reachable.")
                    return
        except Exception:
            pass
        log.info("Waiting for Docling …")
        time.sleep(5)
    log.warning("Docling did not respond in %d s — continuing without sidecar.", timeout)


def call_docling(pdf_path: Path) -> Path | None:
    """POST *pdf_path* to Docling, return the sidecar output path or None."""
    url = f"{DOCLING_BASE_URL}/v1predict"
    try:
        with open(pdf_path, "rb") as f:
            data = f.read()

        req = Request(
            url,
            data=data,
            headers={
                "accept": "application/json",
                "Content-Type": "application/pdf",
            },
        )
        with urlopen(req, timeout=DOCLING_TIMEOUT) as resp:
            resp.raise_for_status()
            result = resp.read().decode("utf-8")

        out = DOCLING_DIR / f"{pdf_path.stem}.docling.json"
        with open(out, "w") as f:
            f.write(result)
        log.info("Docling sidecar written: %s", out)
        return out

    except Exception as exc:
        log.error("Docling call failed: %s", exc)
        err_out = DOCLING_DIR / f"{pdf_path.stem}.docling.error"
        with open(err_out, "w") as f:
            f.write(str(exc))
        return None


def run_ocrmypdf(src: Path, dst: Path) -> bool:
    """Run OCRmyPDF on *src*, writing to *dst*. Return True on success."""

    def _attempt(attempt: int) -> bool:
        cmd = [
            "ocrmypdf",
            "--force-ocr",
            "--output-type", "pdf",
            "--optimize", "off",
            "--plugin", "ocrmypdf_plugin_rapidocr",
            "--rapidocr-config", RAPIDOCR_CONFIG,
            # Pass language for potential non-RapidOCR backends
            "--language", OCR_LANG,
            "--jobs", "1",
            str(src),
            str(dst),
        ]
        log.info("OCR attempt %d/%d: %s", attempt, MAX_RETRIES, " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.warning("OCR attempt %d failed (rc=%d): %s",
                        attempt, result.returncode, result.stderr[:500])
            return False
        log.info("OCR succeeded: %s", dst)
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        if _attempt(attempt):
            return True
        if attempt < MAX_RETRIES:
            log.info("Retrying in %d seconds …", RETRY_DELAY)
            time.sleep(RETRY_DELAY)

    log.error("OCR failed after %d attempts.", MAX_RETRIES)
    return False


def push_to_paperless(src: Path) -> bool:
    """Atomically move *src* into Paperless' consume directory."""
    safe = safe_name(src.name)
    tmp = PAPERLESS_CONSUME / f".tmp.{safe}"
    final = PAPERLESS_CONSUME / safe

    # If a sidecar exists, move it alongside the PDF
    sidecar = DOCLING_DIR / f"{src.stem}.docling.json"
    if sidecar.exists():
        sc_tmp = PAPERLESS_CONSUME / f".tmp.{safe}.docling.json"
        shutil.move(str(sidecar), str(sc_tmp))

    shutil.move(str(src), str(tmp))
    os.rename(str(tmp), str(final))  # atomic on same filesystem

    if sidecar.exists():
        os.rename(str(sc_tmp), str(PAPERLESS_CONSUME / f"{safe}.docling.json"))

    log.info("Pushed to Paperless: %s", final)
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ensure_dirs()
    wait_for_docling()

    log.info("Doc-Worker started. Polling %s every %d s.", INBOX, POLL_INTERVAL)

    while True:
        # Collect PDFs with case-insensitive matching, deduplicated by realpath
        seen = set()
        pdfs = []
        for pattern in ("*.pdf", "*.PDF", "*.Pdf"):
            for p in sorted(INBOX.glob(pattern)):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    pdfs.append(p)

        if not pdfs:
            time.sleep(POLL_INTERVAL)
            continue

        for pdf in pdfs:
            log.info("===== Processing: %s =====", pdf.name)

            # 1. Wait for upload stability
            if not is_stable(pdf):
                log.warning("File disappeared or unstable: %s — skipping", pdf.name)
                continue

            # 2. Move to processing staging (atomic rename on same FS)
            try:
                proc_path = PROCESSING / pdf.name
                shutil.move(str(pdf), str(proc_path))
            except Exception as exc:
                log.error("Failed to move %s to processing: %s", pdf.name, exc)
                continue

            # 3. Docling sidecar (best-effort)
            call_docling(proc_path)

            # 4. OCR
            ocr_output = PROCESSING / f"{proc_path.stem}.ocr.pdf"
            try:
                if not run_ocrmypdf(proc_path, ocr_output):
                    raise RuntimeError("OCRmyPDF failed after retries")
            except Exception as exc:
                log.error("OCR pipeline error for %s: %s", proc_path.name, exc)
                shutil.move(str(proc_path), str(ERROR / proc_path.name))
                continue

            # 5. Push to Paperless
            try:
                if not push_to_paperless(ocr_output):
                    raise RuntimeError("Push to Paperless failed")
            except Exception as exc:
                log.error("Paperless push error for %s: %s", ocr_output.name, exc)
                shutil.move(str(ocr_output), str(ERROR / ocr_output.name))
                continue

            # 6. Done
            final_name = f"{proc_path.stem}.ocr.pdf"
            shutil.move(str(ocr_output), str(DONE / final_name))
            log.info("DONE: %s → %s", pdf.name, final_name)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
