#!/usr/bin/env python3

import importlib
import json
import os
import shutil
import time
from pathlib import Path

import ocrmypdf
import rapidocr
import requests


INBOX = Path("/work/inbox")
PROCESSING = Path("/work/processing")
DONE = Path("/work/done")
ERROR = Path("/work/error")
DOCLING_OUT = Path("/work/docling")

PAPERLESS_CONSUME = Path(os.getenv("PAPERLESS_CONSUME", "/paperless-consume"))
DOCLING_BASE_URL = os.getenv("DOCLING_BASE_URL", "http://docling:5001").rstrip("/")
DOCLING_MODE = os.getenv("DOCLING_MODE", "best_effort")  # "off" | "best_effort" | "required"
OCR_LANG = os.getenv("OCR_LANG", "deu")
OCR_RUNTIME = os.getenv("OCR_RUNTIME", "cpu").lower()  # "cpu" | "cuda"

for path in [INBOX, PROCESSING, DONE, ERROR, DOCLING_OUT, PAPERLESS_CONSUME]:
    path.mkdir(parents=True, exist_ok=True)


def is_stable(path: Path, wait_seconds: int = 3) -> bool:
    try:
        size_1 = path.stat().st_size
        time.sleep(wait_seconds)
        size_2 = path.stat().st_size
        return size_1 == size_2 and size_1 > 0
    except FileNotFoundError:
        return False


def safe_name(name: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- "
    cleaned = "".join(c for c in name if c in allowed).strip()
    return cleaned or "document.pdf"


def wait_for_docling() -> None:
    url = f"{DOCLING_BASE_URL}/health"

    for _ in range(60):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:
                return
        except requests.RequestException:
            pass

        time.sleep(2)

    raise RuntimeError(f"Docling API not reachable at {url}")


def call_docling_api(pdf_path: Path, out_dir: Path) -> None:
    """
    Calls Docling Serve's v1 file conversion endpoint and stores sidecar output.

    Forces Docling to use RapidOCR for extraction sidecars.

    Paperless still gets the OCRmyPDF-produced PDF separately.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"{DOCLING_BASE_URL}/v1/convert/file"

    with pdf_path.open("rb") as f:
        files = {
            "files": (pdf_path.name, f, "application/pdf"),
        }

        data = {
            "from_formats": "pdf",
            "to_formats": ["md", "json"],

            # OCR behavior
            "do_ocr": "true",
            "force_ocr": "true",
            "ocr_engine": "rapidocr",

            # Optional but useful defaults
            "image_export_mode": "embedded",
            "table_mode": "accurate",
        }

        response = requests.post(
            url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=900,
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        error_file = out_dir / "docling-http-error.txt"
        error_file.write_text(
            f"URL: {url}\n"
            f"Status: {response.status_code}\n"
            f"Response:\n{response.text}\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Docling API failed with HTTP {response.status_code}. "
            f"See {error_file}"
        ) from exc

    payload = response.json()

    (out_dir / "docling-response.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    documents = payload.get("document") or payload.get("documents") or payload.get("results")

    if isinstance(documents, dict):
        write_docling_sidecars(documents, out_dir)
    elif isinstance(documents, list):
        for index, doc in enumerate(documents):
            if isinstance(doc, dict):
                item_dir = out_dir / f"document-{index}"
                item_dir.mkdir(exist_ok=True)
                write_docling_sidecars(doc, item_dir)


def write_docling_sidecars(doc: dict, out_dir: Path) -> None:
    md = (
        doc.get("md_content")
        or doc.get("markdown")
        or doc.get("md")
        or doc.get("text")
    )

    if isinstance(md, str) and md.strip():
        (out_dir / "document.md").write_text(md, encoding="utf-8")

    # Store any structured document object as JSON as well.
    structured = (
        doc.get("json_content")
        or doc.get("json")
        or doc.get("document")
        or doc
    )

    (out_dir / "document.json").write_text(
        json.dumps(structured, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _configure_rapidocr_runtime() -> None:
    """Patch RapidOCR to use the selected runtime (CPU or CUDA).

    This is called once at startup. It monkey-patches RapidOCR.__init__ so
    every instance picks up our runtime preference.
    """
    global _rapidocr_configured
    if _rapidocr_configured:
        return
    _rapidocr_configured = True

    # Validate runtime
    if OCR_RUNTIME not in ("cpu", "cuda"):
        print(
            f"WARNING: OCR_RUNTIME='{OCR_RUNTIME}' is invalid, falling back to 'cpu'. "
            "Valid values: cpu, cuda",
            flush=True,
        )
        OCR_RUNTIME = "cpu"

    # If CUDA requested, verify it's actually available
    if OCR_RUNTIME == "cuda":
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                print(
                    f"WARNING: OCR_RUNTIME=cuda requested but CUDAExecutionProvider "
                    f"is not available. Falling back to CPU. "
                    f"Available providers: {providers}",
                    flush=True,
                )
                OCR_RUNTIME = "cpu"
            else:
                print("INFO: CUDAExecutionProvider found — GPU acceleration enabled", flush=True)
        except Exception as exc:
            print(
                f"WARNING: Failed to check CUDA availability ({exc}), falling back to CPU",
                flush=True,
            )
            OCR_RUNTIME = "cpu"

    print(f"INFO: RapidOCR runtime = {OCR_RUNTIME}", flush=True)

    # Build params dict for RapidOCR
    params = {
        "EngineConfig": {
            "onnxruntime": {
                "use_cuda": OCR_RUNTIME == "cuda",
            }
        }
    }

    # Monkey-patch RapidOCR.__init__ to inject our params
    _orig_init = rapidocr.RapidOCR.__init__

    def _patched_init(self, config_path=None, user_params=None):
        merged = {
            "EngineConfig": {
                "onnxruntime": dict(params["EngineConfig"]["onnxruntime"]),
            }
        }
        if user_params:
            for key, value in user_params.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
        _orig_init(self, config_path=config_path, params=merged)

    rapidocr.RapidOCR.__init__ = _patched_init


_rapidocr_configured = False


def run_ocrmypdf(input_pdf: Path, output_pdf: Path) -> None:
    """Run OCRmyPDF with RapidOCR engine via Python API."""
    _configure_rapidocr_runtime()

    # input_file_or_options and output_file are positional args in ocrmypdf.ocr()
    # Everything else is keyword-only (after the * in the signature)
    ocrmypdf.ocr(
        input_pdf,
        output_pdf,
        plugins=["ocrmypdf_rapidocr"],
        language=OCR_LANG,
        force_ocr=True,
        deskew=True,
        clean=True,
        optimize=1,
        rapidocr_config_path=os.environ.get("RAPIDOCR_CONFIG"),
    )

def atomic_move_into_consume(source: Path, final_name: str) -> None:
    tmp_target = PAPERLESS_CONSUME / f".{final_name}.tmp"
    final_target = PAPERLESS_CONSUME / final_name

    if tmp_target.exists():
        tmp_target.unlink()

    shutil.move(str(source), str(tmp_target))
    tmp_target.rename(final_target)


def process_pdf(input_path: Path) -> None:
    cleaned_name = safe_name(input_path.name)

    if not cleaned_name.lower().endswith(".pdf"):
        cleaned_name += ".pdf"

    work_pdf = PROCESSING / cleaned_name
    ocr_pdf = PROCESSING / f"{work_pdf.stem}.ocr.pdf"
    sidecar_dir = DOCLING_OUT / work_pdf.stem

    shutil.move(str(input_path), str(work_pdf))

    print(f"Processing {cleaned_name}", flush=True)

    try:
        if DOCLING_MODE == "best_effort":
            try:
                call_docling_api(work_pdf, sidecar_dir)
            except Exception as exc:
                # Do not block Paperless ingestion if Docling sidecar creation fails.
                (sidecar_dir / "docling-error.txt").write_text(str(exc), encoding="utf-8")
                print(f"Docling failed for {cleaned_name}: {exc}", flush=True)
        elif DOCLING_MODE == "required":
            call_docling_api(work_pdf, sidecar_dir)
        # else "off": skip

        run_ocrmypdf(work_pdf, ocr_pdf)
        atomic_move_into_consume(ocr_pdf, cleaned_name)

        shutil.move(str(work_pdf), str(DONE / cleaned_name))
        print(f"Done {cleaned_name}", flush=True)

    except Exception:
        error_target = ERROR / cleaned_name
        if work_pdf.exists():
            shutil.move(str(work_pdf), str(error_target))
        if ocr_pdf.exists():
            ocr_pdf.unlink()
        raise


def recover_processing_folder() -> int:
    """On startup, recover any files left in the processing folder from a crash.

    Moves PDFs back to the inbox so they get re-processed, and cleans up
    any partial OCR outputs and orphaned sidecar directories.

    Returns the number of files recovered.
    """
    recovered = 0

    # Move any base PDFs back to inbox
    for pdf in PROCESSING.glob("*.pdf"):
        # Skip intermediate OCR outputs — those get cleaned up below
        if pdf.name.endswith(".ocr.pdf"):
            continue

        target = INBOX / pdf.name
        # Avoid overwriting an inbox file that may have already been processed
        if target.exists():
            # Stamp it with a recovery suffix so it still gets picked up
            stem = pdf.stem
            suffix = pdf.suffix
            target = INBOX / f"{stem}.recovered{suffix}"
        pdf.rename(target)
        print(f"Recovered {pdf.name} -> {target.name}", flush=True)
        recovered += 1

    # Clean up partial OCR outputs
    for ocr_pdf in PROCESSING.glob("*.ocr.pdf"):
        ocr_pdf.unlink()
        print(f"Cleaned up partial OCR output: {ocr_pdf.name}", flush=True)

    if recovered:
        print(f"Recovered {recovered} file(s) from processing folder", flush=True)

    return recovered


def main() -> None:
    # --- Crash recovery: pick up files left from a previous run ---
    recover_processing_folder()

    if DOCLING_MODE == "best_effort":
        try:
            wait_for_docling()
        except RuntimeError as exc:
            print(f"Docling not reachable, continuing in best-effort mode: {exc}", flush=True)
    elif DOCLING_MODE == "required":
        wait_for_docling()
    # else "off": skip

    while True:
        pdfs = sorted(INBOX.glob("*.pdf")) + sorted(INBOX.glob("*.PDF"))

        for pdf in pdfs:
            if not pdf.is_file():
                continue

            if not is_stable(pdf):
                continue

            try:
                process_pdf(pdf)
            except Exception as exc:
                print(f"Failed {pdf.name}: {exc}", flush=True)

        time.sleep(5)


if __name__ == "__main__":
    main()
