#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

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


def run_ocrmypdf(input_pdf: Path, output_pdf: Path) -> None:
    rapidocr_config = os.getenv("RAPIDOCR_CONFIG", "/opt/doc-worker/rapidocr.yaml")

    cmd = [
        "ocrmypdf",
        "--plugin", "ocrmypdf_rapidocr",
        "--rapidocr-config-path", rapidocr_config,
        "-l", OCR_LANG,

        "--force-ocr",
        "--deskew",
        "--clean",
        "--optimize", "1",

        str(input_pdf),
        str(output_pdf),
    ]

    subprocess.run(cmd, check=True)

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


def main() -> None:
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
