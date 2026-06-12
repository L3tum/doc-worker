# Doc-Worker

A Docker-based OCR pipeline worker that:

1. **Polls** an inbox directory for new PDF files.
2. **Runs OCR** using [OCRmyPDF](https://github.com/jbarlow83/OCRmyPDF) with the [RapidOCR](https://github.com/RapidAI/RapidOCR) ONNX plugin via Python API. Tesseract is used for deskewing and image optimization (the RapidOCR plugin handles text recognition).
3. **Generates sidecar documents** via the [Docling](https://github.com/DS4SD/docling) API (Markdown + JSON).
4. **Pushes** the processed PDFs into a [Paperless-ngx](https://paperless-ngx.com/) consume directory.

## Architecture

```
INBOX/*.pdf
  │
  ├─ Stability check (waits for upload to finish)
  │
  ├─ PROCESSING/
  │   ├─ Docling API  →  sidecar JSON
  │   ├─ ocrmypdf.ocr() + RapidOCR  →  searchable PDF
  │   └─ Push → Paperless consume/
  │
  ├─ DONE/       (successfully processed)
  └─ ERROR/      (failed processing, for inspection)
```

## Quick Start

### 1. Build the image

```bash
docker build -t doc-worker .
```

### 2. Run with Docker Compose

See [`docker-compose.yml`](docker-compose.yml) for a complete example. The minimal setup:

```yaml
services:
  doc-worker:
    image: doc-worker
    volumes:
      - ./inbox:/work/inbox
      - ./processing:/work/processing
      - ./done:/work/done
      - ./error:/work/error
      - ./docling:/work/docling
      - ./paperless-consume:/paperless-consume
    environment:
      - DOCLING_BASE_URL=http://docling:12000
      - OCR_LANG=deu+eng
      - POLL_INTERVAL=5
      - DOCLING_TIMEOUT=900
      - MAX_RETRIES=3
      - RETRY_DELAY=10
    depends_on:
      - docling
```

### 3. Drop PDFs into the inbox

```bash
cp new-document.pdf ./inbox/
```

The worker will pick it up within `POLL_INTERVAL` seconds.

## Configuration

All settings are environment variables:

| Variable | Default | Description |
|---|---|---|
| `INBOX` | `/work/inbox` | Directory to poll for new PDFs |
| `PROCESSING` | `/work/processing` | Staging area during processing |
| `DONE` | `/work/done` | Successfully processed files |
| `ERROR` | `/work/error` | Failed files (for inspection) |
| `DOCLING_DIR` | `/work/docling` | Docling sidecar output |
| `PAPERLESS_CONSUME` | `/paperless-consume` | Paperless-ngx consume directory |
| `OCR_LANG` | `deu+eng` | OCR language(s) |
| `POLL_INTERVAL` | `5` | Seconds between inbox polls |
| `DOCLING_BASE_URL` | `http://docling:12000` | Docling API endpoint |
| `DOCLING_TIMEOUT` | `900` | Timeout for Docling API calls (seconds) |
| `RAPIDOCR_CONFIG` | *(none)* | Optional path to a custom RapidOCR YAML config. If omitted, RapidOCR uses its built-in defaults. |
| `MAX_RETRIES` | `3` | Max OCR retry attempts |
| `RETRY_DELAY` | `10` | Seconds between OCR retries |

## Volume Mounts

| Path | Purpose | Required |
|---|---|---|
| `/work/inbox` | Input directory for new PDFs | Yes |
| `/work/processing` | Temporary staging area | Yes |
| `/work/done` | Archive of successfully processed files | Yes |
| `/work/error` | Failed files for inspection | Yes |
| `/work/docling` | Docling sidecar output (JSON) | Optional |
| `/paperless-consume` | Paperless-ngx consume directory | Yes |

## Error Handling

- **Unstable files**: If a file's size changes during the stability check (default 30 s), it is skipped and retried on the next poll.
- **Docling failures**: Logged but non-fatal. The pipeline continues with OCR and Paperless ingestion.
- **OCR failures**: Retried up to `MAX_RETRIES` times with `RETRY_DELAY` seconds between attempts. On final failure, the file is moved to `/work/error`.
- **Paperless push failures**: The file is moved to `/work/error` for manual inspection.

## Health Check

The container includes a `HEALTHCHECK` that verifies the worker process is still running. Use `docker inspect` or your orchestrator's health monitoring to check status.

## License

[Add your license here]
