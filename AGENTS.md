# AGENTS.md — Doc-Worker

## What It Is

Docker-based OCR pipeline: polls `INBOX/*.pdf` → OCRmyPDF + PaddleOCR → pushes to Paperless-ngx. Also serves a FastAPI endpoint for Open-WebUI integration (`/layout-parsing`, `/extract`).

## Key Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI: `/health`, `/layout-parsing` (Open-WebUI), `/extract` |
| `worker.py` | File polling worker: inbox → processing → done/error |
| `Dockerfile` | Multi-variant: CPU (default), CUDA via `PADDLE_GPU` build arg |
| `docker-compose.yml` | Full stack example with docling + paperless |
| `pyproject.toml` | Project metadata (minimal) |
| `requirements.txt` | Runtime deps |

## Tech Stack

- **Python 3.12**, **FastAPI** (Uvicorn), **OCRmyPDF** + **PaddleOCR** plugin, **Docling** API client
- **Docker** with `PADDLE_GPU` build arg (`cpu`/`cuda`)
- **CI**: GitHub Actions (`docker.yaml`) — builds CPU + CUDA, pushes to GHCR

## Directory Flow

```
INBOX → stability check → PROCESSING → [Docling sidecar] → [OCR] → [Paperless push] → DONE/ERROR
```

## Env Vars (runtime)

`INBOX`, `PROCESSING`, `DONE`, `ERROR`, `DOCLING_DIR`, `PAPERLESS_CONSUME`, `OCR_LANG` (default `deu`), `OCR_USE_GPU`, `POLL_INTERVAL`, `DOCLING_BASE_URL`, `DOCLING_MODE`, `DOCLING_TIMEOUT`, `MAX_RETRIES`, `RETRY_DELAY`, `PADDLEOCR_VL_TOKEN`, `PADDLEOCR_MODELS` (default `/app/models`)

## Build Args

`PADDLE_GPU`: `cpu` (default), `cuda`

## Critical Gotchas

1. **Open-WebUI sends `Bearer <token>`** — parse auth header accordingly in `server.py`
2. **tesseract-ocr is still required at import time** by OCRmyPDF even when using PaddleOCR backend
3. **ROCm (AMD GPU) not supported** — PaddlePaddle's ROCm wheels are only available via their Docker images, not pip. The wheel index is a JavaScript SPA that pip can't parse.
