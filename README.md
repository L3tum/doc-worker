# Doc-Worker

A Docker-based OCR pipeline worker with a unified sequential processing engine supporting two ingestion methods:

1. **Folder Watcher** — polls an inbox directory for new PDFs, processes them sequentially, and moves them through lifecycle directories.
2. **API Hook** — an HTTP endpoint that accepts document submissions on-demand and returns structured output (OCR'd PDF + Markdown sidecar).

Both ingestion methods share the same processing pipeline, ensuring consistency and simplifying maintenance.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      doc-worker Container                    │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │ Folder Watcher│    │  API Hook    │                       │
│  │  (poller)    │    │  (FastAPI)   │                       │
│  └──────┬───────┘    └──────┬───────┘                       │
│         │                   │                                │
│         └────────┬──────────┘                                │
│                  ▼                                            │
│           ┌──────────────┐                                   │
│           │  Orchestrator │  ← FSM state management,          │
│           │   (pipeline)  │    stage dispatch, retry logic    │
│           └──────┬───────┘                                   │
│                  │                                             │
│         ┌────────┼─────────────────────────────┐              │
│         ▼        ▼                              ▼             │
│   ┌──────────┐ ┌──────────┐  ...       ┌────────────┐        │
│   │ Validate │ │ OCR      │            │ Lifecycle  │        │
│   │ & Triage │ │ Pipeline │            │ & Cleanup  │        │
│   └──────────┘ └──────────┘            └────────────┘        │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │            server:companion (shared backend)          │    │
│  │  • PaddleOCR model loading & inference                │    │
│  │  • File I/O abstraction (disk / in-memory)           │    │
│  │  • Logging, metrics, health checks                    │    │
│  │  • Configuration management                           │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Validate & Triage** — file type check, size limit, document classification (text/scanned/hybrid PDF)
2. **OCR Processing** — runs OCRmyPDF + RapidOCR (current) or PaddleOCR (Phase 4)
3. **Output Assembly** — persists OCR'd PDF, Markdown sidecar, JSON metadata, pushes to Paperless
4. **Lifecycle Management** — moves files through DONE/ or ERROR/, cleans up intermediates

## Quick Start

### 1. Build or pull the image

| Tag | Backend |
|---|---|
| `:latest` | CPU default |
| `:cpu` | CPU explicit |
| `:cuda` | NVIDIA CUDA |
| `:openvino` | OpenVINO *(experimental)* |
| `:rocm` | ROCm *(experimental)* |

```bash
# CPU (default)
docker build -t doc-worker:cpu -t doc-worker:latest .

# CUDA GPU (NVIDIA)
docker build --build-arg ONNX_RUNTIME=cuda -t doc-worker:cuda .
```

### 2. Run with Docker Compose

```bash
docker compose up -d
```

See [`docker-compose.yml`](docker-compose.yml) for the full configuration.

### 3. Use the pipeline

**Folder mode** — drop PDFs into the inbox:
```bash
cp new-document.pdf ./work/inbox/
```

**API mode** — submit via HTTP:
```bash
curl -X POST http://localhost:8000/api/v1/convert \
  -F "file=@new-document.pdf" \
  -F "mode=auto"
```

**Combined mode** (default) — both folder watching and API are active simultaneously.

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `INBOX` | `/work/inbox` | Directory to poll for new files |
| `PROCESSING` | `/work/processing` | Staging area during processing |
| `DONE` | `/work/done` | Successfully processed files |
| `ERROR` | `/work/error` | Failed files |
| `DOCLING_DIR` | `/work/docling` | Sidecar output directory |
| `PAPERLESS_CONSUME` | `/paperless-consume` | Paperless-ngx consume directory |
| `OCR_LANG` | `deu` | OCR language |
| `OCR_RUNTIME` | `cpu` | Backend: `cpu`, `cuda`, `openvino`, `rocm` |
| `PROCESSING_MODE` | `auto` | OCR mode: `auto`, `pp_ocr`, `paddle_vl` |
| `POLL_INTERVAL` | `5` | Seconds between inbox polls |
| `MAX_RETRIES` | `3` | Max retry attempts per job |
| `RETRY_DELAY` | `10` | Base seconds between retries (exponential backoff) |
| `MAX_FILE_SIZE_MB` | `100` | Maximum file size in MB |
| `MAX_CONCURRENT_JOBS` | `1` | Max concurrent pipeline jobs |
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |
| `API_ENABLED` | `true` | Enable API server |
| `LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_JSON` | `true` | Use structured JSON logging |

### Build Arguments

| Argument | Default | Description |
|---|---|---|
| `ONNX_RUNTIME` | `cpu` | ONNX Runtime variant (see [Runtime Backends](#runtime-backends)) |

## API Reference

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/convert` | Submit a document for OCR processing |
| `GET` | `/health` | Health check |
| `GET` | `/ready` | Readiness check (model loaded + orchestrator running) |
| `GET` | `/metrics` | Processing metrics |

### Convert Endpoint

```
POST /api/v1/convert
Content-Type: multipart/form-data

Parameters:
  file: UploadFile (required) — PDF or image to process
  mode: str (optional, default "auto") — processing mode

Response (200):
{
  "job_id": "abc123",
  "filename": "document.pdf",
  "document_type": "scanned_pdf",
  "processing_mode": "full_ocr",
  "elapsed_seconds": 12.34,
  "pdf": {
    "filename": "document_ocr.pdf",
    "size": 123456,
    "data": "<base64-encoded PDF>"
  },
  "markdown": {
    "filename": "document.md",
    "content": "# Extracted text..."
  },
  "metadata": { ... },
  "manifest": { ... }
}
```

### Error Responses

| Status | Condition |
|---|---|
| `400` | Unsupported file type or file too large |
| `429` | Too many concurrent jobs |
| `500` | Processing failed |
| `503` | Orchestrator not initialized |

## Runtime Backends

| Backend | Image Tag | Build Arg | Env Var | Hardware | Status |
|---|---|---|---|---|---|
| **CPU** | `:latest`, `:cpu` | `cpu` | `cpu` | Any CPU | Supported default |
| **CUDA** | `:cuda` | `cuda` | `cuda` | NVIDIA GPU | Supported |
| **OpenVINO** | `:openvino` | `openvino` | `openvino` | Intel GPU/CPU | Experimental |
| **ROCm** | `:rocm` | `rocm` | `rocm` | AMD GPU | Experimental |

### CUDA (NVIDIA)

```yaml
environment:
  - OCR_RUNTIME=cuda
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### OpenVINO (Intel, experimental)

```yaml
image: doc-worker:openvino
environment:
  - OCR_RUNTIME=openvino
```

### ROCm (AMD, experimental)

```yaml
image: doc-worker:rocm
environment:
  - OCR_RUNTIME=rocm
devices:
  - /dev/kfd
  - /dev/dri
```

## Volume Mounts

| Path | Purpose | Required |
|---|---|---|
| `/work/inbox` | Input directory for new files | Yes (folder mode) |
| `/work/processing` | Temporary staging area | Yes |
| `/work/done` | Archive of processed files | Yes |
| `/work/error` | Failed files for inspection | Yes |
| `/work/docling` | Sidecar output (Markdown, JSON) | Optional |
| `/paperless-consume` | Paperless-ngx consume directory | Yes |

## Error Handling

- **Unstable files**: Skipped if file size changes during stability check.
- **Stage failures (transient)**: Retried up to `MAX_RETRIES` with exponential backoff.
- **Stage failures (permanent)**: Job marked as failed, file moved to `ERROR/`.
- **Invalid input**: Failed immediately with descriptive error; no retries.
- **Concurrent limit exceeded**: Jobs queued (folder) or rejected with HTTP 429 (API).
- **Crash recovery**: On restart, leftover files in `PROCESSING/` are moved to `ERROR/`.

## OCR Language

The `ocrmypdf-rapidocr` plugin supports single-language codes:

| Language | Code |
|---|---|
| German | `deu` |
| English | `eng` |
| French | `fra` |
| Spanish | `spa` |
| Italian | `ita` |
| Portuguese | `por` |
| Dutch | `nld` |
| Polish | `pol` |
| Chinese (Simplified) | `ch_sim` |
| Japanese | `jpn` |
| Korean | `kor` |

## Migration Notes

### From v1 (worker.py) to v2 (main.py)

- **Entry point**: `worker.py` → `main.py`
- **Default mode**: Combined (folder + API). Use `--mode folder` for folder-only behavior.
- **Docling**: The external Docling service is deprecated. Sidecar generation is handled by the integrated PaddleOCR-VL model (Phase 4).
- **Configuration**: All existing environment variables are supported. New variables added for API and pipeline control.
- **Backward compatibility**: The folder watcher behavior is identical to v1. The API is an addition, not a replacement.

## License

[Add your license here]