# Doc-Worker

A production-ready Docker-based OCR pipeline with **PaddleOCR-VL** document understanding, supporting dual ingestion methods:

1. **Folder Watcher** — polls an inbox directory for new files, processes them sequentially, and moves them through lifecycle directories.
2. **API Hook** — an HTTP endpoint that accepts document submissions on-demand and returns structured output (OCR'd PDF + Markdown sidecar + JSON metadata).

Both ingestion methods share the same processing pipeline, ensuring consistency and simplifying maintenance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       doc-worker Container                       │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │ Folder Watcher│    │  API Hook    │                          │
│  │  (poller)    │    │  (FastAPI)   │                          │
│  └──────┬───────┘    └──────┬───────┘                          │
│         │                   │                                   │
│         └────────┬──────────┘                                   │
│                  ▼                                               │
│           ┌──────────────┐                                      │
│           │  Orchestrator │  ← FSM state management,             │
│           │   (pipeline)  │    stage dispatch, retry logic       │
│           └──────┬───────┘                                      │
│                  │                                                │
│         ┌────────┼──────────────────────────────┐                │
│         ▼        ▼                               ▼               │
│   ┌──────────┐ ┌──────────┐  ...          ┌────────────┐        │
│   │ Validate │ │ OCR      │               │ Lifecycle  │        │
│   │ & Triage │ │ Pipeline │               │ & Cleanup  │        │
│   └──────────┘ └──────────┘               └────────────┘        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            server:companion (shared backend)              │   │
│  │  • PaddleOCR model loading & inference                    │   │
│  │  • PP-OCRv6 (text detection + recognition)                │   │
│  │  • PaddleOCR-VL (document understanding)                  │   │
│  │  • File I/O abstraction (disk / in-memory)               │   │
│  │  • Logging, metrics, health checks                        │   │
│  │  • Configuration management                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            server:security (hardening)                    │   │
│  │  • Rate limiting                                          │   │
│  │  • Input validation & sanitization                        │   │
│  │  • Secure headers                                         │   │
│  │  • CORS configuration                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Validate & Triage** — file type check, size limit, document classification (text/scanned/hybrid PDF)
2. **OCR Processing** — PP-OCRv6 for text detection/recognition + PaddleOCR-VL for document understanding
3. **Output Assembly** — persists OCR'd PDF, Markdown sidecar, JSON metadata, pushes to Paperless
4. **Lifecycle Management** — moves files through DONE/ or ERROR/, cleans up intermediates

### OCR Engines

| Engine | Purpose | Model |
|---|---|---|
| **PP-OCRv6** | Text detection + recognition | PaddleOCR detection (DB) + recognition (CRNN/SVTR) |
| **PaddleOCR-VL** | Document understanding | PaddleOCR-VL-1.5B (vision-language model) |

Processing modes:
- `auto` — PP-OCR for text, PaddleOCR-VL for complex documents (default)
- `pp_ocr` — PP-OCRv6 only (fast, text-only)
- `paddle_vl` — PaddleOCR-VL for full document understanding

## Quick Start

### 1. Build the image

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

**Folder mode** — drop files into the inbox:
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
| `OUTPUT_DIR` | `/work/output` | Sidecar output directory (Markdown, JSON) |
| `PAPERLESS_CONSUME` | `/paperless-consume` | Paperless-ngx consume directory |
| `OCR_LANG` | `deu` | OCR language (for PP-OCR) |
| `OCR_RUNTIME` | `cpu` | Backend: `cpu`, `cuda`, `openvino`, `rocm` |
| `PROCESSING_MODE` | `auto` | OCR mode: `auto`, `pp_ocr`, `paddle_vl` |
| `PADDLE_OCR_LANG` | `ch` | PaddleOCR language: `ch`, `en`, `german`, etc. |
| `PADDLE_VL_MODEL` | `PaddleOCR-VL-1.5B` | PaddleOCR-VL model name |
| `PADDLE_DEVICE` | `auto` | Device: `auto`, `cpu`, `gpu` |
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
| `GET` | `/ready` | Readiness check (models loaded + orchestrator running) |
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
  "metadata": {
    "engine": "paddleocr",
    "pp_ocr": {
      "block_count": 15,
      "avg_confidence": 0.95
    },
    "paddle_vl": {
      "tables_count": 2,
      "formulas_count": 0
    }
  },
  "manifest": { ... }
}
```

### Error Responses

| Status | Condition |
|---|---|
| `400` | Unsupported file type or file too large |
| `429` | Too many concurrent jobs or rate limited |
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

## Volume Mounts

| Path | Purpose | Required |
|---|---|---|
| `/work/inbox` | Input directory for new files | Yes (folder mode) |
| `/work/processing` | Temporary staging area | Yes |
| `/work/done` | Archive of processed files | Yes |
| `/work/error` | Failed files for inspection | Yes |
| `/work/output` | Sidecar output (Markdown, JSON, manifests) | Optional |
| `/paperless-consume` | Paperless-ngx consume directory | Yes |

## Error Handling

- **Unstable files**: Skipped if file size changes during stability check.
- **Stage failures (transient)**: Retried up to `MAX_RETRIES` with exponential backoff.
- **Stage failures (permanent)**: Job marked as failed, file moved to `ERROR/`.
- **Invalid input**: Failed immediately with descriptive error; no retries.
- **Concurrent limit exceeded**: Jobs queued (folder) or rejected with HTTP 429 (API).
- **Crash recovery**: On restart, leftover files in `PROCESSING/` are moved to `ERROR/`.

## OCR Language

### PP-OCR Languages

| Language | Code |
|---|---|
| German | `german` |
| English | `en` |
| French | `french` |
| Spanish | `spanish` |
| Chinese (Simplified) | `ch` |
| Japanese | `japan` |
| Korean | `korean` |

### PaddleOCR-VL

PaddleOCR-VL is language-agnostic for document understanding. It produces structured markdown output regardless of the document language.

## Testing

### Unit Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

### Load Testing

```bash
# Start Locust web UI
locust -f locustfile.py --host http://localhost:8000

# Headless mode
locust -f locustfile.py --host http://localhost:8000 \
    --users 10 --spawn-rate 2 --run-time 5m --headless
```

### Security Scanning

```bash
# Bandit (Python security linter)
bandit -r server/ -ll

# Trivy (container vulnerability scanner)
trivy image doc-worker:latest --severity CRITICAL,HIGH
```

## Production Deployment

### Docker Compose (Production)

```yaml
services:
  doc-worker:
    image: doc-worker:latest
    container_name: doc-worker
    restart: always
    volumes:
      - ./work/inbox:/work/inbox
      - ./work/processing:/work/processing
      - ./work/done:/work/done
      - ./work/error:/work/error
      - ./work/output:/work/output
      - ./paperless-consume:/paperless-consume
      - paddle-model-cache:/root/.cache/modelscope
    ports:
      - "8000:8000"
    environment:
      - OCR_LANG=deu
      - PROCESSING_MODE=auto
      - MAX_CONCURRENT_JOBS=2
      - LOG_LEVEL=INFO
      - LOG_JSON=true
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### Security Considerations

1. **Rate Limiting**: Built-in rate limiting (60 requests/minute per IP)
2. **Input Validation**: File type and size validation
3. **Secure Headers**: X-Content-Type-Options, X-Frame-Options, HSTS, CSP
4. **CORS**: Configurable CORS policy
5. **Trusted Hosts**: Configurable host validation

### Monitoring

- **Health Check**: `GET /health` — basic health status
- **Readiness**: `GET /ready` — model loading + orchestrator status
- **Metrics**: `GET /metrics` — processing counters and timings
- **Structured Logging**: JSON-formatted logs for log aggregation

## Migration Notes

### From v1 (worker.py) to v2 (main.py)

- **Entry point**: `worker.py` → `main.py`
- **Default mode**: Combined (folder + API). Use `--mode folder` for folder-only behavior.
- **Docling**: The external Docling service has been replaced by integrated PaddleOCR-VL.
- **Configuration**: All existing environment variables are supported. New variables added for PaddleOCR control.
- **Backward compatibility**: The folder watcher behavior is identical to v1. The API is an addition.

### From v2 (RapidOCR) to v3 (PaddleOCR-VL)

- **OCR Engine**: RapidOCR → PaddleOCR (PP-OCRv6 + PaddleOCR-VL)
- **Model Loading**: PaddleOCR models are pre-downloaded at build time
- **Document Understanding**: PaddleOCR-VL provides layout analysis, table detection, and formula recognition
- **Markdown Output**: Generated by PaddleOCR-VL instead of Docling API
- **Docker Image**: Larger due to PaddlePaddle + PaddleOCR-VL models (~2-3 GB)

## License

[Add your license here]