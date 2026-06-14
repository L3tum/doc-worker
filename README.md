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

**CPU (default):**

```bash
docker build -t doc-worker .
```

**CUDA GPU (NVIDIA):**

```bash
docker build --build-arg ONNX_RUNTIME=cuda -t doc-worker .
```

**OpenVINO (Intel GPU/CPU):**

```bash
docker build --build-arg ONNX_RUNTIME=openvino -t doc-worker .
```

**ROCm (AMD GPU):**

```bash
docker build --build-arg ONNX_RUNTIME=rocm -t doc-worker .
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
      - DOCLING_BASE_URL=http://docling:5001
      - DOCLING_MODE=best_effort
      - OCR_LANG=deu
      - OCR_RUNTIME=cpu
      - POLL_INTERVAL=5
      - MAX_RETRIES=3
      - RETRY_DELAY=10
    depends_on:
      - docling
```

For **GPU acceleration**, set `OCR_RUNTIME` to match your build and enable the appropriate GPU passthrough:

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
| `OCR_LANG` | `deu` | OCR language (single language, see note below) |
| `OCR_RUNTIME` | `cpu` | Inference backend: `cpu`, `cuda`, `openvino`, or `rocm` |
| `POLL_INTERVAL` | `5` | Seconds between inbox polls |
| `DOCLING_BASE_URL` | `http://docling:5001` | Docling API endpoint |
| `DOCLING_MODE` | `best_effort` | Docling behavior: `off`, `best_effort`, or `required` (see below) |
| `DOCLING_TIMEOUT` | `900` | Timeout for Docling API calls (seconds, hardcoded) |
| `RAPIDOCR_CONFIG` | *(none)* | Optional path to a custom RapidOCR YAML config. If omitted, RapidOCR uses its built-in defaults. |
| `MAX_RETRIES` | `3` | Max OCR retry attempts |
| `RETRY_DELAY` | `10` | Seconds between OCR retries |

### Build Arguments

| Argument | Default | Description |
|---|---|---|
| `ONNX_RUNTIME` | `cpu` | ONNX Runtime variant (see [Runtime Backends](#runtime-backends)) |

## Runtime Backends

The worker supports four ONNX Runtime backends, selected via the `ONNX_RUNTIME` build arg and `OCR_RUNTIME` runtime env var:

| Backend | Build Arg | Env Var | Hardware | Package | Image Size |
|---|---|---|---|---|---|
| **CPU** | `cpu` | `cpu` | Any CPU | `onnxruntime` | ~1.2 GB |
| **CUDA** | `cuda` | `cuda` | NVIDIA GPU | `onnxruntime-gpu` | ~4.5 GB |
| **OpenVINO** | `openvino` | `openvino` | Intel GPU/CPU | `onnxruntime-openvino` | ~2.5 GB |
| **ROCm** | `rocm` | `rocm` | AMD GPU | `onnxruntime-rocm` | ~3.8 GB |

The worker **auto-detects** provider availability at startup. If the requested provider is not available, it falls back to CPU with a warning.

### CUDA (NVIDIA)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. Uses CUDA 13.3.0.

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

### OpenVINO (Intel)

Works with Intel integrated GPUs (iGPU), discrete GPUs (Arc), and CPUs with AVX-512. No special Docker runtime needed.

```yaml
environment:
  - OCR_RUNTIME=openvino
```

### ROCm (AMD)

Requires AMD GPU with ROCm support. May need additional device passthrough.

```yaml
environment:
  - OCR_RUNTIME=rocm
devices:
  - /dev/kfd
  - /dev/dri
```

## Important Notes

### OCR Language

The `ocrmypdf-rapidocr` plugin currently **does not support multi-language selection** (e.g. `deu+eng`). The `+` separator is a Tesseract convention that RapidOCR does not understand. Set `OCR_LANG` to a **single language code**:

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
| Czech | `ces` |
| Chinese (Simplified) | `ch_sim` |
| Chinese (Traditional) | `ch_tra` |
| Japanese | `jpn` |
| Korean | `kor` |

### Pre-downloaded Models

The Docker image ships with **Latin language models** pre-downloaded. If you use a non-Latin language (e.g. Chinese, Japanese, Korean), the first run will download the corresponding models automatically.

### Docling Modes

The `DOCLING_MODE` environment variable controls how Docling sidecar generation is handled:

| Mode | Behavior |
|---|---|
| `best_effort` (default) | Attempt Docling conversion. If it fails, log a warning and continue with OCR + Paperless. |
| `off` | Skip Docling entirely. No sidecar files are generated. Useful when Docling is not available or not needed. |
| `required` | Attempt Docling conversion. If it fails, move the file to `/work/error` (treat Docling as a hard requirement). |

**To disable Docling:**

```yaml
environment:
  - DOCLING_MODE=off
```

When Docling runs, it generates Markdown (`.md`) and JSON (`.json`) sidecar files in `/work/docling/<filename>/`.

## Volume Mounts

| Path | Purpose | Required |
|---|---|
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
