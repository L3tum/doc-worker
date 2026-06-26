# Doc-Worker

A Docker-based OCR pipeline worker that:

1. **Polls** an inbox directory for new PDF files.
2. **Runs OCR** using [OCRmyPDF](https://github.com/jbarlow83/OCRmyPDF) with the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) plugin via Python API.
3. **Generates sidecar documents** via the [Docling](https://github.com/DS4SD/docling) API (Markdown + JSON).
4. **Pushes** the processed PDFs into a [Paperless-ngx](https://paperless-ngx.com/) consume directory.
5. **Serves a FastAPI endpoint** for on-demand document understanding (Open-WebUI compatible).

## Architecture

```
INBOX/*.pdf
  │
  ├─ Stability check (waits for upload to finish)
  │
  ├─ PROCESSING/
  │   ├─ Docling API  →  sidecar JSON
  │   ├─ ocrmypdf.ocr() + PaddleOCR  →  searchable PDF
  │   └─ Push → Paperless consume/
  │
  ├─ DONE/       (successfully processed)
  └─ ERROR/      (failed processing, for inspection)

API Server (port 8000)
  ├─ GET  /health           — Health check
  ├─ POST /layout-parsing   — Open-WebUI PaddleOCR-VL endpoint
  └─ POST /extract          — Direct text extraction (non-Open-WebUI)
```

## Quick Start

### 1. Build or pull the image

Published image tags:

| Tag | Backend |
|---|---|
| `:latest` | CPU default |
| `:cpu` | CPU explicit |
| `:cuda` | NVIDIA CUDA |

Build locally:

```bash
# CPU (default)
docker build -t doc-worker:cpu -t doc-worker:latest .

# CUDA GPU (NVIDIA)
docker build --build-arg PADDLE_GPU=cuda -t doc-worker:cuda .
```

### 2. Run with Docker Compose

See [`docker-compose.yml`](docker-compose.yml) for a complete example:

```yaml
services:
  doc-worker:
    image: doc-worker
    ports:
      - "8000:8000"    # API server
    volumes:
      - ./inbox:/work/inbox
      - ./processing:/work/processing
      - ./done:/work/done
      - ./error:/work/error
      - ./docling:/work/docling
      - ./paperless-consume:/paperless-consume
    environment:
      - DOCLING_BASE_URL=http://docling:12000
      - DOCLING_MODE=best_effort
      - OCR_LANG=deu
      - OCR_USE_GPU=false
      - POLL_INTERVAL=5
```

For **GPU acceleration (NVIDIA)**:

```yaml
    environment:
      - OCR_USE_GPU=true
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

### 4. Configure Open-WebUI

In Open-WebUI, go to **Admin Settings > Documents**:

1. Set **Content Extraction Engine** to `PaddleOCR-VL`
2. Set **API Base URL** to `http://doc-worker:8000`
3. (Optional) Set **API Token** if you configured `PADDLEOCR_VL_TOKEN`

Open-WebUI will call `POST /layout-parsing` with base64-encoded files.

### 5. Direct API access

```bash
# Health check
curl http://localhost:8000/health

# Extract text (direct API, non-Open-WebUI)
curl -X POST http://localhost:8000/extract \
  -F "file=@document.pdf" | python -m json.tool
```

## Configuration

### Worker (folder polling)

| Variable | Default | Description |
|---|---|---|
| `INBOX` | `/work/inbox` | Directory to poll for new PDFs |
| `PROCESSING` | `/work/processing` | Staging area during processing |
| `DONE` | `/work/done` | Successfully processed files |
| `ERROR` | `/work/error` | Failed files (for inspection) |
| `DOCLING_DIR` | `/work/docling` | Docling sidecar output |
| `PAPERLESS_CONSUME` | `/paperless-consume` | Paperless-ngx consume directory |
| `OCR_LANG` | `deu` | OCR language (see [Language Codes](#ocr-language)) |
| `OCR_USE_GPU` | `false` | Enable GPU acceleration (`true`/`false`) |
| `POLL_INTERVAL` | `5` | Seconds between inbox polls |
| `DOCLING_BASE_URL` | `http://docling:12000` | Docling API endpoint |
| `DOCLING_MODE` | `best_effort` | Sidecar mode: `off`, `best_effort`, `required`, or `native` |
| `DOCLING_TIMEOUT` | `900` | Docling API timeout (seconds) |
| `MAX_RETRIES` | `3` | Max OCR retry attempts |
| `RETRY_DELAY` | `10` | Seconds between OCR retries |
| `PADDLEOCR_VL_TOKEN` | *(none)* | Bearer token for `/layout-parsing` and `/extract` auth (Open-WebUI) |
| `PADDLEOCR_MODELS` | `/app/models` | Path to pre-downloaded PaddleOCR model dirs |
| `MAX_REQUEST_SIZE` | `104857600` | Max request body size in bytes (100 MB default) |

### Build Arguments

| Argument | Default | Description |
|---|---|---|
| `PADDLE_GPU` | `cpu` | PaddlePaddle variant: `cpu` or `cuda` |

## API Endpoints

### `GET /health`

Health check. Returns:
```json
{"status": "ok", "ocr_lang": "deu", "paddleocr_lang": "german", "gpu": false}
```

### `POST /layout-parsing` (Open-WebUI)

The endpoint Open-WebUI calls when PaddleOCR-VL is selected as Content Extraction Engine.

**Request:**
```http
POST /layout-parsing
Authorization: Bearer <PADDLEOCR_VL_TOKEN>
Content-Type: application/json

{
  "file": "<base64-encoded file bytes>",
  "fileType": 0,              // 0=PDF, 1=image
  "useDocOrientationClassify": false,
  "useDocUnwarping": false,
  "useChartRecognition": false
}
```

**Response:**
```json
{
  "result": {
    "layoutParsingResults": [
      { "markdown": { "text": "Page 1 extracted text..." } },
      { "markdown": { "text": "Page 2 extracted text..." } }
    ]
  }
}
```

### `POST /extract` (Direct API)

Upload a PDF or image, extract text via PaddleOCR.

**Request:** `multipart/form-data` with field `file`
**Response:**
```json
{
  "filename": "document.pdf",
  "pages": [
    {
      "page": 1,
      "text": "Full page text...",
      "blocks": [
        {"text": "Line text", "bbox": [x0, y0, x1, y1], "confidence": 0.98}
      ]
    }
  ],
  "full_text": "All pages concatenated"
}
```

## Important Notes

### OCR Language

The `ocrmypdf-paddleocr` plugin supports PaddleOCR language codes:

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
| Turkish | `tur` |
| Chinese (Simplified) | `ch_sim` |
| Chinese (Traditional) | `ch_tra` |
| Japanese | `jpn` |
| Korean | `kor` |

### Pre-downloaded Models

The Docker image ships with **Latin language models** pre-downloaded. Non-Latin languages download on first use.

### Sidecar Modes

| Mode | Behavior |
|---|---|
| `best_effort` (default) | Attempt Docling API. If it fails, continue with OCR. |
| `off` | Skip sidecar generation entirely. |
| `required` | If sidecar generation fails, move to ERROR. |
| `native` | Use local PaddleOCR instead of the Docling API. Same output format, no external dependency. |

## Volume Mounts

| Path | Purpose | Required |
|---|---|---|
| `/work/inbox` | Input directory for new PDFs | Yes |
| `/work/processing` | Temporary staging area | Yes |
| `/work/done` | Archive of processed files | Yes |
| `/work/error` | Failed files for inspection | Yes |
| `/work/docling` | Docling sidecar output | Optional |
| `/paperless-consume` | Paperless-ngx consume directory | Yes |

## Error Handling

- **Unstable files**: Skipped if size changes during stability check.
- **Docling failures**: Non-fatal by default (mode=best_effort).
- **OCR failures**: Retried up to `MAX_RETRIES` times. On final failure → ERROR.
- **Paperless push failures**: File moved to ERROR.

## License

[Add your license here]
