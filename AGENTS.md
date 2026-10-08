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

`INBOX`, `PROCESSING`, `DONE`, `ERROR`, `DOCLING_DIR`, `PAPERLESS_CONSUME`, `OCR_LANG` (default `deu`), `OCR_USE_GPU`, `POLL_INTERVAL`, `DOCLING_BASE_URL`, `DOCLING_MODE`, `DOCLING_TIMEOUT`, `MAX_RETRIES`, `RETRY_DELAY`, `PADDLEOCR_VL_TOKEN`, `PADDLEOCR_MODELS` (default `/app/models`), `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT` (image default `0`, see Version-Specific Hacks), `PADDLE_PDX_CACHE_HOME` (image default `/tmp/.paddlex`), `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` (default `1`, set by `paddlex_helpers.py`)

## Build Args

`PADDLE_GPU`: `cpu` (default), `cuda`

## Code Quality (Required After Each Changeset)

After every set of code changes, run and fix all failures before considering the work done:

```bash
ruff format .
ruff check .
mypy *.py
```

## Critical Gotchas

1. **Open-WebUI sends `Bearer <token>`** — parse auth header accordingly in `server.py`
2. **tesseract-ocr is still required at import time** by OCRmyPDF even when using PaddleOCR backend
3. **ROCm (AMD GPU) not supported** — PaddlePaddle's ROCm wheels are only available via their Docker images, not pip. The wheel index is a JavaScript SPA that pip can't parse.
4. **Forgejo `release` events silently drop jobs with `if:`** — on `release`-triggered runs, Forgejo (observed on 16.0.x) does not create job rows for jobs whose `if:` resolves false at run-creation time, or that reference `needs.<job>.outputs` of a job that was itself skipped — no job, no log, no error in the UI (errors, if any, are server-side only; cf. Forgejo issue #14684). The dropped jobs are invisible, and any `uses:` (reusable workflow) caller whose `needs` then all resolve gets marked *success in <1s without its inner jobs ever running* — a publish that "succeeds" while pushing nothing. **Rule: in this repo, never put `github.event_name`/`needs.*` logic in job-level `if:` — gate in-step instead** (see `filter` / `e2e-ocr` in `docker.yaml`). Re-check after a Forgejo server upgrade whether the original `if:` form works again.

## Version-Specific Hacks (re-verify when bumping paddle/paddlex)

Workarounds for bugs in the pinned upstream versions. **When bumping
`paddlepaddle` / `paddlex` in `requirements.txt` (or the bundled model
pins), walk this list and re-run the `e2e-ocr` CI gate** — it runs the
umlaut fixture through the full pipeline and prints the state of every
patch/flag. Most hacks are monkey-patches with fail-soft shape gates, so
a bad upgrade won't break import — but a patch can silently end up
`skipped`, which shows up in the e2e diagnostics line.

1. **`PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0`** — image `ENV` (Dockerfile).
   PaddleX 3.7.2 picks `run_mode="mkldnn"` on CPU by default, which
   crashes Paddle 3.3.0's PIR oneDNN converter on the first predict:
   `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
   [pir::ArrayAttribute<pir::DoubleAttribute>]` (`onednn_instruction.cc`).
   Note: `FLAGS_use_mkldnn=0` does **not** fix this — PaddleX's runner
   enables mkldnn explicitly via the config API, bypassing the flag.
   **Retire when** the pinned Paddle ships the fixed converter (or
   PaddleX blocklists the PP-OCRv6 models for mkldnn): remove the env
   var and confirm via e2e-ocr. Cost of keeping it: oneDNN CPU speedup.
2. **Word-segmentation patch** — `_patch_paddlex_word_segmentation()` in
   `paddlex_helpers.py`. PaddleX 3.x (3.2.0–3.7.2) classifies word
   characters with an ASCII-only regex, so umlauts/ß each become their
   own one-character "word" (spurious spaces in the PDF text layer).
   State (`applied`/`skipped`/`unpatched`) is printed on every e2e run.
   **Retire when** upstream PaddleX lands the Unicode fix (PaddleX#5188).
3. **Offline model-resolver patch** — `_patch_paddlex_official_models()`
   in `paddlex_helpers.py`. Forces PaddleX's `official_models` lookups to
   the bundled local model dirs (air-gapped / geo-blocked hosting;
   PaddleX#4578, PaddleOCR#16620, PaddleOCR#16639). Shape-gated against
   PaddleX's registry API — if PaddleX changes it, the patch latches
   `skipped` and model init will fail on download attempts instead.
4. **`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1`** — `os.environ.setdefault`
   at `paddlex_helpers.py` module import (must precede any PaddleX
   import). Skips PaddleX's hosting-platform health check in
   air-gapped environments. Re-verify the flag name still exists in
   `paddlex/utils/flags.py` on upgrade.
