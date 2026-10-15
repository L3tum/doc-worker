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

`INBOX`, `PROCESSING`, `DONE`, `ERROR`, `DOCLING_DIR`, `PAPERLESS_CONSUME`, `OCR_LANG` (default `deu`), `OCR_USE_GPU`, `POLL_INTERVAL`, `DOCLING_BASE_URL`, `DOCLING_MODE`, `DOCLING_TIMEOUT`, `MAX_RETRIES`, `RETRY_DELAY`, `PADDLEOCR_VL_TOKEN`, `PADDLEOCR_MODELS` (default `/app/models`), `MAX_OCR_PAGES` (default `500`, server page cap), `RENDER_DPI` (default `300`, server render DPI), `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT` (image default `0`, see Version-Specific Hacks), `PADDLE_PDX_CACHE_HOME` (image default `/tmp/.paddlex`), `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` (default `1`, set by `paddlex_helpers.py`)

## Deployment Notes

- The API is unauthenticated unless `PADDLEOCR_VL_TOKEN` is set (`docker-compose.yml` ships it commented out); there is no rate limiting — set the token for any exposed deployment.
- `MAX_REQUEST_SIZE` bounds request bytes; POSTs without `Content-Length` (chunked) are rejected 413 — the body is fully materialized, so a missing declared length would be unbounded. `MAX_OCR_PAGES` (default 500) caps page count with a 413 before any render/model work.
- PP-StructureV3 pre-renders PDFs at 300 dpi (`paddlex_helpers.py` `RENDER_DPI`, env-overridable to trade OCR quality for staging), ~4× the pixel work of the old 150 dpi: ~4× peak temp-disk staging (all pages held as PNG) and ~4× per-page decode memory.
- `PAPDFTOPPM_TIMEOUT`'s default now scales with DPI (480 s at 300 dpi, derived from the 150-dpi-era 120 s); the env override is unchanged.

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

## Version-Specific Hacks (re-verify when bumping the pinned upstreams)

Workarounds for bugs in the pinned upstream versions (paddle/paddlex for
#1–#4, ocrmypdf/pypdfium2 for #5). **When bumping the corresponding pins
in `requirements.txt` (or the bundled model pins), walk this list and
re-run the `e2e-ocr` CI gate** — it runs the umlaut fixture and the
variants fixtures through the full pipeline and prints the state of every
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
5. **AcroForm pre-flatten** — `worker.py::flatten_acroform_if_needed()`
   (state constant `_ACROFORM_FLATTEN_STATE`, printed on every e2e run).
   ocrmypdf 17's built-in pypdfium2 rasterizer opens documents without
   `init_forms()`, and PDFium's form layer only renders after
   `init_forms()` — so typed-in AcroForm values are absent from the page
   image. Under `force_ocr=True` ocrmypdf merely *warns* about AcroForms,
   making the loss silent. The worker now pre-flattens a temporary copy
   (`<stem>__acroform_flattened__<8-hex>.pdf`, unlinked after OCR; the
   original still lands in DONE/) before `run_ocrmypdf`. Fail-soft: any problem falls back to
   the original. The sidecar (Docling API or native PaddleX) now runs on
   the flattened copy — flatten happens before sidecar generation — so the
   sidecar markdown/JSON and the OCR text layer agree for /V-only widgets;
   the sidecar output remains keyed by the original filename
   (DOCLING_OUT/<original stem>/). **Retire when** ocrmypdf's rasterizer
   calls `init_forms()` (absent across the pinned 17.4.1–17.13.x range):
   delete the helper + state constant + the `process_file` call, remove
   the `pypdfium2` pin from `requirements.txt`, and confirm via the
   variants e2e (check A's red-guard pixel-diff + value assertions).

## Follow-ups (themed review 2026-01-15 — non-blocking)

Addressed in this round: `asyncio.to_thread` around both `run_paddlex_structure_v3`
call sites (loop stall) — and, after the follow-up review, around the
`run_paddleocr` fallback at both sites too; `_PADDLEX_PREDICT_LOCK`
(process-wide) serializes model fetch + predict for both pipelines — the
to_thread offload removed the event loop's implicit one-job-at-a-time
serialization and Paddle's predictor is not thread-safe (one predictor per
thread); the two e2e `docker run`s merged into one (model-init
halved); mocked-HTTP test for the Docling-API `sidecar_name` contract
(`tests/unit/test_docling_api.py`); the intermediate renamed to
`<stem>__acroform_flattened__<8-hex>.pdf` (see hack #5 — no more user-file
collision); stray-directory guard in the flatten except-handler — and the
same guard in `recover_leftover_files` (a marker-named *directory* used to
raise `IsADirectoryError` and crash-loop the worker at startup), plus
name-sorted recovery; `_is_text_content` rejects NUL bytes (7-bit-clean
binaries such as a near-silent WAV no longer pass as "text"); magic-byte
anchors skip leading whitespace/control bytes before the signature (a
header-padded `%PDF` no longer 415s or leaks through as raw text);
degenerate `block_bbox` normalized to a four-float list at extraction, with
dict rejection and `math.isfinite` filtering (NaN/Inf no longer 500 the
response via `json.dumps(allow_nan=False)`); `handle_docling`-raise
leak fixed in `process_file`; `_ACROFORM_FLATTEN_STATE` docstring corrected and
the `unpatched→skipped` transition now tested; `.gitattributes` marks the PDF
fixtures as binary (`-text`) so autocrlf checkouts can't corrupt them. And
the previously parked items: `MAX_OCR_PAGES` page-count cap (default 500,
env-overridable, 413 before any render/model work on both OCR endpoints);
`RENDER_DPI` env-overridable (documented quality/staging tradeoff);
chunked uploads (POST without Content-Length) rejected 413 in the size
middleware; `process_file`'s flattened intermediate unlinked by one
try/finally instead of three per-branch unlinks; `page_width_px` removed
from the page dict (dead production data — the width is still consumed
internally by the reading-order sort); the 503 health guard, content
routing, and OCR core (page cap + kind-suffixed temp file + `to_thread`
pipeline call + idle-clock stamping) deduplicated into
`_require_model_ready` / `_route_upload` / `_run_ocr_pipeline` shared by
both endpoints (also fixing the asymmetric endpoint log wording); the
`_point_worker_at` test helper deduplicated to module level; and tests for
the previously untested paths (`push_to_paperless` success/copy-failure/
replace-failure, `unique_destination` same-second collision — which also
got a numeric-suffix hardening against overwriting, `bzip2`/`lzip`
signatures parametrized, `RENDER_DPI` → `pdftoppm -r`, real-PDF page
counting, `_ensure_paddlex_offline_compat`'s ImportError branch, chunked
413, page-cap 413 on both endpoints); `blocks_to_markdown` marked PUBLIC
API (the module has no `__all__`) and `handle_docling`'s `sidecar_name`
docstring carries the user-supplied-filename/trusted-volume security note.

Still open:

- Dependency: the `pypdfium2>=5,<6` upper bound can hard-conflict with a future ocrmypdf 17.x requiring pypdfium2>=6 while `ocrmypdf>=17.4.1,<18` still permits it — drop the cap or mirror ocrmypdf's floor.
- Ops: monitor temp disk (at 300 dpi a 500-page PDF stages 1–1.5 GB of PNGs; `RENDER_DPI` / `MAX_OCR_PAGES` are the levers).
- Hygiene: consolidate `name→kind` routing across the signature tables (`_FILE_SIGNATURES`, `_IMAGE_SIGNATURE_NAMES`, inline special-cases — the endpoint-level dedupe is done, the table consolidation is not); dedupe ~60 lines of shared boilerplate in the tests/fixtures generators and the duplicated check helpers in `variants_e2e.py`; the `Bearer` scheme match is case-sensitive (RFC-legal `bearer` rejected — unchanged from prior behavior, nit); optional fail-soft symmetry — fall back to the original if the flattened copy fails OCR.
- Nits: log-injection surface via crafted INBOX filenames (log names with `repr()`); exception text echoed in the 500 detail; env-var numeric coercion is unvalidated (`int(os.getenv(...))`); TIFF bytes under a `.png` suffix untested end-to-end; two >88-char literals.
