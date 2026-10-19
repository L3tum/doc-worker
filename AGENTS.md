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

`INBOX`, `PROCESSING`, `DONE`, `ERROR`, `DOCLING_DIR`, `PAPERLESS_CONSUME`, `OCR_LANG` (default `deu`), `OCR_USE_GPU`, `POLL_INTERVAL`, `DOCLING_BASE_URL`, `DOCLING_MODE`, `DOCLING_TIMEOUT`, `MAX_RETRIES`, `RETRY_DELAY`, `PADDLEOCR_VL_TOKEN`, `PADDLEOCR_MODELS` (default `/app/models`), `MAX_OCR_PAGES` (default `500`, server page cap), `RENDER_DPI` (default `300`, server render DPI), `MAX_RENDERED_PIXELS` (default `100000000`, per-page rendered-pixel cap at RENDER_DPI), `MAX_CONCURRENT_OCR_JOBS` (default `2`), `MAX_CONCURRENT_PDF_PARSSES` (default `2`), `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT` (image default `0`, see Version-Specific Hacks), `PADDLE_PDX_CACHE_HOME` (image default `/tmp/.paddlex`), `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` (default `1`, set by `paddlex_helpers.py`)

## Deployment Notes

- The API is unauthenticated unless `PADDLEOCR_VL_TOKEN` is set (`docker-compose.yml` ships it commented out); there is no rate limiting — set the token for any exposed deployment.
- The OCR endpoints (`/layout-parsing`, `/extract`) admit at most `MAX_CONCURRENT_OCR_JOBS` in-flight OCR jobs (default `2`, env-overridable; temp-file staging + pipeline) and return 503 when saturated — the `to_thread` offload removed the event loop's implicit one-job-at-a-time serialization, and queued jobs would otherwise each keep decoded bytes + a ~100 MB temp file alive behind the process-wide predict lock (the API is unauthenticated by default, so this is the memory/temp-disk amplification bound). The pre-slot PDF cap check (`_enforce_pdf_caps`) is bounded separately by `MAX_CONCURRENT_PDF_PARSSES` (default `2`) via the event-loop-side `_pdf_parse_slot` (503 when saturated): it runs BEFORE the job slot (a rejected job must not take a slot), so the job-slot knob cannot bound its fan-out — without it, up to the thread-pool size of concurrent parses, each holding up to `MAX_REQUEST_SIZE` of decoded bytes plus pypdf's working set, could be in flight at once. The slot is acquired in the event loop (never a blocked worker thread) and released before the job slot is taken, so the two semaphores are never nested (no deadlock).
- `MAX_REQUEST_SIZE` bounds request bytes; POSTs without `Content-Length` (chunked) are rejected 413 — the body is fully materialized, so a missing declared length would be unbounded. `MAX_OCR_PAGES` (default 500) caps page count with a 413 before any render/model work; the per-page rendered-pixel cap (`MAX_RENDERED_PIXELS`, default 100 Mpx at `RENDER_DPI`) 413s before staging a tiny PDF with a huge page — page count alone cannot see that. It measures each page's EFFECTIVE box (the larger of MediaBox and CropBox per axis, with page-tree inheritance and corner normalization — measured with pypdf, because pypdfium2's page-level box APIs see neither; PDFium crbug 1786) — the same box poppler renders, which is the point: a spec-legal ~290-byte PDF can otherwise stage ~521 GB.
- The rendered-pixel cap's accuracy assumes poppler ignores `/UserUnit` (verified against poppler upstream master, not the exact base-image package) AND pypdf's page-tree-inheritance + raw-corner semantics (verified against pypdf 6.21.1) — re-verify against the base image's poppler-utils and the `TestEnforcePdfCaps` regression tests when the image, that pin, or the `pypdf>=6.16.1,<7` pin changes (the pin's hard upper bound is the re-verification tripwire).
- The pixel cap's per-page fail-soft skip for the encrypted inline-`/Kids`-leaf shape (a `/Page` dict inlined into `/Kids`: pypdf's forced `_flatten(list_only=True)` walk does not list such a leaf, so that page's measurement is skipped and the page reaches the renderer) is bounded only because poppler rejects every non-indirect `/Kids` entry — verified against the base image's exact poppler 22.09.0 (and 22.12.0) source: `Catalog::cachePageTree` aborts the page fetch, `PDFDoc::getPageMediaWidth` returns 0.0 for the resulting null page, and `displayPageSlice` skips it silently, so pdftoppm writes a 0/1-px stub per page — a poppler upgrade that accepts inline-dict kids would unbound this shape; re-verify against the base image's poppler-utils when bumping the image.
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
5. **ruff silently skips all `EXE` rules on WSL** — ruff ≥ 0.16's flake8-executable rules detect WSL via the "microsoft" kernel string (`/proc/sys/kernel/osrelease`) and disable themselves there, but `EXE*` IS in ruff's default rule set, so plain `ruff check .` enforces them on real Linux CI. An executable `.py` without a shebang (git mode 100755, e.g. a fixture generator) passes local lint on WSL and fails CI with `EXE002`. **Rule: every executable `.py` in this repo keeps a `#!/usr/bin/env python3` first line** (house pattern: the fixture generators).

## Version-Specific Hacks (re-verify when bumping the pinned upstreams)

Workarounds for bugs in the pinned upstream versions (paddle/paddlex for
#1–#4, ocrmypdf/pypdfium2 for #5). **When bumping the corresponding pins
in `requirements.txt` (or the bundled model pins), walk this list and
re-run the `e2e-ocr` CI gate** — it runs the umlaut fixture and the
variants fixtures through the full pipeline and prints the state of every
patch/flag. Most hacks are monkey-patches with fail-soft shape gates, so
a bad upgrade won't break import — but a patch can silently end up
`skipped`, which shows up in the e2e diagnostics line. The variants
e2e has a fail-soft signal of the same kind: `[e2e] check B|C: WARNING:
model geometry disagrees with the fixture layout` means PP-StructureV3
localized a column marker on the wrong side (or with an unusable bbox),
so the column-first **ordering** assertion is disarmed for that run and
the check degrades to marker presence (the sort itself stays
deterministically unit-tested in `tests/test_paddlex_helpers.py`). A
one-off warning is model noise; if it **persists across runs**, the
ordering e2e is not covering the fix — revisit the fixture geometry or
the model before relying on it.

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
   (DOCLING_OUT/<original stem>/). Crash recovery recognizes a leftover as
   a worker-owned intermediate ONLY when the tail after
   `__acroform_flattened__` is exactly 8 hex chars (matching
   `uuid.uuid4().hex[:8]` in `flatten_acroform_if_needed`); marker-named
   files with any other tail are preserved (moved to ERROR/ via
   `unique_destination`) instead of unlinked. **Retire when** ocrmypdf's rasterizer
   calls `init_forms()` (absent across the pinned 17.4.1–17.13.x range):
   delete the helper + state constant + the `process_file` call, remove
   the `pypdfium2` pin from `requirements.txt`, and confirm via the
   variants e2e (check A's red-guard pixel-diff + value assertions).
