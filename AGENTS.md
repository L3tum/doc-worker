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

## Follow-ups (themed review 2026-01-15 — non-blocking)

Addressed in this round: `asyncio.to_thread` around both `run_paddlex_structure_v3`
call sites (loop stall) — and, after the follow-up review, around the
`run_paddleocr` fallback at both sites too; `_PADDLEX_PREDICT_LOCK`
(process-wide) serializes model fetch + predict for both pipelines — the
to_thread offload removed the event loop's implicit one-job-at-a-time
serialization and Paddle's predictor is not thread-safe (one predictor per
thread); the two e2e `docker run`s merged into one (one fewer
container start; models are pre-baked in the image — each e2e script
still does its own model init); mocked-HTTP test for the Docling-API `sidecar_name` contract
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
- Ops: monitor temp disk (at 300 dpi a 500-page PDF stages 1–1.5 GB of PNGs; `RENDER_DPI` / `MAX_OCR_PAGES` / the per-page 100 Mpx `MAX_RENDERED_PIXELS` cap are the levers). Memory-posture note: the pre-slot `_enforce_pdf_caps` parse runs (in a worker thread, on in-memory bytes) *before* the `MAX_CONCURRENT_OCR_JOBS` slot, so concurrent in-flight parses are bounded by the `MAX_CONCURRENT_PDF_PARSSES` parse slot (each parse holding up to `MAX_REQUEST_SIZE` of in-memory bytes), not the job-slot knob — accepted trade (a rejected job must not hold a slot) and transient (pypdfium2 keeps the caller-owned bytes; parse is lazy). Worst-case pre-slot cost (measured): a degenerate-but-spec-legal 99,999-leaf `/Kids` tree (just under pypdf's 100,000-entry guard, ~9 MB) takes ≈3.5–4 s on the pypdf pass (≈0.2 s ctor + ≈3.3 s `_flatten` — pypdf has no early-exit count) — rejected by the page cap under the default `MAX_OCR_PAGES` (413 on the true tree count), off-thread, parse-slot-bounded, no staging.
- Hygiene: consolidate `name→kind` routing across the signature tables (`_FILE_SIGNATURES`, `_IMAGE_SIGNATURE_NAMES`, inline special-cases — the endpoint-level dedupe is done, the table consolidation is not); dedupe ~60 lines of shared boilerplate in the tests/fixtures generators and the duplicated check helpers in `variants_e2e.py`; optional fail-soft symmetry — fall back to the original if the flattened copy fails OCR.
- Nits: log-injection surface via crafted INBOX filenames (log names with `repr()`); env-var numeric coercion: `RENDER_DPI` / `PAPDFTOPPM_TIMEOUT` / `MAX_CONCURRENT_OCR_JOBS` now clamped to >= 1, while `MAX_REQUEST_SIZE` / `MAX_OCR_PAGES` / `MAX_RENDERED_PIXELS` still fail loudly at import on non-numeric values; TIFF bytes under a `.png` suffix untested end-to-end.

Post-review follow-up (2026-10-15): OCR job-slot cap (2 in flight, 503 when saturated; per-event-loop semaphore); raw `rec_boxes`/`rec_polys` geometry routed through finite-float normalization; non-numeric `Content-Length` rejected 400 in the size middleware; sidecar stem normalization (leading-dot filenames can no longer escape DOCLING_OUT/); strict 8-hex flatten-marker recognition + `unique_destination` in crash recovery; plus the medium/low items — page count from in-memory bytes before staging, `RENDER_DPI` clamp + `_pdftoppm_timeout_for_dpi`, `COLUMN_THRESHOLD_PX` module constant, `blocks_to_markdown` public-contract note, section-banner reorder, `_image_width_px` failure warning, and the added tests (real 2-page page-cap fixture incl. the `count == MAX_OCR_PAGES` boundary, endpoint-level concurrency test, degenerate `rec_boxes`, `unique_destination` three-way collision, strict recovery-marker cases, `RENDER_DPI` env contract, non-vacuous predict-overlap test, `_body_size_policy` 400 case).

Post-review fixes (2026-10-15, second pass):

- `tests/fixtures/twopage.pdf` committed (it was untracked, breaking CI test collection on fresh checkout) + new deterministic generator `tests/fixtures/generate_twopage.py` (house style: byte-identical output, self-verifying via pypdfium2).
- `MAX_RENDERED_PIXELS` (default `100000000` = 100 Mpx): per-page rendered-pixel cap at `RENDER_DPI` — each page's LARGER of MediaBox and CropBox is checked via pypdfium2 on the in-memory bytes before staging/render (offloaded to a worker thread alongside the page count; the renderer rasterizes the MediaBox even when a CropBox is set, while pypdfium2's `get_width`/`get_height` report the CropBox — hence the max-of-both measurement, regression-tested against the normal-CropBox/huge-MediaBox bypass); a page whose rendered w_px×h_px exceeds the cap 413s (closes the tiny-PDF-huge-MediaBox DoS — one page `[0 0 100000 100000]` ≈ 521 GB RGB at 300 dpi — that passed both `MAX_REQUEST_SIZE` and `MAX_OCR_PAGES`).
- `MAX_CONCURRENT_OCR_JOBS` now env-overridable (default `2`) — its sibling knobs all were; the 2-job cap / 503-on-saturation behavior is unchanged; the per-event-loop semaphore dict now prunes closed-loop entries on lookup (invariant documented at the code: the cap is per event loop; production runs one uvicorn loop, so the bound holds).
- 500 details no longer echo exception text to unauthenticated callers (`/layout-parsing` → "Layout parsing failed", `/extract` → "Extraction failed"); the full traceback goes server-side via `logger.exception`; the 503 model-init detail is likewise generic ("PaddleX model initialization failed; see server logs") and the init failure is now logged with traceback at capture time in `paddlex_helpers`.
- Magic-byte signature scan is offset-anchored (leading-noise offset computed once with a regex, predicates anchor via `startswith(sig, offset)`) — no more full-body `lstrip` copy of up-to-100-MB uploads; behavior identical for all inputs.
- `paddlex_helpers._finite_geom` now matches `_normalize_bbox`'s dict contract (top-level dict → zero box; nested dict values walked) — a dict-shaped geometry containing NaN/Inf can no longer 500 the response via `SafeJSONResponse` (`allow_nan=False`).
- Flatten-marker crash-recovery recognition is lowercase-hex-only (the exact `uuid.uuid4().hex` shape the worker creates); an uppercase 8-hex tail may be a user file and is now preserved (moved to ERROR/), not unlinked.
- Docling multipart filename strips CR/LF (a crafted INBOX filename can no longer inject headers into the internal Docling request's `Content-Disposition`).
- `PAPDFTOPPM_TIMEOUT` env override clamped to >= 1 s (a 0/negative value no longer makes every render raise `TimeoutExpired` with a misleading error); the `RENDER_DPI` clamp comment reworded to the real hazard (invalid `pdftoppm -r` argument; the derived timeout was already clamped separately).
- Test hardening: flatten-marker length-rule parametrization (7/9-hex → not recognized), failed-job-releases-slot test, deterministic concurrency test (threading.Event barrier on 2 provably-held slots, no sleep margins), 413 tests assert no temp-file staging + no slot taken, raw-ASGI probe pinning non-numeric `Content-Length` → 400 through the real middleware, `_finite_geom` nested-list/dict tests, fresh-interpreter `RENDER_DPI=0` → clamped-1-timeout case, `PAPDFTOPPM_TIMEOUT` clamp cases, `strftime` same-second collision tests pinned to a fixed timestamp, Docling CRLF-strip tests.

Post-review fixes (2026-10-16, third pass):

- `/health` model-init failure now returns the SAME generic 503 detail as the OCR endpoints' `_require_model_ready` guard (shared `MODEL_INIT_FAILURE_DETAIL` constant in `server.py`) — the raw init-exception text (model/temp paths, enriched-download diagnostics) no longer leaks to unauthenticated callers; the full failure is already logged with traceback at capture time in `paddlex_helpers`.
- `MAX_CONCURRENT_OCR_JOBS` clamped to >= 1 at import (`max(1, int(...))`, matching the sibling-knob pattern): `0` no longer 503s every OCR request "busy" forever with no import-time signal, and a negative value no longer raises `ValueError` inside `asyncio.Semaphore` at the first request (an opaque generic 500).
- Page cap + per-page rendered-pixel cap consolidated into ONE pypdfium2 pass (`server._enforce_pdf_caps`, replacing `_count_pdf_pages` + `_enforce_page_limit` + the second pass's per-page pixel-cap helper (it existed only in the uncommitted second-pass state — its intermediate name never reached git history) and the two `to_thread` calls): one parse of the up-to-100-MB request bytes per request instead of two. Semantics unchanged: the count check still runs first (over-cap docs 413 "Too many pages" without per-page work), the same 413 details, the same fail-soft pass-through on unopenable bytes.
- Test updates: `/health` 503 test pins the generic detail + no-leak assertions; fresh-interpreter `MAX_CONCURRENT_OCR_JOBS` clamp contract (0/-3 → 1, 5 → 5, same subprocess pattern as the `RENDER_DPI` tests); `TestEnforcePdfCaps` covers the page cap at unit level (413 + `count == cap` boundary + fail-soft + count-first ordering + single-parse assertion via a counting `pypdfium2.PdfDocument` wrapper); the two mock-based page-cap e2e tests were deleted as strict duplicates of their real-two-page-PDF siblings.

Post-review fixes (fourth pass):

- H1 (DoS bypass — the one genuine defect of this changeset): `_enforce_pdf_caps` measured only pypdfium2's `get_width()`/`get_height()`, which report the **CropBox** when one is set, while `pdftoppm` (no `-cropbox` flag) rasterizes the **MediaBox** — a normal-CropBox/huge-MediaBox page (MediaBox `[0 0 100000 100000]` + CropBox `[0 0 612 792]`, ~250 bytes) passed every cap and staged a ~521 GB render. The per-page cap now measures the LARGER of MediaBox and CropBox per axis (`get_mediabox()`); doc wording corrected in `server.py`/`AGENTS.md`/`README`; regression tests: unit (`test_huge_mediabox_with_normal_cropbox_413` — cap derived BETWEEN the crop- and media-rendered sizes so a CropBox-only measurement provably passes while the max-of-both 413s — plus `test_cropbox_smaller_than_mediabox_passes`) and e2e (`_HUGE_CROP_PDF` → 413, no staging, no slot); `_minimal_pdf` gained a `cropbox` param in both test files. The fail-soft contract is unchanged (a page object without `get_mediabox` still degrades to the pipeline's own error, not a 413/500).
- H2: a >4300-digit `Content-Length` (the `isascii() and isdigit()` guard admits ANY length of digits) made `int()` raise `ValueError` under CPython 3.11+ (repo pins 3.12) → uncaught in the size middleware → 500, contradicting `_body_size_policy`'s "the app must not 500" contract. Now `try/except ValueError` → 400 "Invalid Content-Length header"; pure-function + raw-ASGI regression tests pin a 5000-digit value (the raw-ASGI scope boilerplate factored into a shared `_raw_asgi_post` helper). Real-world exposure stays low (both uvicorn HTTP backends reject >20-digit Content-Lengths first) — this closes the raw-ASGI/transport gap.
- RENDER_DPI-coupled tests de-hard-coded: `test_boundary_exact` and BOTH `*_normal_page_tight_cap` e2e tests (layout-parsing + extract) now derive their caps from the rendered size at the CURRENT `server.RENDER_DPI` instead of hard-coded 300-dpi numbers (they failed spuriously under a documented `RENDER_DPI` override).
- `logger.exception` traceback logging in `paddlex_helpers._retry_model_init` (the server-side compensation for the generic 503) is now pinned by caplog assertions on the existing failure-path tests — permanent branch and exhausted-retries branch, asserting level + `exc_info`.
- Lint: `-> None` on `test_failed_job_releases_slot` / `test_three_concurrent_jobs_two_200_one_503`; stale "anchored at offset 0" docstring corrected to "first non-noise byte"; redundant double `worker` import in `variants_e2e._check_acroform` collapsed.
- README: operator sizing note — tune `RENDER_DPI`/`MAX_OCR_PAGES`/`MAX_RENDERED_PIXELS` together to fit the temp volume (cap envelope 500 pages × 100 Mpx ≈ 150 GB theoretical worst case).
- Carried to the follow-up list (non-blocking): optional dedicated parse semaphore (e.g. 4) to make the pre-slot `_enforce_pdf_caps` executor bound explicit (the ~32 × 100 MB fan-out is a documented accepted trade); `generate_native_sidecar` CRLF strip (log-injection nit, same family as the tracked Docling-sidecar item); test-theme minors (4th-job `Semaphore.release` pin, `MAX_CONCURRENT_OCR_JOBS`→semaphore wiring test with a fresh loop, `_finite_geom` on numpy input, fail-loud side of the five import clamps, closed-loop semaphore-prune branch, ENOSPC-safety reordering pin); awareness-only: `locked()`+`acquire()` atomicity leans on the CPython fast path (availability-only worst case), and the cap fail-softly skips pypdfium2-unopenable-but-pdftoppm-renderable documents.

Post-review fixes (2026-10-16, fifth pass):

- H1 (DoS bypass): the size middleware accepted a forged small `Content-Length` when a `Transfer-Encoding` header was also present — per RFC 7230 §3.3.3 the transport honors `Transfer-Encoding` and IGNORES `Content-Length`, so the body is fully materialized (memory + temp file) no matter what the length claims. `_body_size_policy` now rejects ANY POST with a `Transfer-Encoding` header (413, checked before Content-Length); unit tests + a raw-ASGI regression test (through the real middleware, the pure function's sibling) pin both the rejection and the detail.
- `PAPDFTOPPM_TIMEOUT` moved to an import-time constant, matching its sibling knobs (`RENDER_DPI` / `MAX_REQUEST_SIZE` / `MAX_OCR_PAGES` / `MAX_RENDERED_PIXELS` / `MAX_CONCURRENT_OCR_JOBS`): parsed + clamped to >= 1 s at import, `_pdf_to_images` uses the module constant (no per-call env lookup). The call-site env tests were replaced with the house two-sided pattern: fresh-interpreter env-contract tests (0/-50 → clamped 1 s, 999 passes through unchanged, unset → `DEFAULT_PDFTOPPM_TIMEOUT` DPI-scaled default, non-numeric → import fails loudly with `ValueError` — the carried-over "fail-loud side of the import clamps" item for this knob) + an in-process wiring test (the call site passes the constant to `subprocess.run`'s timeout; monkeypatched constant → captured timeout).
- Endpoint wiring closed (carried-over item): `test_slot_cap_driven_by_max_concurrent_ocr_jobs` pins the `MAX_CONCURRENT_OCR_JOBS` constant → per-event-loop semaphore link — with the constant patched to 3, a fresh `asyncio.run` loop admits exactly 3 concurrent barrier-held jobs and 503s the 4th (distinguishes ignoring the constant, a hardcoded 1, or a hardcoded 3+; barrier-deterministic, no sleep margins).
- Producer → consumer round-trip: `test_acroform_flattened_copy` now asserts `worker._is_flatten_intermediate(out.name, out.suffix) is True` on the REAL `_flatten_acroform_copy` output — the crash-recovery classifier (the consumer of the producer's naming) is pinned against the producer's actual name + suffix, not two hand-rolled literals that both satisfy the rule.
- `TestOcrCapClampContract` now parametrized with both halves of the contract: zero clamps to 1 AND an explicit positive value (5/1234/999) passes through UNCHANGED — the clamp is a floor, not an assignment, and pre-fix only the zero case was exercised (an accidental assignment to 1 would have gone undetected).
- `test_no_overwrite_in_error` modernized to the house style: `tmp_path` + `monkeypatch` (no raw module assignment / `mkdtemp`), `strftime` pinned to a fixed stamp, and the assertions are on the exact recovered filename (`conflict_20250101120000.pdf`) plus both files' bytes.
- New defensive-branch tests for `_enforce_pdf_caps`: a page whose `get_mediabox()` returns `None` still enforces the cap from `get_width`/`get_height` (the `if box is not None` guard must not silently skip the cap), and a page object WITHOUT `get_mediabox` at all degrades fail-soft (the AttributeError is caught, no 413/500).
- Executor-hop kind-gate: `_enforce_pdf_caps` runs on a worker thread only for `kind == "pdf"` — image/text return immediately inside it, so they no longer queue behind in-flight 100 MB PDF parses under executor saturation; the non-PDF unit test now pins the PRODUCTION-REACHABLE floor (`MAX_OCR_PAGES=1` / `MAX_RENDERED_PIXELS=1`, since the import clamp makes 0 unreachable).
- Docstring/comment hygiene: `_ocr_job_slot` docstring corrected (temp-file staging happens AFTER the slot is taken — a queued job holds no temp file, only the in-memory request bytes); dead `else 0` fallback removed from `_leading_noise_offset` (a `*`-quantified regex always matches — the comment now says why no fallback exists); `_HUGE_PDF` comment arithmetic fixed (100000 pt × 300 dpi → ~1.7e11 px → ≈521 GB of RGB, ~290-byte file — the old figures were wrong in both places); `_recording_slot` got its return annotation; the two >88-char literals in the PDF fixture builders wrapped (closes the "two >88-char literals" nit).
- `tests/fixtures/generate_twopage.py` aligned to the house style of the other fixture generators (build → self-verify → write → read-back confirm; dropped the sha256 / diff-summary / exit-code ceremony; offset pins + pypdfium2 self-verification kept) and set to 755 like its siblings. Verified byte-identical output before and after the rewrite. (Stale-index note from the themed review: the rewritten generator was left unstaged while the fixture it guards was staged — now staged; the docstring no longer claims a CI diff-guard step that docker.yaml does not have.)

Post-review fixes (2026-10-17, sixth pass):

- H1 + H2 (DoS bypass — the two blocking findings, both independently verified against the pinned pypdfium2 5.14.0 and poppler source): the per-page rendered-pixel cap was measured with pypdfium2's page-level box APIs, which (a) do not walk `/Parent` — a spec-legal ~290-byte PDF with a huge `MediaBox` on the `/Pages` node and a normal page `CropBox` measures 612x792 to pypdfium2 (PDFium crbug 1786, documented in its own docstring) while pdftoppm (no `-cropbox`) inherits the parent's box and renders 416667x416667 px at 300 dpi (~521 GB) — and (b) return corners as written with raw (un-`abs()`-ed) deltas — a page-level `MediaBox` with swapped/inverted corners (spec-legal per PDF 32000-1) behind a CropBox was likewise invisible. `server._enforce_pdf_caps` is now a two-parser design: pypdfium2 pass for the page count (unchanged semantics, checked first), then a **pypdf** pass that measures each page as the LARGER of its EFFECTIVE MediaBox and CropBox per axis with `abs()` of the corner deltas — pypdf resolves the same page-tree inheritance poppler does and keeps the corners as written, so `abs()` gives the normalized size poppler renders. Per-page try/except keeps the failure semantics fail-soft and PER PAGE (one unresolvable page — missing box, malformed, cyclic `/Parent`, deep page tree — is skipped, poppler renders it at the bounded 612x792 spec default, while the remaining pages are still capped); a document pypdf cannot parse skips the pixel cap only (the page cap already applied). Regression tests: unit (cap-between construction so a no-inheritance / raw-delta / CropBox-only measurement provably passes while the effective-box measurement 413s — DPI-independent) + endpoint-level e2e for both shapes (413 before slot + staging), per-page-vs-document fail-soft, mid-iteration pypdf failure, and one-open-per-parser (replaces the single-parse pin the two-parser design supersedes). Measured cost of the added pypdf pass (off-thread, pre-slot): ≈50 ms at 500 pages / 0.05 MB, ≈190 ms at 2000 pages — negligible next to a render that takes minutes, and documents over the default 500-page cap never reach it.
- New runtime dependency: `pypdf>=6,<7` (pure Python, zero deps; ocrmypdf 17.4.1 does NOT pull it in — it uses pikepdf). The pin's hard upper bound is deliberate: pypdf's inheritance + box semantics are load-bearing for the cap, and its own cycle/depth guards (verified: cyclic `/Parent` → ValueError, >100-deep page tree → LimitReachedError, all catchable, no hangs) must be re-verified at a major bump — the `TestEnforcePdfCaps` regression tests are the re-verification vehicle. The test PDF builders (unit + e2e) now emit REAL xref tables (the old trailer-only fakes were pypdfium2-tolerant but pypdf rejects them, and the cap measures with pypdf); `_FAKE_PDF` is built through the same builder.
- H3 (stale index): `tests/fixtures/generate_twopage.py` (the house-style rewrite) is now staged — it was left as a worktree-only change while `twopage.pdf` was staged, so a commit of the index would have shipped the fixture without its generator. The docstring's "CI can run the generator and fail on any git diff" claim corrected to match docker.yaml (no such step).
- Small items from the same report: (1) `Bearer` auth-scheme now case-insensitive per RFC 6750/7235 (`bearer`/`BEARER` authenticate; the token itself is still compared case-sensitively in constant time) — pinned by a direct `_check_token` unit test; (2) the table parser's `int(colspan_value)` can no longer 500 a request on garbage (`<td colspan="abc">`) — falls back to 1; (3) unpinned `requests` → `requests>=2.32,<3` (supply-chain exposure in a lib the Docling CRLF fix relies on); (4) `_leading_noise_offset`'s `assert m is not None` (stripped under `-O`, the codebase's only assert) → explicit `if m is not None else 0` fallback (the same value the always-matching regex would report).
- Carried / parked (non-blocking, per the synthesis): the pre-slot cap parse fan-out (~32 x 100 MB under executor saturation — optional dedicated parse semaphore); the four non-critical pinned-but-fail-soft knobs (fail-loud on import); the `/health` getter split; the Paperless-fail e2e path; the `_minimal_pdf` duplication between test_server.py and test_vl_api_e2e.py (kept byte-identical by hand for now); and the test-theme minors (barrier-test sleep, dedup of fresh-interpreter env-clamp tests, `Final` for `MODEL_INIT_FAILURE_DETAIL`, worker.INBOX docstring drift, /extract 503 endpoint pin, e2e `&&` chaining, stale-loop prune, the two leftover `mkdtemp` tests).

Post-review fixes (2026-10-17, seventh pass):

- CRITICAL (DoS bypass — verified against the pinned pypdfium2 5.14.0 and pypdf 6.21.1 before and after): the page cap trusted pypdfium2's `len(doc)` → `CountPages()`, which returns the RAW `/Count` whenever `0 < /Count < 0xFFFFF` without cross-checking `/Kids` (and indexes pages the same way, so PDFium never sees the extra pages). A spec-legal few-MB PDF with `/Count=1` and N real leaf pages passed both caps: the lying-small vector is a count-gate + measurement-skip (every consumer of the /Count-gated count sees only the first /Count pages; the "pdftoppm renders all N" framing was a phantom — verified against poppler 22.09/22.12/24.02 source, `Catalog::getNumPages()` returns the raw `/Count` and pdftoppm clamps its render loop to it). The concrete residual is the page-SIZE dimension: a huge box on any of the first /Count pages renders hundreds of GB from a few-KB file — unbounded temp-disk + CPU on the unauthenticated API. `server._enforce_pdf_caps` now enforces `MAX_OCR_PAGES` on the PAGE TREE's count — pypdf's `len(reader.pages)` → `get_num_pages()` → `_flatten` (explicit stack; pypdf's own guards `page_tree_maximum_depth=100` / `page_tree_maximum_entries=100_000`; no recursion, no new parser surface) — checked first, before the per-page pixel loop; the pypdfium2 count is demoted to the FALLBACK applied only when pypdf cannot parse the document (outer handler; pixel cap still skipped). This also fixes the inverted-`/Count` Low finding (overstated `/Count` no longer 413s a valid-but-malformed document; the detail names the true count). ENCRYPTED documents: pypdf's own count is /Count-gated there too (`get_num_pages` returns the raw `/Count` when `is_encrypted`; the page sequence truncates to the same /Count) — the cap decrypts with the empty user password (`decrypt("")` is cheap: ≤50 MD5 rounds even for R≥3, no R=5 2^32 blowup) and forces pypdf's guarded tree walk (`_flatten(list_only=True)`, lightweight), counting + measuring `flattened_pages` — which also fixes the encrypted pixel cap that previously only saw the first /Count pages (page-tree inheritance included — verified). A failed decryption = user password required = unrenderable by pdftoppm anyway → fail-soft return (the pipeline's own error applies). House -O style kept: `flattened_pages is None` raises ValueError → outer fallback (no assert). Cost: the unencrypted tree flatten the pixel loop already triggered; the encrypted branch adds one guarded flatten only for encrypted docs.
- Regression tests (all verified to FAIL against the old gate — the old logic was re-implemented in scratch: understated lies passed, overstated lies 413'd spuriously): unit `TestEnforcePdfCaps` +7 (`test_lying_count_understated_413` — 3 pages / `/Count=1` / cap 1 → 413 "Too many pages: 3 (max 1)", self-verified lie via pypdfium2 reading 1; `test_lying_count_understated_cap_between_counts_413` — cap 2, the discriminator: lying 1 passes, true 3 413s; `test_lying_count_understated_at_cap_passes` — cap 3 boundary on the true count; `test_overstated_count_no_spurious_413` — 1 page / `/Count=10` / cap 5, previously 413, now passes; `test_overstated_count_true_count_413` — detail names the true 3, not the lying 10; `test_lying_count_encrypted_413` — encrypted 3 pages / `/Count=1` / cap 1 → 413, self-verified that BOTH /Count-gated count APIs read 1; `test_lying_count_encrypted_pixel_cap_sees_all_tree_pages` — encrypted 2 pages / `/Count=1`, page 2 inherits a huge MediaBox → 413 naming page 2, pinning that the pixel loop walks the full tree). e2e +2 (`test_page_limit_413_lying_count` on /layout-parsing, `test_extract_page_limit_413_lying_count` on /extract multipart — 413 before slot + staging, pipeline never runs). Builder extension: `count:` parameter on `_pdf_with_pages` / `_minimal_pdf` in BOTH test files (kept byte-identical per the carried dedup note) + `_encrypt_and_lie_count` (owner-only encrypt — empty user password so pdftoppm opens it — then first-occurrence byte replacement of `/Count N` on the unencrypted object dicts, self-verifying the needle). `test_pypdf_failure_skips_pixel_cap_only` (renamed to `test_pypdf_constructor_failure_413_unparseable` in the ninth pass — the constructor-level disagreement now fails CLOSED, see below)'s docstring updated to the fallback semantics.
- pypdf pin floor raised `pypdf>=6,<7` → `pypdf>=6.16.1,<7` (High item): the 2026 crafted-PDF DoS cluster (all web-verified): CVE-2026-54531 (infinite loop, outlines — fixed 6.13.0), CVE-2026-54651 (infinite loop, cyclic Article/Thread — fixed 6.13.1), CVE-2026-59937 (malformed cross-reference streams — fixed 6.14.0), CVE-2026-84310/84311 (GHSA-763m-79hh-57f2, algorithmic-complexity DoS in outline/XObject traversal — fixed 6.16.1). The server feeds this library up-to-100-MB unauthenticated bytes pre-slot, so the floor is load-bearing; fresh installs resolve the latest 6.x, which is why the exposure was limited. Pin-contract re-verification run against the installed pypdf 6.21.1 (satisfies the new floor): `TestEnforcePdfCaps` 24/24 + the e2e page-limit pair.
- Docs updated with the new semantics: `_enforce_pdf_caps` docstring (page-cap + fail-soft paragraphs), the `MAX_OCR_PAGES` constant comment, `_run_ocr_pipeline`'s docstring + call-site comment, README's `MAX_OCR_PAGES` row, this file's live pin reference (Deployment Notes). The sixth-pass bullet's `pypdf>=6,<7` is history and stays as-is.
- Pre-slot cap parse fan-out (carried since the sixth pass) RESOLVED: the new `MAX_CONCURRENT_PDF_PARSSES` knob (default `2`, import-clamped to >= 1 like the sibling knobs — 0 would 503 every PDF request "busy" forever, negative makes `asyncio.Semaphore` raise at the first request) + per-event-loop `_PDF_PARSE_SEMAPHORES` registry (`_pdf_parse_slot` mirrors `_ocr_job_slot`'s per-loop pattern, opportunistic stale-loop prune, and 503-on-saturation; event-loop-side on purpose — a worker thread is never blocked on it, so parse waiters cannot occupy pool threads ahead of pipeline jobs) wraps the `asyncio.to_thread(_enforce_pdf_caps, ...)` hop in `_run_ocr_pipeline` — at most N concurrent in-flight parses instead of the thread-pool-size × MAX_REQUEST_SIZE worst case. Acquired before, and released before, the job slot: no nesting between the two semaphores (no deadlock), and a rejected job still takes no job slot (413s keep their no-staging/no-slot/no-pipeline e2e pins). Tests: `TestPdfParseSlot` +4 (503 when saturated; slot held while admitted and released on exit; release on the 413 rejection path; constant → semaphore link — patched constant 3, fresh registry + fresh loop admits exactly 3 concurrent holds and 503s the 4th), `TestJobSlotsEnvContract` extended to BOTH concurrency knobs (fresh-interpreter clamp contract, zero/negative/explicit each), e2e +3 (`test_page_limit_parse_slot_saturated_503` on /layout-parsing and `test_extract_parse_slot_saturated_503` on /extract — 503 before staging + job slot + pipeline; `test_page_limit_413_releases_parse_slot` — capacity 1, the 413 leaves the loop's slot free). `_encrypt_and_lie_count` hardened: the length-preserving `/Count` byte replacement now asserts equal needle/replacement length (a mismatch would silently corrupt the fixture's xref offsets).
- Carried / parked: unchanged from the sixth pass except the resolved pre-slot parse fan-out (still non-blocking per the synthesis): `pypdfium2>=5,<6` upper-bound conflict watch with a future ocrmypdf 17.x requiring pypdfium2≥6; the four non-critical pinned-but-fail-soft knobs (fail-loud on import); the `/health` getter split; the Paperless-fail e2e path; the `_minimal_pdf` duplication (the builders now also share the `count:` parameter); and the test-theme minors (barrier-test sleep, fresh-interpreter env-clamp test dedup, `Final` for `MODEL_INIT_FAILURE_DETAIL`, worker.INBOX docstring drift, /extract 503 endpoint pin, e2e `&&` chaining, stale-loop prune branch, the two leftover `mkdtemp` tests).

Post-review fixes (2026-10-17, eighth pass):

- `_finite_geom._walk` now walks **tuples** (re-emitted as lists), including at the top-level dispatch: `_jsonable` normally normalizes tuples away, but its `.tolist()` branch returns the result *without* recursing — an object whose `.tolist()` yields a tuple (a custom ndarray wrapper) can deliver one straight to the walk, where a NaN inside it would have passed unsanitized and 500'd the response via `json.dumps(allow_nan=False)`. Unreachable with realistic PaddleX output (ndarray → `_jsonable` → list) — a theoretical hardening, but pinned anyway: `TestFiniteGeom` +2 (nested and top-level `tolist()→tuple` with NaN/Inf → zeroed, list-typed, JSON-safe under `allow_nan=False`).
- Non-ASCII bearer token (review finding) VERIFIED ALREADY SAFE — no code change: `_check_token` encodes both sides to UTF-8 bytes before `hmac.compare_digest`, and the TypeError the finding describes is only raisable for non-ASCII *str* arguments. Empirically probed against the live function: non-ASCII presented token → 403 (no 500), non-ASCII configured token with matching header → authenticates. Pinned by `test_check_token_non_ascii_bearer_no_500` so a future refactor back to str comparison fails loudly instead of re-introducing the 500.
- Other seventh-review findings accepted as documented: the broad fail-closed `PdfReadError` clause (any guarded-walk `PdfReadError` → 413 is the safe direction); the pypdf-unparseable/pypdfium2-openable residual (gated on the /Count count, pixel cap skipped, logged with traceback); and the `MAX_OCR_PAGES` > 100,000 ↔ `page_tree_maximum_entries` interaction (flagged at the constant). Carried / parked: unchanged from the seventh pass.

Post-review fixes (themed review 2026-01-15, ninth pass):

- **The one blocking item — `_enforce_pdf_caps` fail-soft residual (constructor-level parser disagreement):** pypdf's constructor rejects the bytes while pypdfium2 opens the same ones (the verified shape: a spec-legal PDF with a valid xref table + trailer but no `startxref` section — pypdf raises `PdfReadError("startxref not found")`, pypdfium2/poppler read the xref table/trailer directly; empirically verified this round) used to fail soft: page cap on the /Count-gated pdfium count, **pixel cap skipped entirely** — a ~51 KB 500-page × 6000 pt file rendered ≈1.88 GB of RGB per page (~19× the default `MAX_RENDERED_PIXELS` envelope) for the full render timeout, per unauthenticated request. It now fails CLOSED (413 `"Unparseable PDF"`, logged with traceback) — consistent with the page-tree guard trips: pypdf's own guards refusing a degenerate tree already fail-closed the same way, and the /Count-gated count is the attacker-writable value the page-tree gate exists to distrust. The genuinely fail-soft shapes remain (and are documented): bytes NEITHER parser opens (BOTH caps skipped — the pipeline's own error applies), and a document pypdf OPENED but that fails a mid-pass step (page cap on the /Count-gated count where that is the only count available, pixel cap for the pages already measured; logged with traceback). `requirements.txt`'s pypdf comment updated to the new residual semantics. Tests: `test_pypdf_failure_skips_pixel_cap_only` → `test_pypdf_constructor_failure_413_unparseable` (mocked constructor failure + pdfium-openable → 413, fail-closed log pinned) + `test_pypdf_constructor_failure_real_missing_startxref_413` (REAL parsers, self-verifying both sides of the disagreement; huge first page proves the pre-fix hole) + e2e `test_page_limit_413_unparseable_pdfium_only` / `test_extract_page_limit_413_unparseable_pdfium_only` (both endpoints, 413 before slot + staging). The deliberate over-rejection (an UNDERSTATED /Count whose huge box sits beyond the raw /Count 413s on the true count even though poppler clamps its render loop to the raw /Count) is documented at `_enforce_pdf_caps` — the fail-closed direction.
- **New image-canvas cap (`_enforce_image_pixel_cap`):** the review's follow-up item #1 — a highly compressible PNG/TIFF is a few KB/MB on disk yet can declare a 100 000 × 100 000 px canvas (tens of GB at decode) and pass `MAX_REQUEST_SIZE`; the decode happens inside the pipeline. The cap reads the header-declared canvas (`PIL Image.open().size` — metadata only, no pixel decode) under the existing parse slot (a pathological header is O(file size) to parse — must not stall the event loop) and 413s over `MAX_RENDERED_PIXELS` (same knob, same envelope — no new env var), BEFORE the job slot + temp-file staging. Fail-soft on unreadable headers (undecodable by the pipeline too); a canvas above PIL's OWN decompression-bomb ERROR threshold (2 × `MAX_IMAGE_PIXELS`, default ≈179 Mpx) 413s whenever the threshold sits at/above the cap (the default case), so the operator knob stays authoritative in the default configuration. `MAX_RENDERED_PIXELS` / `MAX_CONCURRENT_PDF_PARSSES` comments updated. Tests: unit `TestEnforceImagePixelCap` (6 — both over-cap branches, at/just-over boundary, undecodable fail-soft, cap-above-threshold fail-soft) + e2e on BOTH endpoints (`_HUGE_PNG` bomb-branch + 13000² direct-branch canvases, 413 before slot + staging; parse-slot release on 413; small-canvas 200).
- **Carried-over follow-up pins:** `/extract` job-slot saturation 503 (e2e `test_extract_job_slot_saturated_503` — the parse-slot sibling was pinned on both endpoints; the job gate now is too, same real-slot/patched-factory pattern); server-side `_mark_model_used` stamping in `_run_ocr_pipeline` (unit `TestRunOcrPipelineStamping` — the eighth pass pinned the worker side only); the per-page CropBox-exception fallback branch in `_enforce_pdf_caps` (unit `test_cropbox_exception_falls_back_to_mediabox` — 3-element /CropBox → pypdf `ValueError`, verified — huge MediaBox still 413s on the MediaBox alone, normal one passes; the UNIT builder gained the raw-string `/CropBox` option — the e2e copy mirrors the builder by hand but keeps the 4-tuple-only form, so the "byte-identical by hand" invariant is reworded to "mirrored by hand" at both doc sites).
- **Nits:** the two per-event-loop semaphore blocks deduped into `_per_loop_semaphore` (+ a shared `_saturated_slot` 503 context manager — the public `_ocr_job_semaphore` / `_pdf_parse_semaphore` / `_ocr_job_slot` / `_pdf_parse_slot` names and per-loop semantics are unchanged, so the test patch targets still work); the six fresh-interpreter env-contract subprocess blocks (test_paddlex_helpers.py ×5, test_server.py ×2) collapsed onto one `tests/conftest.py::run_fresh_interpreter(_ok)` helper (the 0/negative twins of the same clamp each keep their zero case — same `max(1, ...)` floor); the duplicated `"Bearer "` scheme string replaced by `_BEARER_PREFIX` (the len() derivation — the 7 was a magic number); the stale `"lazy (PEP 563)"` annotation comment corrected to PEP 526 (local annotations are never evaluated — the lazy import is what that buys); `worker.py`'s `_HEX_DIGITS` is a `frozenset` (immutable membership set); the `pdftoppm_timeout` local alias in `_pdf_to_images` removed (the `PAPDFTOPPM_TIMEOUT` constant directly). README env table updated (`MAX_RENDERED_PIXELS` now also covers image uploads; `MAX_CONCURRENT_PDF_PARSSES` also bounds the image header parse).
