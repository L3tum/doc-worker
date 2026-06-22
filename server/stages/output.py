"""
Stage 3: Output Assembly & Persistence
========================================

Packages results and delivers them to the appropriate destination:
- Folder watcher: write OCR'd PDF, sidecars, push to Paperless
- API hook: bundle outputs into structured response

Also generates a processing manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from server.companion import FileIO, get_config, get_logger
from server.models import Job, JobContext
from server.orchestrator import StageError


def output(job: Job, ctx: JobContext) -> JobContext:
    """Assemble and persist outputs.

    Args:
        job: The current job.
        ctx: The job context.

    Returns:
        Updated context with persisted output paths.

    Raises:
        StageError: If output persistence fails.
    """
    logger = get_logger("doc-worker.stages.output")
    config = get_config()
    filename = job.input.filename
    stem = Path(filename).stem

    logger.info(f"Assembling outputs: {filename}")

    try:
        if job.input.source == "folder":
            ctx = _persist_folder_outputs(job, ctx, config, stem)
        elif job.input.source == "api":
            ctx = _bundle_api_outputs(job, ctx)
        else:
            raise StageError(
                f"Unknown job source: {job.input.source}",
                stage="output",
            )

        # Generate manifest
        ctx.outputs.manifest = _generate_manifest(job, ctx)

        logger.info(f"Outputs assembled: {filename}")
        return ctx

    except StageError:
        raise
    except Exception as exc:
        raise StageError(
            f"Output assembly failed: {exc}",
            retryable=True,
            stage="output",
        )


def _persist_folder_outputs(
    job: Job, ctx: JobContext, config, stem: str
) -> JobContext:
    """Persist outputs for folder-sourced jobs."""
    logger = get_logger("doc-worker.stages.output")

    # Create per-document output directory
    out_dir = config.OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write OCR'd PDF to PROCESSING/ (will be moved by lifecycle stage)
    if ctx.outputs.ocr_pdf:
        ocr_path = config.PROCESSING / f"{stem}_ocr.pdf"
        FileIO.write(ocr_path, ctx.outputs.ocr_pdf)
        ctx.extra["ocr_pdf_path"] = str(ocr_path)
        logger.info(f"  OCR PDF written: {ocr_path}")

    # Write Markdown sidecar to OUTPUT_DIR/
    if ctx.outputs.markdown:
        md_path = out_dir / f"{stem}.md"
        FileIO.write_text(md_path, ctx.outputs.markdown)
        ctx.extra["markdown_path"] = str(md_path)
        logger.info(f"  Markdown written: {md_path}")

    # Write JSON metadata to OUTPUT_DIR/
    if ctx.outputs.json_metadata:
        json_path = out_dir / f"{stem}.json"
        FileIO.write_text(
            json_path,
            json.dumps(ctx.outputs.json_metadata, indent=2, default=str),
        )
        ctx.extra["json_path"] = str(json_path)
        logger.info(f"  JSON metadata written: {json_path}")

    # Write processing manifest to OUTPUT_DIR/
    if ctx.outputs.manifest:
        manifest_path = out_dir / f"{stem}_manifest.json"
        FileIO.write_text(
            manifest_path,
            json.dumps(ctx.outputs.manifest, indent=2, default=str),
        )
        logger.info(f"  Manifest written: {manifest_path}")

    # Push to Paperless
    if ctx.outputs.ocr_pdf and ctx.extra.get("ocr_pdf_path"):
        ocr_path = Path(ctx.extra["ocr_pdf_path"])
        if _push_to_paperless(ocr_path, config):
            logger.info(f"  Pushed to Paperless: {config.PAPERLESS_CONSUME / ocr_path.name}")

    return ctx


def _bundle_api_outputs(job: Job, ctx: JobContext) -> JobContext:
    """Bundle outputs for API-sourced jobs (in-memory)."""
    # For API jobs, outputs stay in memory (bytes/strings).
    # The API adapter will serialize them into the HTTP response.
    ctx.extra["source"] = "api"
    return ctx


def _push_to_paperless(ocr_path: Path, config) -> bool:
    """Upload OCR'd PDF to Paperless-ngx consume directory.

    Uses an atomic write pattern (write to .tmp, then rename) so Paperless
    never picks up a partially-written file.
    """
    dest = config.PAPERLESS_CONSUME / ocr_path.name
    tmp_dest = config.PAPERLESS_CONSUME / f"{ocr_path.name}.tmp"

    try:
        FileIO.atomic_write(dest, ocr_path.read_bytes())
        return True
    except Exception as exc:
        logger = get_logger("doc-worker.stages.output")
        logger.error(f"Paperless push failed: {exc}")
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False


def _generate_manifest(job: Job, ctx: JobContext) -> dict:
    """Generate a processing manifest documenting the job."""
    return {
        "job_id": job.job_id,
        "filename": job.input.filename,
        "source": job.input.source,
        "content_hash": job.input.content_hash,
        "document_type": ctx.document_type.value,
        "processing_mode": ctx.processing_mode.value,
        "state": job.state.value,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "elapsed_seconds": round(job.elapsed_seconds, 2),
        "timings": ctx.timings,
        "errors": ctx.errors,
        "metadata": ctx.outputs.json_metadata,
    }