"""
Stage 4: Lifecycle Management
===============================

Moves the job to its terminal state and cleans up:
- On success: move input to DONE/ (folder) or release buffer (API)
- On failure: move input to ERROR/ (folder) or return error (API)
- Clean up intermediate files in PROCESSING/
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from server.companion import get_config, get_logger
from server.models import Job, JobContext
from server.orchestrator import StageError


def lifecycle(job: Job, ctx: JobContext) -> JobContext:
    """Manage job lifecycle and cleanup.

    Args:
        job: The current job.
        ctx: The job context.

    Returns:
        Updated context with final state.

    Raises:
        StageError: If lifecycle management fails.
    """
    logger = get_logger("doc-worker.stages.lifecycle")
    config = get_config()
    filename = job.input.filename

    logger.info(f"Lifecycle management: {filename} (state={job.state.value})")

    try:
        if job.input.source == "folder":
            ctx = _manage_folder_lifecycle(job, ctx, config)
        elif job.input.source == "api":
            ctx = _manage_api_lifecycle(job, ctx)
        else:
            raise StageError(
                f"Unknown job source: {job.input.source}",
                stage="lifecycle",
            )

        logger.info(
            f"Lifecycle complete: {filename} → {job.state.value} "
            f"({job.elapsed_seconds:.1f}s)"
        )
        return ctx

    except StageError:
        raise
    except Exception as exc:
        raise StageError(
            f"Lifecycle management failed: {exc}",
            retryable=False,
            stage="lifecycle",
        )


def _manage_folder_lifecycle(
    job: Job, ctx: JobContext, config
) -> JobContext:
    """Manage lifecycle for folder-sourced jobs."""
    filename = job.input.filename
    inbox_path = config.INBOX / filename
    processing_path = config.PROCESSING / filename
    error_path = config.ERROR / filename
    done_path = config.DONE / filename

    # Determine current file location
    current_path = None
    for p in (processing_path, inbox_path):
        if p.exists():
            current_path = p
            break

    if job.state == JobState.FAILED:
        # Move to ERROR/
        if current_path:
            dest = error_path
            if dest.exists():
                stem = current_path.stem
                suffix = current_path.suffix
                ts = time.strftime("%Y%m%d%H%M%S")
                dest = config.ERROR / f"{stem}_{ts}{suffix}"
            shutil.move(str(current_path), str(dest))
            ctx.extra["final_path"] = str(dest)
            get_logger("doc-worker.stages.lifecycle").info(
                f"  → ERROR/ ({filename})"
            )

    else:
        # Success — move to DONE/
        if current_path:
            dest = done_path
            if dest.exists():
                stem = current_path.stem
                suffix = current_path.suffix
                ts = time.strftime("%Y%m%d%H%M%S")
                dest = config.DONE / f"{stem}_{ts}{suffix}"
            shutil.move(str(current_path), str(dest))
            ctx.extra["final_path"] = str(dest)
            get_logger("doc-worker.stages.lifecycle").info(
                f"  → DONE/ ({filename})"
            )

    # Clean up intermediate files
    _cleanup_intermediate(config.PROCESSING, filename)

    return ctx


def _manage_api_lifecycle(job: Job, ctx: JobContext) -> JobContext:
    """Manage lifecycle for API-sourced jobs."""
    # For API jobs, there's no filesystem lifecycle.
    # Just release any in-memory buffers.
    if job.input.data is not None:
        job.input.data = None
    ctx.extra["source"] = "api"
    return ctx


def _cleanup_intermediate(processing_dir: Path, filename: str) -> None:
    """Clean up intermediate files in PROCESSING/."""
    stem = Path(filename).stem

    # Remove OCR output and any temp files
    for pattern in [f"{stem}_ocr.pdf", f"{stem}*.tmp"]:
        for f in processing_dir.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass


# Import JobState here to avoid circular imports
from server.models import JobState  # noqa: E402