"""
Doc-Worker — API Hook (FastAPI)
=================================

HTTP endpoint for on-demand document processing:
- POST /api/v1/convert — submit a document for OCR processing
- GET /health — health check
- GET /ready — readiness check
- GET /metrics — processing metrics
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from server.companion import (
    get_config, get_health, get_logger, get_metrics, init_companion,
)
from server.models import DocumentInput, Job
from server.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Doc-Worker API",
    description="OCR processing pipeline API",
    version="1.0.0",
)

_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator not initialized",
        )
    return _orchestrator


def set_orchestrator(orch: Orchestrator) -> None:
    """Set the global orchestrator instance (called by main)."""
    global _orchestrator
    _orchestrator = orch


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    """Initialize the companion module on startup."""
    init_companion()
    get_logger("doc-worker.api").info("API server starting")


# ---------------------------------------------------------------------------
# Health & Readiness
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    health_status = get_health()
    return {
        "status": health_status.status,
        "uptime": round(health_status.uptime, 1),
    }


@app.get("/ready")
async def ready() -> dict[str, Any]:
    """Readiness check endpoint."""
    health_status = get_health()
    orchestrator = None
    try:
        orchestrator = get_orchestrator()
    except HTTPException:
        pass

    ready = (
        health_status.ready
        and orchestrator is not None
    )

    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "ready": ready,
            "model_loaded": health_status.model_loaded,
            "orchestrator_running": orchestrator is not None
                and orchestrator._running if orchestrator else False,
        },
    )


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Processing metrics endpoint."""
    return get_metrics().summary()


# ---------------------------------------------------------------------------
# Convert Endpoint
# ---------------------------------------------------------------------------

@app.post("/api/v1/convert")
async def convert(
    file: UploadFile = File(..., description="Document to process"),
    mode: str = Form("auto", description="Processing mode: auto, pp_ocr, paddle_vl"),
) -> dict[str, Any]:
    """Submit a document for OCR processing.

    Accepts a PDF or image file, processes it through the OCR pipeline,
    and returns the OCR'd PDF along with a Markdown sidecar.
    """
    logger = get_logger("doc-worker.api")

    # Validate file type
    ext = Path(file.filename or "").suffix.lower()
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".webp"}
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {supported}",
        )

    # Read file content
    content = await file.read()

    # Check file size
    config = get_config()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB > {config.MAX_FILE_SIZE_MB}MB",
        )

    logger.info(f"API convert request: {file.filename} ({size_mb:.1f}MB)")

    # Create job
    document = DocumentInput(
        filename=file.filename or f"upload{ext}",
        source="api",
        data=content,
        metadata={"mode": mode},
    )

    job = Job(input=document)

    # Get orchestrator and process
    try:
        orchestrator = get_orchestrator()
    except HTTPException:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator not initialized",
        )

    if not orchestrator.can_accept:
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent jobs. Try again later.",
        )

    # Process synchronously (for API, we want to return the result)
    # In a future iteration, this could be async with a job status endpoint
    orchestrator._run_pipeline(job)

    if job.state.value == "failed":
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Processing failed",
                "job_id": job.job_id,
                "errors": job.context.errors,
            },
        )

    # Build response
    response: dict[str, Any] = {
        "job_id": job.job_id,
        "filename": job.input.filename,
        "document_type": job.context.document_type.value,
        "processing_mode": job.context.processing_mode.value,
        "elapsed_seconds": round(job.elapsed_seconds, 2),
    }

    # Include OCR'd PDF as base64
    if job.context.outputs.ocr_pdf:
        response["pdf"] = {
            "filename": f"{Path(job.input.filename).stem}_ocr.pdf",
            "size": len(job.context.outputs.ocr_pdf),
            "data": base64.b64encode(job.context.outputs.ocr_pdf).decode("ascii"),
        }

    # Include Markdown
    if job.context.outputs.markdown:
        response["markdown"] = {
            "filename": f"{Path(job.input.filename).stem}.md",
            "content": job.context.outputs.markdown,
        }

    # Include metadata
    if job.context.outputs.json_metadata:
        response["metadata"] = job.context.outputs.json_metadata

    # Include manifest
    if job.context.outputs.manifest:
        response["manifest"] = job.context.outputs.manifest

    logger.info(f"API convert complete: {file.filename} ({job.elapsed_seconds:.1f}s)")
    return response