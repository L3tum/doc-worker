"""
Doc-Worker — Core Data Models
==============================

Defines the job lifecycle types used across the pipeline:
- JobState: FSM states a job can be in.
- Job: the top-level handle that tracks a document through processing.
- JobContext: mutable context passed between pipeline stages.
- DocumentInput: ingestion-layer input abstraction.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class JobState(str, Enum):
    """Finite states for the job lifecycle."""

    QUEUED = "queued"
    VALIDATING = "validating"
    PROCESSING = "processing"
    DELIVERING = "delivering"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Classifications determined during validation/triage."""

    UNKNOWN = "unknown"
    TEXT_PDF = "text_pdf"
    SCANNED_PDF = "scanned_pdf"
    HYBRID_PDF = "hybrid_pdf"
    IMAGE = "image"


class ProcessingMode(str, Enum):
    """OCR processing strategies."""

    SKIP = "skip"  # text PDF — extract directly
    FULL_OCR = "full_ocr"  # scanned — run full OCR pipeline
    OVERLAY = "overlay"  # hybrid — overlay text layer


@dataclass
class DocumentInput:
    """Ingestion-layer document input.

    Supports both filesystem-backed paths (folder watcher) and
    in-memory byte buffers (API hook).
    """

    filename: str
    source: str  # "folder" | "api"
    path: Path | None = None  # filesystem-backed
    data: bytes | None = None  # in-memory buffer
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the document content."""
        if self.data:
            return hashlib.sha256(self.data).hexdigest()
        elif self.path and self.path.exists():
            return hashlib.sha256(self.path.read_bytes()).hexdigest()
        return "unknown"


@dataclass
class StageOutput:
    """Outputs produced by a pipeline stage."""

    ocr_pdf: bytes | None = None
    markdown: str | None = None
    json_metadata: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None


@dataclass
class JobContext:
    """Mutable context passed between pipeline stages.

    Each stage reads from and writes to this context.
    """

    job_id: str
    document_type: DocumentType = DocumentType.UNKNOWN
    processing_mode: ProcessingMode = ProcessingMode.FULL_OCR
    outputs: StageOutput = field(default_factory=StageOutput)
    errors: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    """Top-level job handle.

    Created by an ingestion adapter, passed through the orchestrator,
    and delivered to the appropriate destination.
    """

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    input: DocumentInput = field(
        default_factory=lambda: DocumentInput(filename="", source="")
    )
    context: JobContext = field(default_factory=lambda: JobContext(job_id=""))
    state: JobState = JobState.QUEUED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        """Ensure context references this job."""
        self.context.job_id = self.job_id

    def transition(self, new_state: JobState) -> None:
        """Transition to a new state and record the timestamp."""
        self.state = new_state
        self.updated_at = time.time()
        if new_state == JobState.COMPLETED or new_state == JobState.FAILED:
            self.completed_at = time.time()

    @property
    def elapsed_seconds(self) -> float:
        """Time elapsed since job creation."""
        end = self.completed_at or time.time()
        return end - self.created_at
