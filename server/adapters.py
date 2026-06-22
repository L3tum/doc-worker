"""
Doc-Worker — Ingestion Adapters
=================================

Defines the IngestionAdapter protocol and provides implementations:
- FolderWatcherAdapter: polls INBOX/ for new files
- APIAdapter: accepts documents via HTTP (FastAPI)

Neither adapter contains pipeline logic — they construct Job objects
and submit them to the orchestrator.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from server.companion import get_config, get_logger
from server.models import DocumentInput, Job

if TYPE_CHECKING:
    from server.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class IngestionAdapter(Protocol):
    """Interface for document ingestion sources."""

    def submit(self, document: DocumentInput) -> Job:
        """Submit a document for processing. Returns a Job handle."""
        ...

    def run(self) -> None:
        """Start the ingestion loop (blocking)."""
        ...


# ---------------------------------------------------------------------------
# Folder Watcher Adapter
# ---------------------------------------------------------------------------


class FolderWatcherAdapter:
    """Polls INBOX/ for new files and submits them to the orchestrator.

    Supports PDF and common image formats:
    - PDF: .pdf, .PDF
    - Images: .png, .jpg, .jpeg, .tiff, .webp (and uppercase variants)
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".PDF",
        ".png",
        ".PNG",
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG",
        ".tiff",
        ".TIFF",
        ".webp",
        ".WEBP",
    }

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.config = get_config()
        self._logger = get_logger("doc-worker.folder")

    def submit(self, document: DocumentInput) -> Job:
        """Submit a document for processing."""
        job = Job(input=document)
        self.orchestrator.enqueue(job)
        return job

    def run(self) -> None:
        """Main polling loop."""
        self._logger.info(f"Folder watcher started: polling {self.config.INBOX}")

        while True:
            try:
                self._poll()
            except Exception as exc:
                self._logger.error(f"Folder watcher error: {exc}")

            time.sleep(self.config.POLL_INTERVAL)

    def _poll(self) -> None:
        """Single poll iteration: find new files and submit them."""
        files = self._find_new_files()

        for filepath in files:
            if not self._wait_for_stable(filepath):
                self._logger.warning(f"Skipping unstable file: {filepath.name}")
                continue

            if not self.orchestrator.can_accept:
                self._logger.warning(
                    f"Orchestrator at capacity, queuing {filepath.name}"
                )

            document = DocumentInput(
                filename=filepath.name,
                source="folder",
                path=filepath,
            )

            try:
                job = self.submit(document)
                self._logger.info(f"Submitted {filepath.name} as job {job.job_id}")
            except RuntimeError as exc:
                self._logger.error(f"Failed to submit {filepath.name}: {exc}")

    def _find_new_files(self) -> list[Path]:
        """Find new files in the inbox directory."""
        if not self.config.INBOX.exists():
            return []

        files = set()
        for ext in self.SUPPORTED_EXTENSIONS:
            files.update(self.config.INBOX.glob(f"*{ext}"))

        return sorted(files, key=lambda p: p.name.lower())

    def _wait_for_stable(self, filepath: Path) -> bool:
        """Wait until the file size stops changing.

        Polls the file size for up to STABILITY_TIMEOUT seconds.
        If the size stays the same for the full period, the file is
        considered stable (upload complete).
        """
        timeout = self.config.STABILITY_TIMEOUT
        try:
            last_size = filepath.stat().st_size
            start = time.time()

            while time.time() - start < timeout:
                time.sleep(1)
                current_size = filepath.stat().st_size
                if current_size != last_size:
                    last_size = current_size
                    start = time.time()
                    continue

            self._logger.debug(f"File stable: {filepath.name}")
            return True

        except FileNotFoundError:
            self._logger.warning(
                f"File disappeared during stability check: {filepath.name}"
            )
            return False


# ---------------------------------------------------------------------------
# Crash Recovery
# ---------------------------------------------------------------------------


def recover_leftover_files(config=None) -> None:
    """Move stale files from PROCESSING/ to ERROR/ on startup.

    This ensures that files left behind by a previous crash are not
    silently lost. They are moved to ERROR/ for inspection.
    """
    if config is None:
        config = get_config()

    processing = config.PROCESSING
    error = config.ERROR
    logger = get_logger("doc-worker.recovery")

    leftovers = list(processing.iterdir()) if processing.exists() else []
    if not leftovers:
        return

    logger.info(f"Crash recovery: found {len(leftovers)} file(s) in PROCESSING/")

    for f in leftovers:
        dest = error / f.name
        if dest.exists():
            stem = f.stem
            suffix = f.suffix
            ts = time.strftime("%Y%m%d%H%M%S")
            dest = error / f"{stem}_{ts}{suffix}"
        shutil.move(str(f), str(dest))
        logger.info(f"  Recovered: {f.name} → ERROR/")
