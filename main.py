#!/usr/bin/env python3
"""
Doc-Worker — Unified Processing Pipeline
==========================================

Entry point for the refactored doc-worker. Supports two modes:
- Folder watcher (default): polls INBOX/ for new files
- API server: HTTP endpoint for on-demand processing
- Both: run folder watcher + API server concurrently

Usage:
    python main.py              # folder watcher + API
    python main.py --mode folder  # folder watcher only
    python main.py --mode api     # API server only
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import uvicorn

from server.adapters import FolderWatcherAdapter, recover_leftover_files
from server.api import app, set_orchestrator
from server.companion import (
    get_config, get_logger, init_companion,
)
from server.orchestrator import Orchestrator
from server.stages import lifecycle, ocr, output, validate


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def build_orchestrator() -> Orchestrator:
    """Build the orchestrator with all pipeline stages."""
    stages = [
        ("validate", validate.validate),
        ("ocr", ocr.ocr),
        ("output", output.output),
        ("lifecycle", lifecycle.lifecycle),
    ]
    return Orchestrator(stages=stages)


# ---------------------------------------------------------------------------
# Folder watcher mode
# ---------------------------------------------------------------------------

def run_folder_mode(orchestrator: Orchestrator) -> None:
    """Run the folder watcher with periodic processing."""
    config = get_config()
    logger = get_logger("doc-worker.main")

    # Crash recovery
    recover_leftover_files(config)

    orchestrator.start()
    adapter = FolderWatcherAdapter(orchestrator)

    logger.info("Entering folder watcher mode...")

    while True:
        try:
            # Find and submit new files
            adapter._poll()

            # Process queued jobs
            while orchestrator.can_accept and orchestrator.queue_size > 0:
                job = orchestrator.process_next()
                if job:
                    logger.info(
                        f"Job {job.job_id} finished: {job.state.value} "
                        f"({job.elapsed_seconds:.1f}s)"
                    )

        except Exception as exc:
            logger.error(f"Main loop error: {exc}")

        time.sleep(config.POLL_INTERVAL)


# ---------------------------------------------------------------------------
# API mode
# ---------------------------------------------------------------------------

def run_api_mode(orchestrator: Orchestrator) -> None:
    """Run the API server."""
    config = get_config()
    logger = get_logger("doc-worker.main")

    orchestrator.start()
    set_orchestrator(orchestrator)

    logger.info(f"Starting API server on {config.API_HOST}:{config.API_PORT}")

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower(),
    )


# ---------------------------------------------------------------------------
# Combined mode (folder + API)
# ---------------------------------------------------------------------------

def run_combined_mode(orchestrator: Orchestrator) -> None:
    """Run both folder watcher and API server concurrently."""
    config = get_config()
    logger = get_logger("doc-worker.main")

    # Crash recovery
    recover_leftover_files(config)

    orchestrator.start()
    set_orchestrator(orchestrator)

    # Start API server in a thread
    api_thread = threading.Thread(
        target=lambda: uvicorn.run(
            app,
            host=config.API_HOST,
            port=config.API_PORT,
            log_level=config.LOG_LEVEL.lower(),
        ),
        daemon=True,
    )
    api_thread.start()
    logger.info(f"API server started on {config.API_HOST}:{config.API_PORT}")

    # Run folder watcher in main thread
    adapter = FolderWatcherAdapter(orchestrator)
    logger.info("Entering combined mode (folder + API)...")

    while True:
        try:
            adapter._poll()

            while orchestrator.can_accept and orchestrator.queue_size > 0:
                job = orchestrator.process_next()
                if job:
                    logger.info(
                        f"Job {job.job_id} finished: {job.state.value} "
                        f"({job.elapsed_seconds:.1f}s)"
                    )

        except Exception as exc:
            logger.error(f"Main loop error: {exc}")

        time.sleep(config.POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Doc-Worker OCR Pipeline")
    parser.add_argument(
        "--mode",
        choices=["folder", "api", "combined"],
        default="combined",
        help="Running mode (default: combined)",
    )
    args = parser.parse_args()

    # Initialize
    init_companion()
    config = get_config()
    logger = get_logger("doc-worker.main")

    logger.info("=" * 60)
    logger.info("Doc-Worker starting")
    logger.info(f"  Mode:        {args.mode}")
    logger.info(f"  INBOX:       {config.INBOX}")
    logger.info(f"  PROCESSING:  {config.PROCESSING}")
    logger.info(f"  DONE:        {config.DONE}")
    logger.info(f"  ERROR:       {config.ERROR}")
    logger.info(f"  DOCLING_OUT: {config.DOCLING_OUT}")
    logger.info(f"  PAPERLESS:   {config.PAPERLESS_CONSUME}")
    logger.info(f"  OCR_LANG:    {config.OCR_LANG}")
    logger.info(f"  OCR_RUNTIME: {config.OCR_RUNTIME}")
    logger.info(f"  API:         {config.API_HOST}:{config.API_PORT}")
    logger.info(f"  MAX_RETRIES: {config.MAX_RETRIES}")
    logger.info("=" * 60)

    # Build orchestrator
    orchestrator = build_orchestrator()

    # Run in selected mode
    if args.mode == "folder":
        run_folder_mode(orchestrator)
    elif args.mode == "api":
        run_api_mode(orchestrator)
    elif args.mode == "combined":
        run_combined_mode(orchestrator)


if __name__ == "__main__":
    main()