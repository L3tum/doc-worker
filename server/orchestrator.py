"""
Doc-Worker — Orchestrator
==========================

Manages the job lifecycle through a finite state machine:
  queued → validating → processing → delivering → completed | failed

Dispatches pipeline stages sequentially, handles per-stage retry logic
with configurable backoff, and emits structured logs at each transition.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Callable, Type

from server.companion import (
    get_config, get_logger, get_metrics, log_with_context,
)
from server.models import Job, JobContext, JobState


# ---------------------------------------------------------------------------
# Stage Protocol
# ---------------------------------------------------------------------------

StageFn = Callable[[Job, JobContext], JobContext]
"""A pipeline stage function. Receives job + context, returns updated context."""


class StageError(Exception):
    """Raised by a stage to indicate failure.

    Attributes:
        retryable: If True, the orchestrator will retry.
        stage: The stage name where the error occurred.
    """

    def __init__(self, message: str, retryable: bool = True, stage: str = ""):
        super().__init__(message)
        self.retryable = retryable
        self.stage = stage


class StageValidationError(StageError):
    """Non-retryable validation error (invalid input)."""

    def __init__(self, message: str, stage: str = ""):
        super().__init__(message, retryable=False, stage=stage)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """FSM-based pipeline orchestrator.

    Manages job state transitions, stage dispatch, retry logic,
    and concurrency control.
    """

    # Valid state transitions
    TRANSITIONS: dict[JobState, set[JobState]] = {
        JobState.QUEUED: {JobState.VALIDATING, JobState.FAILED},
        JobState.VALIDATING: {JobState.PROCESSING, JobState.FAILED},
        JobState.PROCESSING: {JobState.DELIVERING, JobState.FAILED},
        JobState.DELIVERING: {JobState.COMPLETED, JobState.FAILED},
        JobState.COMPLETED: set(),
        JobState.FAILED: set(),
    }

    def __init__(
        self,
        stages: list[tuple[str, StageFn]],
        max_concurrent: int | None = None,
        max_retries: int | None = None,
        retry_delay: int | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            stages: List of (stage_name, stage_function) tuples, executed in order.
            max_concurrent: Maximum concurrent jobs (from config if None).
            max_retries: Max retries per job (from config if None).
            retry_delay: Base retry delay in seconds (from config if None).
        """
        config = get_config()
        self.stages = stages
        self.max_concurrent = max_concurrent or config.MAX_CONCURRENT_JOBS
        self.max_retries = max_retries or config.MAX_RETRIES
        self.retry_delay = retry_delay or config.RETRY_DELAY

        self._queue: deque[Job] = deque()
        self._active_jobs: dict[str, Job] = {}
        self._logger = get_logger("doc-worker.orchestrator")
        self._running = False

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def active_count(self) -> int:
        return len(self._active_jobs)

    @property
    def can_accept(self) -> bool:
        """Whether the orchestrator can accept new jobs."""
        return (
            self._running
            and self.active_count < self.max_concurrent
        )

    def start(self) -> None:
        """Start the orchestrator."""
        self._running = True
        self._logger.info(
            f"Orchestrator started: {len(self.stages)} stages, "
            f"max_concurrent={self.max_concurrent}"
        )

    def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        self._logger.info("Orchestrator stopped")

    def enqueue(self, job: Job) -> Job:
        """Add a job to the processing queue.

        Args:
            job: The job to enqueue.

        Returns:
            The job (for chaining).

        Raises:
            RuntimeError: If the queue is full.
        """
        if not self._running:
            raise RuntimeError("Orchestrator is not running")

        if self.queue_size + self.active_count >= self.max_concurrent * 2:
            self._logger.warning(
                f"Queue full ({self.queue_size} queued, "
                f"{self.active_count} active), rejecting job {job.job_id}"
            )
            job.transition(JobState.FAILED)
            job.context.errors.append("Queue full — rejected")
            raise RuntimeError("Job queue is full")

        self._queue.append(job)
        self._logger.info(
            f"Job {job.job_id} enqueued (queue={self.queue_size}, "
            f"active={self.active_count})"
        )
        return job

    def process_next(self) -> Job | None:
        """Process the next job in the queue (if any).

        Returns:
            The completed/failed job, or None if no job was processed.
        """
        if not self._queue:
            return None

        job = self._queue.popleft()
        self._active_jobs[job.job_id] = job
        job.started_at = time.time()

        try:
            self._run_pipeline(job)
        except Exception as exc:
            self._logger.error(
                f"Job {job.job_id} failed unexpectedly: {exc}",
                extra={"job_id": job.job_id},
            )
            job.transition(JobState.FAILED)
            job.context.errors.append(f"Unexpected error: {exc}")
        finally:
            self._active_jobs.pop(job.job_id, None)

        # Record metrics
        elapsed = job.elapsed_seconds
        if job.state == JobState.COMPLETED:
            get_metrics().record_success(elapsed)
        else:
            get_metrics().record_failure(
                job.context.errors[-1] if job.context.errors else "unknown"
            )

        return job

    def _run_pipeline(self, job: Job) -> None:
        """Run the full pipeline for a job, with retry logic."""
        for stage_name, stage_fn in self.stages:
            # State transition before each stage
            self._transition(job, self._next_state(stage_name))

            # Run stage with retries
            self._run_stage_with_retry(job, stage_name, stage_fn)

            # If job failed during this stage, stop
            if job.state == JobState.FAILED:
                return

        # All stages completed
        self._transition(job, JobState.COMPLETED)
        self._logger.info(
            f"Job {job.job_id} completed successfully in "
            f"{job.elapsed_seconds:.1f}s"
        )

    def _run_stage_with_retry(
        self, job: Job, stage_name: str, stage_fn: StageFn
    ) -> None:
        """Run a single stage with retry logic."""
        start = time.time()
        retries = 0

        while True:
            try:
                job.context = stage_fn(job, job.context)
                duration = time.time() - start
                get_metrics().record_stage_timing(stage_name, duration)
                self._logger.debug(
                    f"Stage '{stage_name}' for job {job.job_id}: "
                    f"OK ({duration:.2f}s)"
                )
                return

            except StageValidationError as exc:
                # Non-retryable — fail immediately
                self._logger.error(
                    f"Stage '{stage_name}' validation failed for job "
                    f"{job.job_id}: {exc}"
                )
                job.transition(JobState.FAILED)
                job.context.errors.append(
                    f"[{stage_name}] Validation: {exc}"
                )
                return

            except StageError as exc:
                # Retryable stage error
                retries += 1
                if retries <= job.max_retries:
                    delay = self.retry_delay * (2 ** (retries - 1))
                    self._logger.warning(
                        f"Stage '{stage_name}' failed for job "
                        f"{job.job_id} (attempt {retries}/{job.max_retries}): "
                        f"{exc}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    self._logger.error(
                        f"Stage '{stage_name}' failed after "
                        f"{job.max_retries} retries for job {job.job_id}: {exc}"
                    )
                    job.transition(JobState.FAILED)
                    job.context.errors.append(
                        f"[{stage_name}] Failed after {job.max_retries} retries: {exc}"
                    )
                    return

            except Exception as exc:
                # Unexpected error — retry if within limit
                retries += 1
                if retries <= job.max_retries:
                    delay = self.retry_delay * (2 ** (retries - 1))
                    self._logger.warning(
                        f"Stage '{stage_name}' unexpected error for job "
                        f"{job.job_id} (attempt {retries}/{job.max_retries}): "
                        f"{exc}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    self._logger.error(
                        f"Stage '{stage_name}' failed after "
                        f"{job.max_retries} retries for job {job.job_id}: {exc}"
                    )
                    job.transition(JobState.FAILED)
                    job.context.errors.append(
                        f"[{stage_name}] Unexpected error after "
                        f"{job.max_retries} retries: {exc}"
                    )
                    return

    def _transition(self, job: Job, new_state: JobState) -> None:
        """Transition a job to a new state, validating the transition."""
        allowed = self.TRANSITIONS.get(job.state, set())
        if new_state not in allowed:
            self._logger.warning(
                f"Invalid transition for job {job.job_id}: "
                f"{job.state} → {new_state} (allowed: {allowed})"
            )
            return
        job.transition(new_state)
        self._logger.info(
            f"Job {job.job_id}: {job.state} → {new_state}"
        )

    @staticmethod
    def _next_state(stage_name: str) -> JobState:
        """Map stage name to the FSM state it represents."""
        mapping = {
            "validate": JobState.VALIDATING,
            "ocr": JobState.PROCESSING,
            "output": JobState.DELIVERING,
            "lifecycle": JobState.DELIVERING,
        }
        return mapping.get(stage_name, JobState.PROCESSING)