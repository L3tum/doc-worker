"""
Tests for server.orchestrator — FSM pipeline orchestrator.
"""

from __future__ import annotations


import pytest

from server.models import DocumentInput, Job, JobState
from server.orchestrator import (
    Orchestrator, StageError, StageValidationError,
)


class TestStageError:
    """Test StageError exceptions."""

    def test_stage_error_default(self):
        err = StageError("Test error")
        assert str(err) == "Test error"
        assert err.retryable is True
        assert err.stage == ""

    def test_stage_error_params(self):
        err = StageError("Test error", retryable=False, stage="ocr")
        assert err.retryable is False
        assert err.stage == "ocr"

    def test_validation_error(self):
        err = StageValidationError("Invalid input", stage="validate")
        assert err.retryable is False
        assert err.stage == "validate"


class TestOrchestrator:
    """Test Orchestrator FSM and job processing."""

    def _make_job(self, temp_dir, sample_pdf_bytes):
        """Helper to create a test job."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(sample_pdf_bytes)

        doc = DocumentInput(
            filename="test.pdf",
            source="folder",
            path=pdf_path,
        )
        return Job(input=doc)

    def test_initial_state(self):
        """Test orchestrator initial state."""
        orch = Orchestrator(stages=[])

        assert orch.queue_size == 0
        assert orch.active_count == 0
        assert orch._running is False

    def test_start_stop(self):
        """Test start and stop."""
        orch = Orchestrator(stages=[])

        orch.start()
        assert orch._running is True

        orch.stop()
        assert orch._running is False

    def test_enqueue_requires_running(self):
        """Test that enqueue requires orchestrator to be running."""
        orch = Orchestrator(stages=[])

        with pytest.raises(RuntimeError, match="not running"):
            orch.enqueue(Job())

    def test_enqueue_and_process(self, temp_dir, sample_pdf_bytes):
        """Test basic enqueue and process flow."""
        processed_jobs = []

        def mock_stage(job, ctx):
            processed_jobs.append(job.job_id)
            return ctx

        orch = Orchestrator(
            stages=[("test_stage", mock_stage)],
            max_concurrent=1,
        )
        orch.start()

        job = self._make_job(temp_dir, sample_pdf_bytes)
        orch.enqueue(job)

        assert orch.queue_size == 1

        result = orch.process_next()

        assert result is not None
        assert result.job_id == job.job_id
        assert result.state == JobState.COMPLETED
        assert orch.queue_size == 0
        assert len(processed_jobs) == 1

    def test_stage_failure(self, temp_dir, sample_pdf_bytes):
        """Test that stage failure marks job as failed."""
        def failing_stage(job, ctx):
            raise StageError("Test failure", stage="test")

        orch = Orchestrator(
            stages=[("failing", failing_stage)],
            max_concurrent=1,
            max_retries=0,  # No retries
        )
        orch.start()

        job = self._make_job(temp_dir, sample_pdf_bytes)
        orch.enqueue(job)

        result = orch.process_next()

        assert result.state == JobState.FAILED
        assert len(result.context.errors) > 0

    def test_validation_error_no_retry(self, temp_dir, sample_pdf_bytes):
        """Test that validation errors fail immediately."""
        def validation_stage(job, ctx):
            raise StageValidationError("Invalid file", stage="validate")

        orch = Orchestrator(
            stages=[("validate", validation_stage)],
            max_concurrent=1,
            max_retries=3,
        )
        orch.start()

        job = self._make_job(temp_dir, sample_pdf_bytes)
        orch.enqueue(job)

        result = orch.process_next()

        assert result.state == JobState.FAILED
        assert "Invalid file" in result.context.errors[0]

    def test_retry_logic(self, temp_dir, sample_pdf_bytes):
        """Test that retryable errors are retried."""
        attempt_count = 0

        def flaky_stage(job, ctx):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise StageError("Transient error", stage="flaky")
            return ctx

        orch = Orchestrator(
            stages=[("flaky", flaky_stage)],
            max_concurrent=1,
            max_retries=3,
            retry_delay=0,  # No delay for testing
        )
        orch.start()

        job = self._make_job(temp_dir, sample_pdf_bytes)
        orch.enqueue(job)

        result = orch.process_next()

        assert result.state == JobState.COMPLETED
        assert attempt_count == 3

    def test_exhaust_retries(self, temp_dir, sample_pdf_bytes):
        """Test that job fails after exhausting retries."""
        def always_failing_stage(job, ctx):
            raise StageError("Permanent error", stage="fail")

        orch = Orchestrator(
            stages=[("fail", always_failing_stage)],
            max_concurrent=1,
            max_retries=2,
            retry_delay=0,
        )
        orch.start()

        job = self._make_job(temp_dir, sample_pdf_bytes)
        orch.enqueue(job)

        result = orch.process_next()

        assert result.state == JobState.FAILED
        assert "2 retries" in result.context.errors[0]

    def test_multiple_stages(self, temp_dir, sample_pdf_bytes):
        """Test that multiple stages are executed in order."""
        stage_order = []

        def stage1(job, ctx):
            stage_order.append("stage1")
            return ctx

        def stage2(job, ctx):
            stage_order.append("stage2")
            return ctx

        def stage3(job, ctx):
            stage_order.append("stage3")
            return ctx

        orch = Orchestrator(
            stages=[("s1", stage1), ("s2", stage2), ("s3", stage3)],
            max_concurrent=1,
        )
        orch.start()

        job = self._make_job(temp_dir, sample_pdf_bytes)
        orch.enqueue(job)

        result = orch.process_next()

        assert result.state == JobState.COMPLETED
        assert stage_order == ["stage1", "stage2", "stage3"]

    def test_state_transitions(self):
        """Test valid state transitions."""
        transitions = Orchestrator.TRANSITIONS

        assert JobState.VALIDATING in transitions[JobState.QUEUED]
        assert JobState.FAILED in transitions[JobState.QUEUED]
        assert JobState.PROCESSING in transitions[JobState.VALIDATING]
        assert JobState.DELIVERING in transitions[JobState.PROCESSING]
        assert JobState.COMPLETED in transitions[JobState.DELIVERING]
        assert transitions[JobState.COMPLETED] == set()
        assert transitions[JobState.FAILED] == set()

    def test_next_state_mapping(self):
        """Test stage name to state mapping."""
        assert Orchestrator._next_state("validate") == JobState.VALIDATING
        assert Orchestrator._next_state("ocr") == JobState.PROCESSING
        assert Orchestrator._next_state("output") == JobState.DELIVERING
        assert Orchestrator._next_state("lifecycle") == JobState.DELIVERING
        assert Orchestrator._next_state("unknown") == JobState.PROCESSING