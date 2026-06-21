"""
Doc-Worker — server:companion (Shared Backend)
================================================

Provides:
- Configuration management (env vars → validated config object)
- Structured JSON logging
- Health check endpoints
- Model lifecycle (lazy-loading, inference abstraction)
- File I/O abstraction (filesystem / in-memory)
- Metrics tracking
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from server.models import JobContext, ProcessingMode


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Validated configuration from environment variables."""

    # Directory paths
    INBOX: Path = Path("/work/inbox")
    PROCESSING: Path = Path("/work/processing")
    DONE: Path = Path("/work/done")
    ERROR: Path = Path("/work/error")
    DOCLING_OUT: Path = Path("/work/docling")
    PAPERLESS_CONSUME: Path = Path("/paperless-consume")

    # OCR settings
    OCR_LANG: str = "deu"
    OCR_RUNTIME: str = "cpu"  # cpu | cuda | openvino | rocm

    # Processing mode
    PROCESSING_MODE: str = "auto"  # auto | pp_ocr | paddle_vl

    # Pipeline settings
    POLL_INTERVAL: int = 5
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 10
    STABILITY_TIMEOUT: int = 10
    MAX_FILE_SIZE_MB: int = 100
    MAX_CONCURRENT_JOBS: int = 1

    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_ENABLED: bool = True

    # Docling (legacy compatibility)
    DOCLING_BASE_URL: str = ""
    DOCLING_MODE: str = "off"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True

    @classmethod
    def from_env(cls) -> Config:
        """Read configuration from environment variables."""
        config = cls()

        # Directory paths
        for attr in ("INBOX", "PROCESSING", "DONE", "ERROR",
                      "DOCLING_OUT", "PAPERLESS_CONSUME"):
            val = os.getenv(attr)
            if val:
                setattr(config, attr, Path(val))

        # OCR settings
        config.OCR_LANG = os.getenv("OCR_LANG", config.OCR_LANG)
        config.OCR_RUNTIME = os.getenv("OCR_RUNTIME", "cpu").lower()

        # Processing mode
        config.PROCESSING_MODE = os.getenv("PROCESSING_MODE", "auto")

        # Pipeline settings
        config.POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
        config.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        config.RETRY_DELAY = int(os.getenv("RETRY_DELAY", "10"))
        config.STABILITY_TIMEOUT = int(os.getenv("STABILITY_TIMEOUT", "10"))
        config.MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
        config.MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

        # API settings
        config.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        config.API_PORT = int(os.getenv("API_PORT", "8000"))
        config.API_ENABLED = os.getenv("API_ENABLED", "true").lower() != "false"

        # Docling (legacy)
        config.DOCLING_BASE_URL = os.getenv("DOCLING_BASE_URL", "")
        config.DOCLING_MODE = os.getenv("DOCLING_MODE", "off")

        # Logging
        config.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        config.LOG_JSON = os.getenv("LOG_JSON", "true").lower() == "true"

        return config

    def ensure_directories(self) -> None:
        """Create all required directories."""
        for path in (self.INBOX, self.PROCESSING, self.DONE,
                      self.ERROR, self.DOCLING_OUT, self.PAPERLESS_CONSUME):
            path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_logger: logging.Logger | None = None


def get_logger(name: str = "doc-worker") -> logging.Logger:
    """Get or create a structured logger."""
    global _logger
    if _logger is not None:
        return logging.getLogger(name)

    _logger = logging.getLogger(name)
    _logger.setLevel(get_config().LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    if get_config().LOG_JSON:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

    _logger.addHandler(handler)
    return _logger


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Add extra fields if present
        if hasattr(record, "job_id"):
            log_entry["job_id"] = record.job_id
        if hasattr(record, "stage"):
            log_entry["stage"] = record.stage
        if hasattr(record, "source"):
            log_entry["source"] = record.source
        if hasattr(record, "filename"):
            log_entry["filename"] = record.filename
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def log_with_context(
    logger: logging.Logger, level: int, msg: str,
    job_id: str | None = None, stage: str | None = None,
    source: str | None = None, filename: str | None = None,
) -> None:
    """Log with structured context fields."""
    extra = {}
    if job_id:
        extra["job_id"] = job_id
    if stage:
        extra["stage"] = stage
    if source:
        extra["source"] = source
    if filename:
        extra["filename"] = filename
    logger.log(level, msg, extra=extra)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    """Simple in-process metrics tracker."""

    files_processed: int = 0
    files_failed: int = 0
    total_processing_time: float = 0.0
    stage_timings: dict[str, list[float]] = field(default_factory=dict)
    model_load_time: float | None = None
    errors: list[str] = field(default_factory=list)

    def record_stage_timing(self, stage: str, duration: float) -> None:
        self.stage_timings.setdefault(stage, []).append(duration)

    def record_success(self, elapsed: float) -> None:
        self.files_processed += 1
        self.total_processing_time += elapsed

    def record_failure(self, error: str) -> None:
        self.files_failed += 1
        self.errors.append(error)

    def summary(self) -> dict[str, Any]:
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "total_processing_time": round(self.total_processing_time, 2),
            "avg_processing_time": (
                round(self.total_processing_time / max(self.files_processed, 1), 2)
            ),
            "model_load_time": self.model_load_time,
            "errors": self.errors[-10:],  # last 10
        }


# ---------------------------------------------------------------------------
# Model Lifecycle
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages PaddleOCR model lifecycle.

    Handles lazy-loading, warm-up, and provides a unified inference interface.
    """

    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._model_type: str | None = None
        self._loaded = False
        self._logger = get_logger("doc-worker.model")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, model_type: str | None = None) -> None:
        """Load the specified model.

        Args:
            model_type: "pp_ocr" for PP-OCRv6, "paddle_vl" for PaddleOCR-VL.
                        If None, uses config.PROCESSING_MODE.
        """
        if self._loaded:
            if model_type and model_type != self._model_type:
                self._logger.warning(
                    f"Model already loaded as {self._model_type}, "
                    f"ignoring request for {model_type}"
                )
            return

        start = time.time()
        self._logger.info(f"Loading model: {model_type or self.config.PROCESSING_MODE}")

        try:
            self._model_type = model_type or self.config.PROCESSING_MODE
            self._model = self._load_model(self._model_type)
            self._loaded = True
            load_time = time.time() - start
            self._logger.info(f"Model loaded in {load_time:.2f}s: {self._model_type}")
        except Exception as exc:
            self._logger.error(f"Failed to load model: {exc}")
            raise

    def _load_model(self, model_type: str) -> Any:
        """Load the actual model. Returns the model instance."""
        # This is a placeholder that will be filled in during Phase 4
        # when we integrate the actual PaddleOCR models.
        # For now, it returns a mock that can be replaced.
        if model_type == "paddle_vl":
            self._logger.info("PaddleOCR-VL integration pending (Phase 4)")
            return None
        elif model_type == "pp_ocr":
            self._logger.info("PP-OCRv6 integration pending (Phase 4)")
            return None
        else:
            self._logger.info(f"Using auto mode (model type: {model_type})")
            return None

    def infer(self, input_data: bytes | Path, ctx: JobContext) -> dict[str, Any]:
        """Run inference on the given input.

        Args:
            input_data: Either a file path or raw bytes.
            ctx: The job context for tracking results.

        Returns:
            Dict with inference results (text, layout, confidence, etc.)
        """
        if not self._loaded:
            self.load()

        start = time.time()
        try:
            result = self._run_inference(input_data)
            duration = time.time() - start
            ctx.timings["inference"] = duration
            return result
        except Exception as exc:
            self._logger.error(f"Inference failed: {exc}")
            raise

    def _run_inference(self, input_data: bytes | Path) -> dict[str, Any]:
        """Run the actual inference. Placeholder for Phase 4."""
        # This will be replaced with actual PaddleOCR inference code
        return {
            "text": "",
            "layout": [],
            "confidence": 0.0,
            "model": self._model_type,
        }

    def unload(self) -> None:
        """Unload the model and free resources."""
        if self._loaded:
            self._model = None
            self._model_type = None
            self._loaded = False
            self._logger.info("Model unloaded")


# ---------------------------------------------------------------------------
# File I/O Abstraction
# ---------------------------------------------------------------------------

class FileIO:
    """Unified file I/O abstraction supporting disk and memory."""

    @staticmethod
    def read(input_path: Path | None, input_data: bytes | None) -> bytes:
        """Read file content from path or return in-memory data."""
        if input_data is not None:
            return input_data
        if input_path is not None and input_path.exists():
            return input_path.read_bytes()
        raise ValueError("No input data available (neither path nor bytes)")

    @staticmethod
    def write(path: Path, data: bytes) -> None:
        """Write bytes to a file path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    @staticmethod
    def write_text(path: Path, text: str) -> None:
        """Write text to a file path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def atomic_write(path: Path, data: bytes) -> None:
        """Atomic write: write to .tmp then rename."""
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            FileIO.write(tmp_path, data)
            os.replace(str(tmp_path), str(path))
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise


# ---------------------------------------------------------------------------
# Health Checks
# ---------------------------------------------------------------------------

@dataclass
class HealthStatus:
    """Health check result."""

    status: str = "healthy"  # healthy | degraded | unhealthy
    ready: bool = True
    model_loaded: bool = False
    uptime: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

_config: Config | None = None
_metrics: Metrics | None = None
_model_manager: ModelManager | None = None
_start_time: float = time.time()


def get_config() -> Config:
    """Get the global configuration (lazy-init)."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def get_metrics() -> Metrics:
    """Get the global metrics tracker (lazy-init)."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


def get_model_manager() -> ModelManager:
    """Get the global model manager (lazy-init)."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(get_config())
    return _model_manager


def get_health() -> HealthStatus:
    """Get current health status."""
    model_mgr = get_model_manager()
    return HealthStatus(
        status="healthy",
        ready=True,
        model_loaded=model_mgr.is_loaded,
        uptime=time.time() - _start_time,
        metrics=get_metrics().summary(),
    )


def init_companion() -> None:
    """Initialize the companion module (call once at startup)."""
    config = get_config()
    config.ensure_directories()
    logger = get_logger()
    logger.info("server:companion initialized")
    logger.info(f"Config: INBOX={config.INBOX}, PROCESSING={config.PROCESSING}, "
                f"DONE={config.DONE}, ERROR={config.ERROR}")
    logger.info(f"OCR_RUNTIME={config.OCR_RUNTIME}, PROCESSING_MODE={config.PROCESSING_MODE}")