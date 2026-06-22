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

Phase 4: PaddleOCR-VL integration replaces RapidOCR as the primary OCR engine.
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
    OUTPUT_DIR: Path = Path("/work/output")
    PAPERLESS_CONSUME: Path = Path("/paperless-consume")

    # OCR settings
    OCR_LANG: str = "deu"
    OCR_RUNTIME: str = "cpu"  # cpu | cuda | openvino | rocm

    # Processing mode
    PROCESSING_MODE: str = "auto"  # auto | pp_ocr | paddle_vl

    # PaddleOCR settings
    PADDLE_OCR_USE_ANGLE: bool = True
    PADDLE_OCR_USE_DLL: bool = False  # Differential evolution for layout
    PADDLE_OCR_USE_DB: bool = True    # Detection algorithm
    PADDLE_OCR_USE_REC: bool = True   # Recognition
    PADDLE_OCR_LANG: str = "ch"       # ch | en | french | german | japan | korean
    PADDLE_VL_MODEL: str = "PaddleOCR-VL-1.5B"  # Default VL model
    PADDLE_DEVICE: str = ""           # auto | cpu | gpu | xpu

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

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True

    @classmethod
    def from_env(cls) -> Config:
        """Read configuration from environment variables."""
        config = cls()

        # Directory paths
        for attr in ("INBOX", "PROCESSING", "DONE", "ERROR",
                      "OUTPUT_DIR", "PAPERLESS_CONSUME"):
            val = os.getenv(attr)
            if val:
                setattr(config, attr, Path(val))

        # OCR settings
        config.OCR_LANG = os.getenv("OCR_LANG", config.OCR_LANG)
        config.OCR_RUNTIME = os.getenv("OCR_RUNTIME", "cpu").lower()

        # Processing mode
        config.PROCESSING_MODE = os.getenv("PROCESSING_MODE", "auto")

        # PaddleOCR settings
        config.PADDLE_OCR_USE_ANGLE = os.getenv("PADDLE_OCR_USE_ANGLE", "true").lower() == "true"
        config.PADDLE_OCR_USE_DLL = os.getenv("PADDLE_OCR_USE_DLL", "false").lower() == "true"
        config.PADDLE_OCR_USE_DB = os.getenv("PADDLE_OCR_USE_DB", "true").lower() == "true"
        config.PADDLE_OCR_USE_REC = os.getenv("PADDLE_OCR_USE_REC", "true").lower() == "true"
        config.PADDLE_OCR_LANG = os.getenv("PADDLE_OCR_LANG", "ch")
        config.PADDLE_VL_MODEL = os.getenv("PADDLE_VL_MODEL", "PaddleOCR-VL-1.5B")
        config.PADDLE_DEVICE = os.getenv("PADDLE_DEVICE", "auto")

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

        # Logging
        config.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        config.LOG_JSON = os.getenv("LOG_JSON", "true").lower() == "true"

        return config

    def ensure_directories(self) -> None:
        """Create all required directories."""
        for path in (self.INBOX, self.PROCESSING, self.DONE,
                      self.ERROR, self.OUTPUT_DIR, self.PAPERLESS_CONSUME):
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
# Model Lifecycle — PaddleOCR-VL Integration
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages PaddleOCR model lifecycle.

    Handles lazy-loading, warm-up, and provides a unified inference interface
    for both PP-OCRv6 (text recognition) and PaddleOCR-VL (document understanding).

    Architecture:
    - PP-OCRv6: Detection (DB) + Recognition (CRNN/SVTR) for text extraction
    - PaddleOCR-VL: Vision-Language model for layout analysis, table detection,
      formula recognition, and markdown generation

    The model selection is driven by PROCESSING_MODE config:
    - "auto": Use PP-OCR for text, PaddleOCR-VL for complex documents
    - "pp_ocr": Use only PP-OCRv6 (fast, text-only)
    - "paddle_vl": Use PaddleOCR-VL for full document understanding
    """

    def __init__(self, config: Config):
        self.config = config
        self._pp_ocr_engine = None
        self._vl_model = None
        self._vl_processor = None
        self._model_type: str | None = None
        self._loaded = False
        self._logger = get_logger("doc-worker.model")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, model_type: str | None = None) -> None:
        """Load the specified model(s).

        Args:
            model_type: "pp_ocr" for PP-OCRv6, "paddle_vl" for PaddleOCR-VL,
                        or "auto" for both. If None, uses config.PROCESSING_MODE.
        """
        if self._loaded:
            if model_type and model_type != self._model_type:
                self._logger.warning(
                    f"Model already loaded as {self._model_type}, "
                    f"ignoring request for {model_type}"
                )
            return

        start = time.time()
        effective_type = model_type or self.config.PROCESSING_MODE
        self._logger.info(f"Loading model(s): {effective_type}")

        try:
            self._model_type = effective_type

            # Always load PP-OCR for text recognition
            self._load_pp_ocr()

            # Load PaddleOCR-VL if needed
            if effective_type in ("paddle_vl", "auto"):
                self._load_paddle_vl()

            self._loaded = True
            load_time = time.time() - start
            self._logger.info(f"All models loaded in {load_time:.2f}s: {self._model_type}")

            # Record metrics
            metrics = get_metrics()
            metrics.model_load_time = load_time

        except Exception as exc:
            self._logger.error(f"Failed to load model(s): {exc}")
            raise

    def _load_pp_ocr(self) -> None:
        """Load PP-OCRv6 engine for text detection and recognition."""
        self._logger.info("Loading PP-OCRv6 engine...")

        try:
            from paddleocr import PaddleOCR

            # Map our language codes to PaddleOCR language codes
            lang_map = {
                "deu": "german",
                "eng": "en",
                "fra": "french",
                "spa": "spanish",
                "ita": "italian",
                "por": "portuguese",
                "nld": "dutch",
                "pol": "polish",
                "ch_sim": "ch",
                "jpn": "japan",
                "kor": "korean",
                "ch": "ch",
            }
            paddle_lang = lang_map.get(self.config.OCR_LANG, self.config.OCR_LANG)

            # Determine device
            device = self._resolve_device()

            self._pp_ocr_engine = PaddleOCR(
                use_angle_cls=self.config.PADDLE_OCR_USE_ANGLE,
                use_doc_unwarping=False,
                use_dll=self.config.PADDLE_OCR_USE_DLL,
                use_db=self.config.PADDLE_OCR_USE_DB,
                use_rec=self.config.PADDLE_OCR_USE_REC,
                lang=paddle_lang,
                show_log=False,
                use_gpu="gpu" in device.lower(),
                device=device,
            )

            self._logger.info(f"PP-OCRv6 loaded (lang={paddle_lang}, device={device})")

        except ImportError:
            self._logger.warning(
                "paddleocr not installed. PP-OCR will be unavailable. "
                "Install with: pip install paddleocr paddlepaddle"
            )
            self._pp_ocr_engine = None
        except Exception as exc:
            self._logger.error(f"Failed to load PP-OCR: {exc}")
            self._pp_ocr_engine = None

    def _load_paddle_vl(self) -> None:
        """Load PaddleOCR-VL model for document understanding."""
        self._logger.info(f"Loading PaddleOCR-VL model: {self.config.PADDLE_VL_MODEL}")

        try:
            from modelscope import snapshot_download
            from transformers import AutoModel, AutoProcessor

            # Download model from ModelScope
            model_dir = snapshot_download(
                self.config.PADDLE_VL_MODEL,
                cache_dir="/root/.cache/modelscope",
            )

            device = self._resolve_device()

            self._vl_model = AutoModel.from_pretrained(
                model_dir,
                trust_remote_code=True,
            ).eval()

            self._vl_processor = AutoProcessor.from_pretrained(
                model_dir,
                trust_remote_code=True,
            )

            # Move model to device
            if "gpu" in device.lower():
                self._vl_model = self._vl_model.cuda()

            self._logger.info(
                f"PaddleOCR-VL loaded (model={self.config.PADDLE_VL_MODEL}, "
                f"device={device})"
            )

        except ImportError:
            self._logger.warning(
                "transformers/modelscope not installed. PaddleOCR-VL will be unavailable. "
                "Install with: pip install transformers modelscope"
            )
            self._vl_model = None
            self._vl_processor = None
        except Exception as exc:
            self._logger.error(f"Failed to load PaddleOCR-VL: {exc}")
            self._vl_model = None
            self._vl_processor = None

    def _resolve_device(self) -> str:
        """Resolve the device to use for inference."""
        if self.config.PADDLE_DEVICE and self.config.PADDLE_DEVICE != "auto":
            return self.config.PADDLE_DEVICE

        # Auto-detect
        try:
            import paddle
            if paddle.is_compiled_with_cuda():
                return "gpu"
        except Exception:
            pass
        return "cpu"

    def run_ocr(self, input_data: bytes | Path, ctx: JobContext) -> dict[str, Any]:
        """Run PP-OCR text detection and recognition.

        Args:
            input_data: Either a file path or raw bytes.
            ctx: The job context for tracking results.

        Returns:
            Dict with OCR results (text blocks, bounding boxes, confidence).
        """
        if not self._loaded:
            self.load()

        start = time.time()
        try:
            result = self._run_pp_ocr(input_data)
            duration = time.time() - start
            ctx.timings["pp_ocr"] = duration
            self._logger.debug(f"PP-OCR completed in {duration:.2f}s")
            return result
        except Exception as exc:
            self._logger.error(f"PP-OCR failed: {exc}")
            raise

    def run_vl_understanding(self, input_data: bytes | Path, ctx: JobContext) -> dict[str, Any]:
        """Run PaddleOCR-VL document understanding.

        Produces:
        - Structured markdown with layout preservation
        - Table detection and extraction
        - Formula recognition
        - Block-level confidence scores

        Args:
            input_data: Either a file path or raw bytes.
            ctx: The job context for tracking results.

        Returns:
            Dict with VL results (markdown, layout, tables, formulas).
        """
        if not self._loaded:
            self.load()

        if self._vl_model is None or self._vl_processor is None:
            self._logger.warning("PaddleOCR-VL not available, falling back to PP-OCR only")
            return self.run_ocr(input_data, ctx)

        start = time.time()
        try:
            result = self._run_paddle_vl(input_data)
            duration = time.time() - start
            ctx.timings["paddle_vl"] = duration
            self._logger.debug(f"PaddleOCR-VL completed in {duration:.2f}s")
            return result
        except Exception as exc:
            self._logger.error(f"PaddleOCR-VL failed: {exc}")
            # Fallback to PP-OCR
            self._logger.info("Falling back to PP-OCR")
            return self.run_ocr(input_data, ctx)

    def _run_pp_ocr(self, input_data: bytes | Path) -> dict[str, Any]:
        """Run PP-OCR inference on the input."""
        if self._pp_ocr_engine is None:
            raise RuntimeError("PP-OCR engine not loaded")

        # Write to temp file if bytes
        import tempfile
        tmp_path = None
        if isinstance(input_data, bytes):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(input_data)
                tmp_path = Path(f.name)
            input_path = str(tmp_path)
        else:
            input_path = str(input_data)

        try:
            results = self._pp_ocr_engine.ocr(input_path, cls=True)

            # Parse results
            text_blocks = []
            full_text = []

            if results and results[0]:
                for line in results[0]:
                    bbox = line[0]  # bounding box
                    text = line[1][0]  # recognized text
                    confidence = line[1][1]  # confidence score

                    text_blocks.append({
                        "text": text,
                        "bbox": bbox,
                        "confidence": confidence,
                    })
                    full_text.append(text)

            return {
                "text": "\n".join(full_text),
                "text_blocks": text_blocks,
                "block_count": len(text_blocks),
                "avg_confidence": (
                    sum(b["confidence"] for b in text_blocks) / len(text_blocks)
                    if text_blocks else 0.0
                ),
                "model": "pp_ocr",
            }

        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    def _run_paddle_vl(self, input_data: bytes | Path) -> dict[str, Any]:
        """Run PaddleOCR-VL inference for document understanding."""
        if self._vl_model is None or self._vl_processor is None:
            raise RuntimeError("PaddleOCR-VL model not loaded")

        import torch
        import tempfile
        from PIL import Image

        # Write to temp file if bytes
        tmp_path = None
        if isinstance(input_data, bytes):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(input_data)
                tmp_path = Path(f.name)
            image_path = str(tmp_path)
        else:
            image_path = str(input_data)

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Build prompt for document understanding
            system_prompt = (
                "You are a document analysis assistant. Analyze the document and "
                "return structured markdown output. Include:\n"
                "1. Document title and headings\n"
                "2. Body text with proper formatting\n"
                "3. Tables in markdown table format\n"
                "4. Lists with proper indentation\n"
                "5. Mathematical formulas in LaTeX format\n"
                "6. Footnotes and references\n\n"
                "Preserve the document structure and formatting as much as possible."
            )

            user_prompt = "Please analyze this document and extract its content as structured markdown."

            # Process with VL model
            inputs = self._vl_processor(
                image=image,
                text=user_prompt,
                system=system_prompt,
                return_tensors="pt",
            )

            # Move to device
            device = next(self._vl_model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self._vl_model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                    temperature=0.1,
                )

            # Decode response
            response = self._vl_processor.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            # Parse the response into structured components
            parsed = self._parse_vl_response(response)

            return {
                "markdown": parsed["markdown"],
                "layout": parsed["layout"],
                "tables": parsed["tables"],
                "formulas": parsed["formulas"],
                "text": parsed["plain_text"],
                "confidence": parsed.get("confidence", 0.9),
                "model": "paddle_vl",
            }

        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    def _parse_vl_response(self, response: str) -> dict[str, Any]:
        """Parse the VL model response into structured components."""
        import re

        markdown = response
        plain_text = re.sub(r'[#*\-\[\]`_~>]', '', response).strip()

        # Extract tables
        tables = []
        table_pattern = r'\|.*\|\n\|[-\s|:]+\|.*(?:\n\|.*\|)*'
        for match in re.finditer(table_pattern, response):
            tables.append(match.group())

        # Extract formulas (LaTeX)
        formulas = []
        formula_pattern = r'(?:(?:\$)(.*?)(?:\$)|(?:\$\$)(.*?)(?:\$\$))'
        for match in re.finditer(formula_pattern, response):
            formulas.append(match.group(1) or match.group(2))

        # Extract layout blocks
        layout = []
        heading_pattern = r'^(#{1,6}\s+.+)$'
        for i, line in enumerate(response.split('\n')):
            if re.match(heading_pattern, line):
                layout.append({
                    "type": "heading",
                    "level": len(re.match(r'^(#+)', line).group()),
                    "text": line.lstrip('# ').strip(),
                    "line": i,
                })
            elif line.strip().startswith('|') and '|' in line:
                layout.append({
                    "type": "table",
                    "line": i,
                })
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                layout.append({
                    "type": "list_item",
                    "text": line.strip()[2:],
                    "line": i,
                })

        return {
            "markdown": markdown,
            "plain_text": plain_text,
            "tables": tables,
            "formulas": formulas,
            "layout": layout,
        }

    def unload(self) -> None:
        """Unload all models and free resources."""
        if self._loaded:
            self._pp_ocr_engine = None
            self._vl_model = None
            self._vl_processor = None
            self._model_type = None
            self._loaded = False
            self._logger.info("All models unloaded")

            # Force garbage collection for GPU memory
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


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
    pp_ocr_loaded: bool = False
    vl_model_loaded: bool = False
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
        pp_ocr_loaded=model_mgr._pp_ocr_engine is not None,
        vl_model_loaded=model_mgr._vl_model is not None,
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
    logger.info(f"PADDLE_OCR_LANG={config.PADDLE_OCR_LANG}, PADDLE_VL_MODEL={config.PADDLE_VL_MODEL}")