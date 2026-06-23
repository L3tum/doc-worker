# =============================================================================
# Doc-Worker — Dockerfile (Phase 4-6: PaddleOCR-VL)
# =============================================================================
# Multi-stage: CPU (default) or GPU build.
#
# Usage:
#   CPU (default, tags :latest/:cpu):
#     docker build -t doc-worker:latest -t doc-worker:cpu .
#
#   CUDA GPU (NVIDIA, tag :cuda):
#     docker build --build-arg ONNX_RUNTIME=cuda -t doc-worker:cuda .
#
#   OpenVINO (Intel GPU/CPU, experimental, tag :openvino):
#     docker build --build-arg ONNX_RUNTIME=openvino -t doc-worker:openvino .
#
#   ROCm (AMD GPU, experimental, tag :rocm):
#     docker build --build-arg ONNX_RUNTIME=rocm -t doc-worker:rocm .
#
# The ONNX_RUNTIME build arg controls the ONNX Runtime package installed:
#   cpu       — onnxruntime (CPU-only, smallest image)
#   cuda      — onnxruntime-gpu (CUDA 12.8.1 + cuDNN, NVIDIA GPU)
#   openvino  — onnxruntime-openvino (Intel GPU/CPU via OpenVINO)
#   rocm      — onnxruntime-rocm (AMD GPU via ROCm)
#
# Modes (via CMD or --mode flag):
#   combined  — folder watcher + API server (default)
#   folder    — folder watcher only
#   api       — API server only
# =============================================================================

# ---------------------------------------------------------------------------
# Select base image based on runtime
# ---------------------------------------------------------------------------
ARG ONNX_RUNTIME=cpu

# CPU base (default)
FROM python:3.12-slim-bookworm AS base-cpu

# CUDA base (NVIDIA GPU) — ONNX Runtime PyPI GPU wheels require CUDA 12.x + cuDNN 9.x.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS base-cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# OpenVINO base (Intel GPU/CPU)
FROM python:3.12-slim-bookworm AS base-openvino

# ROCm base (AMD GPU)
FROM python:3.12-slim-bookworm AS base-rocm

# Pick the correct base
FROM base-${ONNX_RUNTIME} AS base

ARG ONNX_RUNTIME=cpu

# Ubuntu 24.04 (CUDA base) enforces PEP 668 — allow pip to install system-wide.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    ghostscript \
    fonts-dejavu \
    fonts-noto-cjk \
    # PDF processing helpers
    qpdf libgl1 \
    # PaddleOCR dependencies
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Swap in GPU runtime if requested
# NOTE: onnxruntime-gpu >=1.27 requires CUDA 13 (libcudart.so.13), but our
# base image (nvidia/cuda:12.8.1) only provides CUDA 12. Pin to <1.27 to keep
# CUDA 12 compatibility.
RUN if [ "$ONNX_RUNTIME" = "cuda" ]; then \
      pip install --no-cache-dir 'onnxruntime-gpu<1.27'; \
    elif [ "$ONNX_RUNTIME" = "openvino" ]; then \
      pip install --no-cache-dir onnxruntime-openvino; \
    elif [ "$ONNX_RUNTIME" = "rocm" ]; then \
      pip install --no-cache-dir onnxruntime-rocm; \
    fi

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
WORKDIR /app
COPY . .

# ---------------------------------------------------------------------------
# Pre-download PaddleOCR models (avoids first-run download delay / offline fail)
# ---------------------------------------------------------------------------
RUN python3 << 'PYEOF'
import os
os.environ['PADDLE_DEVICE'] = 'cpu'
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False, use_gpu=False)
    print('PP-OCR models pre-downloaded successfully')
except Exception as e:
    print(f'PP-OCR pre-download skipped: {e}')
PYEOF

# ---------------------------------------------------------------------------
# Runtime metadata label
# ---------------------------------------------------------------------------
LABEL onnx-runtime="${ONNX_RUNTIME}"
LABEL paddle-ocr="pp-ocr-v6 + paddleocr-vl"

# ---------------------------------------------------------------------------
# Expose API port
# ---------------------------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ---------------------------------------------------------------------------
# Entrypoint — unified pipeline (combined mode: folder + API)
# ---------------------------------------------------------------------------
CMD ["python", "-u", "main.py", "--mode", "combined"]