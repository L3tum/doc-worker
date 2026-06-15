# =============================================================================
# Doc-Worker — Dockerfile
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
# OpenVINO EP is provided by pip (onnxruntime-openvino), no system package needed.
FROM python:3.12-slim-bookworm AS base-openvino

# ROCm base (AMD GPU)
FROM python:3.12-slim-bookworm AS base-rocm

# Pick the correct base
FROM base-${ONNX_RUNTIME} AS base

ARG ONNX_RUNTIME=cpu

# Ubuntu 24.04 (CUDA base) enforces PEP 668 — allow pip to install system-wide.
# Safe here: this is a container, not a host system.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    ghostscript \
    fonts-dejavu \
    fonts-noto-cjk \
    # OCRmyPDF helpers (deskew, clean, optimize)
    qpdf unpaper pngquant jbig2dec libgl1 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Swap in GPU runtime if requested
RUN if [ "$ONNX_RUNTIME" = "cuda" ]; then \
      pip install --no-cache-dir onnxruntime-gpu; \
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
# Pre-download RapidOCR models (avoids first-run download delay / offline fail)
# Use the OCRmyPDF plugin helper so model selection matches runtime behavior:
# German maps to RapidOCR's Latin recognition model.
# ---------------------------------------------------------------------------
RUN python -c "from ocrmypdf_rapidocr.engine import get_rapidocr_engine; get_rapidocr_engine('deu', None)"

# ---------------------------------------------------------------------------
# Runtime metadata label
# ---------------------------------------------------------------------------
LABEL onnx-runtime="${ONNX_RUNTIME}"

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
CMD ["python", "-u", "worker.py"]
