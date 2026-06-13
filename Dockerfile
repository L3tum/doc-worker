# =============================================================================
# Doc-Worker — Dockerfile
# =============================================================================
# Multi-stage: CPU (default) or GPU build.
#
# Usage:
#   CPU (default):
#     docker build -t doc-worker .
#
#   CUDA GPU (NVIDIA):
#     docker build --build-arg ONNX_RUNTIME=cuda -t doc-worker .
#
#   OpenVINO (Intel GPU/CPU):
#     docker build --build-arg ONNX_RUNTIME=openvino -t doc-worker .
#
#   ROCm (AMD GPU):
#     docker build --build-arg ONNX_RUNTIME=rocm -t doc-worker .
#
# The ONNX_RUNTIME build arg controls the ONNX Runtime package installed:
#   cpu       — onnxruntime (CPU-only, smallest image)
#   cuda      — onnxruntime-gpu (CUDA 13.3.0, NVIDIA GPU)
#   openvino  — onnxruntime-openvino (Intel GPU/CPU via OpenVINO)
#   rocm      — onnxruntime-rocm (AMD GPU via ROCm)
# =============================================================================

# ---------------------------------------------------------------------------
# Select base image based on runtime
# ---------------------------------------------------------------------------
ARG ONNX_RUNTIME=cpu

# CPU base (default)
FROM python:3.12-slim-bookworm AS base-cpu

# CUDA base (NVIDIA GPU) — install Python 3.12 on top of CUDA 13.3.0
FROM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu24.04 AS base-cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# OpenVINO base (Intel GPU/CPU)
FROM python:3.12-slim-bookworm AS base-openvino

RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-openvino-runtime \
    && rm -rf /var/lib/apt/lists/*

# ROCm base (AMD GPU)
FROM python:3.12-slim-bookworm AS base-rocm

# Pick the correct base
FROM base-${ONNX_RUNTIME} AS base

ARG ONNX_RUNTIME=cpu

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    ghostscript \
    fonts-dejavu \
    fonts-noto-cjk \
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
# Runtime metadata label
# ---------------------------------------------------------------------------
LABEL onnx-runtime="${ONNX_RUNTIME}"

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
CMD ["python", "-u", "worker.py"]
