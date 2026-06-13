# =============================================================================
# Doc-Worker — Dockerfile
# =============================================================================
# Multi-stage: CPU (default) or CUDA GPU build.
#
# Usage:
#   CPU (default):
#     docker build -t doc-worker .
#
#   CUDA GPU:
#     docker build --build-arg ONNX_RUNTIME=cuda -t doc-worker .
#
# The ONNX_RUNTIME build arg controls the ONNX Runtime package installed:
#   cpu   — onnxruntime (CPU-only, smaller image)
#   cuda  — onnxruntime-gpu (CUDA 12.6.2, requires NVIDIA GPU at runtime)
# =============================================================================

# ---------------------------------------------------------------------------
# Select base image based on runtime
# ---------------------------------------------------------------------------
ARG ONNX_RUNTIME=cpu

# CPU base (default)
FROM python:3.12-slim-bookworm AS base-cpu

# CUDA base (GPU)
FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04 AS base-cuda

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
