# =============================================================================
# Doc-Worker — Dockerfile
# =============================================================================
# Multi-stage: CPU (default) or CUDA (NVIDIA GPU) build.
#
# Usage:
#   CPU (default, tags :latest/:cpu):
#     docker build -t doc-worker:latest -t doc-worker:cpu .
#
#   CUDA GPU (NVIDIA, tag :cuda):
#     docker build --build-arg PADDLE_GPU=cuda -t doc-worker:cuda .
#
# PaddlePaddle handles GPU detection internally. The PADDLE_GPU build arg
# controls which PaddlePaddle package is installed:
#   cpu  — paddlepaddle (CPU-only, smallest image)
#   cuda — paddlepaddle-gpu (NVIDIA GPU, CUDA 12.9)
#
# Note: ROCm (AMD GPU) is not supported — PaddlePaddle's ROCm wheels are only
# available via their Docker images, not pip.
# =============================================================================

ARG PADDLE_GPU=cpu

# CPU base (default)
FROM python:3.12-slim-bookworm AS base-cpu

# CUDA base (NVIDIA GPU)
FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04 AS base-cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Pick the correct base
FROM base-${PADDLE_GPU} AS base

ARG PADDLE_GPU=cpu

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
    # PDF processing helpers — tesseract-ocr required by OCRmyPDF at import time
    qpdf libgl1 tesseract-ocr \
    # tini for process supervision, wget for model downloads
    tini wget \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Swap in GPU PaddlePaddle if requested
# Note: PaddlePaddle GPU wheels are built against CUDA 12.9 (cu129), and the
# base image uses CUDA 12.9.2 for compatible runtime.
RUN if [ "$PADDLE_GPU" = "cuda" ]; then \
      pip uninstall -y paddlepaddle && \
      pip install --no-cache-dir paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/; \
    fi

# ---------------------------------------------------------------------------
# Pre-download PaddleOCR models (bypasses runtime download on first request)
# Uses wget + tar — no Python or PaddleX dependency needed
# ---------------------------------------------------------------------------
ENV PADDLEOCR_MODELS=/app/models

# PaddleX / PaddleOCR cache — must be writable and set before importing PaddleOCR
ENV HOME=/tmp
ENV PADDLE_PDX_CACHE_HOME=/tmp/.paddlex

# Verify after download that inference.yml keeps Paddle's logical model names
# without the *_infer directory suffix.
RUN mkdir -p "${PADDLEOCR_MODELS}" "${PADDLE_PDX_CACHE_HOME}" && \
    chmod 1777 "${PADDLE_PDX_CACHE_HOME}" && \
    base="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0" && \
    for name in PP-OCRv6_medium_det_infer PP-OCRv6_medium_rec_infer PP-LCNet_x1_0_textline_ori_infer PP-DocLayout-L_infer; do \
        echo "Downloading ${name}..." && \
        wget -q "${base}/${name}.tar" -O "/tmp/${name}.tar" && \
        tar -xf "/tmp/${name}.tar" -C "${PADDLEOCR_MODELS}" && \
        rm "/tmp/${name}.tar"; \
    done && \
    for spec in \
        "PP-OCRv6_medium_det_infer PP-OCRv6_medium_det" \
        "PP-OCRv6_medium_rec_infer PP-OCRv6_medium_rec" \
        "PP-LCNet_x1_0_textline_ori_infer PP-LCNet_x1_0_textline_ori" \
        "PP-DocLayout-L_infer PP-DocLayout-L"; do \
        set -- ${spec}; dir="$1"; model="$2"; \
        if ! grep -Eq "^[[:space:]]*model_name:[[:space:]]*${model}$" "${PADDLEOCR_MODELS}/${dir}/inference.yml"; then \
            echo "ERROR: ${dir}/inference.yml does not declare model_name: ${model}" && exit 1; \
        fi; \
    done

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
WORKDIR /app
COPY . .

# ---------------------------------------------------------------------------
# Expose API ports
# ---------------------------------------------------------------------------
EXPOSE 8000 8001

# ---------------------------------------------------------------------------
# Runtime metadata label
# ---------------------------------------------------------------------------
LABEL paddle-gpu="${PADDLE_GPU}"

# ---------------------------------------------------------------------------
# Health check — verifies the health check server (port 8001) is responsive
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# ---------------------------------------------------------------------------
# Entrypoint — tini manages signals, shell script forwards to child processes
# ---------------------------------------------------------------------------
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", " \
    python -u server.py & SERVER_PID=$! && \
    echo $SERVER_PID > /tmp/doc-server.pid && \
    python -u worker.py & WORKER_PID=$! && \
    echo $WORKER_PID > /tmp/doc-worker.pid && \
    trap 'kill $SERVER_PID $WORKER_PID 2>/dev/null; wait' TERM INT && \
    wait \
"]
