# =============================================================================
# Doc-Worker — OCR + Docling pipeline for Paperless-ngx
# =============================================================================
FROM python:3.12-slim-bookworm

# ---------------------------------------------------------------------------
# 1. Build-time constants
# ---------------------------------------------------------------------------
ARG PIP_NO_CACHE_DIR=1

# ---------------------------------------------------------------------------
# 2. System packages (no Tesseract — we use RapidOCR exclusively)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    ghostscript \
    fonts-liberation \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 3. Python packages
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    ocrmypdf \
    ocrmypdf_plugin_rapidocr \
    rapidocr_onnxruntime-gpu

# ---------------------------------------------------------------------------
# 4. Download RapidOCR models from ModelScope
# ---------------------------------------------------------------------------
RUN wget -qO- https://modelscope.cn/api/v1/studio/RapidAI/RapidOCR/repository?Revision=master \
    | jq -r '.entries[].key' \
    | grep -E '\.onnx$' \
    > /tmp/rapidocr_files.txt

# Download every ONNX file into /app/models
RUN mkdir -p /app/models && \
    while IFS= read -r file; do \
        echo ">> Downloading $file"; \
        wget -q "https://modelscope.cn/api/v1/studio/RapidAI/RapidOCR/repository?Revision=master&FilePath=/$file" \
            -O "/app/models/$file"; \
    done < /tmp/rapidocr_files.txt

# ---------------------------------------------------------------------------
# 5. Discover model paths & build RapidOCR YAML config at build time
# ---------------------------------------------------------------------------
# Detect the revision (e.g. "ppocr-v5") from the filename pattern
#   det_server_ppocrv5_infer.onnx  ->  ppocr-v5
#   rec_server_ppocrv5_infer.onnx  ->  ppocr-v5
#   cls_server_ppocrv3_infer.onnx  ->  ppocr-v3  (classification)
RUN DET_MODEL=$(ls /app/models/det_server_*_infer.onnx | head -1) && \
    REC_MODEL=$(ls /app/models/rec_server_*_infer.onnx | head -1) && \
    CLS_MODEL=$(ls /app/models/cls_server_*_infer.onnx | head -1) && \
    DET_REV=$(echo "$DET_MODEL" | grep -oP 'ppocr\Kv\d+') && \
    REC_REV=$(echo "$REC_MODEL" | grep -oP 'ppocr\Kv\d+') && \
    CLS_REV=$(echo "$CLS_MODEL" | grep -oP 'ppocr\Kv\d+') && \
    printf '[RapidOCR]\n'\
           'det_model_path = /app/models/det_server_ppocrv%s_infer.onnx\n'\
           'rec_model_path = /app/models/rec_server_ppocrv%s_infer.onnx\n'\
           'cls_model_path = /app/models/cls_server_ppocrv%s_infer.onnx\n' \
           "$DET_REV" "$REC_REV" "$CLS_REV" > /app/rapidocr.yaml

# ---------------------------------------------------------------------------
# 6. Verify the plugin loads
# ---------------------------------------------------------------------------
RUN ocrmypdf --help > /dev/null 2>&1 || { echo "ocrmypdf failed!"; exit 1; }

# ---------------------------------------------------------------------------
# 7. Application code
# ---------------------------------------------------------------------------
WORKDIR /app
COPY worker.py .
RUN chmod +x worker.py

# ---------------------------------------------------------------------------
# 8. Volumes & metadata
# ---------------------------------------------------------------------------
VOLUME ["/work", "/paperless-consume"]
LABEL maintainer="l3tum"
LABEL description="OCR pipeline worker: inbox -> RapidOCR -> Paperless-ngx"

# ---------------------------------------------------------------------------
# 9. Healthcheck — verifies the worker process is alive
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import os, signal; os.kill(1, 0)" || exit 1

# ---------------------------------------------------------------------------
# 10. Run
# ---------------------------------------------------------------------------
CMD ["python3", "worker.py"]
