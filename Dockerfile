FROM python:3.12-slim-bookworm

# Ensure unbuffered Python output for real-time Docker logs
ENV PYTHONUNBUFFERED=1
# Disable pip cache to reduce image size
ENV PIP_NO_CACHE_DIR=1
# Set working directory
ENV HOME=/opt/doc-worker

# Install system dependencies for OCRmyPDF
# Note: tesseract-ocr is required for deskew/clean/optimization features,
# even when using the RapidOCR plugin (which only replaces the OCR engine).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    qpdf \
    unpaper \
    pngquant \
    jbig2dec \
    libgl1 \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /opt/doc-worker/requirements.txt
RUN pip install --no-cache-dir -r /opt/doc-worker/requirements.txt

# Verify RapidOCR models are available (they are bundled with the package)
RUN python - <<'PY'
import tempfile
from pathlib import Path
from PIL import Image

# Create a minimal test image
tmp = Path(tempfile.mkdtemp()) / "test.png"
img = Image.new("RGB", (200, 50), color="white")
img.save(str(tmp))

# Instantiate RapidOCR to verify models load correctly
from rapidocr import RapidOCR
ocr = RapidOCR()
result = ocr(str(tmp))
print("RapidOCR models loaded successfully.")
PY

# Copy worker script
COPY worker.py /usr/local/bin/worker.py
RUN chmod +x /usr/local/bin/worker.py

# Run the worker
ENTRYPOINT ["python", "/usr/local/bin/worker.py"]
