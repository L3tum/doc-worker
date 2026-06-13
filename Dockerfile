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

# Pre-download RapidOCR models for the target language(s)
# This triggers the download of language-specific recognition models at build time,
# avoiding slow first-run downloads at container startup.
RUN python -c "from PIL import Image; Image.new('RGB', (200, 50), 'white').save('/tmp/test.png')" \
    && ocrmypdf --plugin ocrmypdf_rapidocr -l deu --image-dpi 300 -f /tmp/test.png /tmp/test_ocr.pdf \
    && rm -f /tmp/test.png /tmp/test_ocr.pdf \
    && echo "RapidOCR models preloaded successfully."

# Copy worker script
COPY worker.py /usr/local/bin/worker.py
RUN chmod +x /usr/local/bin/worker.py

# Run the worker
ENTRYPOINT ["python", "/usr/local/bin/worker.py"]
