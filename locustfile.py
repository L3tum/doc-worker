"""
Locust load testing file for Doc-Worker API.

Usage:
    locust -f locustfile.py --host http://localhost:8000

This will start the Locust web UI at http://localhost:8089 where you can
configure and run load tests.

Command-line usage (headless):
    locust -f locustfile.py --host http://localhost:8000 \
        --users 10 --spawn-rate 2 --run-time 5m --headless
"""

from __future__ import annotations

from locust import HttpUser, task, between

# Generate a minimal PDF for testing
MINIMAL_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer << /Size 4 /Root 1 0 R >>
startxref
190
%%EOF"""

# Generate a minimal PNG for testing (1x1 pixel)
MINIMAL_PNG = bytes(
    [
        0x89,
        0x50,
        0x4E,
        0x47,
        0x0D,
        0x0A,
        0x1A,
        0x0A,
        0x00,
        0x00,
        0x00,
        0x0D,
        0x49,
        0x48,
        0x44,
        0x52,
        0x00,
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x01,
        0x08,
        0x02,
        0x00,
        0x00,
        0x00,
        0x90,
        0x77,
        0x53,
        0xDE,
        0x00,
        0x00,
        0x00,
        0x0C,
        0x49,
        0x44,
        0x41,
        0x54,
        0x08,
        0xD7,
        0x63,
        0xF8,
        0xFF,
        0xFF,
        0xFF,
        0x00,
        0x05,
        0xFE,
        0x02,
        0xFE,
        0xA7,
        0x9E,
        0x9D,
        0x89,
        0x00,
        0x00,
        0x00,
        0x00,
        0x49,
        0x45,
        0x4E,
        0x44,
        0xAE,
        0x42,
        0x60,
        0x82,
    ]
)


class DocWorkerUser(HttpUser):
    """Simulated user for Doc-Worker API load testing."""

    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks

    @task(3)
    def convert_pdf_auto(self):
        """Submit a PDF for OCR processing (auto mode)."""
        with self.client.post(
            "/api/v1/convert",
            files={"file": ("test.pdf", MINIMAL_PDF, "application/pdf")},
            data={"mode": "auto"},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)
    def convert_pdf_pp_ocr(self):
        """Submit a PDF for OCR processing (PP-OCR mode)."""
        with self.client.post(
            "/api/v1/convert",
            files={"file": ("test.pdf", MINIMAL_PDF, "application/pdf")},
            data={"mode": "pp_ocr"},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def convert_image(self):
        """Submit an image for OCR processing."""
        with self.client.post(
            "/api/v1/convert",
            files={"file": ("test.png", MINIMAL_PNG, "image/png")},
            data={"mode": "auto"},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def health_check(self):
        """Check health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy: {data}")
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def readiness_check(self):
        """Check readiness endpoint."""
        with self.client.get("/ready", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("ready"):
                    response.success()
                else:
                    response.failure(f"Not ready: {data}")
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def metrics_check(self):
        """Check metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    def on_start(self):
        """Called when a user starts."""
        # Initial health check
        self.client.get("/health")
