"""
Doc-Worker — Security Hardening Module
========================================

Provides security utilities for production deployment:
- Input validation and sanitization
- Rate limiting
- Request size limits
- Secure headers
- CORS configuration

Phase 6: Security hardening for production deployment.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import starlette.middleware.base


# ---------------------------------------------------------------------------
# Rate Limiting Middleware
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple in-memory rate limiter.

    Uses a sliding window algorithm to limit requests per IP address.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window.
            window_seconds: Window size in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if a request from the given key is allowed.

        Args:
            key: Typically the client IP address.

        Returns:
            True if the request is allowed, False otherwise.
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old entries
        self._requests[key] = [
            t for t in self._requests[key] if t > window_start
        ]

        if len(self._requests[key]) >= self.max_requests:
            return False

        self._requests[key].append(now)
        return True

    def cleanup(self) -> None:
        """Remove expired entries."""
        now = time.time()
        window_start = now - self.window_seconds

        keys_to_remove = [
            key for key, times in self._requests.items()
            if not any(t > window_start for t in times)
        ]
        for key in keys_to_remove:
            del self._requests[key]


_rate_limiter: RateLimiter | None = None


def get_rate_limiter(max_requests: int = 60, window_seconds: int = 60) -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_requests, window_seconds)
    return _rate_limiter


async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting."""
    # Skip rate limiting for health checks
    if request.url.path in ("/health", "/ready", "/metrics"):
        return await call_next(request)

    # Get client IP
    client_ip = request.client.host if request.client else "unknown"

    # Check rate limit
    limiter = get_rate_limiter()
    if not limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Try again later."},
        )

    response = await call_next(request)
    return response


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".webp"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def validate_filename(filename: str) -> str:
    """Validate and sanitize a filename.

    Args:
        filename: The filename to validate.

    Returns:
        Sanitized filename.

    Raises:
        ValueError: If the filename is invalid.
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    # Check for path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("Filename contains invalid characters")

    # Limit filename length
    if len(filename) > 255:
        raise ValueError("Filename too long")

    return filename


def validate_file_size(size: int, max_size: int = MAX_FILE_SIZE) -> None:
    """Validate file size.

    Args:
        size: File size in bytes.
        max_size: Maximum allowed size in bytes.

    Raises:
        ValueError: If the file is too large.
    """
    if size > max_size:
        max_mb = max_size / (1024 * 1024)
        raise ValueError(f"File too large: {size / (1024 * 1024):.1f}MB > {max_mb:.0f}MB")


def compute_content_hash(data: bytes) -> str:
    """Compute SHA-256 hash of file content.

    Args:
        data: File content bytes.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Secure Headers Middleware
# ---------------------------------------------------------------------------

SECURE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


async def secure_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    for header, value in SECURE_HEADERS.items():
        response.headers[header] = value

    return response


# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------

def configure_cors(app: FastAPI, allowed_origins: list[str] | None = None) -> None:
    """Configure CORS for the FastAPI app.

    Args:
        app: The FastAPI application.
        allowed_origins: List of allowed origins. If None, allows all origins.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Trusted Host Middleware
# ---------------------------------------------------------------------------

def configure_trusted_hosts(
    app: FastAPI, allowed_hosts: list[str] | None = None
) -> None:
    """Configure trusted host middleware.

    Args:
        app: The FastAPI application.
        allowed_hosts: List of allowed host patterns.
    """
    if allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=allowed_hosts,
        )


# ---------------------------------------------------------------------------
# Apply Security Hardening
# ---------------------------------------------------------------------------

def apply_security_hardening(
    app: FastAPI,
    rate_limit_max: int = 60,
    rate_limit_window: int = 60,
    allowed_origins: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
) -> None:
    """Apply all security hardening measures to the app.

    Args:
        app: The FastAPI application.
        rate_limit_max: Maximum requests per window.
        rate_limit_window: Window size in seconds.
        allowed_origins: CORS allowed origins.
        allowed_hosts: Trusted host patterns.
    """
    # Rate limiting
    global _rate_limiter
    _rate_limiter = RateLimiter(rate_limit_max, rate_limit_window)

    app.middleware("http")(rate_limit_middleware)
    app.middleware("http")(secure_headers_middleware)

    # CORS
    configure_cors(app, allowed_origins)

    # Trusted hosts
    configure_trusted_hosts(app, allowed_hosts)