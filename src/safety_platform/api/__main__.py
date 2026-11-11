"""Run the safety platform API with uvicorn.

    PYTHONPATH=src python -m safety_platform.api

Bind address defaults to localhost. Set `SAFETY_API_HOST=0.0.0.0` only behind a
TLS-terminating reverse proxy that enforces authentication and rate limits.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "safety_platform.api.app:app",
        host=os.environ.get("SAFETY_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("SAFETY_API_PORT", "8000")),
        log_level=os.environ.get("SAFETY_API_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
