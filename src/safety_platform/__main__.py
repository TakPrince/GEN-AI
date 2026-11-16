"""`python -m safety_platform` entry point (P2-04 CLI)."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
