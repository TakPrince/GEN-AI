"""Bounded dead-letter storage with rotation and non-blocking I/O support.

`write_dead_letter` appends one JSONL record and rotates the active file
when it would exceed its size bound. Rotation never discards or overwrites
an existing rotated file (atomic `os.link` + `os.unlink`), and total
retention is bounded: a write that would exceed the quota is refused, and
the refusal is surfaced by the caller through `events_lost` /
`last_error="dead_letter_full"` — never reported as persisted.

The write runs in a worker thread (`asyncio.to_thread` on the caller
side); a module-level lock serializes append and rotation across every
consumer sharing a dead-letter path, and the worker — not the awaiting
coroutine — releases it, so cancellation cannot corrupt files.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from threading import Lock

from .db_sink import _row
from .events import StageEvent
from .replay import discover_dead_letter_files

# Bounded dead-letter storage defaults: active file and total retention.
# Nothing is ever deleted to free space; over-quota writes are refused.
DEFAULT_DEAD_LETTER_MAX_ACTIVE_BYTES = 8 * 1024 * 1024
DEFAULT_DEAD_LETTER_MAX_TOTAL_BYTES = 64 * 1024 * 1024

# One writer at a time across every consumer sharing a dead-letter path.
_FILE_LOCK = Lock()


def write_dead_letter(
    path: str | Path,
    events: tuple[StageEvent, ...],
    error: str,
    *,
    max_active_bytes: int,
    max_total_bytes: int,
) -> str:
    """Append one JSONL dead-letter record; rotate the active file when full.

    Returns "written" or "refused"; raises OSError on filesystem failure.
    Refusal leaves every existing file untouched (no silent record loss).
    """
    payload = {
        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "error": error,
        "event_ids": [event.event_id for event in events],
        # `rows` mirror the sink schema so replay is copy-paste-safe.
        "rows": [_row(event) for event in events],
    }
    blob = json.dumps(payload, separators=(",", ":")) + "\n"
    size = len(blob.encode("utf-8"))
    active = Path(path)
    with _FILE_LOCK:
        active_size = active.stat().st_size if active.exists() else 0
        rotated_total = 0
        for rotated in discover_dead_letter_files(active)[:-1]:
            rotated_total += rotated.stat().st_size
        if active_size + rotated_total + size > max_total_bytes:
            return "refused"
        if active_size and active_size + size > max_active_bytes:
            _rotate(active)
        with open(active, "a", encoding="utf-8") as handle:
            handle.write(blob)
        return "written"


def _rotate(active: Path) -> None:
    """Rename the active file to the next free `<path>.<n>` slot.

    `os.link` fails atomically when the target exists, so an existing
    rotated file is never overwritten; the linked copy then replaces the
    active file via unlink.
    """
    for index in itertools.count(1):
        target = active.with_name(f"{active.name}.{index}")
        try:
            os.link(active, target)
        except FileExistsError:
            continue
        os.unlink(active)
        return


__all__ = [
    "DEFAULT_DEAD_LETTER_MAX_ACTIVE_BYTES",
    "DEFAULT_DEAD_LETTER_MAX_TOTAL_BYTES",
    "write_dead_letter",
]
