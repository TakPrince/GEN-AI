"""Tolerant JSON-object extraction shared by judges and Tier 3 connectors.

Lives outside `evaluation/` so guardrail modules can import it without pulling
in the evaluation package (which imports the guardrails back).
"""

from __future__ import annotations

import json


def json_object(text: str) -> dict[str, object] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


__all__ = ["json_object"]
