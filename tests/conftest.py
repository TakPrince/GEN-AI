"""Shared test configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_echo_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    # The speculative echo probe calls the target model. Pipelines are cached per
    # database path and read this variable when they are built, which can happen in
    # a fixture before the test body runs. Ordinary tests must never reach Ollama;
    # tests that exercise the probe set the variable themselves.
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
