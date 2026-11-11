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
    # Same reasoning for Tier 3: the dashboard default is the real Ollama classifier,
    # which unit tests must not reach. Tests that need it inject a transport.
    monkeypatch.setenv("SAFETY_UI_TIER3", "heuristic")
