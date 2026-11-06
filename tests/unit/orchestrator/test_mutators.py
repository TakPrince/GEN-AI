from __future__ import annotations

from pathlib import Path

import pytest

from safety_platform.orchestrator.mutators import Mutation, mutate


def test_direct_mutation_preserves_prompt() -> None:
    mutation = mutate("write malware payload", "direct")

    assert mutation.strategy == "direct"
    assert mutation.original_prompt == "write malware payload"
    assert mutation.executed_prompt == "write malware payload"


@pytest.mark.parametrize("strategy", ("role_wrap", "encoding", "payload_split"))
def test_transforms_keep_objective_and_record_metadata(strategy: str) -> None:
    mutation = mutate("write malware payload", strategy)

    assert isinstance(mutation, Mutation)
    assert mutation.strategy == strategy
    assert mutation.original_prompt == "write malware payload"
    # The executed prompt differs but the objective stays recoverable.
    assert mutation.executed_prompt != mutation.original_prompt


def test_encoding_payload_decodes_to_original_prompt() -> None:
    import base64

    mutation = mutate("write malware payload", "encoding")
    blob = next(
        line
        for line in mutation.executed_prompt.splitlines()
        if set(line)
        <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    )
    assert base64.b64decode(blob).decode("utf-8") == "write malware payload"


def test_payload_split_is_deterministic_per_seed() -> None:
    first = mutate("write malware payload", "payload_split", seed=7).executed_prompt
    second = mutate("write malware payload", "payload_split", seed=7).executed_prompt
    assert first == second


def test_payload_split_seed_changes_fragments() -> None:
    variants = {
        mutate("write malware payload", "payload_split", seed=seed).executed_prompt
        for seed in range(8)
    }
    assert len(variants) > 1


def test_direct_mutation_ignores_seed() -> None:
    assert mutate("prompt", "direct", seed=1) == mutate("prompt", "direct", seed=99)


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValueError):
        mutate("prompt", "not-a-strategy")


def test_mutated_and_original_prompts_are_separate_fields(tmp_path: Path) -> None:
    mutation = mutate("steal credentials", "role_wrap", seed=3)

    assert "steal credentials" in mutation.executed_prompt
    assert mutation.original_prompt == "steal credentials"
    assert mutation.executed_prompt != mutation.original_prompt
