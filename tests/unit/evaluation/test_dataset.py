import json
from pathlib import Path
from typing import cast

import pytest

from safety_platform.evaluation import dataset as module
from safety_platform.evaluation.dataset import (
    EvaluationCase,
    dataset_digest,
    fetch_jbb_dataset,
    load_cases,
)


def case(case_id: str = "attack-000") -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        prompt="prompt",
        label="attack",
        category="Jailbreak",
        behavior="behavior",
        split="train",
        source="source",
    )


def test_evaluation_case_validates_fields() -> None:
    with pytest.raises(ValueError, match="prompt"):
        EvaluationCase("a", "", "attack", "c", "b", "train", "s")
    with pytest.raises(ValueError, match="label"):
        EvaluationCase("a", "p", "other", "c", "b", "train", "s")
    with pytest.raises(ValueError, match="split"):
        EvaluationCase("a", "p", "attack", "c", "b", "other", "s")


def test_load_cases_reads_jsonl_and_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    payload = {
        "case_id": "attack-000",
        "prompt": "prompt",
        "label": "attack",
        "category": "Jailbreak",
        "behavior": "behavior",
        "split": "train",
        "source": "source",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert load_cases(path) == (case(),)

    path.write_text(
        json.dumps(payload) + "\n" + json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_cases(path)


def test_load_cases_rejects_invalid_and_empty_files(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_cases(path)

    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_cases(path)

    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        load_cases(path)


def test_dataset_digest_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("payload\n", encoding="utf-8")

    assert dataset_digest(path) == dataset_digest(path)
    assert len(dataset_digest(path)) == 64


def test_fetch_jbb_dataset_selects_disjoint_goals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jailbreaks = tuple(
        cast(
            dict[str, object],
            {
                "Index": index,
                "goal": f"goal-{index}",
                "prompt": f"prompt-{index}",
            },
        )
        for index in range(40)
    )
    benign = tuple(
        cast(
            dict[str, object],
            {
                "Goal": f"benign-{index}",
                "Category": "Benign",
                "Behavior": f"behavior-{index}",
                "Source": "source",
            },
        )
        for index in range(40)
    )
    calls: list[tuple[str, str, int]] = []

    def fake_fetch(
        config: str,
        split: str,
        length: int,
    ) -> tuple[dict[str, object], ...]:
        calls.append((config, split, length))
        return jailbreaks if config == "judge_comparison" else benign

    monkeypatch.setattr(module, "_fetch_rows", fake_fetch)

    path = fetch_jbb_dataset(
        tmp_path,
        train_per_label=10,
        test_per_label=30,
    )
    cases = load_cases(path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert calls == [
        ("judge_comparison", "test", 300),
        ("behaviors", "benign", 100),
    ]
    assert sum(case.label == "attack" and case.split == "train" for case in cases) == 10
    assert sum(case.label == "attack" and case.split == "test" for case in cases) == 30
    assert sum(case.label == "benign" and case.split == "train" for case in cases) == 10
    assert sum(case.label == "benign" and case.split == "test" for case in cases) == 30
    attack_goals = {case.behavior for case in cases if case.label == "attack"}
    assert len(attack_goals) == 40
    assert manifest["counts"]["attack"] == {"train": 10, "test": 30}
    assert manifest["digest"] == dataset_digest(path)


def test_fetch_jbb_dataset_validates_sizes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        fetch_jbb_dataset(tmp_path, train_per_label=0)
    with pytest.raises(ValueError, match="100"):
        fetch_jbb_dataset(tmp_path, train_per_label=60, test_per_label=41)
