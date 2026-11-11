from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from safety_platform.evaluation import manifest as module
from safety_platform.evaluation.manifest import (
    SCHEMA,
    ManifestError,
    build_manifest,
    config_fingerprint,
    load_manifest,
    preflight,
    prompt_identity,
    source_identity,
    write_manifest,
)


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "sample"
    root.mkdir(parents=True)
    rows = [
        ("a-1", "attack", "train"),
        ("a-2", "attack", "test"),
        ("b-1", "benign", "train"),
        ("b-2", "benign", "test"),
    ]
    lines = [
        json.dumps(
            {
                "case_id": case_id,
                "prompt": f"prompt {case_id}",
                "label": label,
                "category": "test",
                "behavior": "behavior",
                "split": split,
                "source": "unit",
            }
        )
        for case_id, label, split in rows
    ]
    path = root / "cases.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"dataset": "sample", "digest": "abc"}), encoding="utf-8"
    )
    return path


def _database(tmp_path: Path, dataset: Path) -> tuple[Path, str]:
    database = tmp_path / "results.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE run (run_id TEXT PRIMARY KEY, started_ns INTEGER NOT NULL,"
        " completed_ns INTEGER, target_model TEXT NOT NULL,"
        " embed_model TEXT NOT NULL, tier3_model TEXT NOT NULL,"
        " judge_model TEXT NOT NULL, dataset_path TEXT NOT NULL,"
        " dataset_digest TEXT NOT NULL, config_json TEXT NOT NULL);"
    )
    run_id = "20260101T000000-abcdef12"
    connection.execute(
        "INSERT INTO run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            1,
            2,
            "llama3.2:latest",
            "llama3.2:latest",
            "qwen3:8b",
            "qwen3:8b",
            str(dataset),
            "digest",
            json.dumps({"tier2_floor": 0.45, "tier2_block": 0.8}),
        ),
    )
    connection.commit()
    connection.close()
    return database, run_id


@pytest.fixture()
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(module, "source_identity", lambda root: {"available": False})
    dataset = _dataset(tmp_path)
    database, run_id = _database(tmp_path, dataset)
    return {
        "document": build_manifest(database, run_id),
        "database": database,
        "run_id": run_id,
        "dataset": dataset,
    }


def test_manifest_records_sections_and_schema(built) -> None:
    document = built["document"]

    assert document["schema"] == SCHEMA
    assert document["run"]["run_id"] == built["run_id"]
    assert document["dataset"]["rows"] == 4
    assert document["dataset"]["splits"] == {"train": 2, "test": 2}
    assert document["models"]["target"]["name"] == "llama3.2:latest"
    assert document["prompts"]["classifier"]["sha256"]
    assert document["execution"]["recorded"]["tier2_floor"] == 0.45
    command = document["execution"]["reproduction_command"]
    assert command.startswith("safety-eval run")
    assert "--tier2-floor 0.45" in command
    assert document["artifacts"][0]["path"] == str(built["database"])


def test_missing_model_and_git_metadata_is_unavailable_not_invented(
    built, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_ollama_version", lambda: "unavailable")
    document = module.build_manifest(built["database"], built["run_id"])

    assert document["source"]["available"] is False
    assert document["models"]["tier3"]["digest"] == "unavailable"
    assert document["models"]["tier3"]["quantization"] == "unavailable"
    assert document["runtime"]["ollama"] == "unavailable"


def test_config_fingerprint_is_stable_and_config_sensitive(
    built, tmp_path: Path
) -> None:
    document = built["document"]
    assert config_fingerprint(document) == config_fingerprint(document)

    changed = json.loads(json.dumps(document))
    changed["execution"]["recorded"]["tier2_floor"] = 0.40
    assert config_fingerprint(changed) != config_fingerprint(document)

    unrelated = json.loads(json.dumps(document))
    unrelated["runtime"]["python"] = "9.9.9"
    assert config_fingerprint(unrelated) == config_fingerprint(document)


def test_dataset_and_prompt_fingerprints_change_with_content(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    before = module.dataset_identity(dataset)
    dataset.write_text(
        dataset.read_text(encoding="utf-8").replace('"prompt a-1"', '"prompt X"'),
        encoding="utf-8",
    )
    after = module.dataset_identity(dataset)
    assert before["file_sha256"] != after["file_sha256"]
    assert before["split_fingerprint"] == after["split_fingerprint"]

    remixed_text = dataset.read_text(encoding="utf-8").replace(
        '"split": "train"', '"split": "test"', 1
    )
    dataset.write_text(remixed_text, encoding="utf-8")
    remixed = module.dataset_identity(dataset)
    assert before["split_fingerprint"] != remixed["split_fingerprint"]
    assert before["provenance"]["available"] is True
    assert prompt_identity()["classifier"]["sha256"]


def test_dirty_git_state_and_missing_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, cwd):
        if command[:2] == ["git", "rev-parse"]:
            return "deadbeef"
        if command[:2] == ["git", "status"]:
            return "?? src/new.py\n M src/old.py"
        return None

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"stdout": b"diff"})(),
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "new.py").write_text("x", encoding="utf-8")

    identity = source_identity(tmp_path)
    assert identity["dirty"] is True
    assert identity["commit"] == "deadbeef"
    assert identity["untracked"] == [
        {"path": "src/new.py", "sha256": module.sha256_bytes(b"x")}
    ]

    monkeypatch.setattr(module, "_run", lambda command, cwd: None)
    assert source_identity(tmp_path)["available"] is False


def test_write_refuses_to_overwrite_and_load_validates(built, tmp_path: Path) -> None:
    out = write_manifest(built["document"], tmp_path / "run.manifest.json")
    assert load_manifest(out)["schema"] == SCHEMA
    with pytest.raises(ManifestError, match="overwrite"):
        write_manifest(built["document"], out)

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError, match="unreadable"):
        load_manifest(bad)
    bad.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    with pytest.raises(ManifestError, match="schema"):
        load_manifest(bad)
    bad.write_text(json.dumps({"schema": SCHEMA}), encoding="utf-8")
    with pytest.raises(ManifestError, match="missing section"):
        load_manifest(bad)


def test_preflight_detects_modified_artifact_and_dataset(built, tmp_path: Path) -> None:
    out = write_manifest(built["document"], tmp_path / "run.manifest.json")
    result = preflight(out)
    assert result["config_reproducible"] is True
    assert result["exact_reproducible"] is False
    assert result["approximate_model_outputs"] is True
    assert any("model digests unavailable" in item for item in result["unavailable"])

    built["database"].write_bytes(b"tampered")
    assert any("artifact modified" in item for item in preflight(out)["mismatches"])

    built["dataset"].write_text(
        built["dataset"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert any(
        "dataset file hash differs" in item for item in preflight(out)["mismatches"]
    )


def test_preflight_reports_prompt_and_runtime_mismatch(
    built, tmp_path: Path, monkeypatch
) -> None:
    out = write_manifest(built["document"], tmp_path / "run.manifest.json")
    changed = prompt_identity()
    changed["classifier"]["sha256"] = "0" * 64
    monkeypatch.setattr(module, "prompt_identity", lambda: changed)

    result = preflight(out)
    assert any(
        "prompt classifier hash differs" in item for item in result["mismatches"]
    )
    assert result["config_reproducible"] is False


def test_legacy_run_without_manifest_is_reported(built, tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="unreadable"):
        preflight(tmp_path / "absent.manifest.json")


def test_cli_manifest_and_preflight(tmp_path: Path, built, capsys) -> None:
    from safety_platform.evaluation.__main__ import main

    out = tmp_path / "run.manifest.json"
    main(
        [
            "manifest",
            "--db",
            str(built["database"]),
            "--run-id",
            built["run_id"],
            "--out",
            str(out),
        ]
    )
    assert out.is_file()
    capsys.readouterr()

    main(["preflight", "--manifest", str(out)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == built["run_id"]
