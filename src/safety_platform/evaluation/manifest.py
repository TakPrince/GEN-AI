"""Versioned experiment-run manifest and reproduction preflight (GEN-13).

A manifest is a JSON document that records what a stored evaluation run
depended on. It is derived from the run row plus the current checkout, so
fields that cannot be recovered honestly are marked "unavailable" instead of
being invented. Nothing here calls a model or mutates a result artifact.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..guardrails.tier3 import guard_classifier as _guard
from . import llm as _llm
from .dataset import load_cases

SCHEMA = "gen13.manifest.v1"
_UNAVAILABLE = "unavailable"
_MAX_HASH_BYTES = 64 * 1024 * 1024
_MAX_UNTRACKED = 50


class ManifestError(ValueError):
    """Manifest is malformed or unreadable."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _run(command: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.decode("utf-8", "replace").strip()


def source_identity(root: Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], root)
    if commit is None:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "diff_sha256": None,
            "untracked": [],
        }
    status = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], root)
    diff = subprocess.run(
        ["git", "diff", "HEAD"], cwd=root, capture_output=True, check=False, timeout=10
    ).stdout
    untracked = []
    for line in (status or "").splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        candidate = root / relative
        if Path(relative).parts[0] not in ("src", "tests"):
            continue
        if candidate.is_file() and candidate.stat().st_size <= _MAX_HASH_BYTES:
            untracked.append({"path": relative, "sha256": sha256_file(candidate)})
        if len(untracked) >= _MAX_UNTRACKED:
            break
    return {
        "available": True,
        "commit": commit,
        "dirty": bool((status or "").strip()),
        "diff_sha256": sha256_bytes(diff),
        "untracked": untracked,
    }


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return _UNAVAILABLE


def _ollama_version() -> str:
    if shutil.which("ollama") is None:
        return _UNAVAILABLE
    return _run(["ollama", "--version"], Path.cwd()) or _UNAVAILABLE


def runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": f"{platform.system()} {platform.release()}",
        "packages": {
            "safety-platform": _version("safety-platform"),
            "streamlit": _version("streamlit"),
        },
        "ollama": _ollama_version(),
        "device": _UNAVAILABLE,
    }


def model_identity(name: str) -> dict[str, Any]:
    # Ollama model digests/quantization need a live `ollama show`; recording a
    # guess would be worse than recording that it is unknown.
    return {
        "name": name,
        "provider": "ollama",
        "digest": _UNAVAILABLE,
        "quantization": _UNAVAILABLE,
        "runtime": "ollama",
    }


def prompt_identity() -> dict[str, Any]:
    classifier = _guard._INSTRUCTION
    judge = _llm._JUDGE_INSTRUCTION
    conversation = _llm._CONVERSATION_JUDGE_INSTRUCTION
    return {
        "classifier": {
            "sha256": sha256_bytes(classifier.encode()),
            "length": len(classifier),
        },
        "judge": {"sha256": sha256_bytes(judge.encode()), "length": len(judge)},
        "conversation_judge": {
            "sha256": sha256_bytes(conversation.encode()),
            "length": len(conversation),
        },
        "output_schema": "guard-verdict.v1",
    }


def dataset_identity(path: Path) -> dict[str, Any]:
    cases = load_cases(path)
    splits = {
        split: [case.case_id for case in cases if case.split == split]
        for split in ("train", "test")
    }
    labels = {
        label: sum(1 for case in cases if case.label == label)
        for label in ("attack", "benign")
    }
    fingerprint = sha256_bytes(
        json.dumps(
            sorted([case.case_id, case.label, case.split] for case in cases),
            sort_keys=True,
        ).encode()
    )
    return {
        "name": path.parent.name,
        "path": str(path),
        "file_sha256": sha256_file(path),
        "rows": len(cases),
        "splits": {split: len(ids) for split, ids in splits.items()},
        "split_fingerprint": fingerprint,
        "labels": labels,
        "provenance": _dataset_provenance(path.parent / "manifest.json"),
    }


def _dataset_provenance(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {"available": False}
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False}
    return {
        "available": True,
        "dataset": recorded.get("dataset"),
        "digest": recorded.get("digest"),
        "source_sha256": recorded.get("source_sha256"),
        "revision": recorded.get("revision"),
    }


def _load_run(database: Path, run_id: str) -> dict[str, Any]:
    import sqlite3

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=10.0)
    try:
        row = connection.execute(
            "SELECT run_id, started_ns, completed_ns, target_model, embed_model,"
            " tier3_model, judge_model, dataset_path, dataset_digest, config_json"
            " FROM run WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    except sqlite3.DatabaseError as error:
        raise ManifestError(f"cannot read run table: {error}") from error
    finally:
        connection.close()
    if row is None:
        raise ManifestError(f"unknown run_id: {run_id}")
    keys = (
        "run_id",
        "started_ns",
        "completed_ns",
        "target_model",
        "embed_model",
        "tier3_model",
        "judge_model",
        "dataset_path",
        "dataset_digest",
        "config_json",
    )
    return dict(zip(keys, row, strict=False))


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "sha256": _UNAVAILABLE, "exists": False}
    if path.stat().st_size > _MAX_HASH_BYTES:
        return {
            "path": str(path),
            "sha256": _UNAVAILABLE,
            "exists": True,
            "note": "too large to hash",
        }
    return {"path": str(path), "sha256": sha256_file(path), "exists": True}


def _dataset_section(path: Path) -> dict[str, Any]:
    try:
        return dataset_identity(path)
    except (OSError, ValueError) as error:
        raise ManifestError(f"cannot fingerprint dataset {path}: {error}") from error


def config_fingerprint(manifest: dict[str, Any]) -> str:
    """Stable hash of the fields that determine evaluation behaviour.

    Excludes run id, timestamps, runtime host details and artifact hashes so the
    same configuration fingerprints identically on another machine.
    """
    basis = {
        "source_commit": manifest["source"].get("commit"),
        "prompts": manifest["prompts"],
        "dataset_sha256": manifest["dataset"]["file_sha256"],
        "split_fingerprint": manifest["dataset"]["split_fingerprint"],
        "models": {key: value["name"] for key, value in manifest["models"].items()},
        "execution": manifest["execution"],
    }
    return sha256_bytes(json.dumps(basis, sort_keys=True).encode())


def _reproduction_command(run: dict[str, Any], execution: dict[str, Any]) -> str:
    parts = [
        "safety-eval run",
        f"--dataset {run['dataset_path']}",
        f"--db {run['database_path']}",
        f"--target {run['target_model']}",
        f"--embed {run['embed_model']}",
        f"--tier3 {run['tier3_model']}",
        f"--judge {run['judge_model']}",
    ]
    if execution.get("per_label") is not None:
        parts.append(f"--per-label {execution['per_label']}")
    for flag in ("tier2_block", "tier2_floor", "tier2_gap", "tier3_confidence"):
        if execution.get(flag) is not None:
            parts.append(f"--{flag.replace('_', '-')} {execution[flag]}")
    if execution.get("reuse_run_id"):
        parts.append(f"--reuse-run {execution['reuse_run_id']}")
    return " ".join(parts)


def build_manifest(
    database: str | Path,
    run_id: str,
    *,
    artifacts: tuple[str | Path, ...] = (),
    root: str | Path = ".",
) -> dict[str, Any]:
    database = Path(database)
    root = Path(root)
    run = _load_run(database, run_id)
    run["database_path"] = str(database)
    dataset_path = Path(run["dataset_path"])
    try:
        execution = json.loads(run["config_json"])
    except json.JSONDecodeError as error:
        raise ManifestError(f"run config_json is not valid JSON: {error}") from error
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "manifest_origin": "reconstructed",
        "run": {
            "run_id": run["run_id"],
            "started_ns": run["started_ns"],
            "completed_ns": run["completed_ns"],
            "completed": run["completed_ns"] is not None,
        },
        "source": source_identity(root),
        "models": {
            "target": model_identity(run["target_model"]),
            "embedding": model_identity(run["embed_model"]),
            "tier3": model_identity(run["tier3_model"]),
            "judge": model_identity(run["judge_model"]),
        },
        "prompts": prompt_identity(),
        "dataset": _dataset_section(dataset_path),
        "execution": {
            "recorded": execution,
            "generation": {"temperature": 0, "seed": 7},
            "embedding": {"normalization": _UNAVAILABLE, "dimensions": _UNAVAILABLE},
            "centroid_fingerprint": _UNAVAILABLE,
            "reproduction_command": _reproduction_command(run, execution),
        },
        "runtime": runtime_identity(),
        "artifacts": [
            _artifact(Path(path)) for path in (database, *map(Path, artifacts))
        ],
    }
    manifest["config_fingerprint"] = config_fingerprint(manifest)
    return manifest


def write_manifest(manifest: dict[str, Any], destination: str | Path) -> Path:
    destination = Path(destination)
    if destination.exists():
        raise ManifestError(f"refusing to overwrite existing manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"unreadable manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        recorded = (
            manifest.get("schema") if isinstance(manifest, dict) else "not an object"
        )
        raise ManifestError(f"unsupported manifest schema: {recorded}")
    expected = {
        "source": dict,
        "models": dict,
        "prompts": dict,
        "dataset": dict,
        "execution": dict,
        "runtime": dict,
        "artifacts": list,
    }
    for key, kind in expected.items():
        if not isinstance(manifest.get(key), kind):
            raise ManifestError(f"manifest missing section: {key}")
    return manifest


def preflight(path: str | Path, *, root: str | Path = ".") -> dict[str, Any]:
    """Compare a recorded manifest with the current checkout. Read-only."""
    manifest = load_manifest(path)
    root = Path(root)
    mismatches: list[str] = []
    unavailable: list[str] = []

    current_source = source_identity(root)
    if not manifest["source"].get("available"):
        unavailable.append("source: recorded git identity unavailable")
    elif not current_source["available"]:
        unavailable.append("source: git unavailable in current checkout")
    elif current_source["commit"] != manifest["source"].get("commit"):
        recorded_commit = manifest["source"].get("commit")
        mismatches.append(
            f"source commit {current_source['commit']} != recorded {recorded_commit}"
        )
    elif current_source["diff_sha256"] != manifest["source"].get("diff_sha256"):
        mismatches.append("source working-tree diff differs from recorded state")

    current_prompts = prompt_identity()
    for name, recorded in manifest["prompts"].items():
        if not isinstance(recorded, dict):
            continue
        if recorded.get("sha256") != current_prompts[name]["sha256"]:
            mismatches.append(f"prompt {name} hash differs")

    dataset_path = Path(manifest["dataset"].get("path", ""))
    if not dataset_path.is_file():
        unavailable.append(f"dataset file missing: {dataset_path}")
    elif sha256_file(dataset_path) != manifest["dataset"].get("file_sha256"):
        mismatches.append("dataset file hash differs")

    for artifact in manifest["artifacts"]:
        artifact_path = Path(artifact.get("path", ""))
        if not artifact_path.is_file():
            unavailable.append(f"artifact missing: {artifact_path}")
        elif (
            artifact.get("sha256") not in (_UNAVAILABLE, None)
            and sha256_file(artifact_path) != artifact["sha256"]
        ):
            mismatches.append(f"artifact modified: {artifact_path}")

    current_runtime = runtime_identity()
    if current_runtime["python"] != manifest["runtime"].get("python"):
        recorded_python = manifest["runtime"].get("python")
        mismatches.append(
            f"python {current_runtime['python']} != recorded {recorded_python}"
        )
    for package, version in manifest["runtime"].get("packages", {}).items():
        if version in (_UNAVAILABLE, None):
            unavailable.append(f"package version unavailable: {package}")
        elif current_runtime["packages"].get(package) != version:
            current = current_runtime["packages"].get(package)
            mismatches.append(f"package {package} {current} != recorded {version}")
    if manifest["runtime"].get("ollama") in (_UNAVAILABLE, None):
        unavailable.append("ollama version unavailable")

    model_digests = [model.get("digest") for model in manifest["models"].values()]
    if all(digest in (_UNAVAILABLE, None) for digest in model_digests):
        unavailable.append(
            "model digests unavailable; outputs are only approximately reproducible"
        )

    command = manifest["execution"].get("reproduction_command")
    if command in (_UNAVAILABLE, None):
        unavailable.append("reproduction command not recorded")

    return {
        "schema": manifest["schema"],
        "run_id": manifest["run"].get("run_id"),
        "exact_reproducible": not mismatches and not unavailable,
        "config_reproducible": not mismatches,
        "approximate_model_outputs": True,
        "mismatches": mismatches,
        "unavailable": unavailable,
        "reproduction_command": command,
    }
