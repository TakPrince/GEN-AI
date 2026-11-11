"""Frozen evaluation datasets and JailbreakBench ingestion."""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

_JBB_DATASET = "JailbreakBench/JBB-Behaviors"
_JBB_URL = "https://datasets-server.huggingface.co/rows"
_XSTEST_REPO = "paul-rottger/xstest"
XSTEST_SEED = 20261109
_XSTEST_LABELS = {"unsafe": "attack", "safe": "benign"}


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    prompt: str
    label: str
    category: str
    behavior: str
    split: str
    source: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.case_id, "case_id"),
            (self.prompt, "prompt"),
            (self.category, "category"),
            (self.behavior, "behavior"),
            (self.source, "source"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.label not in ("attack", "benign"):
            raise ValueError("label must be attack or benign")
        if self.split not in ("train", "test"):
            raise ValueError("split must be train or test")


def load_cases(path: str | Path) -> tuple[EvaluationCase, ...]:
    source = Path(path)
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must contain an object")
            case = EvaluationCase(
                case_id=_text(payload, "case_id"),
                prompt=_text(payload, "prompt"),
                label=_text(payload, "label"),
                category=_text(payload, "category"),
                behavior=_text(payload, "behavior"),
                split=_text(payload, "split"),
                source=_text(payload, "source"),
            )
            if case.case_id in seen:
                raise ValueError(f"duplicate case_id: {case.case_id}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError("dataset must contain at least one case")
    return tuple(cases)


def dataset_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_jbb_dataset(
    destination: str | Path,
    *,
    train_per_label: int = 10,
    test_per_label: int = 30,
    overwrite: bool = False,
) -> Path:
    if train_per_label <= 0 or test_per_label <= 0:
        raise ValueError("split sizes must be positive")
    required = train_per_label + test_per_label
    if required > 100:
        raise ValueError("JailbreakBench provides only 100 rows per label")

    root = Path(destination)
    cases_path = root / "cases.jsonl"
    manifest_path = root / "manifest.json"
    if not overwrite and (cases_path.exists() or manifest_path.exists()):
        raise ValueError(
            "refusing to overwrite frozen dataset artifacts; "
            "pass overwrite=True or --force to refetch"
        )

    jailbreaks = _fetch_rows("judge_comparison", "test", 300)
    benign = _fetch_rows("behaviors", "benign", 100)
    root.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    counters: dict[str, int] = {"attack": 0, "benign": 0}
    attack_rows = _select_jailbreak_rows(
        jailbreaks,
        train_per_label,
        test_per_label,
    )
    selected: list[tuple[str, str, dict[str, object]]] = [
        ("attack", split, row) for split, row in attack_rows
    ]
    selected.extend(
        ("benign", "train" if index < train_per_label else "test", row)
        for index, row in enumerate(benign[:required])
    )
    for label, split, row in selected:
        if label == "attack":
            prompt = _text(row, "prompt")
            behavior = _text(row, "goal")
            category = "Jailbreak"
            origin = f"judge_comparison:{row.get('Index', '')}"
        else:
            prompt = _text(row, "Goal")
            category = _text(row, "Category")
            behavior = _text(row, "Behavior")
            origin = _text(row, "Source")
        index = counters[label]
        counters[label] += 1
        case = EvaluationCase(
            case_id=f"{label}-{index:03d}",
            prompt=prompt,
            label=label,
            category=category,
            behavior=behavior,
            split=split,
            source=f"{_JBB_DATASET}:{origin}",
        )
        lines.append(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "prompt": case.prompt,
                    "label": case.label,
                    "category": case.category,
                    "behavior": case.behavior,
                    "split": case.split,
                    "source": case.source,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    cases_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = {
        "attack": {"train": train_per_label, "test": test_per_label},
        "benign": {"train": train_per_label, "test": test_per_label},
    }
    manifest = {
        "dataset": _JBB_DATASET,
        "dataset_url": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
        "attack_source": "judge_comparison:test",
        "benign_source": "behaviors:benign",
        "digest": dataset_digest(cases_path),
        "counts": counts,
        "source_split_size": {"attack": 300, "benign": 100},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cases_path


def build_xstest_cases(
    source_csv: str | Path,
    destination: str | Path,
    *,
    revision: str,
    train_per_label: int = 20,
    seed: int = XSTEST_SEED,
    overwrite: bool = False,
) -> Path:
    """Derive frozen cases from the unmodified XSTest CSV.

    unsafe -> attack (direct harmful requests), safe -> benign (over-refusal
    probes, the FPR set). The split is a seeded shuffle per label; the seed,
    source digest and revision are recorded in manifest.json.
    """
    if train_per_label <= 0:
        raise ValueError("train_per_label must be positive")
    root = Path(destination)
    cases_path = root / "cases.jsonl"
    manifest_path = root / "manifest.json"
    if not overwrite and (cases_path.exists() or manifest_path.exists()):
        raise ValueError(
            "refusing to overwrite frozen dataset artifacts; "
            "pass overwrite=True to rebuild"
        )
    source = Path(source_csv)
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = {"attack": [], "benign": []}
    for row in rows:
        label = _XSTEST_LABELS.get(row.get("label", "").strip())
        if label is None:
            raise ValueError(f"unknown XSTest label: {row.get('label')!r}")
        grouped[label].append(row)

    rng = random.Random(seed)
    lines: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for label in ("attack", "benign"):
        ordered = sorted(grouped[label], key=lambda row: int(row["id"]))
        if len(ordered) <= train_per_label:
            raise ValueError(f"not enough {label} rows for train_per_label")
        rng.shuffle(ordered)
        counts[label] = {
            "train": train_per_label,
            "test": len(ordered) - train_per_label,
        }
        for index, row in enumerate(ordered):
            case = {
                "case_id": f"xstest-{label}-{int(row['id']):03d}",
                "prompt": row["prompt"].strip(),
                "label": label,
                "category": row["type"].strip(),
                "behavior": (row["focus"] or row["type"]).strip(),
                "split": "train" if index < train_per_label else "test",
                "source": f"{_XSTEST_REPO}:xstest_prompts.csv:id={row['id']}",
            }
            lines.append(json.dumps(case, ensure_ascii=False, separators=(",", ":")))

    root.mkdir(parents=True, exist_ok=True)
    cases_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "dataset": "XSTest",
        "dataset_url": f"https://github.com/{_XSTEST_REPO}",
        "source_file": "xstest_prompts.csv",
        "revision": revision,
        "acquired_on": datetime.date.today().isoformat(),
        "license": "CC-BY-4.0",
        "source_sha256": dataset_digest(source),
        "source_bytes": source.stat().st_size,
        "digest": dataset_digest(cases_path),
        "seed": seed,
        "train_per_label": train_per_label,
        "counts": counts,
        "label_mapping": {"unsafe": "attack", "safe": "benign"},
        "purpose": (
            "benign set for over-refusal/FPR; unsafe set tests direct-request "
            "refusal only, not jailbreak resistance"
        ),
        "builder": "safety_platform.evaluation.dataset.build_xstest_cases",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cases_path


def _select_jailbreak_rows(
    rows: tuple[dict[str, object], ...],
    train_count: int,
    test_count: int,
) -> list[tuple[str, dict[str, object]]]:
    by_goal: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_goal.setdefault(_text(row, "goal"), []).append(row)
    goals = sorted(by_goal)
    required = train_count + test_count
    if len(goals) < required:
        raise ValueError(
            f"JailbreakBench provides only {len(goals)} distinct jailbreak goals"
        )
    selected: list[tuple[str, dict[str, object]]] = []
    for index, goal in enumerate(goals[:required]):
        split = "train" if index < train_count else "test"
        ordered = sorted(by_goal[goal], key=_row_index)
        selected.append((split, ordered[0]))
    return selected


def _fetch_rows(
    config: str,
    split: str,
    length: int,
) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    offset = 0
    while offset < length:
        page = min(100, length - offset)
        query = (
            f"?dataset={quote(_JBB_DATASET, safe='')}"
            f"&config={config}"
            f"&split={split}&offset={offset}&length={page}"
        )
        request = Request(
            f"{_JBB_URL}{query}",
            headers={"User-Agent": "safety-platform-eval/0.1"},
        )
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("dataset response must be an object")
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("dataset response contains no rows")
        for entry in rows:
            if not isinstance(entry, dict):
                raise ValueError("dataset row must be an object")
            row = entry.get("row")
            if not isinstance(row, dict):
                raise ValueError("dataset row payload must be an object")
            result.append(row)
        offset += len(rows)
        if len(rows) < page:
            break
    return tuple(result)


def _row_index(row: dict[str, object]) -> int:
    value = row.get("Index")
    return value if isinstance(value, int) else 0


def _text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


__all__ = [
    "EvaluationCase",
    "dataset_digest",
    "fetch_jbb_dataset",
    "load_cases",
]
