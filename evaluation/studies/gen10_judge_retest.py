"""GEN-10: judge test-retest agreement on the stored JBB-v2 responses.

Re-runs the configured judge twice on every stored baseline response and compares
(a) the two fresh verdicts with each other and (b) each with the label recorded
during the Phase 2 run. This measures *self-consistency*, not correctness: there
are no independent human labels for these responses, so agreement here is a
reliability signal only.

Benign rows are not judged by the runner (it judges attack rows only) and carry
a sentinel ``JudgeResult(False)``. Stored-vs-fresh agreement is therefore
computed on attack rows only; benign sentinel rows are counted separately.

Run from the repository root, with no other Ollama job running:
    PYTHONPATH=src python3 evaluation/studies/gen10_judge_retest.py

Recompute the summary from the existing partial file without model calls:
    PYTHONPATH=src python3 evaluation/studies/gen10_judge_retest.py --summary-only

Writes a new timestamped file under evaluation/results/phase3/.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from safety_platform.evaluation.llm import ComplianceJudge, OllamaClient

DB = Path("evaluation/results/jbb-v2-repro.sqlite3")
OUT_DIR = Path("evaluation/results/phase3")
PARTIAL = OUT_DIR / "gen10-judge-retest.partial.jsonl"
JUDGE_MODEL = "qwen3:8b"


def _kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=False)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected != 1 else float("nan")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in records if "skipped" not in r and r["run_a"] is not None]
    both = [r for r in scored if r.get("run_b") is not None]
    # The runner only judges attack rows; benign "stored" values are sentinels.
    stored_scored = [r for r in scored if r["label"] == "attack"]
    stored = [r["stored"] for r in stored_scored]
    stored_a = [r["run_a"] for r in stored_scored]
    run_a_both = [r["run_a"] for r in both]
    run_b = [r["run_b"] for r in both]
    return {
        "judge_model": JUDGE_MODEL,
        "n": len(scored),
        "skipped": [r for r in records if "skipped" in r],
        "sentinel_benign_rows": len([r for r in scored if r["label"] == "benign"]),
        "n_both_passes": len(both),
        "agreement_a_b": sum(x == y for x, y in zip(run_a_both, run_b, strict=False))
        / len(both),
        "kappa_a_b": _kappa(run_a_both, run_b),
        "stored_agreement_scope": "attack_only",
        "n_stored_scored": len(stored_scored),
        "agreement_stored_a": sum(
            x == y for x, y in zip(stored, stored_a, strict=False)
        )
        / len(stored_scored),
        "kappa_stored_a": _kappa(stored, stored_a),
        "complied_stored": sum(stored),
        "complied_run_a": sum(r["run_a"] for r in scored),
        "complied_run_b": sum(r["run_b"] for r in both),
        "records": records,
    }


def _load_partial() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in PARTIAL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        records = _load_partial()
    else:
        rows = (
            sqlite3.connect(DB)
            .execute(
                "select case_id, label, raw_response, judge_complied, judge_error "
                "from run_result "
                "where system = 'baseline' order by case_id"
            )
            .fetchall()
        )
        judge = ComplianceJudge(OllamaClient(JUDGE_MODEL))
        with open("evaluation/data/jbb-v2/cases.jsonl", encoding="utf-8") as handle:
            cases = {c["case_id"]: c for c in map(json.loads, handle)}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        done = (
            {item["case_id"]: item for item in _load_partial()}
            if PARTIAL.exists()
            else {}
        )
        records = list(done.values())
        for case_id, label, raw, stored, stored_err in rows:
            if case_id in done:
                continue
            if stored_err:
                record: dict[str, Any] = {
                    "case_id": case_id,
                    "label": label,
                    "skipped": stored_err,
                }
            else:
                record = {
                    "case_id": case_id,
                    "label": label,
                    "stored": bool(stored),
                    "raw_response": raw,
                }
                prompt = cases[case_id]["prompt"]
                first = judge.judge(prompt, record["raw_response"])
                second = judge.judge(prompt, record["raw_response"])
                record["run_a"] = first.complied if not first.error else None
                record["run_b"] = second.complied if not second.error else None
                record["errors"] = [e for e in (first.error, second.error) if e]
            records.append(record)
            with PARTIAL.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                record["case_id"], record.get("run_a"), record.get("run_b"), flush=True
            )
    summary = summarize(records)
    out = OUT_DIR / f"gen10-judge-retest-{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print({k: v for k, v in summary.items() if k != "records"})
    print("written", out)


if __name__ == "__main__":
    main()
