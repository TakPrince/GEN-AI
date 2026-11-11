"""GEN-02 summary: Phase 2 vs Phase 3 on identical held-out cases (reads stored DBs).

Run from the repository root:
    PYTHONPATH=src python3 evaluation/studies/gen02_summary.py

Writes a new timestamped file under evaluation/results/phase3/.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from safety_platform.guardrails.tier1.rules import defense_regex_filter


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def summarize(db: Path, run_id: str | None = None) -> dict:
    con = sqlite3.connect(db)
    rid = (
        run_id
        or con.execute(
            "select run_id from run order by started_ns desc limit 1"
        ).fetchone()[0]
    )
    rows = con.execute(
        "select system,label,category,case_id,final_route,blocked,tier3_invoked,"
        "judge_complied,judge_error,guard_ms,generation_ms,raw_response,final_response "
        "from run_result where run_id=?",
        (rid,),
    ).fetchall()
    out: dict = {"db": str(db), "run_id": rid, "systems": {}}
    by_sys: dict[str, list] = defaultdict(list)
    for r in rows:
        by_sys[r[0]].append(r)
    for system, items in sorted(by_sys.items()):
        atk = [r for r in items if r[1] == "attack"]
        ben = [r for r in items if r[1] == "benign"]
        blocked_atk = sum(1 for r in atk if r[5])
        blocked_ben = sum(1 for r in ben if r[5])
        complied_delivered = sum(1 for r in atk if r[7] == 1 and not r[5])
        tier3 = sum(1 for r in items if r[6])
        errors = sum(1 for r in items if r[8])
        guard = sorted(r[9] for r in items if r[9] is not None)
        gen = sorted(r[10] for r in items if r[10] is not None)

        def pct(xs, q):
            return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None

        cats: dict[str, dict] = defaultdict(lambda: {"n": 0, "blocked": 0})
        for r in atk:
            cats[r[2]]["n"] += 1
            cats[r[2]]["blocked"] += int(bool(r[5]))
        out["systems"][system] = {
            "attack_n": len(atk),
            "attack_blocked": blocked_atk,
            "detection": round(blocked_atk / len(atk), 3) if atk else None,
            "detection_ci95": wilson(blocked_atk, len(atk)),
            "complied_and_delivered": complied_delivered,
            "benign_n": len(ben),
            "benign_blocked": blocked_ben,
            "fpr": round(blocked_ben / len(ben), 3) if ben else None,
            "fpr_ci95": wilson(blocked_ben, len(ben)),
            "tier3_invocations": tier3,
            "judge_errors": errors,
            "guard_ms_p50": pct(guard, 0.5),
            "guard_ms_p95": pct(guard, 0.95),
            "generation_ms_p50": pct(gen, 0.5),
            "per_category_attack": {
                k: {**v, "ci95": wilson(v["blocked"], v["n"])}
                for k, v in sorted(cats.items())
            },
        }
    return out


rules_before = defense_regex_filter()  # current rules
phase2 = summarize(Path("evaluation/results/diagnostic-heldout.sqlite3"))
phase3 = summarize(Path("evaluation/results/phase3/diagnostic-heldout-phase3.sqlite3"))

# Deterministic Tier 1 attribution on the held-out prompts: which attacks the
# three Phase 3 rules add, independent of any model.
cases = [
    json.loads(line)
    for line in Path("evaluation/data/diagnostic-heldout-v1/cases.jsonl")
    .read_text()
    .splitlines()
    if line.strip()
]
new_rule_ids = {
    "block.indirect_injection",
    "block.injected_instructions",
    "block.persona_override",
}
added = []
for c in cases:
    if c["split"] != "test" or c["label"] != "attack":
        continue
    res = rules_before.evaluate(c["prompt"])
    if set(res.rule_ids) & new_rule_ids:
        added.append({"case_id": c["case_id"], "rules": list(res.rule_ids)})

result = {
    "phase2": phase2,
    "phase3": phase3,
    "tier1_new_rule_hits_on_heldout_attacks": added,
}
Path(
    f"evaluation/results/phase3/gen02-diagnostic-heldout-summary-{time.strftime('%Y%m%dT%H%M%S')}.json"
).write_text(json.dumps(result, indent=1), encoding="utf-8")
print(json.dumps(result, indent=1)[:4000])
