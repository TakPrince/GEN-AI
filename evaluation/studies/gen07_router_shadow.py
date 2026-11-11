"""GEN-07 shadow comparison. Development leave-one-attack-out only.

Train attacks are their own centroids in the normal cascade, so routing cannot
be judged on that split. This holds each development attack out of the
centroid set. It does not read or tune on held-out test cases. Missing train
embeddings are fetched once into memory; historical databases are read-only.

    PYTHONPATH=src python3 evaluation/studies/gen07_router_shadow.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from safety_platform.evaluation.cascade import build_cascade
from safety_platform.evaluation.dataset import load_cases
from safety_platform.evaluation.llm import OllamaClient, OllamaEmbedder
from safety_platform.evaluation.runner import _load_embedding_cache
from safety_platform.guardrails.tier1.rules import defense_regex_filter
from safety_platform.research.ml_router import (
    ModelDrivenRouter,
    RouterFeatures,
    Tier3BypassRouter,
    benchmark_router,
)

DATASETS = {
    "jbb-v2": "evaluation/data/jbb-v2/cases.jsonl",
    "diagnostic-heldout-v1": "evaluation/data/diagnostic-heldout-v1/cases.jsonl",
}
CACHES = (
    "evaluation/results/jbb-v2-repro.sqlite3",
    "evaluation/results/diagnostic-heldout.sqlite3",
)
# Benign escalation may not more than double the development baseline.
BENIGN_BUDGET = 2.0


def embedder() -> OllamaEmbedder:
    model = OllamaEmbedder(OllamaClient("llama3.2:latest", timeout_s=300.0))
    for database in CACHES:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        model.seed(_load_embedding_cache(connection, "llama3.2:latest"))
    return model


def extra(features: dict[str, float | bool | str]) -> bool:
    passed = features["tier2"] == "pass"
    gap = float(features["gap"])
    similarity = float(features["similarity"])
    return {
        "baseline": False,
        "gap_0.04_sim_0.35": passed and gap <= 0.04 and similarity >= 0.35,
        "gap_0.08_sim_0.38": passed and gap <= 0.08 and similarity >= 0.38,
        "floor_0.42": similarity >= 0.42 and not features["baseline"],
    }


def evaluate(name: str, path: str, model: OllamaEmbedder) -> dict[str, object]:
    train = [case for case in load_cases(path) if case.split == "train"]
    missing = [case.prompt for case in train if case.prompt not in dict(model._cache)]
    if missing:
        model.embed_many(tuple(dict.fromkeys(missing)))
    attacks = [case for case in train if case.label == "attack"]
    benign = [case for case in train if case.label == "benign"]
    regex = defense_regex_filter(max_input_bytes=16384)
    router = ModelDrivenRouter(Tier3BypassRouter(benchmark_router()))
    policies = ("baseline", "gap_0.04_sim_0.35", "gap_0.08_sim_0.38", "floor_0.42")
    counts = {
        policy: {
            "attacks": 0,
            "attack_escalated": 0,
            "benign_pairs": 0,
            "benign_escalated": 0,
        }
        for policy in policies
    }
    misses = []
    for held in attacks:
        others = [case for case in attacks if case.case_id != held.case_id]
        runtime = build_cascade(others, model, None)

        def row(case, runtime=runtime):
            first = regex.evaluate(case.prompt)
            second = runtime.vector.evaluate(case.prompt)
            decision = router.route(
                RouterFeatures(
                    bool(first.rule_ids),
                    second.top_similarity,
                    second.top_gap,
                    second.verdict.value == "ambiguous",
                    len(case.prompt.encode()),
                )
            )
            baseline = (
                bool(decision.invoke_tier3)
                or second.verdict.value in ("ambiguous", "block")
                or first.verdict.value == "block"
            )
            return {
                "tier2": second.verdict.value,
                "similarity": second.top_similarity,
                "gap": second.top_gap,
                "rho": decision.probability,
                "baseline": baseline,
            }

        held_row = row(held)
        if not held_row["baseline"]:
            misses.append(
                {
                    "case_id": held.case_id,
                    "similarity": round(held_row["similarity"], 3),
                    "gap": round(held_row["gap"], 3),
                    "rho": round(held_row["rho"], 3),
                }
            )
        for policy, enabled in extra(held_row).items():
            counts[policy]["attacks"] += 1
            counts[policy]["attack_escalated"] += int(held_row["baseline"] or enabled)
        for case in benign:
            benign_row = row(case)
            for policy, enabled in extra(benign_row).items():
                counts[policy]["benign_pairs"] += 1
                escalated = benign_row["baseline"] or enabled
                counts[policy]["benign_escalated"] += int(escalated)
    return {"dataset": name, "routing_misses": misses, "policies": counts}


def main() -> None:
    started = time.perf_counter()
    model = embedder()
    datasets = [evaluate(name, path, model) for name, path in DATASETS.items()]
    document = {
        "study": "gen07_router_shadow",
        "held_out_used_for_tuning": False,
        "benign_escalation_budget": "at most 2x development baseline",
        "activated_policy": None,
        "reason": (
            "Every candidate that catches development routing misses "
            "more than doubles benign Tier 3 escalation."
        ),
        "datasets": datasets,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = Path(f"evaluation/results/phase3/gen07-router-shadow-{stamp}.json")
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({item["dataset"]: item["routing_misses"] for item in datasets}))
    print("written", out)


if __name__ == "__main__":
    main()
