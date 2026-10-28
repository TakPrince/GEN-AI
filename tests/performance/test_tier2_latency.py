import math
import time

from safety_platform.config.registry import CascadeThresholds, RoutingMargins
from safety_platform.guardrails.tier2.vector_filter import Centroid, VectorFilter


class LocalEmbedder:
    def __init__(self, vector: tuple[float, ...]) -> None:
        self.vector = vector

    def embed(self, text: str) -> tuple[float, ...]:
        return self.vector


def p95(samples: list[float]) -> float:
    ordered = sorted(samples)
    index = max(0, int(len(ordered) * 0.95) - 1)
    return ordered[index]


def centroid(index: int) -> Centroid:
    vector = tuple(
        math.sin((index + 1) * (dimension + 1) * 0.017)
        + math.cos((dimension + 1) * 0.011)
        for dimension in range(64)
    )
    return Centroid(f"attack_{index:03d}", vector)


def query_vector() -> tuple[float, ...]:
    return tuple(
        math.cos((dimension + 1) * 0.023) for dimension in range(64)
    )


def test_tier2_vector_filter_p95_under_20ms() -> None:
    detector = VectorFilter(
        LocalEmbedder(query_vector()),
        tuple(centroid(index) for index in range(128)),
        CascadeThresholds(0.8, 0.5, 0.75),
        RoutingMargins(0.1, 0.4),
    )
    text = ("benign local prompt " * 80)[:2048]

    for _ in range(25):
        detector.evaluate(text)

    samples: list[float] = []
    for _ in range(200):
        started = time.perf_counter_ns()
        detector.evaluate(text)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)

    assert p95(samples) < 20.0
