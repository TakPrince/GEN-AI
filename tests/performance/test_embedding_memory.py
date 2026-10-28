import os
import subprocess
import sys
import textwrap
from pathlib import Path

_MAX_RSS_MIB = 512.0
_SCRIPT = textwrap.dedent(
    """
    import gc
    import resource
    import sys
    from collections.abc import Sequence

    from safety_platform.config.registry import CascadeThresholds, RoutingMargins
    from safety_platform.guardrails.tier2.vector_filter import Centroid, VectorFilter

    class StableEmbedder:
        def embed(self, text: str) -> Sequence[float]:
            return (0.75, 0.25, 0.5, 0.5)

    centroids = tuple(
        Centroid(f"centroid-{index}", (1.0, float(index + 1), 0.5, 0.25))
        for index in range(128)
    )
    detector = VectorFilter(
        StableEmbedder(),
        centroids,
        CascadeThresholds(0.85, 0.5, 0.75),
        RoutingMargins(0.1, 0.4),
    )
    for _ in range(10_000):
        detector.evaluate("stable prompt")
    gc.collect()
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        peak = peak / (1024.0 * 1024.0)
    else:
        peak = peak / 1024.0
    print(peak)
    """
)


def test_embedding_loop_peak_rss_under_512miB() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root / "src"), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert float(completed.stdout.strip()) <= _MAX_RSS_MIB
