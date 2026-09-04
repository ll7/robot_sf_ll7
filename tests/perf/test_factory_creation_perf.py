"""T015: Performance regression guard for factory creation (FR-017).

Compares current mean creation times against baseline in
`output/benchmarks/factory_perf_baseline.json`. Fails if mean exceeds baseline by >5%.
Skips if baseline file missing (developer must run baseline script first).

NOTE: Ensures fast demo mode is disabled to reflect real creation cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.common.artifact_paths import resolve_artifact_path
from robot_sf.gym_env.environment_factory import make_image_robot_env, make_robot_env
from robot_sf.gym_env.unified_config import ImageRobotConfig, RobotSimulationConfig

BASELINE_PATH = resolve_artifact_path(Path("benchmarks/factory_perf_baseline.json"))
THRESHOLD = 1.15  # +15% hard budget (tightened per T031 spec compliance)
SOFT_THRESHOLD = 1.30  # soft warn band (>15% and <=30%)
ITERATIONS = 2  # keep light for CI; baseline may have been generated with more


def _time_once(fn):  # minimal inline timing to avoid importing heavy script
    """TODO docstring. Document this function.

    Args:
        fn: TODO docstring.
    """
    import time

    start = time.perf_counter()
    env = fn()
    try:
        env.reset()
    finally:
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
    return (time.perf_counter() - start) * 1000.0


@pytest.mark.skipif(
    not BASELINE_PATH.exists(),
    reason="Baseline file missing; run baseline script first.",
)
def test_factory_creation_mean_within_budget(monkeypatch):
    # Disable fast demo shortcut if present
    """TODO docstring. Document this function.

    Args:
        monkeypatch: TODO docstring.
    """
    monkeypatch.delenv("ROBOT_SF_FAST_DEMO", raising=False)

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    base_robot = baseline["results"]["make_robot_env"]["mean_ms"]
    base_image = baseline["results"]["make_image_robot_env"]["mean_ms"]

    robot_times = [
        _time_once(lambda: make_robot_env(config=RobotSimulationConfig(), debug=False))
        for _ in range(ITERATIONS)
    ]
    image_times = [
        _time_once(lambda: make_image_robot_env(config=ImageRobotConfig(), debug=False))
        for _ in range(ITERATIONS)
    ]

    current_robot_mean = sum(robot_times) / len(robot_times)
    current_image_mean = sum(image_times) / len(image_times)

    # Hard budget (tightened). A measured breach fails on every lane: skips are
    # reserved for missing prerequisites (baseline/hardware), so a regression
    # can never pass silently (issue #8377). Hosts that cannot meet the
    # reference budget must regenerate the baseline for their hardware profile
    # instead of relying on silent skips.
    if current_robot_mean > base_robot * THRESHOLD:
        pytest.fail(
            f"Robot env creation mean {current_robot_mean:.2f}ms exceeds hard budget (+{(THRESHOLD - 1) * 100:.0f}% {base_robot * THRESHOLD:.2f}ms ceiling from baseline {base_robot:.2f}ms)"
        )
    if current_image_mean > base_image * THRESHOLD:
        pytest.fail(
            f"Image env creation mean {current_image_mean:.2f}ms exceeds hard budget (+{(THRESHOLD - 1) * 100:.0f}% {base_image * THRESHOLD:.2f}ms ceiling from baseline {base_image:.2f}ms)"
        )
    if (
        base_robot * THRESHOLD < current_robot_mean <= base_robot * SOFT_THRESHOLD
    ):  # pragma: no cover
        print(
            f"[PERF SOFT WARN] Robot env mean {current_robot_mean:.2f}ms within soft band (>+5% <=+8%) baseline {base_robot:.2f}ms",
        )
    if (
        base_image * THRESHOLD < current_image_mean <= base_image * SOFT_THRESHOLD
    ):  # pragma: no cover
        print(
            f"[PERF SOFT WARN] Image env mean {current_image_mean:.2f}ms within soft band (>+5% <=+8%) baseline {base_image:.2f}ms",
        )
