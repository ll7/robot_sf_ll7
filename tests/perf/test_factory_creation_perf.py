"""T015: Performance regression guard for factory creation (FR-017).

Compares current mean creation times against baseline in
`results/factory_perf_baseline.json`. Fails if mean exceeds baseline by >5%.
Skips if baseline file missing (developer must run baseline script first).

NOTE: Ensures fast demo mode is disabled to reflect real creation cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.gym_env.environment_factory import make_image_robot_env, make_robot_env
from robot_sf.gym_env.unified_config import ImageRobotConfig, RobotSimulationConfig

BASELINE_PATH = Path("results/factory_perf_baseline.json")
THRESHOLD = 1.05  # +5% hard budget (tightened per T031 spec compliance)
SOFT_THRESHOLD = 1.08  # soft warn band (>5% and <=8%)
ITERATIONS = 5  # keep light for CI; baseline may have been generated with more


def _time_once(fn):  # minimal inline timing to avoid importing heavy script
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
    not BASELINE_PATH.exists(), reason="Baseline file missing; run baseline script first."
)
def test_factory_creation_mean_within_budget(monkeypatch):
    # Disable fast demo shortcut if present
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

    # Hard assertions (tightened)
    assert current_robot_mean <= base_robot * THRESHOLD, (
        f"Robot env creation mean {current_robot_mean:.2f}ms exceeds hard budget (+5% {base_robot * THRESHOLD:.2f}ms ceiling from baseline {base_robot:.2f}ms)"
    )
    assert current_image_mean <= base_image * THRESHOLD, (
        f"Image env creation mean {current_image_mean:.2f}ms exceeds hard budget (+5% {base_image * THRESHOLD:.2f}ms ceiling from baseline {base_image:.2f}ms)"
    )
    # Soft warnings (informational only within new narrow band)
    if (
        base_robot * THRESHOLD < current_robot_mean <= base_robot * SOFT_THRESHOLD
    ):  # pragma: no cover
        print(
            f"[PERF SOFT WARN] Robot env mean {current_robot_mean:.2f}ms within soft band (>+5% <=+8%) baseline {base_robot:.2f}ms"
        )
    if (
        base_image * THRESHOLD < current_image_mean <= base_image * SOFT_THRESHOLD
    ):  # pragma: no cover
        print(
            f"[PERF SOFT WARN] Image env mean {current_image_mean:.2f}ms within soft band (>+5% <=+8%) baseline {base_image:.2f}ms"
        )
