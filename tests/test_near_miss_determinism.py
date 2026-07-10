"""Focused tests for the near-miss rerun nondeterminism harness (issue #5140).

These tests are CPU-level only: they prove (1) the metric reduction is
deterministic on fixed input, (2) the exact-repeat measurement reports the
correct deviation shape and detects injected variance, (3) the SNQI propagation
bound matches the documented formula and fails closed on bad inputs, and
(4) the headline empirical claim from the repro-contract update (exact-repeat
near-miss deviation is zero on the supported scenarios) holds.

No campaigns, no Slurm, no training: every case runs a handful of short
headless episodes or pure-NumPy reductions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robot_sf.benchmark.near_miss_determinism import (
    CANONICAL_W_NEAR_V3,
    DEFAULT_TRACKED_METRICS,
    SCHEMA_VERSION,
    measure_exact_repeat_nondeterminism,
    metric_path_is_deterministic,
    snqi_near_miss_propagation_bound,
)

SMOKE_SCENARIO: dict[str, Any] = {
    "id": "smoke-uni-low-open",
    "density": "low",
    "flow": "uni",
    "obstacle": "open",
    "groups": 0.0,
    "speed_var": "low",
    "goal_topology": "point",
    "robot_context": "embedded",
    "repeats": 1,
}


def _straddling_trajectories() -> tuple[np.ndarray, np.ndarray]:
    """Trajectories whose surface clearance straddles the 0.5 m near-miss band."""
    rng = np.random.default_rng(0)
    T, K = 60, 5
    robot_pos = rng.uniform(0.0, 5.0, size=(T, 2))
    peds_pos = rng.uniform(0.0, 5.0, size=(T, K, 2))
    # Force one pedestrian to hover near the robot so clearances cross the band.
    peds_pos[:, 0, :] = robot_pos + np.array([0.75, 0.0])
    return robot_pos, peds_pos


# --- metric_path_is_deterministic -------------------------------------------


def test_metric_path_is_deterministic_on_straddling_input():
    """Repeated metric reduction on a fixed input must be bit-identical."""
    robot_pos, peds_pos = _straddling_trajectories()
    report = metric_path_is_deterministic(
        robot_pos,
        peds_pos,
        robot_radius=0.25,
        ped_radius=0.25,
        n_invocations=20,
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["n_invocations"] == 20
    assert report["is_deterministic"] is True
    assert report["n_distinct_results"] == 1
    # The straddling input must actually exercise the near-miss counter, else the
    # proof would be vacuous.
    assert report["sample"]["near_misses"] > 0
    assert "near_misses" in report["tracked_aggregates"]


def test_metric_path_is_deterministic_detects_mutation():
    """Internal copies mean feeding the same objects twice is still deterministic."""
    robot_pos, peds_pos = _straddling_trajectories()
    report = metric_path_is_deterministic(
        robot_pos,
        peds_pos,
        robot_radius=0.25,
        ped_radius=0.25,
        n_invocations=3,
    )
    assert report["is_deterministic"] is True


def test_metric_path_rejects_invalid_arguments():
    """Invalid invocation counts and empty trajectories must raise."""
    robot_pos, peds_pos = _straddling_trajectories()
    with pytest.raises(ValueError, match="n_invocations"):
        metric_path_is_deterministic(
            robot_pos, peds_pos, robot_radius=0.25, ped_radius=0.25, n_invocations=1
        )
    with pytest.raises(ValueError, match="empty"):
        metric_path_is_deterministic(
            np.zeros((0, 2)), np.zeros((0, 1, 2)), robot_radius=0.25, ped_radius=0.25
        )


# --- measure_exact_repeat_nondeterminism ------------------------------------


def _fake_runner_factory(deviate_after: int, metric_value: float, deviated_value: float):
    """Build a deterministic stub runner that flips one metric on a chosen repeat.

    This lets the measurement harness be tested for variance detection without
    depending on genuine simulation nondeterminism.
    """

    state = {"i": 0}

    def _run(scenario, *, seed, horizon, dt, record_forces):
        del scenario, seed, horizon, dt, record_forces  # stub signature mirrors the real runner
        i = state["i"]
        state["i"] += 1
        nm = deviated_value if i >= deviate_after else metric_value
        # All tracked metrics present; near_misses flips.
        return {
            "metrics": {k: nm if k == "near_misses" else 0.0 for k in DEFAULT_TRACKED_METRICS},
        }

    return _run


def test_exact_repeat_measurement_reports_zero_deviation_when_deterministic():
    """A deterministic runner yields zero deviation and a bit-identical report."""
    runner = _fake_runner_factory(deviate_after=999, metric_value=5.0, deviated_value=5.0)
    report = measure_exact_repeat_nondeterminism(
        SMOKE_SCENARIO, seed=123, n_repeats=4, horizon=5, run_episode=runner
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["summary"]["n_repeats"] == 4
    assert report["summary"]["any_nondeterministic_metric"] is False
    assert report["summary"]["near_misses_max_deviation"] == 0.0
    nm = report["exact_repeat"]["near_misses"]
    assert nm["bit_identical"] is True
    assert nm["n_distinct_values"] == 1
    assert nm["available"] is True
    # Diagnostics captured for cross-machine legibility.
    assert "numpy_version" in report["diagnostics"]


def test_exact_repeat_measurement_detects_injected_variance():
    """A runner that flips near_misses on a later repeat is detected as nondeterministic."""
    runner = _fake_runner_factory(deviate_after=2, metric_value=5.0, deviated_value=7.0)
    report = measure_exact_repeat_nondeterminism(
        SMOKE_SCENARIO, seed=123, n_repeats=4, horizon=5, run_episode=runner
    )

    assert report["summary"]["any_nondeterministic_metric"] is True
    assert report["summary"]["near_misses_max_deviation"] == pytest.approx(2.0)
    nm = report["exact_repeat"]["near_misses"]
    assert nm["bit_identical"] is False
    assert nm["n_distinct_values"] == 2
    assert nm["min"] == 5.0
    assert nm["max"] == 7.0


def test_exact_repeat_measurement_rejects_invalid_arguments():
    """Too few repeats and an empty metric list must raise."""
    runner = _fake_runner_factory(999, 1.0, 1.0)
    with pytest.raises(ValueError, match="n_repeats"):
        measure_exact_repeat_nondeterminism(SMOKE_SCENARIO, seed=1, n_repeats=1, run_episode=runner)
    with pytest.raises(ValueError, match="tracked_metrics"):
        measure_exact_repeat_nondeterminism(
            SMOKE_SCENARIO, seed=1, n_repeats=2, tracked_metrics=(), run_episode=runner
        )


def test_exact_repeat_measurement_fails_closed_on_missing_required_metric():
    """A missing required metric must raise rather than look like zero deviation."""

    def _runner(scenario, *, seed, horizon, dt, record_forces):
        del scenario, seed, horizon, dt, record_forces  # stub signature mirrors the real runner
        # Emits near_misses but omits the other required metrics.
        return {"metrics": {"near_misses": 1.0}}

    with pytest.raises(ValueError, match="required metric"):
        measure_exact_repeat_nondeterminism(
            SMOKE_SCENARIO, seed=1, n_repeats=2, horizon=5, run_episode=_runner
        )


def test_exact_repeat_measurement_skips_optional_absent_metric():
    """A tracked-but-optional metric is recorded unavailable, not raised."""

    def _runner(scenario, *, seed, horizon, dt, record_forces):
        del scenario, seed, horizon, dt, record_forces  # stub signature mirrors the real runner
        # All required metrics present; the optional time_to_goal absent.
        metrics = dict.fromkeys(DEFAULT_TRACKED_METRICS, 1.0)
        metrics.pop("time_to_goal", None)
        return {"metrics": metrics}

    report = measure_exact_repeat_nondeterminism(
        SMOKE_SCENARIO, seed=1, n_repeats=2, horizon=5, run_episode=_runner
    )
    ttg = report["exact_repeat"]["time_to_goal"]
    assert ttg["available"] is False
    assert ttg["values"] == []
    assert report["exact_repeat"]["near_misses"]["available"] is True


@pytest.mark.timeout(120)
def test_exact_repeat_near_miss_is_deterministic_on_smoke_scenario():
    """Headline empirical claim: exact-repeat near-miss deviation is zero.

    This is the executable evidence cited by the updated repro contract. It runs
    a small number of short headless episodes; on the supported CPU it must be
    bit-identical for ``near_misses``. If a future machine/compiler surfaces
    genuine fastmath-driven divergence, this test becomes the detector (mirrors
    issue #4978's exact-repeat determinism contract).
    """
    report = measure_exact_repeat_nondeterminism(SMOKE_SCENARIO, seed=123, n_repeats=5, horizon=30)
    assert report["exact_repeat"]["near_misses"]["bit_identical"] is True
    assert report["summary"]["near_misses_max_deviation"] == 0.0


# --- snqi_near_miss_propagation_bound ---------------------------------------


def test_snqi_bound_linear_region_matches_formula():
    """In the linear region the bound equals w_near * tol / (p95 - med)."""
    bound = snqi_near_miss_propagation_bound(
        near_miss_tolerance=2.0,
        w_near=CANONICAL_W_NEAR_V3,
        baseline_med=10.0,
        baseline_p95=30.0,
    )
    expected_linear = CANONICAL_W_NEAR_V3 * 2.0 / (30.0 - 10.0)
    assert bound["linear_region_bound"] == pytest.approx(expected_linear)
    # Linear bound is well below the clip cap, so it wins.
    assert bound["propagation_bound"] == pytest.approx(expected_linear)
    assert bound["clip_capped_bound"] == pytest.approx(CANONICAL_W_NEAR_V3)


def test_snqi_bound_clips_when_linear_exceeds_weight():
    """When the linear bound exceeds w_near, the [0,1] clamp caps it at w_near."""
    # A huge tolerance relative to the spread would exceed w_near; the clip caps it.
    bound = snqi_near_miss_propagation_bound(
        near_miss_tolerance=1_000.0,
        w_near=0.31,
        baseline_med=0.0,
        baseline_p95=1.0,
    )
    assert bound["linear_region_bound"] == pytest.approx(0.31 * 1000.0)
    assert bound["clip_capped_bound"] == pytest.approx(0.31)
    assert bound["propagation_bound"] == pytest.approx(0.31)


def test_snqi_bound_without_baseline_is_clip_capped_only():
    """Without baseline stats only the unconditional clip-capped bound is reported."""
    bound = snqi_near_miss_propagation_bound(near_miss_tolerance=0.0)
    assert bound["linear_region_bound"] is None
    assert bound["clip_capped_bound"] == pytest.approx(CANONICAL_W_NEAR_V3)
    assert bound["propagation_bound"] == pytest.approx(CANONICAL_W_NEAR_V3)


def test_snqi_bound_zero_tolerance_is_zero_linear():
    """A zero near-miss tolerance yields a zero SNQI propagation bound."""
    bound = snqi_near_miss_propagation_bound(
        near_miss_tolerance=0.0, baseline_med=10.0, baseline_p95=30.0
    )
    assert bound["linear_region_bound"] == 0.0
    assert bound["propagation_bound"] == 0.0


def test_snqi_bound_rejects_invalid_arguments():
    """Invalid tolerances, weights, and baseline spreads must raise."""
    with pytest.raises(ValueError, match="non-negative"):
        snqi_near_miss_propagation_bound(-1.0)
    with pytest.raises(ValueError, match="w_near"):
        snqi_near_miss_propagation_bound(1.0, w_near=-0.5)
    with pytest.raises(ValueError, match="spread"):
        snqi_near_miss_propagation_bound(1.0, baseline_med=5.0, baseline_p95=5.0)
