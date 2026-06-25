"""Tests for the stream_gap gate-threshold calibration sweep (#3558).

This exercises the deferred-run layer: the #3471 episode harness driven across a grid of
uncertainty-gate thresholds and fed into the merged ``calibrate_stream_gap_gate`` decision
layer. Evidence tier is diagnostic (controlled crossing scenario), matching the harness.
"""

from __future__ import annotations

import pytest

from scripts.validation.run_scenario_belief_episode_safety_issue_3471 import (
    EpisodeParams,
    _planner_config,
    run_episode,
)
from scripts.validation.run_stream_gap_gate_threshold_sweep_issue_3558 import (
    SCHEMA_VERSION,
    run_sweep,
)

# Episodes are deterministic per seed, so a small matrix is enough to fix the safe/unsafe
# classification; the sweep's value is the threshold contrast, not statistical power.
_FAST = EpisodeParams(max_steps=50)
_SEEDS = [101, 102, 103]


def test_gate_threshold_override_changes_dropping_behavior() -> None:
    """A permissive existence threshold (<= degraded 0.2) must retain the agent; a strict one drops it.

    Retaining is behaviorally conservative retention, so it must match the gate-off contrast on
    the safety metrics; dropping must be strictly worse.
    """
    permissive = run_episode(
        "uncertain_dropped",
        seed=101,
        params=_FAST,
        gate_thresholds={"uncertainty_min_existence_probability": 0.1},
    )
    strict = run_episode(
        "uncertain_dropped",
        seed=101,
        params=_FAST,
        gate_thresholds={"uncertainty_min_existence_probability": 0.5},
    )
    retained = run_episode("uncertain_retained", seed=101, params=_FAST)

    # Permissive gate keeps the degraded agent -> identical to conservative retention.
    assert permissive["unsafe_commit_steps"] == retained["unsafe_commit_steps"]
    assert permissive["collision"] == retained["collision"]
    # Strict gate drops it -> at least as much unsafe commitment as retention.
    assert strict["unsafe_commit_steps"] >= retained["unsafe_commit_steps"]


def test_unknown_gate_threshold_override_fails_closed() -> None:
    """An unknown gate-threshold key must raise rather than be silently ignored."""
    with pytest.raises(ValueError, match="unknown gate threshold"):
        _planner_config("uncertain_dropped", {"not_a_real_threshold": 0.5})


def test_sweep_finds_safe_region_below_degraded_existence() -> None:
    """The sweep must classify <=0.2 thresholds safe and >0.2 thresholds less safe."""
    report = run_sweep(_SEEDS, _FAST, existence_grid=(0.1, 0.2, 0.3, 0.5))

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["issue"] == 3558
    assert report["evidence_tier"] == "diagnostic"
    assert report["active_gate_axis"] == "uncertainty_min_existence_probability"

    calibration = report["calibration"]
    by_existence = {
        row["thresholds"]["uncertainty_min_existence_probability"]: row["classification"]
        for row in calibration["settings"]
    }
    assert by_existence[0.1] == "at_least_as_safe"
    assert by_existence[0.2] == "at_least_as_safe"
    assert by_existence[0.3] == "less_safe"
    assert by_existence[0.5] == "less_safe"


def test_sweep_recommends_a_safe_setting_and_concludes_region_exists() -> None:
    """When a safe region exists the calibration must recommend a member of it."""
    report = run_sweep(_SEEDS, _FAST, existence_grid=(0.1, 0.2, 0.5))
    calibration = report["calibration"]

    assert calibration["conclusion"] == "safe_region_exists"
    recommended = calibration["recommended_setting"]
    assert recommended is not None
    # The recommended (safest) setting must be permissive enough to retain the degraded agent.
    assert recommended["thresholds"]["uncertainty_min_existence_probability"] <= 0.2


def test_sweep_reports_strict_default_gate_is_less_safe() -> None:
    """The production default (0.5) existence gate must land in the less-safe set, per #3471."""
    report = run_sweep(_SEEDS, _FAST, existence_grid=(0.5,))
    only = report["calibration"]["settings"][0]

    assert only["thresholds"]["uncertainty_min_existence_probability"] == 0.5
    assert only["classification"] == "less_safe"


def test_sweep_rejects_empty_seeds() -> None:
    """An empty seed set cannot be swept."""
    with pytest.raises(ValueError):
        run_sweep([], _FAST)
