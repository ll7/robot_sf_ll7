"""Tests for the stream_gap gate-threshold calibration decision layer (issue #3558)."""

from __future__ import annotations

import pytest

from robot_sf.planner.stream_gap_gate_calibration import (
    AT_LEAST_AS_SAFE,
    CONSERVATIVE_RETENTION_DOMINATES,
    LESS_SAFE,
    STREAM_GAP_CALIBRATION_SCHEMA,
    GateSettingResult,
    SafetyTolerance,
    calibrate_stream_gap_gate,
    classify_setting_safety,
)

_BASELINE = GateSettingResult(
    thresholds={"retain": 1.0},
    unsafe_commit_rate=0.10,
    collision_rate=0.02,
    min_separation_m=0.50,
)


def _setting(name: str, unsafe: float, collision: float, sep: float) -> GateSettingResult:
    """Build a swept gate setting with the given safety aggregates."""
    return GateSettingResult(
        thresholds={"name": name},
        unsafe_commit_rate=unsafe,
        collision_rate=collision,
        min_separation_m=sep,
    )


def test_setting_no_worse_than_baseline_is_at_least_as_safe() -> None:
    """A setting that does not worsen any safety axis is at least as safe as retention."""
    setting = _setting("s", unsafe=0.10, collision=0.02, sep=0.50)

    assert classify_setting_safety(setting, _BASELINE) == AT_LEAST_AS_SAFE


def test_setting_worse_on_any_axis_is_less_safe() -> None:
    """Worsening any single safety axis must classify the setting as less safe."""
    worse_unsafe = _setting("a", unsafe=0.20, collision=0.02, sep=0.50)
    worse_collision = _setting("b", unsafe=0.10, collision=0.05, sep=0.50)
    worse_sep = _setting("c", unsafe=0.10, collision=0.02, sep=0.30)

    assert classify_setting_safety(worse_unsafe, _BASELINE) == LESS_SAFE
    assert classify_setting_safety(worse_collision, _BASELINE) == LESS_SAFE
    assert classify_setting_safety(worse_sep, _BASELINE) == LESS_SAFE


def test_tolerance_allows_small_regression() -> None:
    """A small regression within tolerance must still count as at least as safe."""
    setting = _setting("s", unsafe=0.11, collision=0.02, sep=0.50)
    tolerance = SafetyTolerance(unsafe_commit_abs=0.02)

    assert classify_setting_safety(setting, _BASELINE, tolerance) == AT_LEAST_AS_SAFE


def test_calibration_recommends_safest_member_of_safe_region() -> None:
    """When a safe region exists, the safest setting must be recommended."""
    settings = [
        _setting("strict", unsafe=0.05, collision=0.01, sep=0.60),  # safe, safest
        _setting("mid", unsafe=0.09, collision=0.02, sep=0.55),  # safe
        _setting("loose", unsafe=0.30, collision=0.06, sep=0.20),  # unsafe
    ]
    report = calibrate_stream_gap_gate(settings, _BASELINE)

    assert report["schema_version"] == STREAM_GAP_CALIBRATION_SCHEMA
    assert report["conclusion"] == "safe_region_exists"
    assert len(report["safe_region"]) == 2
    assert report["recommended_setting"]["thresholds"]["name"] == "strict"


def test_calibration_concludes_retention_dominates_when_none_safe() -> None:
    """If no setting clears the bar, conservative retention must dominate."""
    settings = [
        _setting("a", unsafe=0.30, collision=0.05, sep=0.30),
        _setting("b", unsafe=0.40, collision=0.07, sep=0.20),
    ]
    report = calibrate_stream_gap_gate(settings, _BASELINE)

    assert report["conclusion"] == CONSERVATIVE_RETENTION_DOMINATES
    assert report["safe_region"] == []
    assert report["recommended_setting"] is None


def test_calibration_rejects_empty_sweep() -> None:
    """An empty sweep cannot be calibrated."""
    with pytest.raises(ValueError):
        calibrate_stream_gap_gate([], _BASELINE)
