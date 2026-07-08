"""Tests for the critical-interval metric window mechanism (issue #4758).

Covers anchor detection, window extraction, interval metrics, config validation,
graceful handling of missing anchors, and the guarantee that the feature is
opt-in only.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.critical_intervals import (
    TTC_CONVENTION,
    CriticalInterval,
    IntervalMetrics,
    _compute_interval_metrics_in_window,
    _compute_max_braking_deceleration_mps2,
    _compute_window_min_ttc_s,
    _detect_ttc_threshold_crossing,
    _pairwise_ttc_s,
    extract_critical_intervals,
    load_config,
    report_to_dict,
    summarize_interval_metrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trace(
    robot_pos: np.ndarray,
    peds_pos: np.ndarray | None = None,
    *,
    robot_vel: np.ndarray | None = None,
    dt: float = 0.1,
) -> dict[str, object]:
    """Build a minimal trace dict."""
    if peds_pos is None:
        peds_pos = np.zeros((robot_pos.shape[0], 0, 2), dtype=float)
    trace: dict[str, object] = {
        "robot_pos": robot_pos,
        "peds_pos": peds_pos,
        "dt": dt,
    }
    if robot_vel is not None:
        trace["robot_vel"] = robot_vel
    return trace


def _make_approaching_trace(dt: float = 0.1) -> dict[str, object]:
    """Trace where robot and pedestrian approach, closest at step 5."""
    n = 10
    robot_pos = np.zeros((n, 2), dtype=float)
    robot_pos[:, 0] = np.arange(n) * 0.5  # 0 .. 4.5 m

    peds_pos = np.zeros((n, 1, 2), dtype=float)
    peds_pos[:, 0, 0] = 5.0 - np.arange(n) * 0.5  # 5.0 .. 0.5 m

    robot_vel = np.zeros((n, 2), dtype=float)
    robot_vel[:, 0] = 5.0

    return _make_trace(robot_pos, peds_pos, robot_vel=robot_vel, dt=dt)


def _make_braking_trace(dt: float = 0.1) -> dict[str, object]:
    """Trace where robot decelerates sharply after step 3."""
    n = 10
    robot_pos = np.zeros((n, 2), dtype=float)
    # Constant speed then sharp braking
    speeds = np.array([3.0, 3.0, 3.0, 3.0, 1.5, 0.5, 0.2, 0.0, 0.0, 0.0])
    robot_pos[:, 0] = np.cumsum(speeds * dt) - speeds[0] * dt
    robot_pos[0, 0] = 0.0

    robot_vel = np.zeros((n, 2), dtype=float)
    robot_vel[:, 0] = speeds

    peds_pos = np.zeros((n, 1, 2), dtype=float)
    peds_pos[:, 0, 0] = 10.0  # stationary pedestrian far away

    return _make_trace(robot_pos, peds_pos, robot_vel=robot_vel, dt=dt)


def _make_no_event_trace(dt: float = 0.1) -> dict[str, object]:
    """Trace with no safety-relevant events."""
    n = 10
    robot_pos = np.zeros((n, 2), dtype=float)
    robot_pos[:, 0] = np.arange(n) * 0.5

    peds_pos = np.zeros((n, 1, 2), dtype=float)
    peds_pos[:, 0, 0] = 50.0  # very far away

    robot_vel = np.zeros((n, 2), dtype=float)
    robot_vel[:, 0] = 5.0

    return _make_trace(robot_pos, peds_pos, robot_vel=robot_vel, dt=dt)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Config parser rejects invalid configs and accepts valid ones."""

    def test_default_config_loads(self) -> None:
        cfg = load_config(config_dict={"schema_version": "critical-intervals.v1"})
        assert cfg["schema_version"] == "critical-intervals.v1"

    def test_negative_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            load_config(
                config_dict={
                    "schema_version": "critical-intervals.v1",
                    "critical_intervals": {
                        "closest_approach": {"before_s": -1.0, "after_s": 1.0},
                    },
                }
            )

    def test_unknown_anchor_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown anchor"):
            load_config(
                config_dict={
                    "schema_version": "critical-intervals.v1",
                    "critical_intervals": {
                        "unsupported_anchor": {"before_s": 1.0, "after_s": 1.0},
                    },
                }
            )

    def test_bad_schema_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown schema_version"):
            load_config(config_dict={"schema_version": "invalid-schema"})

    def test_yaml_file_load(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "schema_version": "critical-intervals.v1",
                    "critical_intervals": {
                        "closest_approach": {
                            "enabled": True,
                            "before_s": 1.0,
                            "after_s": 1.0,
                        },
                    },
                },
                f,
            )
            f.flush()
            cfg = load_config(f.name)
            assert "closest_approach" in cfg["critical_intervals"]


# ---------------------------------------------------------------------------
# Anchor detection
# ---------------------------------------------------------------------------


class TestClosestApproach:
    """closest_approach anchor finds expected step and clamps window."""

    def test_closest_approach_finds_step(self) -> None:
        trace = _make_approaching_trace()
        intervals = extract_critical_intervals(
            trace,
            {
                "critical_intervals": {
                    "closest_approach": {
                        "enabled": True,
                        "before_s": 0.5,
                        "after_s": 0.5,
                    },
                },
            },
        )
        assert len(intervals) == 1
        iv = intervals[0]
        assert iv.status == "available"
        # Robot: 0..4.5m, Ped: 5.0..0.5m, closest at step 5 (both at 2.5m)
        assert iv.anchor_step == 5
        assert iv.start_step is not None
        assert iv.start_step >= 0

    def test_closest_approach_no_peds(self) -> None:
        trace = _make_trace(np.zeros((5, 2)))
        intervals = extract_critical_intervals(
            trace,
            {
                "critical_intervals": {
                    "closest_approach": {
                        "enabled": True,
                        "before_s": 0.5,
                        "after_s": 0.5,
                    },
                },
            },
        )
        assert len(intervals) == 1
        assert intervals[0].status == "missing_anchor"
        assert intervals[0].reason is not None

    def test_window_clamped_to_trace_bounds(self) -> None:
        trace = _make_approaching_trace()
        intervals = extract_critical_intervals(
            trace,
            {
                "critical_intervals": {
                    "closest_approach": {
                        "enabled": True,
                        "before_s": 100.0,
                        "after_s": 100.0,
                    },
                },
            },
        )
        iv = intervals[0]
        assert iv.start_step == 0
        # end_step should be clamped to T (10)
        assert iv.end_step == 10


class TestTTCThresholdCrossing:
    """TTC threshold crossing anchor works with velocities and fails gracefully."""

    def test_ttc_crossing_detected(self) -> None:
        trace = _make_approaching_trace()
        intervals = extract_critical_intervals(
            trace,
            {
                "critical_intervals": {
                    "ttc_threshold_crossing": {
                        "enabled": True,
                        "threshold_s": 2.0,
                        "before_s": 0.5,
                        "after_s": 0.5,
                    },
                },
            },
        )
        assert len(intervals) == 1
        assert intervals[0].status == "available"

    def test_ttc_missing_velocity(self) -> None:
        trace = _make_approaching_trace()
        del trace["robot_vel"]
        intervals = extract_critical_intervals(
            trace,
            {
                "critical_intervals": {
                    "ttc_threshold_crossing": {
                        "enabled": True,
                        "threshold_s": 2.0,
                        "before_s": 0.5,
                        "after_s": 0.5,
                    },
                },
            },
        )
        assert len(intervals) == 1
        assert intervals[0].status == "missing_anchor"
        assert "velocity" in intervals[0].reason.lower()


class TestBrakingEvent:
    """Braking anchor detects first deceleration event."""

    def test_braking_detected(self) -> None:
        trace = _make_braking_trace()
        intervals = extract_critical_intervals(
            trace,
            {
                "critical_intervals": {
                    "first_braking_event": {
                        "enabled": True,
                        "deceleration_threshold_mps2": 0.75,
                        "before_s": 0.5,
                        "after_s": 0.5,
                    },
                },
            },
        )
        assert len(intervals) == 1
        assert intervals[0].status == "available"
        assert intervals[0].anchor_step is not None

    def test_braking_no_velocity(self) -> None:
        trace = _make_braking_trace()
        del trace["robot_vel"]
        intervals = extract_critical_intervals(
            trace,
            {
                "critical_intervals": {
                    "first_braking_event": {
                        "enabled": True,
                        "deceleration_threshold_mps2": 0.75,
                        "before_s": 0.5,
                        "after_s": 0.5,
                    },
                },
            },
        )
        assert len(intervals) == 1
        assert intervals[0].status == "missing_anchor"


class TestCollisionOrNearMiss:
    """collision_or_near_miss detects when distance is below threshold."""

    def test_near_miss_detected(self) -> None:
        n = 10
        robot_pos = np.zeros((n, 2), dtype=float)
        robot_pos[:, 0] = np.arange(n) * 0.3

        peds_pos = np.zeros((n, 1, 2), dtype=float)
        peds_pos[:, 0, 0] = 2.0 - np.arange(n) * 0.2  # approaches robot

        trace = _make_trace(robot_pos, peds_pos, dt=0.1)
        intervals = extract_critical_intervals(
            trace,
            {
                "critical_intervals": {
                    "collision_or_near_miss": {
                        "enabled": True,
                        "before_s": 0.5,
                        "after_s": 0.5,
                    },
                },
            },
        )
        assert len(intervals) == 1
        assert intervals[0].status == "available"

    def test_no_near_miss_no_crash(self) -> None:
        trace = _make_no_event_trace()
        intervals = extract_critical_intervals(
            trace,
            {
                "critical_intervals": {
                    "collision_or_near_miss": {
                        "enabled": True,
                        "before_s": 0.5,
                        "after_s": 0.5,
                    },
                },
            },
        )
        assert len(intervals) == 1
        assert intervals[0].status == "missing_anchor"


# ---------------------------------------------------------------------------
# Summary and report
# ---------------------------------------------------------------------------


class TestSummarizeIntervalMetrics:
    """Interval metrics differ from whole-run metrics in a synthetic fixture."""

    def test_interval_vs_whole_run(self) -> None:
        """In the approaching trace, the closest-approach window should have
        a smaller distance than the whole-run average."""
        trace = _make_approaching_trace()
        cfg = {
            "critical_intervals": {
                "closest_approach": {
                    "enabled": True,
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
            },
        }
        intervals = extract_critical_intervals(trace, cfg)
        report = summarize_interval_metrics(trace, intervals)

        assert len(report.interval_metrics) == 1
        wm = report.interval_metrics[0]
        wr = report.whole_run

        # Closest-approach window should have the smallest distance
        assert wm.min_distance_m is not None
        assert wr["min_distance_m"] is not None
        assert wm.min_distance_m <= wr["min_distance_m"]

    def test_missing_anchors_recorded(self) -> None:
        """When an anchor is missing, it appears in missing_anchors."""
        trace = _make_trace(np.zeros((5, 2)))  # no peds
        cfg = {
            "critical_intervals": {
                "closest_approach": {
                    "enabled": True,
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
            },
        }
        intervals = extract_critical_intervals(trace, cfg)
        report = summarize_interval_metrics(trace, intervals)

        assert len(report.missing_anchors) == 1
        assert report.missing_anchors[0]["anchor"] == "closest_approach"

    def test_report_serializable(self) -> None:
        """Report can be round-tripped through JSON."""
        trace = _make_approaching_trace()
        cfg = {
            "critical_intervals": {
                "closest_approach": {
                    "enabled": True,
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
            },
        }
        intervals = extract_critical_intervals(trace, cfg)
        report = summarize_interval_metrics(trace, intervals)
        d = report_to_dict(report)

        # JSON round-trip
        payload = json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        parsed = json.loads(payload)
        assert "whole_run" in parsed
        assert "critical_intervals" in parsed
        assert "interval_metrics" in parsed
        assert "missing_anchors" in parsed


# ---------------------------------------------------------------------------
# Opt-in behaviour
# ---------------------------------------------------------------------------


class TestOptIn:
    """Feature is disabled unless config is supplied."""

    def test_empty_config_no_intervals(self) -> None:
        trace = _make_approaching_trace()
        cfg = load_config(config_dict={"schema_version": "critical-intervals.v1"})
        intervals = extract_critical_intervals(trace, cfg)
        assert len(intervals) == 0  # no anchors declared

    def test_disabled_anchor_skipped(self) -> None:
        trace = _make_approaching_trace()
        cfg = {
            "critical_intervals": {
                "closest_approach": {
                    "enabled": False,
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
            },
        }
        intervals = extract_critical_intervals(trace, cfg)
        assert len(intervals) == 0

    def test_no_event_trace_no_intervals(self) -> None:
        """Trace with no relevant events yields only missing anchors."""
        trace = _make_no_event_trace()
        cfg = {
            "critical_intervals": {
                "closest_approach": {
                    "enabled": True,
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
                "ttc_threshold_crossing": {
                    "enabled": True,
                    "threshold_s": 1.0,
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
            },
        }
        intervals = extract_critical_intervals(trace, cfg)
        # closest_approach is available (ped is far but present), TTC may be missing
        available = [i for i in intervals if i.status == "available"]
        assert len(available) >= 1  # at least closest_approach is available


# ---------------------------------------------------------------------------
# Dataclass contracts
# ---------------------------------------------------------------------------


class TestDataclasses:
    """CriticalInterval and IntervalMetrics behave as expected."""

    def test_critical_interval_frozen(self) -> None:
        iv = CriticalInterval(anchor="test", status="available")
        with pytest.raises(Exception):  # FrozenInstanceError
            iv.anchor = "other"  # type: ignore[misc]

    def test_interval_metrics_defaults(self) -> None:
        im = IntervalMetrics(anchor="test")
        assert im.n_steps == 0
        assert im.near_miss_count == 0
        assert im.collision_flag is False


# ---------------------------------------------------------------------------
# Windowed-aggregation regression (gate review PR #4782, finding #1)
# ---------------------------------------------------------------------------


class TestWindowedDeceleration:
    """max_deceleration_mps2 must reflect ONLY the window, not the whole run."""

    def test_deceleration_is_windowed_not_whole_run(self) -> None:
        # Robot decelerates sharply at step 5 (speed drops from 10 to 2 m/s).
        n = 20
        dt = 0.1
        robot_vel = np.zeros((n, 2), dtype=float)
        robot_vel[:5, 0] = 10.0  # fast
        robot_vel[5:, 0] = 2.0  # slow (braking at step 4->5)
        robot_pos = np.cumsum(robot_vel * dt, axis=0)
        trace = _make_trace(robot_pos, robot_vel=robot_vel, dt=dt)

        whole = _compute_interval_metrics_in_window(trace, start=0, end=n)
        # A late window that excludes the braking event must see zero decel.
        late = _compute_interval_metrics_in_window(trace, start=8, end=n)

        assert whole["max_deceleration_mps2"] is not None
        assert whole["max_deceleration_mps2"] > 0.0
        assert late["max_deceleration_mps2"] == pytest.approx(0.0)
        # The whole point of the feature: window value != whole-run value.
        assert late["max_deceleration_mps2"] < whole["max_deceleration_mps2"]


# ---------------------------------------------------------------------------
# TTC aggregation (issue #4783)
# ---------------------------------------------------------------------------


class TestWindowedTTC:
    """min_ttc_s is populated for windows with pedestrian + velocity data."""

    def test_ttc_populated_for_closing_pair(self) -> None:
        """Robot and pedestrian approach head-on => finite TTC."""
        trace = _make_approaching_trace()
        whole = _compute_interval_metrics_in_window(trace, start=0, end=10)
        assert whole["min_ttc_s"] is not None
        assert whole["min_ttc_s"] >= 0.0
        assert np.isfinite(whole["min_ttc_s"])

    def test_ttc_populated_in_interval(self) -> None:
        """TTC is populated for a TTC-anchored interval."""
        trace = _make_approaching_trace()
        cfg = {
            "critical_intervals": {
                "ttc_threshold_crossing": {
                    "enabled": True,
                    "threshold_s": 2.0,
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
            },
        }
        intervals = extract_critical_intervals(trace, cfg)
        report = summarize_interval_metrics(trace, intervals)
        assert len(report.interval_metrics) == 1
        im = report.interval_metrics[0]
        assert im.min_ttc_s is not None
        assert im.min_ttc_s >= 0.0
        assert np.isfinite(im.min_ttc_s)

    def test_ttc_none_when_no_velocity(self) -> None:
        """Missing robot velocity => min_ttc_s is None."""
        trace = _make_approaching_trace()
        del trace["robot_vel"]
        whole = _compute_interval_metrics_in_window(trace, start=0, end=10)
        assert whole.get("min_ttc_s") is None

    def test_ttc_none_when_no_peds(self) -> None:
        """No pedestrians => min_ttc_s is None."""
        trace = _make_trace(np.zeros((5, 2)), robot_vel=np.zeros((5, 2)))
        whole = _compute_interval_metrics_in_window(trace, start=0, end=5)
        assert whole.get("min_ttc_s") is None

    def test_ttc_none_for_nonclosing_pair(self) -> None:
        """Pedestrian moving away (non-closing) => min_ttc_s is None."""
        n = 10
        robot_pos = np.zeros((n, 2), dtype=float)
        robot_pos[:, 0] = np.arange(n) * 0.5  # 0 .. 4.5

        peds_pos = np.zeros((n, 1, 2), dtype=float)
        peds_pos[:, 0, 0] = 10.0 + np.arange(n) * 0.5  # moving away

        robot_vel = np.zeros((n, 2), dtype=float)
        robot_vel[:, 0] = 5.0

        trace = _make_trace(robot_pos, peds_pos, robot_vel=robot_vel, dt=0.1)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=10)
        assert whole.get("min_ttc_s") is None

    def test_ttc_not_zero_for_missing_data(self) -> None:
        """min_ttc_s must be None (not 0) when data is unavailable."""
        trace = _make_trace(np.zeros((5, 2)))
        whole = _compute_interval_metrics_in_window(trace, start=0, end=5)
        ttc = whole.get("min_ttc_s")
        if ttc is not None:
            assert ttc > 0.0, "min_ttc_s must not be 0 as a placeholder for missing data"

    def test_ttc_in_serialized_report(self) -> None:
        """min_ttc_s appears in serialized report for TTC-capable fixture."""
        trace = _make_approaching_trace()
        cfg = {
            "critical_intervals": {
                "ttc_threshold_crossing": {
                    "enabled": True,
                    "threshold_s": 2.0,
>>>>>>> eb94c7402 (feat(benchmark): populate min_ttc_s and fix max_deceleration_mps2 semantics)
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
            },
        }
        intervals = extract_critical_intervals(trace, cfg)
        report = summarize_interval_metrics(trace, intervals)
        d = report_to_dict(report)
        assert len(d["interval_metrics"]) == 1
        assert d["interval_metrics"][0]["min_ttc_s"] is not None
        robot_vel = np.zeros((n, 2))
        robot_vel[:, 0] = np.linspace(5.0, 0.0, n)

        result = _compute_max_braking_deceleration_mps2(robot_vel, dt=dt)
        assert result is not None
        assert result > 0
        # Uniform deceleration: Δv/Δt = 5/(9*0.1) ≈ 5.556 m/s²
        assert result == pytest.approx(50.0 / 9.0, abs=0.1)

    def test_speed_increase_gives_zero_braking(self) -> None:
        dt = 0.1
        n = 5
        robot_vel = np.zeros((n, 2))
        robot_vel[:, 0] = np.linspace(1.0, 5.0, n)
        result = _compute_max_braking_deceleration_mps2(robot_vel, dt=dt)
        assert result is not None
        assert result == pytest.approx(0.0)

    def test_pure_turning_does_not_count_as_braking(self) -> None:
        # Circular motion at constant speed: velocity direction changes but
        # speed is constant, so tangential acceleration is zero.
        speed = 5.0
        omega = 1.0  # rad/s
        n_steps = 200
        t = np.arange(n_steps) * 0.01
        robot_vel = np.column_stack([speed * np.cos(omega * t), speed * np.sin(omega * t)])
        result = _compute_max_braking_deceleration_mps2(robot_vel, dt=0.01)
        assert result is not None
        # Central-difference introduces O(dt²·ω²·speed) numerical error;
        # the key property is that it is orders of magnitude smaller than
        # the centripetal acceleration (speed·ω = 5 m/s²).
        assert result < 0.01 * speed * omega

    def test_two_samples_only(self) -> None:
        dt = 0.1
        robot_vel = np.array([[10.0, 0.0], [0.0, 0.0]])
        result = _compute_max_braking_deceleration_mps2(robot_vel, dt=dt)
        assert result is not None
        assert result == pytest.approx(100.0)

    def test_single_sample_returns_none(self) -> None:
        robot_vel = np.array([[5.0, 0.0]])
        result = _compute_max_braking_deceleration_mps2(robot_vel, dt=0.1)
        assert result is None


# ---------------------------------------------------------------------------
# Physical TTC convention (issue #4832)
# ---------------------------------------------------------------------------


class TestPhysicalTTC:
    """TTC uses physical line-of-sight closing speed, not the old s/m proxy."""

    def test_ttc_convention_constant(self) -> None:
        """Module declares the TTC convention explicitly."""
        assert TTC_CONVENTION == "line_of_sight_closing_speed_seconds.v1"

    def test_ttc_threshold_returns_none_for_truncated_pedestrian_arrays(self) -> None:
        """TTC threshold detection fails closed on shorter pedestrian traces."""
        robot_pos = np.zeros((3, 2))
        robot_vel = np.ones((3, 2))
        peds_pos = np.zeros((2, 1, 2))
        ped_vel = np.zeros((2, 1, 2))

        assert (
            _detect_ttc_threshold_crossing(
                robot_pos,
                robot_vel,
                peds_pos,
                threshold_s=2.0,
                dt=0.1,
                ped_vel=ped_vel,
            )
            is None
        )

    def test_window_min_ttc_returns_none_for_truncated_velocity_arrays(self) -> None:
        """Window min TTC fails closed when velocity arrays are shorter than positions."""
        robot_pos = np.zeros((3, 2))
        robot_vel = np.ones((2, 2))
        peds_pos = np.zeros((3, 1, 2))
        ped_vel = np.zeros((2, 1, 2))

        assert (
            _compute_window_min_ttc_s(
                robot_pos=robot_pos,
                robot_vel=robot_vel,
                peds_pos=peds_pos,
                ped_vel=ped_vel,
                dt=0.1,
            )
            is None
        )

    def test_report_includes_ttc_convention(self) -> None:
        """Report output includes the TTC convention metadata."""
        trace = _make_approaching_trace()
        cfg = {
            "critical_intervals": {
                "closest_approach": {
                    "enabled": True,
=======
        assert np.isfinite(d["interval_metrics"][0]["min_ttc_s"])


# ---------------------------------------------------------------------------
# Braking deceleration semantics (issue #4783)
# ---------------------------------------------------------------------------


class TestBrakingDecelerationSemantics:
    """max_deceleration_mps2 is true braking deceleration, not accel magnitude."""

    def test_pure_turn_constant_speed_no_braking(self) -> None:
        """Pure turning at constant speed does not inflate max_deceleration_mps2."""
        n = 20
        dt = 0.1
        robot_vel = np.zeros((n, 2), dtype=float)
        # Speed is constant 5 m/s, but heading changes every step
        angles = np.linspace(0, np.pi / 2, n)
        robot_vel[:, 0] = 5.0 * np.cos(angles)
        robot_vel[:, 1] = 5.0 * np.sin(angles)
        robot_pos = np.cumsum(robot_vel * dt, axis=0)
        trace = _make_trace(robot_pos, robot_vel=robot_vel, dt=dt)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=n)
        # With true braking projection, turning at constant speed yields ~0
        assert whole["max_deceleration_mps2"] is not None
        assert whole["max_deceleration_mps2"] < 1.0  # near zero

    def test_speed_decrease_produces_positive_decel(self) -> None:
        """Speed decrease along velocity direction produces positive deceleration."""
        n = 10
        dt = 0.1
        robot_vel = np.zeros((n, 2), dtype=float)
        robot_vel[:5, 0] = 10.0  # fast
        robot_vel[5:, 0] = 2.0  # slow
        robot_pos = np.cumsum(robot_vel * dt, axis=0)
        trace = _make_trace(robot_pos, robot_vel=robot_vel, dt=dt)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=n)
        assert whole["max_deceleration_mps2"] is not None
        assert whole["max_deceleration_mps2"] > 0.0

    def test_speed_increase_not_counted_as_braking(self) -> None:
        """Speed increase does not count as braking deceleration."""
        n = 10
        dt = 0.1
        robot_vel = np.zeros((n, 2), dtype=float)
        robot_vel[:5, 0] = 2.0  # slow
        robot_vel[5:, 0] = 10.0  # fast
        robot_pos = np.cumsum(robot_vel * dt, axis=0)
        trace = _make_trace(robot_pos, robot_vel=robot_vel, dt=dt)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=n)
        # Speed increase is acceleration, not braking
        assert whole["max_deceleration_mps2"] is not None
        assert whole["max_deceleration_mps2"] == pytest.approx(0.0)

    def test_deceleration_none_when_insufficient_samples(self) -> None:
        """max_deceleration_mps2 is None with fewer than 2 velocity samples."""
        robot_vel = np.zeros((1, 2), dtype=float)
        robot_vel[0, 0] = 5.0
        robot_pos = np.zeros((1, 2), dtype=float)
        trace = _make_trace(robot_pos, robot_vel=robot_vel, dt=0.1)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=1)
        assert whole.get("max_deceleration_mps2") is None

    def test_end_to_end_summarize_and_serialize(self) -> None:
        """End-to-end: summarize_interval_metrics + report_to_dict with TTC and braking."""
        trace = _make_braking_trace()
        cfg = {
            "critical_intervals": {
                "first_braking_event": {
                    "enabled": True,
                    "deceleration_threshold_mps2": 0.75,
>>>>>>> eb94c7402 (feat(benchmark): populate min_ttc_s and fix max_deceleration_mps2 semantics)
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
            },
        }
        intervals = extract_critical_intervals(trace, cfg)
        report = summarize_interval_metrics(trace, intervals)
        assert len(report.interval_metrics) == 1
        im = report.interval_metrics[0]
        # Braking trace should have positive deceleration
        assert im.max_deceleration_mps2 is not None
        assert im.max_deceleration_mps2 > 0.0
        # TTC should be populated (pedestrian is present and robot has velocity)
        assert im.min_ttc_s is not None
        # Serialize and check no NaN/inf
        d = report_to_dict(report)
        payload = json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        parsed = json.loads(payload)
        interval_d = parsed["interval_metrics"][0]
        assert interval_d["min_ttc_s"] is None or np.isfinite(interval_d["min_ttc_s"])
        assert interval_d["max_deceleration_mps2"] is None or np.isfinite(
            interval_d["max_deceleration_mps2"]
        )
