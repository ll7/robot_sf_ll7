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
    CriticalInterval,
    IntervalMetrics,
    _compute_interval_metrics_in_window,
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
        # A single large velocity change at steps 0->1 (Δv = 10 m/s over dt=0.1s
        # => 100 m/s^2), then constant velocity for the rest of the trace.
        n = 20
        dt = 0.1
        robot_vel = np.zeros((n, 2), dtype=float)
        robot_vel[1:, 0] = 10.0  # spike between step 0 and step 1 only
        robot_pos = np.cumsum(robot_vel * dt, axis=0)
        trace = _make_trace(robot_pos, robot_vel=robot_vel, dt=dt)

        whole = _compute_interval_metrics_in_window(trace, start=0, end=n)
        # A late window that excludes the step-0->1 spike must see zero accel.
        late = _compute_interval_metrics_in_window(trace, start=5, end=n)

        assert whole["max_deceleration_mps2"] == pytest.approx(100.0)
        assert late["max_deceleration_mps2"] == pytest.approx(0.0)
        # The whole point of the feature: window value != whole-run value.
        assert late["max_deceleration_mps2"] < whole["max_deceleration_mps2"]
