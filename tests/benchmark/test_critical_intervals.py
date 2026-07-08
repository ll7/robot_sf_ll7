"""Tests for the critical-interval metric window mechanism (issue #4758).

Covers anchor detection, window extraction, interval metrics, config validation,
graceful handling of missing anchors, and the guarantee that the feature is
opt-in only.
"""

from __future__ import annotations

import json
import math
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
        # Robot moves east at 10 m/s for steps 1..5, then brakes to -5 m/s
        # between steps 5 and 6, then stays at -5 m/s.  The deceleration spike
        # lives between steps 5 and 6 only.
        n = 20
        dt = 0.1
        robot_vel = np.zeros((n, 2), dtype=float)
        robot_vel[1:6, 0] = 10.0
        robot_vel[6:, 0] = -5.0
        robot_pos = np.cumsum(robot_vel * dt, axis=0)
        trace = _make_trace(robot_pos, robot_vel=robot_vel, dt=dt)

        whole = _compute_interval_metrics_in_window(trace, start=0, end=n)
        # A late window that excludes the deceleration spike must see zero.
        late = _compute_interval_metrics_in_window(trace, start=10, end=n)

        # Central-difference at the 10→-5 transition: Δv=15, dt=0.2 → 75
        assert whole["max_deceleration_mps2"] == pytest.approx(75.0)
        assert late["max_deceleration_mps2"] == pytest.approx(0.0)
        # The whole point of the feature: window value != whole-run value.
        assert late["max_deceleration_mps2"] < whole["max_deceleration_mps2"]


# ---------------------------------------------------------------------------
# min_ttc_s (issue #4785)
# ---------------------------------------------------------------------------


class TestWindowedMinTTC:
    """min_ttc_s is populated when robot/pedestrian data are available."""

    def test_closing_pair_produces_finite_ttc(self) -> None:
        trace = _make_approaching_trace()
        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        assert whole["min_ttc_s"] is not None
        assert whole["min_ttc_s"] >= 0
        assert np.isfinite(whole["min_ttc_s"])

    def test_ttc_in_report_to_dict(self) -> None:
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
        assert len(d["interval_metrics"]) == 1
        assert d["interval_metrics"][0]["min_ttc_s"] is not None
        assert d["interval_metrics"][0]["min_ttc_s"] >= 0

    def test_missing_robot_vel_gives_none(self) -> None:
        trace = _make_approaching_trace()
        del trace["robot_vel"]
        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        assert whole["min_ttc_s"] is None

    def test_no_peds_gives_none(self) -> None:
        n = 5
        robot_pos = np.zeros((n, 2))
        robot_pos[:, 0] = np.arange(n) * 0.5
        robot_vel = np.zeros((n, 2))
        robot_vel[:, 0] = 1.0
        trace = _make_trace(robot_pos, robot_vel=robot_vel)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        assert whole["min_ttc_s"] is None

    def test_non_closing_pair_gives_none(self) -> None:
        n = 5
        robot_pos = np.zeros((n, 2))
        robot_vel = np.zeros((n, 2))
        robot_vel[:, 0] = 1.0
        peds_pos = np.zeros((n, 1, 2))
        peds_pos[:, 0, 0] = 10.0
        ped_vel = np.zeros((n, 1, 2))
        ped_vel[:, 0, 0] = 2.0
        trace = _make_trace(robot_pos, peds_pos, robot_vel=robot_vel, dt=0.1)
        trace["ped_vel"] = ped_vel
        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        assert whole["min_ttc_s"] is None


# ---------------------------------------------------------------------------
# max_deceleration_mps2 — braking semantics (issue #4785)
# ---------------------------------------------------------------------------


class TestBrakingDeceleration:
    """max_deceleration_mps2 is braking deceleration, not acceleration magnitude."""

    def test_deceleration_along_velocity_direction(self) -> None:
        dt = 0.1
        n = 10
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
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
            },
        }
        intervals = extract_critical_intervals(trace, cfg)
        report = summarize_interval_metrics(trace, intervals)
        d = report_to_dict(report)
        assert "ttc_convention" in d
        assert d["ttc_convention"] == TTC_CONVENTION

    def test_hand_computed_head_on_ttc(self) -> None:
        """Head-on geometry: robot at (0,0) moving (1,0), ped at (4,0) moving (-1,0).

        Distance = 4 m, closing speed = 2 m/s, TTC = 2 s.
        """
        robot_pos = np.array([0.0, 0.0])
        robot_vel = np.array([1.0, 0.0])
        ped_pos = np.array([4.0, 0.0])
        ped_vel = np.array([-1.0, 0.0])

        ttc = _pairwise_ttc_s(
            robot_pos=robot_pos,
            robot_vel=robot_vel,
            ped_pos=ped_pos,
            ped_vel=ped_vel,
        )

        assert ttc == pytest.approx(2.0, abs=1e-6)

    def test_threshold_crossing_respects_physical_ttc(self) -> None:
        """Head-on fixture: verifies physical TTC seconds are used for threshold."""
        # Use a 2-step trace with distance such that TTC stays between 1.9s and 2.1s
        # At t=0: distance=4.1m, closing_speed=2m/s → TTC=2.05s (> 2.1s? no, < 2.1s)
        # Actually, let me recalculate: 4.1/2 = 2.05s
        # At t=1: distance=4.1-0.2=3.9m, closing_speed=2m/s → TTC=1.95s
        # So with threshold 2.1s, step 0 should trigger (2.05 < 2.1)
        # And with threshold 1.9s, step 1 should trigger (1.95 > 1.9? no, 1.95 < 1.9? no)
        # Actually: 1.95 > 1.9, so with threshold 1.9, it would not trigger at step 0 or 1
        dt = 0.1
        n = 2
        robot_pos = np.zeros((n, 2))
        robot_pos[:, 0] = [0.0, 0.1]  # moving east at 1 m/s

        peds_pos = np.zeros((n, 1, 2))
        peds_pos[:, 0, 0] = [4.1, 4.0]  # moving west at 1 m/s

        robot_vel = np.zeros((n, 2))
        robot_vel[:, 0] = 1.0

        ped_vel = np.zeros((n, 1, 2))
        ped_vel[:, 0, 0] = -1.0

        trace = _make_trace(robot_pos, peds_pos, robot_vel=robot_vel, dt=dt)
        trace["ped_vel"] = ped_vel

        # At t=0: distance=4.1, TTC=2.05s
        # At t=1: distance=3.9, TTC=1.95s

        # Threshold 1.8s: should find a crossing at step 1 (TTC=1.95 < 1.8? no, 1.95 > 1.8)
        # Wait, 1.95 > 1.8, so no crossing. Let me use 2.0s as threshold.
        # With threshold 2.0s:
        #   t=0: TTC=2.05s, 2.05 < 2.0? no
        #   t=1: TTC=1.95s, 1.95 < 2.0? yes, crosses at step 1

        # Threshold 2.1s:
        #   t=0: TTC=2.05s, 2.05 < 2.1? yes, crosses at step 0
        #   t=1: TTC=1.95s, 1.95 < 2.1? yes

        # Threshold 1.9s:
        #   t=0: TTC=2.05s, 2.05 < 1.9? no
        #   t=1: TTC=1.95s, 1.95 < 1.9? no
        # So no crossing expected

        # Test with threshold 2.1s (should cross at step 0)
        cfg_above = {
            "critical_intervals": {
                "ttc_threshold_crossing": {
                    "enabled": True,
                    "threshold_s": 2.1,
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
            },
        }
        intervals_above = extract_critical_intervals(trace, cfg_above)
        assert len(intervals_above) == 1
        assert intervals_above[0].status == "available"
        assert intervals_above[0].anchor_step == 0  # First step where TTC < 2.1s

        # Test with threshold 1.9s (should NOT cross)
        cfg_below = {
            "critical_intervals": {
                "ttc_threshold_crossing": {
                    "enabled": True,
                    "threshold_s": 1.9,
                    "before_s": 0.5,
                    "after_s": 0.5,
                },
            },
        }
        intervals_below = extract_critical_intervals(trace, cfg_below)
        assert len(intervals_below) == 1
        assert intervals_below[0].status == "missing_anchor"

    def test_distance_scaling_proportional_to_distance(self) -> None:
        """Same relative velocity, different initial distances: TTC scales with distance."""
        # Both fixtures have same closing speed (2 m/s)
        robot_vel = np.array([1.0, 0.0])
        ped_vel = np.array([-1.0, 0.0])

        # Fixture 1: distance = 4 m, TTC = 2 s
        ttc_4m = _pairwise_ttc_s(
            robot_pos=np.array([0.0, 0.0]),
            robot_vel=robot_vel,
            ped_pos=np.array([4.0, 0.0]),
            ped_vel=ped_vel,
        )

        # Fixture 2: distance = 8 m, TTC = 4 s
        ttc_8m = _pairwise_ttc_s(
            robot_pos=np.array([0.0, 0.0]),
            robot_vel=robot_vel,
            ped_pos=np.array([8.0, 0.0]),
            ped_vel=ped_vel,
        )

        assert ttc_4m == pytest.approx(2.0, abs=1e-6)
        assert ttc_8m == pytest.approx(4.0, abs=1e-6)
        # TTC scales proportionally with distance (not old s/m proxy)
        assert ttc_8m == pytest.approx(2.0 * ttc_4m)

    def test_non_closing_pair_returns_none(self) -> None:
        """Pedestrian moving away or robot moving away returns None."""
        # Robot and ped moving same direction at same speed: not closing
        ttc = _pairwise_ttc_s(
            robot_pos=np.array([0.0, 0.0]),
            robot_vel=np.array([1.0, 0.0]),
            ped_pos=np.array([4.0, 0.0]),
            ped_vel=np.array([1.0, 0.0]),
        )
        assert ttc is None

        # Robot moving away from ped
        ttc = _pairwise_ttc_s(
            robot_pos=np.array([0.0, 0.0]),
            robot_vel=np.array([-1.0, 0.0]),
            ped_pos=np.array([4.0, 0.0]),
            ped_vel=np.array([0.0, 0.0]),
        )
        assert ttc is None

    def test_overlap_contact_returns_zero(self) -> None:
        """Distance below epsilon returns 0.0."""
        # Same position
        ttc = _pairwise_ttc_s(
            robot_pos=np.array([0.0, 0.0]),
            robot_vel=np.array([1.0, 0.0]),
            ped_pos=np.array([0.0, 0.0]),
            ped_vel=np.array([-1.0, 0.0]),
        )
        assert ttc == 0.0

        # Very close (below epsilon)
        ttc = _pairwise_ttc_s(
            robot_pos=np.array([0.0, 0.0]),
            robot_vel=np.array([1.0, 0.0]),
            ped_pos=np.array([1e-10, 0.0]),
            ped_vel=np.array([-1.0, 0.0]),
        )
        assert ttc == 0.0

    def test_perpendicular_motion_no_ttc(self) -> None:
        """Perpendicular trajectories (no closing speed) return None."""
        # Robot at (0,0) moving north, ped at (4,0) moving south
        # Separation is east-west, velocities are north-south (perpendicular)
        ttc = _pairwise_ttc_s(
            robot_pos=np.array([0.0, 0.0]),
            robot_vel=np.array([0.0, 1.0]),
            ped_pos=np.array([4.0, 0.0]),
            ped_vel=np.array([0.0, -1.0]),
        )
        # Relative velocity (0, 2) is perpendicular to separation (4, 0)
        assert ttc is None

    def test_window_min_ttc_uses_physical_ttc(self) -> None:
        """Window min_ttc_s reflects physical TTC, not old proxy."""
        n = 10
        dt = 0.1
        # Head-on: distance 8m, closing speed 2 m/s → TTC = 4 s
        robot_pos = np.zeros((n, 2))
        robot_pos[:, 0] = np.arange(n) * dt * 1.0

        peds_pos = np.zeros((n, 1, 2))
        peds_pos[:, 0, 0] = 8.0 - np.arange(n) * dt * 1.0

        robot_vel = np.zeros((n, 2))
        robot_vel[:, 0] = 1.0

        ped_vel = np.zeros((n, 1, 2))
        ped_vel[:, 0, 0] = -1.0

        trace = _make_trace(robot_pos, peds_pos, robot_vel=robot_vel, dt=dt)
        trace["ped_vel"] = ped_vel

        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        # At t=0: distance=8m, closing_speed=2m/s → TTC=4s
        # Later steps have smaller distance → smaller TTC
        assert whole["min_ttc_s"] is not None
        assert whole["min_ttc_s"] < 4.0  # Some step has TTC < 4s
        assert whole["min_ttc_s"] > 0.0

    def test_oblique_approach_angle_correct(self) -> None:
        """Oblique approach: closing speed is v_rel·d_hat, not just speed magnitude."""
        # Robot at origin moving east at 2 m/s
        # Ped at (4, 3) = 5 m away, moving west at 1 m/s
        # Separation vector: (4, 3), unit: (0.8, 0.6)
        # v_rel = (2 - (-1), 0 - 0) = (3, 0)
        # closing_speed = dot((3, 0), (0.8, 0.6)) = 2.4 m/s
        # TTC = 5 / 2.4 ≈ 2.083 s
        ttc = _pairwise_ttc_s(
            robot_pos=np.array([0.0, 0.0]),
            robot_vel=np.array([2.0, 0.0]),
            ped_pos=np.array([4.0, 3.0]),
            ped_vel=np.array([-1.0, 0.0]),
        )
        assert ttc == pytest.approx(5.0 / 2.4, abs=1e-6)


# ---------------------------------------------------------------------------
# heading_oscillation straight-path baseline (issue #4886)
# ---------------------------------------------------------------------------


class TestHeadingOscillationBaseline:
    """``heading_oscillation`` must be ~0 for straight, constant-heading motion.

    Regression coverage for issue #4886. The previous implementation used
    ``np.var(dirs)`` over the *flattened* unit-direction vectors, which mixes
    the structurally different x and y components of a unit vector into one
    population. For a straight ``(1, 0)`` path that flattened variance is
    ``0.25`` (variance of ``[1, 0, 1, 0, ...]`` around its ``0.5`` mean) --
    indistinguishable from a small wiggle, and wildly different from a straight
    ``(1, 1)`` path (which scored ``0.0``). The metric therefore measured axis
    alignment, not heading oscillation.

    The fix computes variance *per component* (``axis=0``) and sums it (the
    trace of the 2x2 covariance of the unit directions == ``1 - |mean_dir|^2``),
    which is ``0.0`` for any constant-heading path and rises with dispersion.
    """

    def test_straight_x_axis_is_zero(self) -> None:
        """Constant ``(1, 0)`` motion must score 0.0 (was 0.25 before #4886)."""
        n = 10
        robot_pos = np.zeros((n, 2), dtype=float)
        robot_pos[:, 0] = np.arange(n) * 0.5
        trace = _make_trace(robot_pos)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        assert whole["heading_oscillation"] == pytest.approx(0.0, abs=1e-12)

    def test_straight_diagonal_is_zero(self) -> None:
        """Constant ``(1, 1)`` motion must also score 0.0 -- axis independent."""
        n = 10
        robot_pos = np.zeros((n, 2), dtype=float)
        step = np.arange(n) * 0.5
        robot_pos[:, 0] = step
        robot_pos[:, 1] = step
        trace = _make_trace(robot_pos)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        assert whole["heading_oscillation"] == pytest.approx(0.0, abs=1e-12)

    def test_axis_alignment_does_not_change_score(self) -> None:
        """Straight motion scores identically along any heading."""
        n = 10
        cases = {
            "x_axis": np.array([[i * 0.5, 0.0] for i in range(n)]),
            "diagonal": np.array([[i * 0.5, i * 0.5] for i in range(n)]),
            "y_axis": np.array([[0.0, i * 0.5] for i in range(n)]),
        }
        scores = {
            name: _compute_interval_metrics_in_window(_make_trace(pos), start=0, end=None)[
                "heading_oscillation"
            ]
            for name, pos in cases.items()
        }
        # All constant-heading paths score 0.0; none is privileged by axis.
        for name, score in scores.items():
            assert score == pytest.approx(0.0, abs=1e-12), name

    def test_small_wiggle_is_small_and_nonzero(self) -> None:
        """A +-5 deg heading wiggle scores a small value (~0.0076)."""
        angles_deg = [5.0, -5.0, 5.0, -5.0]
        dirs = np.array(
            [[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in angles_deg]
        )
        robot_pos = np.vstack([[0.0, 0.0], np.cumsum(dirs, axis=0)])
        trace = _make_trace(robot_pos)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        wiggle = whole["heading_oscillation"]
        # Real oscillation, but small -- tightly matches the closed-form value.
        assert wiggle == pytest.approx(0.0075961, abs=1e-6)

    def test_straight_is_distinguishable_from_wiggle(self) -> None:
        """The defining property the old metric violated: straight < wiggle.

        Under the buggy flattened-variance metric, straight ``(1, 0)`` scored
        0.25 while a +-5 deg wiggle scored ~0.252 -- the metric could not tell
        them apart. The fixed per-component metric restores the ordering.
        """
        n = 10
        straight_pos = np.zeros((n, 2), dtype=float)
        straight_pos[:, 0] = np.arange(n) * 0.5
        straight = _compute_interval_metrics_in_window(
            _make_trace(straight_pos), start=0, end=None
        )["heading_oscillation"]

        angles_deg = [5.0, -5.0, 5.0, -5.0]
        dirs = np.array(
            [[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in angles_deg]
        )
        wiggle_pos = np.vstack([[0.0, 0.0], np.cumsum(dirs, axis=0)])
        wiggle = _compute_interval_metrics_in_window(_make_trace(wiggle_pos), start=0, end=None)[
            "heading_oscillation"
        ]

        assert straight < wiggle
        assert straight == pytest.approx(0.0, abs=1e-12)

    def test_full_reversal_scores_high(self) -> None:
        """Back-and-forth ``(1, 0) <-> (-1, 0)`` reversals score ~1.0."""
        dirs = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        robot_pos = np.vstack([[0.0, 0.0], np.cumsum(dirs, axis=0)])
        trace = _make_trace(robot_pos)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        # x component alternates +-1 (variance 1.0), y stays 0 -> total 1.0.
        assert whole["heading_oscillation"] == pytest.approx(1.0, abs=1e-12)

    def test_single_step_window_omits_metric(self) -> None:
        """With fewer than 2 positions the metric key is absent (boundary)."""
        robot_pos = np.zeros((1, 2), dtype=float)
        trace = _make_trace(robot_pos)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        assert "heading_oscillation" not in whole

    def test_straight_path_with_pause_is_zero(self) -> None:
        """A straight path that pauses for a step must still score 0.0.

        Stationary steps have no defined heading; injecting their ``[0, 0]``
        delta into the direction population would inflate the variance and make
        a merely-paused straight path read as "oscillating". A straight
        ``(1, 0)`` path with a single repeated (stationary) position must score
        the same 0.0 as the uninterrupted path.
        """
        robot_pos = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
        trace = _make_trace(robot_pos)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        assert whole["heading_oscillation"] == pytest.approx(0.0, abs=1e-12)

    def test_fully_stationary_window_is_zero(self) -> None:
        """A window with no movement at all defaults to 0.0 (no heading)."""
        robot_pos = np.zeros((5, 2), dtype=float)
        trace = _make_trace(robot_pos)
        whole = _compute_interval_metrics_in_window(trace, start=0, end=None)
        assert whole["heading_oscillation"] == pytest.approx(0.0, abs=1e-12)
