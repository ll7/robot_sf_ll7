"""Critical-interval metric windows around safety-relevant events (experimental).

Background
----------
Global episode averages hide short but important failures around interaction
onset, near-miss, braking, occlusion, and recovery.  This module provides an
opt-in, offline mechanism that extracts time windows around event anchors in a
recorded trace, so metric reports can include both whole-run and critical-window
values.

The API accepts a generic trace dict (as produced by the map runner JSONL
exports or EpisodeData-backed synthetic traces) together with a YAML config that
declares which anchors are enabled and how wide each window should be.

Status
------
Experimental and disabled by default.  Missing anchors are reported explicitly
rather than fabricated.  No change to existing benchmark metrics or release
gates.

References
----------
- Original issue: https://github.com/ll7/robot_sf_ll7/issues/4758
- Critical Interval MSE: https://arxiv.org/abs/2606.29898
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "critical-intervals.v1"

VALID_ANCHORS = frozenset(
    [
        "closest_approach",
        "ttc_threshold_crossing",
        "first_braking_event",
        "collision_or_near_miss",
        "recovery_after_avoidance",
        "planner_mode_switch",
        "pedestrian_deviation_onset",
        "stuck_oscillation_onset",
    ]
)

DEFAULT_NEAR_MISS_DIST = 0.7  # metres, aligned with benchmark constants


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriticalInterval:
    """Resolved critical interval around a single event anchor.

    Attributes
    ----------
    anchor :
        Event anchor name (e.g. ``"closest_approach"``).
    status :
        One of ``"available"``, ``"missing_anchor"``, or ``"invalid_trace"``.
    start_step :
        Inclusive step index for the window start (clamped to trace bounds).
    end_step :
        Exclusive step index for the window end (clamped to trace bounds).
    anchor_step :
        Step index of the anchor event itself.
    reason :
        Human-readable explanation when the anchor is missing or invalid.
    """

    anchor: str
    status: Literal["available", "missing_anchor", "invalid_trace"]
    start_step: int | None = None
    end_step: int | None = None
    anchor_step: int | None = None
    reason: str | None = None


@dataclass
class IntervalMetrics:
    """Metrics computed over a single critical interval window."""

    anchor: str
    n_steps: int = 0
    min_clearance_m: float | None = None
    min_distance_m: float | None = None
    # NOTE: min_ttc_s is not yet computed by _compute_interval_metrics_in_window
    # (always None as shipped). Windowed TTC aggregation is tracked as a
    # follow-up; kept in the schema so the field is stable when populated.
    min_ttc_s: float | None = None
    mean_speed_ms: float | None = None
    max_speed_ms: float | None = None
    # NOTE: this is the max L2 magnitude of per-step velocity change (|Δv|/dt),
    # i.e. total acceleration magnitude within the window — NOT a signed
    # deceleration projected onto the braking direction. Rename/redefinition is
    # tracked as a follow-up.
    max_deceleration_mps2: float | None = None
    heading_oscillation: float | None = None
    near_miss_count: int = 0
    collision_flag: bool = False


@dataclass
class CriticalIntervalReport:
    """Complete report with whole-run baselines and per-interval summaries.

    Attributes
    ----------
    whole_run :
        Dictionary of whole-run aggregated metrics.
    intervals :
        List of resolved intervals.
    interval_metrics :
        Per-interval metric results.
    missing_anchors :
        Anchors that were requested but could not be resolved, with reasons.
    """

    whole_run: dict[str, Any] = field(default_factory=dict)
    intervals: list[CriticalInterval] = field(default_factory=list)
    interval_metrics: list[IntervalMetrics] = field(default_factory=list)
    missing_anchors: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(
    path: str | None = None, *, config_dict: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load and validate a critical-intervals configuration.

    Parameters
    ----------
    path :
        Path to the YAML configuration file (optional, when ``config_dict`` is
        not provided).
    config_dict :
        Pre-parsed configuration dictionary.

    Returns
    -------
    Validated configuration dictionary.

    Raises
    ------
    ValueError
        If the schema version is unknown, negative windows are used, or an
        unknown anchor is referenced.
    """

    if config_dict is not None:
        cfg = dict(config_dict)
    elif path is not None:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    schema = cfg.get("schema_version", "")
    if schema and schema != SCHEMA_VERSION:
        raise ValueError(f"Unknown schema_version {schema!r}; expected {SCHEMA_VERSION!r}")

    for anchor, spec in cfg.get("critical_intervals", {}).items():
        if anchor not in VALID_ANCHORS:
            raise ValueError(f"Unknown anchor {anchor!r}; valid: {sorted(VALID_ANCHORS)}")
        if not isinstance(spec, dict):
            raise ValueError(f"Anchor {anchor!r} must map to a dict, got {type(spec).__name__}")

        for key in ("before_s", "after_s"):
            val = spec.get(key)
            if val is not None and float(val) < 0:
                raise ValueError(f"Anchor {anchor!r}: {key} must be >= 0, got {val!r}")

    return cfg


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def _get_trace_arrays(
    trace: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract robot position, pedestrian positions, and dt from a trace dict.

    Returns
    -------
    (robot_pos, peds_pos, dt)
    """

    robot_pos = np.asarray(trace.get("robot_pos", []), dtype=float)
    if robot_pos.ndim == 1:
        robot_pos = robot_pos.reshape(1, -1)

    peds_pos_raw = trace.get("peds_pos", [])
    if len(peds_pos_raw) == 0:
        peds_pos = np.zeros((max(robot_pos.shape[0], 0), 0, 2), dtype=float)
    else:
        peds_pos = np.asarray(peds_pos_raw, dtype=float)
        if peds_pos.ndim == 2:
            peds_pos = peds_pos[:, np.newaxis, :]

    dt = trace.get("dt", 0.1)
    if not isinstance(dt, (int, float)) or dt <= 0:
        dt = 0.1

    return robot_pos, peds_pos, dt


# ---------------------------------------------------------------------------
# Anchor detection
# ---------------------------------------------------------------------------


def _detect_closest_approach(robot_pos: np.ndarray, peds_pos: np.ndarray) -> int | None:
    """Return step index of the minimum robot-pedestrian distance.

    If no pedestrians are present, the anchor cannot be resolved and ``None``
    is returned.
    """

    n_peds = peds_pos.shape[1] if peds_pos.ndim >= 3 else 0
    if n_peds == 0 or robot_pos.shape[0] == 0:
        return None

    diffs = robot_pos[:, np.newaxis, :] - peds_pos  # (T, K, 2)
    dists = np.linalg.norm(diffs, axis=2)  # (T, K)
    min_dist_per_step = np.min(dists, axis=1)  # (T,)

    if not np.isfinite(min_dist_per_step).any():
        return None

    return int(np.argmin(min_dist_per_step))


def _detect_ttc_threshold_crossing(  # noqa: C901
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    peds_pos: np.ndarray,
    *,
    threshold_s: float,
    dt: float,
    ped_vel: np.ndarray | None = None,
) -> int | None:
    """Return the first step where TTC drops below *threshold_s*.

    TTC is computed using the constant-velocity convention relative to the line
    of approach.  If velocity data is missing for either robot or pedestrians,
    no anchor can be resolved.
    """

    T = robot_pos.shape[0]
    n_peds = peds_pos.shape[1] if peds_pos.ndim >= 3 else 0
    if n_peds == 0 or T < 2:
        return None

    if ped_vel is None:
        if peds_pos.shape[0] >= 2:
            ped_vel = np.diff(peds_pos, axis=0) / dt
            ped_vel = np.vstack([ped_vel[:1], ped_vel])
        else:
            return None

    if robot_vel.shape[0] < T:
        return None

    _MIN_SPEED = 1e-9
    _MIN_DIST = 1e-9

    for t in range(T):
        for k in range(n_peds):
            d_vec = peds_pos[t, k] - robot_pos[t]
            dist = float(np.linalg.norm(d_vec))
            if dist < _MIN_DIST:
                return t

            v_rel = robot_vel[t] - ped_vel[t, k]
            speed = float(np.linalg.norm(v_rel))
            if speed < _MIN_SPEED:
                continue

            closing = float(np.dot(v_rel, d_vec))
            if closing <= 0:
                continue

            ttc = dist / closing
            if ttc < threshold_s:
                return t

    return None


def _detect_first_braking_event(
    robot_vel: np.ndarray,
    *,
    decel_threshold: float,
    dt: float,
) -> int | None:
    """Return the first step where robot deceleration exceeds *decel_threshold*.

    Deceleration is approximated via the magnitude of negative tangential
    acceleration, derived by central-difference on velocity.
    """

    T = robot_vel.shape[0]
    if T < 2:
        return None

    acc = np.zeros_like(robot_vel)
    acc[0] = (robot_vel[1] - robot_vel[0]) / dt
    acc[-1] = (robot_vel[-1] - robot_vel[-2]) / dt
    if T > 2:
        acc[1:-1] = (robot_vel[2:] - robot_vel[:-2]) / (2.0 * dt)

    speed = np.linalg.norm(robot_vel, axis=1)
    for t in range(T):
        if speed[t] > 1e-9:
            vel_dir = robot_vel[t] / speed[t]
            tangential_a = float(np.dot(acc[t], vel_dir))
            if tangential_a < -decel_threshold:
                return t

    return None


def _detect_collision_or_near_miss(
    robot_pos: np.ndarray,
    peds_pos: np.ndarray,
    *,
    near_miss_dist: float = DEFAULT_NEAR_MISS_DIST,
) -> int | None:
    """Return the first step where any robot-pedestrian pair is a near-miss or closer."""

    n_peds = peds_pos.shape[1] if peds_pos.ndim >= 3 else 0
    if n_peds == 0 or robot_pos.shape[0] == 0:
        return None

    diffs = robot_pos[:, np.newaxis, :] - peds_pos
    dists = np.linalg.norm(diffs, axis=2)

    for t in range(robot_pos.shape[0]):
        if np.any(dists[t] < near_miss_dist):
            return t

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_critical_intervals(  # noqa: C901, PLR0912, PLR0915
    trace: dict[str, Any], config: dict[str, Any]
) -> list[CriticalInterval]:
    """Extract critical intervals from a recorded trace.

    Parameters
    ----------
    trace :
        Dictionary containing keys such as ``robot_pos``, ``peds_pos``,
        ``robot_vel``, and ``dt``.
    config :
        Validated configuration dictionary (from :func:`load_config`).

    Returns
    -------
    List of :class:`CriticalInterval` objects, one per enabled anchor.
    """

    robot_pos, peds_pos, dt = _get_trace_arrays(trace)
    T = robot_pos.shape[0]

    robot_vel: np.ndarray | None = None
    if "robot_vel" in trace:
        rv = np.asarray(trace["robot_vel"], dtype=float)
        if rv.ndim == 1:
            rv = rv.reshape(1, -1)
        robot_vel = rv

    ped_vel: np.ndarray | None = None
    if "ped_vel" in trace:
        ped_vel = np.asarray(trace["ped_vel"], dtype=float)
    elif peds_pos.ndim >= 3 and peds_pos.shape[0] >= 2:
        ped_diffs = np.diff(peds_pos, axis=0) / dt
        ped_vel = np.vstack([ped_diffs[:1], ped_diffs])

    intervals: list[CriticalInterval] = []

    for anchor, spec in config.get("critical_intervals", {}).items():
        if not spec.get("enabled", False):
            continue

        before_n_steps = int(spec.get("before_s", 1.0) / dt) if dt > 0 else 0
        after_n_steps = int(spec.get("after_s", 1.0) / dt) if dt > 0 else 0

        anchor_step: int | None = None
        reason: str | None = None
        status: Literal["available", "missing_anchor", "invalid_trace"] = "available"

        if anchor == "closest_approach":
            if peds_pos.size == 0:
                status = "missing_anchor"
                reason = "no pedestrian positions in trace"
            else:
                anchor_step = _detect_closest_approach(robot_pos, peds_pos)
                if anchor_step is None:
                    status = "invalid_trace"
                    reason = "could not compute closest approach (invalid data)"

        elif anchor == "ttc_threshold_crossing":
            threshold_s = spec.get("threshold_s", 1.5)
            if robot_vel is None:
                status = "missing_anchor"
                reason = "robot velocity data required for TTC computation"
            elif ped_vel is None and peds_pos.size == 0:
                status = "missing_anchor"
                reason = "pedestrian velocity data required for TTC computation"
            else:
                anchor_step = _detect_ttc_threshold_crossing(
                    robot_pos,
                    robot_vel,
                    peds_pos,
                    threshold_s=threshold_s,
                    dt=dt,
                    ped_vel=ped_vel,
                )
                if anchor_step is None:
                    status = "missing_anchor"
                    reason = f"no TTC crossing below {threshold_s}s found"

        elif anchor == "first_braking_event":
            decel_thr = spec.get("deceleration_threshold_mps2", 0.75)
            if robot_vel is None:
                status = "missing_anchor"
                reason = "robot velocity data required for braking detection"
            elif robot_vel.shape[0] < 2:
                status = "missing_anchor"
                reason = "trace too short to compute braking"
            else:
                anchor_step = _detect_first_braking_event(
                    robot_vel,
                    decel_threshold=decel_thr,
                    dt=dt,
                )
                if anchor_step is None:
                    status = "missing_anchor"
                    reason = f"no braking event above {decel_thr} m/s^2"

        elif anchor == "collision_or_near_miss":
            if peds_pos.size == 0:
                status = "missing_anchor"
                reason = "no pedestrian positions in trace"
            else:
                anchor_step = _detect_collision_or_near_miss(robot_pos, peds_pos)
                if anchor_step is None:
                    status = "missing_anchor"
                    reason = "no collision or near-miss event detected"

        elif anchor == "recovery_after_avoidance":
            nm_step = _detect_collision_or_near_miss(robot_pos, peds_pos)
            if nm_step is None or peds_pos.size == 0:
                status = "missing_anchor"
                reason = "no near-miss to recover from"
            else:
                anchor_step = nm_step
                status = "available"

        elif anchor == "planner_mode_switch":
            mode_trace = trace.get("planner_mode")
            if mode_trace is None:
                status = "missing_anchor"
                reason = "planner mode trace not available"
            else:
                try:
                    modes = list(mode_trace)
                    for i in range(1, len(modes)):
                        if modes[i] != modes[i - 1]:
                            anchor_step = i
                            break
                    if anchor_step is None:
                        status = "missing_anchor"
                        reason = "no mode switch detected"
                except (TypeError, ValueError):
                    status = "invalid_trace"
                    reason = "planner_mode trace has unsupported type"

        else:
            status = "missing_anchor"
            reason = f"anchor {anchor!r} not yet implemented in detection"

        if status == "available" and anchor_step is not None:
            start = max(0, anchor_step - before_n_steps)
            end = min(T, anchor_step + 1 + after_n_steps)
        else:
            start = None
            end = None

        intervals.append(
            CriticalInterval(
                anchor=anchor,
                status=status,
                start_step=start,
                end_step=end,
                anchor_step=anchor_step,
                reason=reason,
            )
        )

    return intervals


def _compute_interval_metrics_in_window(
    trace: dict[str, Any],
    *,
    start: int,
    end: int | None = None,
) -> dict[str, Any]:
    """Compute metrics for a single window [start, end).

    Returns
    -------
    dict with metric keys and float values for the given slice.
    """

    robot_pos, peds_pos, dt = _get_trace_arrays(trace)
    T = robot_pos.shape[0]
    if end is None:
        end = T
    end = min(end, T)
    n_peds = peds_pos.shape[1] if peds_pos.ndim >= 3 else 0

    sl = slice(start, end)
    sub_robot_pos = robot_pos[sl]
    sub_peds_pos = peds_pos[sl]

    result: dict[str, Any] = {
        "n_steps": max(0, end - start),
    }

    if sub_robot_pos.shape[0] == 0:
        return result

    # Distance / clearance
    if n_peds > 0:
        diffs = sub_robot_pos[:, np.newaxis, :] - sub_peds_pos
        dists = np.linalg.norm(diffs, axis=2)
        min_dist_per_step = np.min(dists, axis=1)
        result["min_distance_m"] = float(np.min(min_dist_per_step))
        result["min_clearance_m"] = float(np.min(min_dist_per_step))

        near_miss_d = trace.get("near_miss_dist_m", DEFAULT_NEAR_MISS_DIST)
        result["near_miss_count"] = int(np.sum(np.min(dists, axis=1) < near_miss_d))
        result["collision_flag"] = bool(np.any(np.min(dists, axis=1) < 0.3))
    else:
        result["min_distance_m"] = None
        result["min_clearance_m"] = None

    # Speed
    robot_vel_raw = trace.get("robot_vel")
    if robot_vel_raw is not None:
        full_vel = np.asarray(robot_vel_raw, dtype=float)
        if full_vel.ndim == 1:
            full_vel = full_vel.reshape(1, -1)
        # Slice to the window so per-interval metrics reflect ONLY the window,
        # not the whole run (otherwise the critical-window value always equals
        # the whole-run value and the comparison is meaningless).
        sub_vel = full_vel[sl]
        speeds = np.linalg.norm(sub_vel, axis=1)
        result["mean_speed_ms"] = float(np.mean(speeds))
        result["max_speed_ms"] = float(np.max(speeds))

        if sub_vel.shape[0] >= 2:
            acc_diffs = np.diff(sub_vel, axis=0) / dt
            acc_norms = np.linalg.norm(acc_diffs, axis=1)
            result["max_deceleration_mps2"] = float(np.max(acc_norms))
    elif sub_robot_pos.shape[0] >= 2:
        vel_approx = np.diff(sub_robot_pos, axis=0) / dt
        speeds = np.linalg.norm(vel_approx, axis=1)
        result["mean_speed_ms"] = float(np.mean(speeds))
        result["max_speed_ms"] = float(np.max(speeds))
    else:
        result["mean_speed_ms"] = None
        result["max_speed_ms"] = None

    # Heading oscillation proxy (variance of heading direction)
    if sub_robot_pos.shape[0] >= 2:
        deltas = np.diff(sub_robot_pos, axis=0)
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        dirs = deltas / norms
        result["heading_oscillation"] = float(np.var(dirs))

    return result


def summarize_interval_metrics(
    trace: dict[str, Any], intervals: list[CriticalInterval]
) -> CriticalIntervalReport:
    """Summarize metrics for whole run and each available critical interval.

    Parameters
    ----------
    trace :
        Trace dictionary.
    intervals :
        Resolved intervals from :func:`extract_critical_intervals`.

    Returns
    -------
    :class:`CriticalIntervalReport`
    """

    report = CriticalIntervalReport()

    # Whole-run metrics
    report.whole_run = _compute_interval_metrics_in_window(trace, start=0, end=None)

    # Per-interval metrics
    missing_anchors: list[dict[str, str]] = []

    for interval in intervals:
        if interval.status == "available" and interval.start_step is not None:
            end_s = interval.end_step if interval.end_step is not None else 0
            window_metrics = _compute_interval_metrics_in_window(
                trace,
                start=interval.start_step,
                end=end_s,
            )
            im = IntervalMetrics(
                anchor=interval.anchor,
                n_steps=window_metrics.get("n_steps", 0),
                min_clearance_m=window_metrics.get("min_clearance_m"),
                min_distance_m=window_metrics.get("min_distance_m"),
                mean_speed_ms=window_metrics.get("mean_speed_ms"),
                max_speed_ms=window_metrics.get("max_speed_ms"),
                max_deceleration_mps2=window_metrics.get("max_deceleration_mps2"),
                heading_oscillation=window_metrics.get("heading_oscillation"),
                near_miss_count=window_metrics.get("near_miss_count", 0),
                collision_flag=window_metrics.get("collision_flag", False),
            )
            report.interval_metrics.append(im)
        else:
            missing_anchors.append(
                {
                    "anchor": interval.anchor,
                    "status": interval.status,
                    "reason": interval.reason or "unknown",
                }
            )

    report.intervals = intervals
    report.missing_anchors = missing_anchors

    return report


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def report_to_dict(report: CriticalIntervalReport) -> dict[str, Any]:
    """Convert a :class:`CriticalIntervalReport` to a JSON-serializable dict.

    Returns
    -------
    dict with keys ``whole_run``, ``critical_intervals``,
    ``interval_metrics``, and ``missing_anchors``.
    """

    def _interval_to_dict(iv: CriticalInterval) -> dict[str, Any]:
        return {
            "anchor": iv.anchor,
            "status": iv.status,
            "start_step": iv.start_step,
            "end_step": iv.end_step,
            "anchor_step": iv.anchor_step,
            "reason": iv.reason,
        }

    def _metrics_to_dict(im: IntervalMetrics) -> dict[str, Any]:
        d: dict[str, Any] = {"anchor": im.anchor, "n_steps": im.n_steps}
        for k in (
            "min_clearance_m",
            "min_distance_m",
            "min_ttc_s",
            "mean_speed_ms",
            "max_speed_ms",
            "max_deceleration_mps2",
            "heading_oscillation",
            "near_miss_count",
            "collision_flag",
        ):
            v = getattr(im, k)
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                v = None
            d[k] = v
        return d

    return {
        "whole_run": report.whole_run,
        "critical_intervals": [_interval_to_dict(iv) for iv in report.intervals],
        "interval_metrics": [_metrics_to_dict(m) for m in report.interval_metrics],
        "missing_anchors": report.missing_anchors,
    }
