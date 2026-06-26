"""Calibrate stream_gap uncertainty-gate thresholds for safe agent-dropping (issue #3558).

#3471 found that dropping uncertain agents at the **current default** ``stream_gap``
uncertainty-gate thresholds increases unsafe commitment. That is a finding about the defaults,
not about gating in general. This module is the pure **decision layer** that turns a
gate-threshold sweep (run over the #3471 episode harness) into actionable guidance: it classifies
each swept setting against the conservative-retention baseline and reports whether any safe
operating region exists, or confirms that conservative retention dominates.

The sweep that produces the per-setting safety aggregates needs benchmark runs and is deferred;
this layer is pure and side-effect free, mirroring the failure-cause classifier in
``robot_sf/scenario_certification/failure_cause.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot_sf.benchmark.finite_checks import require_finite_fields

STREAM_GAP_CALIBRATION_SCHEMA = "stream_gap_gate_calibration.v1"

_SETTING_METRIC_FIELDS = ("unsafe_commit_rate", "collision_rate", "min_separation_m")

AT_LEAST_AS_SAFE = "at_least_as_safe"
LESS_SAFE = "less_safe"
CONSERVATIVE_RETENTION_DOMINATES = "conservative_retention_dominates"


@dataclass(frozen=True, slots=True)
class GateSettingResult:
    """Per-setting safety aggregates from one point of the gate-threshold sweep.

    Attributes:
        thresholds: The four gate thresholds defining this setting.
        unsafe_commit_rate: Fraction of episodes with an unsafe commitment (lower is safer).
        collision_rate: Collision probability (lower is safer).
        min_separation_m: Minimum robot-pedestrian separation (higher is safer).
    """

    thresholds: dict[str, float]
    unsafe_commit_rate: float
    collision_rate: float
    min_separation_m: float


@dataclass(frozen=True, slots=True)
class SafetyTolerance:
    """Slack allowed when comparing a dropping setting to the retained baseline."""

    unsafe_commit_abs: float = 0.0
    collision_abs: float = 0.0
    min_separation_abs: float = 0.0


def _require_finite_setting(result: GateSettingResult, label: str) -> None:
    """Fail closed on a non-finite (NaN/Inf) safety aggregate.

    A degraded sweep point or baseline can carry NaN/Inf; the ``<=`` / ``>=`` safety
    comparisons then evaluate ``False`` and the setting is silently classified
    ``less_safe`` (or, for the baseline, lets settings pass) without flagging the
    bad input. Raising names the offending field so the caller drops the trace.

    Raises:
        ValueError: If any safety aggregate of ``result`` is not finite.
    """
    require_finite_fields(label, result, _SETTING_METRIC_FIELDS)


def classify_setting_safety(
    setting: GateSettingResult,
    baseline: GateSettingResult,
    tolerance: SafetyTolerance | None = None,
) -> str:
    """Classify a dropping setting against the conservative-retention baseline.

    A setting is ``at_least_as_safe`` only when it does not worsen any safety axis beyond the
    allowed tolerance (unsafe-commit, collision, and minimum separation).

    Returns:
        str: ``at_least_as_safe`` or ``less_safe``.
    """
    _require_finite_setting(setting, "setting")
    _require_finite_setting(baseline, "baseline")
    tolerance = tolerance or SafetyTolerance()
    safe = (
        setting.unsafe_commit_rate <= baseline.unsafe_commit_rate + tolerance.unsafe_commit_abs
        and setting.collision_rate <= baseline.collision_rate + tolerance.collision_abs
        and setting.min_separation_m >= baseline.min_separation_m - tolerance.min_separation_abs
    )
    return AT_LEAST_AS_SAFE if safe else LESS_SAFE


def calibrate_stream_gap_gate(
    settings: list[GateSettingResult],
    baseline: GateSettingResult,
    tolerance: SafetyTolerance | None = None,
) -> dict[str, Any]:
    """Turn a gate-threshold sweep into actionable safe-region guidance.

    Identifies the safe operating region (settings at least as safe as retention) and recommends
    the safest member, or concludes that conservative retention dominates.

    Returns:
        dict[str, Any]: Versioned report with per-setting classifications, the safe region, and a
        recommended setting or the ``conservative_retention_dominates`` conclusion.
    """
    if not settings:
        raise ValueError("at least one swept setting is required")
    tolerance = tolerance or SafetyTolerance()
    classified = [
        {
            "thresholds": dict(setting.thresholds),
            "unsafe_commit_rate": setting.unsafe_commit_rate,
            "collision_rate": setting.collision_rate,
            "min_separation_m": setting.min_separation_m,
            "classification": classify_setting_safety(setting, baseline, tolerance),
        }
        for setting in settings
    ]
    safe_region = [row for row in classified if row["classification"] == AT_LEAST_AS_SAFE]

    recommendation: dict[str, Any] | None = None
    conclusion = CONSERVATIVE_RETENTION_DOMINATES
    if safe_region:
        # Recommend the safe setting with the strongest safety margin (fewest unsafe commits,
        # then fewest collisions, then largest separation).
        recommendation = min(
            safe_region,
            key=lambda row: (
                row["unsafe_commit_rate"],
                row["collision_rate"],
                -row["min_separation_m"],
            ),
        )
        conclusion = "safe_region_exists"

    return {
        "schema_version": STREAM_GAP_CALIBRATION_SCHEMA,
        "evidence_kind": "diagnostic_proxy",
        "baseline": {
            "unsafe_commit_rate": baseline.unsafe_commit_rate,
            "collision_rate": baseline.collision_rate,
            "min_separation_m": baseline.min_separation_m,
        },
        "n_settings": len(settings),
        "settings": classified,
        "safe_region": safe_region,
        "recommended_setting": recommendation,
        "conclusion": conclusion,
    }


__all__ = [
    "AT_LEAST_AS_SAFE",
    "CONSERVATIVE_RETENTION_DOMINATES",
    "LESS_SAFE",
    "STREAM_GAP_CALIBRATION_SCHEMA",
    "GateSettingResult",
    "SafetyTolerance",
    "calibrate_stream_gap_gate",
    "classify_setting_safety",
]
