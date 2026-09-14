"""Signed temporal-logic robustness objectives for adversarial falsification.

Implements per-property signed robustness semantics (negative = violation,
positive = satisfaction margin) for social-navigation safety properties:

1. **Clearance** (always d_clearance > d_safe): ``rho = min_clearance - NEAR_MISS_DIST``
2. **TTC** (always TTC > tau while closing): ``rho = min_ttc - tau``
3. **Goal reaching** (eventually reach goal within T): ``rho = T - T_actual``
4. **Progress** (avoid sustained low-progress): ``rho = -failure_to_progress``
5. **Collision** (never collide): ``rho = -collision_count``

Per-property robustness values and critical timestamps are preserved in
:class:`RobustnessReport` and written to a sidecar JSON file in the candidate
bundle directory.  The registered objective returns an aggregated scalar for
search optimisation (maximising finds worst violations).

Literature basis: S-TaLiRo-style robustness semantics (Annpureddy et al.,
TACAS 2011); AST (Koren et al., IV 2018).

Status: research/exploratory.  These objectives are search objectives, not
reported benchmark metrics.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.adversarial.io import read_first_jsonl_record
from robot_sf.benchmark.constants import NEAR_MISS_DIST
from robot_sf.benchmark.near_miss_ttc import DIAGNOSTIC_TTC_THRESHOLD_S

if TYPE_CHECKING:
    from robot_sf.adversarial.config import CandidateEvaluation

_ROBUSTNESS_SCHEMA_VERSION = "robustness-report.v1"


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    """Read a finite metric scalar, preserving missingness as unavailable."""
    value = metrics.get(key)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_int(value: Any, default: int) -> int:
    """Read a positive integer, falling back for malformed record metadata."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _derived_dt(record: dict[str, Any]) -> float:
    """Derive a timestep from recorded physical/simulation time, never wall time."""
    steps = record.get("steps")
    try:
        steps_value = float(steps)
    except (TypeError, ValueError):
        return 0.1
    if not math.isfinite(steps_value) or steps_value <= 0.0:
        return 0.1

    timing = record.get("timing") if isinstance(record.get("timing"), dict) else {}
    trace = record.get("algorithm_metadata")
    trace = trace.get("paired_effect_native_trace") if isinstance(trace, dict) else {}
    candidates = (
        record.get("dt_s"),
        timing.get("dt_s"),
        trace.get("dt_s"),
    )
    for value in candidates:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0.0:
            return parsed

    for key in (
        "physical_time_sec",
        "physical_time_s",
        "simulation_time_sec",
        "simulation_time_s",
        "sim_time_sec",
        "sim_time_s",
    ):
        try:
            duration = float(record.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0.0:
            dt = duration / steps_value
            if math.isfinite(dt) and dt > 0.0:
                return dt
    return 0.1


def _validated_positive_float(value: float, *, name: str) -> float:
    """Validate an explicit semantic parameter rather than masking a bad configuration."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive float") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive float")
    return parsed


@dataclass(frozen=True)
class PropertyRobustness:
    """Signed robustness for a single temporal-logic property.

    Attributes
    ----------
    property_name : str
        Identifier for the property (e.g. ``"clearance"``).
    robustness : float
        Signed robustness value: negative = violation, positive = satisfaction.
    critical_time_s : float | None
        Timestamp (seconds into episode) of the critical event, if available.
    violated : bool
        ``True`` when robustness < 0.
    detail : str
        Human-readable description of the robustness computation.
    """

    property_name: str
    robustness: float | None
    critical_time_s: float | None = None
    violated: bool = False
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        return asdict(self)


@dataclass(frozen=True)
class RobustnessReport:
    """Per-property robustness report with aggregated scalar.

    Attributes
    ----------
    schema_version : str
        Fixed schema identifier.
    properties : tuple[PropertyRobustness, ...]
        Per-property robustness values, preserving critical timestamps.
    overall_robustness : float
        Minimum robustness across all properties (worst-case).
    objective_value : float
        Value returned by the registered search objective (negated for
        maximisation so the search finds worst violations).
    """

    schema_version: str = _ROBUSTNESS_SCHEMA_VERSION
    properties: tuple[PropertyRobustness, ...] = ()
    overall_robustness: float | None = 0.0
    objective_value: float | None = 0.0

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable report payload."""
        return {
            "schema_version": self.schema_version,
            "properties": [p.to_json() for p in self.properties],
            "overall_robustness": self.overall_robustness,
            "objective_value": self.objective_value,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RobustnessReport:
        """Reconstruct a report from a JSON payload."""
        properties = tuple(PropertyRobustness(**entry) for entry in payload.get("properties", []))
        return cls(
            schema_version=payload.get("schema_version", _ROBUSTNESS_SCHEMA_VERSION),
            properties=properties,
            overall_robustness=(
                float(payload["overall_robustness"])
                if payload.get("overall_robustness") is not None
                else None
            ),
            objective_value=(
                float(payload["objective_value"])
                if payload.get("objective_value") is not None
                else None
            ),
        )


def _clearance_robustness(
    metrics: dict[str, Any],
    event_ledger: dict[str, Any],
) -> PropertyRobustness:
    """Always maintain clearance: rho = min_clearance - NEAR_MISS_DIST."""
    min_clearance = _metric(metrics, "min_clearance")
    if min_clearance is None:
        return PropertyRobustness("clearance", None, detail="unavailable: min_clearance")
    rho = min_clearance - NEAR_MISS_DIST
    critical_time = _first_collision_time(event_ledger) if rho < 0 else None
    return PropertyRobustness(
        property_name="clearance",
        robustness=rho,
        critical_time_s=critical_time,
        violated=rho < 0,
        detail=f"min_clearance={min_clearance:.4f}m, d_safe={NEAR_MISS_DIST:.4f}m",
    )


def _ttc_robustness(
    metrics: dict[str, Any],
    tau: float,
    event_ledger: dict[str, Any],
) -> PropertyRobustness:
    """Always maintain TTC > tau while closing: rho = min_ttc - tau."""
    min_ttc = _metric(metrics, "time_to_collision_min")
    if min_ttc is None:
        return PropertyRobustness("ttc", None, detail="unavailable: time_to_collision_min")
    rho = min_ttc - tau
    critical_time = _first_collision_time(event_ledger) if rho < 0 else None
    return PropertyRobustness(
        property_name="ttc",
        robustness=rho,
        critical_time_s=critical_time,
        violated=rho < 0,
        detail=f"min_ttc={min_ttc:.4f}s, tau={tau:.4f}s",
    )


def _goal_robustness(
    metrics: dict[str, Any],
    outcome: dict[str, Any],
    horizon: int,
    dt: float,
) -> PropertyRobustness:
    """Eventually reach goal within T: rho = T - T_actual.

    When the goal is reached at time T_actual < T, rho > 0 (satisfaction).
    When the goal is not reached, model it as one unresolved timestep beyond
    the horizon. This keeps the signed contract intact: non-completion is a
    negative violation rather than a zero-valued tie with a boundary success.
    """
    t_total = horizon * dt
    route_complete = bool(outcome.get("route_complete"))
    time_to_goal_norm = _metric(metrics, "time_to_goal_norm")
    if route_complete:
        if time_to_goal_norm is None:
            return PropertyRobustness("goal", None, detail="unavailable: time_to_goal_norm")
        t_actual = time_to_goal_norm * t_total
        rho = t_total - t_actual
        critical_time = t_actual
    else:
        rho = -dt
        critical_time = t_total
    time_to_goal_detail = "unavailable" if time_to_goal_norm is None else f"{time_to_goal_norm:.4f}"
    return PropertyRobustness(
        property_name="goal",
        robustness=rho,
        critical_time_s=critical_time,
        violated=rho < 0,
        detail=f"route_complete={route_complete}, time_to_goal_norm={time_to_goal_detail}",
    )


def _progress_robustness(
    metrics: dict[str, Any],
) -> PropertyRobustness:
    """Avoid sustained low-progress intervals: rho = -failure_to_progress."""
    ftp = _metric(metrics, "failure_to_progress")
    if ftp is None:
        return PropertyRobustness("progress", None, detail="unavailable: failure_to_progress")
    rho = -ftp
    return PropertyRobustness(
        property_name="progress",
        robustness=rho,
        critical_time_s=None,
        violated=rho < 0,
        detail=f"failure_to_progress={ftp:.0f}",
    )


def _collision_robustness(
    metrics: dict[str, Any],
    outcome: dict[str, Any],
    event_ledger: dict[str, Any],
) -> PropertyRobustness:
    """Never collide: rho = -collision_count."""
    collision_count = _metric(metrics, "total_collision_count")
    if collision_count is None:
        if not any(name in outcome for name in ("collision", "collision_event")):
            return PropertyRobustness("collision", None, detail="unavailable: collision evidence")
        collision_flag = bool(outcome.get("collision") or outcome.get("collision_event"))
        collision_count = 1.0 if collision_flag else 0.0
    rho = -collision_count
    critical_time = _first_collision_time(event_ledger) if rho < 0 else None
    return PropertyRobustness(
        property_name="collision",
        robustness=rho,
        critical_time_s=critical_time,
        violated=rho < 0,
        detail=f"collision_count={collision_count:.0f}",
    )


def _first_collision_time(event_ledger: dict[str, Any]) -> float | None:
    """Extract the timestamp of the first collision event, if available."""
    collision_events = event_ledger.get("collision_events")
    if not isinstance(collision_events, list):
        return None
    for event in collision_events:
        if not isinstance(event, dict):
            continue
        try:
            timestamp = float(event.get("collision_time"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(timestamp):
            return timestamp
    return None


def compute_robustness_report(
    record: dict[str, Any],
    *,
    tau: float = DIAGNOSTIC_TTC_THRESHOLD_S,
    dt: float | None = None,
) -> RobustnessReport:
    """Compute signed robustness for all properties from an episode record.

    Parameters
    ----------
    record : dict
        Episode JSONL record (first record from the file).
    tau : float
        TTC safety threshold in seconds.  Defaults to the diagnostic placeholder.
    dt : float | None
        Timestep in seconds.  If ``None``, derived from ``horizon`` and
        ``timestamps`` or defaults to 0.1.

    Returns
    -------
    RobustnessReport
        Per-property robustness values with critical timestamps and aggregated
        scalar.
    """
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
    event_ledger = (
        record.get("event_ledger") if isinstance(record.get("event_ledger"), dict) else {}
    )
    horizon = _positive_int(record.get("horizon"), 200)

    if dt is None:
        dt = _derived_dt(record)
    else:
        dt = _validated_positive_float(dt, name="dt")
    tau = _validated_positive_float(tau, name="tau")

    properties = (
        _clearance_robustness(metrics, event_ledger),
        _ttc_robustness(metrics, tau, event_ledger),
        _goal_robustness(metrics, outcome, horizon, dt),
        _progress_robustness(metrics),
        _collision_robustness(metrics, outcome, event_ledger),
    )

    robustness_values = [p.robustness for p in properties]
    overall = (
        min(robustness_values)
        if robustness_values and all(value is not None for value in robustness_values)
        else None
    )
    objective = -overall if overall is not None else None

    return RobustnessReport(
        properties=properties,
        overall_robustness=overall,
        objective_value=objective,
    )


def write_robustness_report(report: RobustnessReport, path: Path) -> Path:
    """Write a robustness report to a JSON sidecar file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    return path


def temporal_robustness_objective(evaluation: CandidateEvaluation) -> float | None:
    """Signed temporal-logic robustness objective for adversarial search.

    Computes per-property signed robustness, writes the full report to a
    sidecar ``robustness_report.json`` in the candidate bundle directory,
    and returns the aggregated scalar for search optimisation.

    The search maximises the returned value; more negative robustness
    (larger violations) produces a larger returned value.
    """
    if evaluation.episode_record_path is None:
        return None
    record = read_first_jsonl_record(evaluation.episode_record_path)
    if record is None:
        return None

    report = compute_robustness_report(record)

    if evaluation.bundle_path is not None:
        write_robustness_report(report, evaluation.bundle_path / "robustness_report.json")

    return report.objective_value
