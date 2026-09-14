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


def _positive_recorded_float(value: Any) -> float | None:
    """Parse a positive recorded timing value without accepting booleans."""
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _nonnegative_recorded_float(value: Any) -> float | None:
    """Parse a finite non-negative recorded value without accepting booleans."""
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _consistent_timing_value(values: list[float]) -> float | None:
    """Return one timing value when all recorded sources agree."""
    if not values:
        return None
    first = values[0]
    return (
        first
        if all(math.isclose(value, first, rel_tol=0.0, abs_tol=1.0e-12) for value in values[1:])
        else None
    )


def _mapping_dt_values(container: Any, keys: tuple[str, ...]) -> tuple[list[float], bool]:
    """Read positive timestep fields from one mapping."""
    if not isinstance(container, dict):
        return [], False
    values: list[float] = []
    malformed = False
    for key in keys:
        if key not in container:
            continue
        parsed = _positive_recorded_float(container[key])
        if parsed is None:
            malformed = True
        else:
            values.append(parsed)
    return values, malformed


def _declared_dt_candidates(record: dict[str, Any]) -> tuple[list[float], bool]:
    """Collect declared evaluation timesteps and report malformed declarations."""
    sources: list[tuple[Any, tuple[str, ...]]] = [
        (record, ("dt_s", "dt")),
        (record.get("timing"), ("dt_s", "dt")),
    ]

    algorithm_metadata = record.get("algorithm_metadata")
    if isinstance(algorithm_metadata, dict):
        sources.extend(
            (algorithm_metadata.get(trace_name), ("dt_s", "dt"))
            for trace_name in (
                "paired_effect_native_trace",
                "analysis_trace",
                "simulation_step_trace",
            )
        )

    scenario_params = record.get("scenario_params")
    if isinstance(scenario_params, dict):
        direct_keys = ("run_dt", "dt_s", "dt")
        keys = (
            direct_keys
            if any(key in scenario_params for key in direct_keys)
            else (
                "time_per_step_in_secs",
                "dt_s",
                "dt",
            )
        )
        sources.append(
            (
                scenario_params
                if keys is direct_keys
                else scenario_params.get("simulation_config"),
                keys,
            )
        )

    candidates: list[float] = []
    malformed = False
    for container, keys in sources:
        values, source_malformed = _mapping_dt_values(container, keys)
        candidates.extend(values)
        malformed |= source_malformed
    return candidates, malformed


def _derived_dt_status(record: dict[str, Any]) -> tuple[float | None, bool]:
    """Derive the evaluation timestep and flag malformed recorded timing."""
    candidates, malformed = _declared_dt_candidates(record)
    if malformed:
        return None, True

    steps_value = _positive_recorded_float(record.get("steps"))
    if steps_value is not None:
        for key in (
            "physical_time_sec",
            "physical_time_s",
            "simulation_time_sec",
            "simulation_time_s",
            "sim_time_sec",
            "sim_time_s",
        ):
            if key not in record:
                continue
            duration = _positive_recorded_float(record[key])
            if duration is None:
                return None, True
            derived = duration / steps_value
            if not math.isfinite(derived) or derived <= 0.0:
                return None, True
            candidates.append(derived)

    if not candidates:
        return None, False
    recorded_dt = _consistent_timing_value(candidates)
    return recorded_dt, recorded_dt is None


def _derived_dt(record: dict[str, Any]) -> float | None:
    """Derive the evaluation timestep from recorded simulation timing, never wall time."""
    return _derived_dt_status(record)[0]


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
    dt: float | None,
) -> PropertyRobustness:
    """Eventually reach goal within T: rho = T - T_actual.

    When the goal is reached at time T_actual < T, rho > 0 (satisfaction).
    When the goal is not reached, model it as one unresolved timestep beyond
    the horizon. This keeps the signed contract intact: non-completion is a
    negative violation rather than a zero-valued tie with a boundary success.
    """
    if dt is None:
        return PropertyRobustness("goal", None, detail="unavailable: evaluation dt")
    t_total = horizon * dt
    route_complete, malformed_route = _strict_bool_field(outcome, "route_complete")
    if malformed_route or route_complete is None:
        return PropertyRobustness("goal", None, detail="unavailable: route_complete")
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


def _strict_bool_field(container: Any, key: str) -> tuple[bool | None, bool]:
    """Return ``(value, malformed)`` for a present boolean field."""
    if not isinstance(container, dict) or key not in container:
        return None, False
    value = container[key]
    return (value, False) if isinstance(value, bool) else (None, True)


def _collision_count_value(metrics: dict[str, Any]) -> tuple[float | None, bool]:
    """Read a non-negative integer collision count and report malformed values."""
    counts: list[float] = []
    for key in ("collisions", "total_collision_count"):
        if key not in metrics:
            continue
        value = metrics.get(key)
        count = _metric(metrics, key)
        valid = (
            not isinstance(value, bool)
            and count is not None
            and count >= 0.0
            and math.isclose(count, round(count), rel_tol=0.0, abs_tol=1.0e-12)
        )
        if not valid:
            return None, True
        counts.append(count)
    if not counts:
        return None, False
    return (_consistent_timing_value(counts), False) if len(set(counts)) == 1 else (None, True)


def _collision_flag_value(container: Any) -> tuple[bool | None, bool]:
    """Reconcile legacy and canonical collision flags."""
    flags = [_strict_bool_field(container, key) for key in ("collision_event", "collision")]
    if any(malformed for _value, malformed in flags):
        return None, True
    values = [value for value, _malformed in flags if value is not None]
    if values and any(value != values[0] for value in values[1:]):
        return None, True
    return (values[0], False) if values else (None, False)


def _typed_collision_count(event_ledger: dict[str, Any]) -> tuple[int | None, bool]:
    """Read positive evidence from a typed collision-event ledger."""
    if "collision_events" not in event_ledger:
        return None, False
    collision_events = event_ledger["collision_events"]
    if not isinstance(collision_events, list):
        return None, True
    for event in collision_events:
        if (
            not isinstance(event, dict)
            or _nonnegative_recorded_float(event.get("collision_time")) is None
        ):
            return None, True
    # An empty list means that no typed event detail was retained; it is not an
    # authoritative negative collision assertion.
    return (len(collision_events), False) if collision_events else (None, False)


def _reconciled_collision_count(
    collision_count: float | None,
    outcome_collision: bool | None,
    ledger_count: int | None,
    exact_collision: bool | None,
) -> tuple[float | None, str | None]:
    """Reconcile collision evidence and return a count or an unavailable reason."""
    if (
        collision_count is not None
        and ledger_count is not None
        and not math.isclose(collision_count, ledger_count, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        return None, "unavailable: collision count/event mismatch"

    evidence = [value > 0.0 for value in (collision_count,) if value is not None]
    evidence.extend(value for value in (outcome_collision, exact_collision) if value is not None)
    if ledger_count is not None:
        evidence.append(ledger_count > 0)
    if not evidence:
        return None, "unavailable: collision evidence"
    if any(value != evidence[0] for value in evidence[1:]):
        return None, "unavailable: contradictory collision evidence"

    if collision_count is None:
        collision_count = float(
            ledger_count if ledger_count is not None else (1 if evidence[0] else 0)
        )
    return collision_count, None


def _collision_robustness(
    metrics: dict[str, Any],
    outcome: dict[str, Any],
    event_ledger: dict[str, Any],
) -> PropertyRobustness:
    """Never collide: rho = -collision_count, with fail-closed reconciliation."""

    collision_count, malformed_count = _collision_count_value(metrics)
    if malformed_count:
        return PropertyRobustness(
            "collision", None, detail="unavailable: malformed collision count"
        )

    outcome_collision, malformed_outcome = _collision_flag_value(outcome)
    if malformed_outcome:
        return PropertyRobustness("collision", None, detail="unavailable: malformed collision flag")

    ledger_count, malformed_ledger = _typed_collision_count(event_ledger)
    if malformed_ledger:
        return PropertyRobustness(
            "collision", None, detail="unavailable: malformed collision event ledger"
        )

    exact_events = event_ledger.get("exact_events")
    if "exact_events" in event_ledger and not isinstance(exact_events, dict):
        return PropertyRobustness("collision", None, detail="unavailable: malformed exact events")
    exact_collision, exact_malformed = _strict_bool_field(exact_events, "collision")
    if exact_malformed:
        return PropertyRobustness("collision", None, detail="unavailable: malformed exact events")

    collision_count, unavailable_detail = _reconciled_collision_count(
        collision_count, outcome_collision, ledger_count, exact_collision
    )
    if unavailable_detail is not None:
        return PropertyRobustness("collision", None, detail=unavailable_detail)
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
        Timestep in seconds. If ``None``, it must be present in recorded
        simulation timing or the goal property is unavailable. An explicit
        value must agree with a recorded evaluation timestep when one exists.

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

    recorded_dt, malformed_timing = _derived_dt_status(record)
    if dt is None:
        dt = recorded_dt
    else:
        dt = _validated_positive_float(dt, name="dt")
        if malformed_timing:
            raise ValueError("dt cannot override malformed or conflicting recorded timing")
        if recorded_dt is not None and not math.isclose(
            dt, recorded_dt, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("dt does not match the recorded evaluation timestep")
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
