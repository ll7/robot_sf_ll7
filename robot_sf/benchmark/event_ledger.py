"""Canonical per-episode safety-event ledger helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from robot_sf.benchmark.spawn_validity import (
    RESET_CLEARANCE_UNAVAILABLE_REASON,
    SPAWN_VALIDITY_SCHEMA_VERSION,
    record_has_invalid_spawn,
)
from robot_sf.common.validation import finite_float as _finite_float
from robot_sf.nav.spawn_clearance import SPAWN_OVERLAP_INVALID_REASON

EPISODE_EVENT_LEDGER_SCHEMA_VERSION = "EpisodeEventLedger.v2"
SUPPORTED_EVENT_LEDGER_SCHEMA_VERSIONS = frozenset(
    {
        "EpisodeEventLedger.v1",
        EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
    }
)
COLLISION_PARTNER_TYPES = (
    "pedestrian",
    "static_geometry",
    "boundary",
    "goal_artifact",
)
CONTACT_PROVENANCE_SCHEMA_VERSION = "contact_provenance.v1"
STATIONARY_ROBOT_SPEED_THRESHOLD_M_S = 0.05


@dataclass(frozen=True, slots=True)
class ExactEvents:
    """Exact simulator outcome events for one episode."""

    collision: bool
    goal_reached: bool
    timeout: bool
    invalid_run: bool = False


@dataclass(frozen=True, slots=True)
class SurrogateEvents:
    """Derived diagnostic safety events for one episode."""

    near_miss: bool = False
    clearance_breach: bool = False
    ttc_breach: bool = False
    oscillation: bool = False
    late_evasive: bool = False
    occlusion_near_miss: bool = False


@dataclass(frozen=True, slots=True)
class CollisionEventRecord:
    """Typed exact-collision record for one collision event."""

    collision_partner_type: str
    collision_partner_id: str | None
    collision_time: float
    relative_speed_at_contact: float | None
    clearance_series_source: str
    exact_event_source: str


@dataclass(frozen=True, slots=True)
class EpisodeEventLedger:
    """Versioned source-of-truth event block for benchmark episode records."""

    schema_version: str
    scenario_id: str
    seed: int
    planner: str
    software_commit: str | None
    exact_events: ExactEvents
    collision_events: list[CollisionEventRecord] = field(default_factory=list)
    surrogate_events: SurrogateEvents = field(default_factory=SurrogateEvents)
    metric_definitions: dict[str, dict[str, str]] = field(default_factory=dict)
    reconciliation: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable ledger payload."""
        return asdict(self)


def _bool_at(payload: Mapping[str, Any], key: str, *, default: bool = False) -> bool:
    """Read a boolean-like field from a mapping.

    Returns:
        Parsed boolean value, or ``default`` when the key is absent or unrecognized.
    """
    if key not in payload:
        return default
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


def _safe_int(value: Any, *, default: int = 0) -> int:
    """Return an integer value when available.

    Returns:
        Parsed integer, or ``default`` when parsing fails.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _metric_value(metrics: Mapping[str, Any], *keys: str) -> tuple[float | None, str | None]:
    """Return the first finite metric value and source path."""
    for key in keys:
        if key not in metrics:
            continue
        value = _finite_float(metrics.get(key))
        if value is not None:
            return value, f"metrics.{key}"
    return None, None


def _safety_predicate_records(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return safety predicate records from an episode record."""
    predicates = record.get("safety_predicates")
    return dict(predicates) if isinstance(predicates, Mapping) else {}


def _safety_wrapper_summary(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return episode-level safety-wrapper summary from algorithm metadata."""

    algorithm_metadata = record.get("algorithm_metadata")
    if not isinstance(algorithm_metadata, Mapping):
        return None
    safety_wrapper = algorithm_metadata.get("safety_wrapper")
    if not isinstance(safety_wrapper, Mapping):
        return None
    return dict(safety_wrapper)


def _cbf_safety_filter_summary(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return episode-level CBF safety-filter summary from algorithm metadata."""

    algorithm_metadata = record.get("algorithm_metadata")
    if not isinstance(algorithm_metadata, Mapping):
        return None
    cbf_safety_filter = algorithm_metadata.get("cbf_safety_filter")
    if not isinstance(cbf_safety_filter, Mapping):
        return None
    return dict(cbf_safety_filter)


def _predicate_event(
    predicates: Mapping[str, Any],
    predicate_key: str,
    event_key: str,
) -> tuple[bool, str | None]:
    """Return a predicate event boolean and source path when present.

    Returns ``(False, None)`` when the predicate record is missing or does not carry
    ``event_key``. Reporting ``None`` (rather than a source path that points at an absent
    field) keeps the reconciliation source honest as ``missing`` and lets the affected
    surrogate event fall back to its metric-derived value instead of a defaulted ``False``.
    """
    payload = predicates.get(predicate_key)
    if not isinstance(payload, Mapping) or event_key not in payload:
        return False, None
    return _bool_at(payload, event_key), f"safety_predicates.{predicate_key}.{event_key}"


def _normalize_collision_partner_type(value: Any) -> str:
    """Return a validated collision partner type."""
    partner_type = str(value).strip()
    if partner_type not in COLLISION_PARTNER_TYPES:
        allowed = ", ".join(COLLISION_PARTNER_TYPES)
        raise ValueError(f"collision_partner_type must be one of {allowed}; got {partner_type!r}")
    return partner_type


def _normalize_collision_event_record(event: Mapping[str, Any]) -> CollisionEventRecord:
    """Validate and normalize one typed collision-event payload.

    Returns:
        Normalized collision-event record ready for JSON serialization.
    """
    collision_time = _finite_float(event.get("collision_time"))
    if collision_time is None or collision_time < 0.0:
        raise ValueError("collision_time must be a finite value >= 0.")
    clearance_series_source = str(event.get("clearance_series_source") or "").strip()
    if not clearance_series_source:
        raise ValueError("clearance_series_source must be a non-empty string.")
    exact_event_source = str(event.get("exact_event_source") or "").strip()
    if not exact_event_source:
        raise ValueError("exact_event_source must be a non-empty string.")
    collision_partner_id = event.get("collision_partner_id")
    normalized_partner_id = (
        str(collision_partner_id).strip() if collision_partner_id is not None else None
    )
    if normalized_partner_id == "":
        normalized_partner_id = None
    return CollisionEventRecord(
        collision_partner_type=_normalize_collision_partner_type(
            event.get("collision_partner_type")
        ),
        collision_partner_id=normalized_partner_id,
        collision_time=collision_time,
        relative_speed_at_contact=_finite_float(event.get("relative_speed_at_contact")),
        clearance_series_source=clearance_series_source,
        exact_event_source=exact_event_source,
    )


def _normalize_collision_events(
    collision_events: Sequence[Mapping[str, Any]] | None,
) -> list[CollisionEventRecord]:
    """Return validated typed collision events."""
    if collision_events is None:
        return []
    normalized: list[CollisionEventRecord] = []
    for event in collision_events:
        if not isinstance(event, Mapping):
            raise ValueError("collision event records must be mappings.")
        normalized.append(_normalize_collision_event_record(event))
    normalized.sort(key=lambda event: (event.collision_time, event.collision_partner_type))
    return normalized


def _contact_provenance(
    record: Mapping[str, Any], collision_events: Sequence[Mapping[str, Any]] | None
) -> dict[str, Any]:
    """Classify opt-in contacts without changing exact events or collision totals.

    Returns:
        Versioned diagnostic provenance for every collision event.
    """
    spawn = record.get("spawn_validity")
    spawn = spawn if isinstance(spawn, Mapping) else {}
    reset_clearance = spawn.get("reset_clearance")
    reset_clearance = reset_clearance if isinstance(reset_clearance, Mapping) else {}
    reset_row_values = reset_clearance.get("overlapping_pedestrian_rows")
    reset_rows = (
        {str(row) for row in reset_row_values} if isinstance(reset_row_values, list) else set()
    )
    respawn_matches = spawn.get("respawn_overlap_collisions")
    respawn_matches = respawn_matches if isinstance(respawn_matches, list) else []
    contacts: list[dict[str, Any]] = []
    sorted_events = _sorted_contact_sources(collision_events)
    for index, event in enumerate(sorted_events):
        speed = _finite_float(event.get("robot_speed_at_contact_m_s"))
        if speed is not None and speed < 0.0:
            speed = None
        step_index = event.get("contact_step_index")
        if not isinstance(step_index, int) or isinstance(step_index, bool) or step_index < 0:
            step_index = None
        event_time = _finite_float(event.get("collision_time"))
        partner_type = str(event.get("collision_partner_type") or "")
        partners = _contact_partner_ids(event)
        matched = next(
            (
                match
                for match in respawn_matches
                if isinstance(match, Mapping)
                and event_time is not None
                and _finite_float(match.get("collision_time_s")) is not None
                and abs(float(match["collision_time_s"]) - event_time) <= 1.0e-6
                and str(match.get("ped_row")) in partners
            ),
            None,
        )
        reset_match = (
            partner_type == "pedestrian"
            and spawn.get("reset_overlap") is True
            and reset_clearance.get("overlap") is True
            and step_index == 0
            and bool(partners & reset_rows)
        )
        if speed is None:
            contact_class = "unresolved_missing_robot_speed"
        elif step_index is None:
            contact_class = "unresolved_missing_step_index"
        elif partner_type != "pedestrian":
            contact_class = "geometry_contact"
        elif reset_match:
            contact_class = "reset_overlap_contact"
        elif matched is not None:
            contact_class = "respawn_contact"
        elif speed <= STATIONARY_ROBOT_SPEED_THRESHOLD_M_S:
            contact_class = "pedestrian_on_stationary_robot"
        else:
            contact_class = "moving_robot_pedestrian_contact"
        contacts.append(
            {
                "collision_event_index": index,
                "collision_time": event_time,
                "contact_step_index": step_index,
                "collision_partner_type": partner_type,
                "robot_speed_at_contact_m_s": speed,
                "robot_speed_source": "runtime.step.robot_position_delta/dt",
                "robot_speed_measurement_window": "simulation_step",
                "contact_class": contact_class,
                "reset_overlap_match": reset_match,
                "respawn_match": dict(matched) if matched is not None else None,
            }
        )
    return {
        "schema_version": CONTACT_PROVENANCE_SCHEMA_VERSION,
        "stationary_robot_speed_threshold_m_s": STATIONARY_ROBOT_SPEED_THRESHOLD_M_S,
        "reset_overlap": spawn.get("reset_overlap") is True,
        "spawn_invalid_run": spawn.get("invalid_run") is True,
        "reset_clearance_status": spawn.get("reset_clearance_status"),
        "contacts": contacts,
        "classification_status": _contact_classification_status(spawn, contacts),
        "planner_causation_admitted": False,
    }


def _sorted_contact_sources(
    collision_events: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    """Use the canonical ledger event order for source/provenance pairing.

    Returns:
        Raw source events in ledger order.
    """
    return sorted(
        collision_events or [],
        key=lambda event: (
            _finite_float(event.get("collision_time"))
            if _finite_float(event.get("collision_time")) is not None
            else math.inf,
            str(event.get("collision_partner_type")),
        ),
    )


def _contact_partner_ids(event: Mapping[str, Any]) -> set[str]:
    """Use typed contact IDs or the canonical nearest partner as fallback.

    Returns:
        Set of pedestrian row identifiers supplied by the saved event.
    """
    partner_values = event.get("contact_partner_ids")
    if not isinstance(partner_values, list):
        partner_values = [event.get("collision_partner_id")]
    return {str(partner) for partner in partner_values if partner is not None}


def _contact_classification_status(
    spawn: Mapping[str, Any], contacts: list[Mapping[str, Any]]
) -> str:
    """Fail closed on invalid starts, unavailable clearance, or missing measurements.

    Returns:
        Diagnostic status for the whole contact-provenance block.
    """
    if spawn.get("reset_overlap") is True:
        return "invalid_start"
    if spawn.get("invalid_run") is True or spawn.get("spawn_invalid_run") is True:
        return "invalid_spawn"
    if spawn.get("reset_clearance_status") != "available":
        return "unresolved_reset_clearance"
    if any(
        not isinstance(c, Mapping) or str(c.get("contact_class", "")).startswith("unresolved_")
        for c in contacts
    ):
        return "unresolved_contact_measurement"
    return "diagnostic_classified"


def build_event_ledger(
    record: Mapping[str, Any],
    *,
    collision_events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an ``EpisodeEventLedger.v2`` payload from an episode record.

    Returns:
        JSON-serializable ledger payload.
    """
    metrics = record.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    outcome = record.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    termination_reason = str(record.get("termination_reason") or "")

    collision_value, collision_source = _metric_value(
        metrics,
        "total_collision_count",
        "collisions",
        "collision_count",
        "collision_rate",
    )
    near_miss_value, near_miss_source = _metric_value(metrics, "near_misses", "near_miss_count")
    min_clearance, min_clearance_source = _metric_value(metrics, "min_clearance", "min_distance")
    ttc_value, ttc_source = _metric_value(metrics, "ttc_breach_count", "ttc_violations")
    oscillation_value, oscillation_source = _metric_value(
        metrics,
        "oscillation_count",
        "heading_rate_sign_changes",
    )
    safety_predicates = _safety_predicate_records(record)
    safety_wrapper = _safety_wrapper_summary(record)
    cbf_safety_filter = _cbf_safety_filter_summary(record)
    predicate_oscillation, predicate_oscillation_source = _predicate_event(
        safety_predicates,
        "oscillatory_control_predicate",
        "oscillation",
    )
    late_evasive, late_evasive_source = _predicate_event(
        safety_predicates,
        "late_evasive_predicate",
        "late_evasive",
    )
    occlusion_near_miss, occlusion_near_miss_source = _predicate_event(
        safety_predicates,
        "occlusion_near_miss_predicate",
        "occlusion_near_miss",
    )
    if predicate_oscillation_source is not None:
        oscillation_source = predicate_oscillation_source
    normalized_collision_events = _normalize_collision_events(collision_events)

    metric_definitions = {
        "collision_count": {
            "kind": "exact_or_sampled",
            "source": collision_source or "missing",
        },
        "near_miss": {
            "kind": "derived",
            "source": near_miss_source or "missing",
        },
        "clearance_breach": {
            "kind": "derived",
            "source": min_clearance_source or "missing",
        },
        "ttc_breach": {
            "kind": "derived",
            "source": ttc_source or "missing",
        },
        "oscillation": {
            "kind": "derived",
            "source": oscillation_source or "missing",
        },
        "late_evasive": {
            "kind": "derived",
            "source": late_evasive_source or "missing",
        },
        "occlusion_near_miss": {
            "kind": "derived",
            "source": occlusion_near_miss_source or "missing",
        },
    }
    spawn_invalid = record_has_invalid_spawn(record)
    spawn_block = record.get("spawn_validity")
    spawn_block = spawn_block if isinstance(spawn_block, Mapping) else {}
    exact = ExactEvents(
        collision=_bool_at(outcome, "collision_event") or termination_reason == "collision",
        goal_reached=_bool_at(outcome, "route_complete") or termination_reason == "success",
        timeout=_bool_at(outcome, "timeout_event")
        or termination_reason in {"max_steps", "truncated"},
        invalid_run=termination_reason == "error"
        or record.get("status") in {"error", "invalid"}
        or spawn_invalid,
    )
    surrogate = SurrogateEvents(
        near_miss=near_miss_value is not None and near_miss_value > 0.0,
        clearance_breach=min_clearance is not None and min_clearance <= 0.0,
        ttc_breach=ttc_value is not None and ttc_value > 0.0,
        oscillation=predicate_oscillation
        if predicate_oscillation_source is not None
        else oscillation_value is not None and oscillation_value > 0.0,
        late_evasive=late_evasive,
        occlusion_near_miss=occlusion_near_miss,
    )
    reconciliation = {
        "collision_metric_value": collision_value,
        "collision_metric_source": collision_source,
        "audit_result": "unchecked",
    }
    ledger = EpisodeEventLedger(
        schema_version=EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
        scenario_id=str(record.get("scenario_id") or "unknown"),
        seed=_safe_int(record.get("seed"), default=0),
        planner=str(record.get("algo") or record.get("planner") or "unknown"),
        software_commit=(
            str(record.get("git_hash")) if record.get("git_hash") is not None else None
        ),
        exact_events=exact,
        collision_events=normalized_collision_events,
        surrogate_events=surrogate,
        metric_definitions=metric_definitions,
        reconciliation=reconciliation,
        provenance={"source": "episode_record"},
    )
    payload = ledger.to_dict()
    if spawn_invalid:
        # The observed route outcome is preserved even when the start is invalid.
        payload["provenance"]["invalid_reason"] = spawn_block.get("invalid_reason")
        payload["provenance"]["spawn_validity_schema_version"] = spawn_block.get("schema_version")
    if safety_wrapper is not None:
        payload["provenance"]["safety_wrapper"] = safety_wrapper
    if cbf_safety_filter is not None:
        payload["provenance"]["cbf_safety_filter"] = cbf_safety_filter
    if safety_predicates:
        payload["surrogate_events"].update(safety_predicates)
    scenario_params = record.get("scenario_params")
    if (
        isinstance(scenario_params, Mapping)
        and scenario_params.get("collision_attribution_version") == "v1"
    ):
        for event, source in zip(
            payload["collision_events"], _sorted_contact_sources(collision_events), strict=True
        ):
            # Keep the step measurement and contact set next to the exact event.
            # The separate classification can then be reconciled against this
            # source; release admission still requires trace-derived remeasurement.
            event["robot_speed_at_contact_m_s"] = source.get("robot_speed_at_contact_m_s")
            event["contact_step_index"] = source.get("contact_step_index")
            if isinstance(source.get("contact_partner_ids"), list):
                event["contact_partner_ids"] = list(source["contact_partner_ids"])
        payload["contact_provenance"] = _contact_provenance(record, payload["collision_events"])
    payload["reconciliation"]["audit_result"] = (
        "pass" if not reconcile_event_ledger(payload) else "fail"
    )
    return payload


def ensure_event_ledger(record: dict[str, Any]) -> dict[str, Any]:
    """Attach a canonical event ledger to an episode record when missing.

    Returns:
        The existing or newly attached event ledger payload.
    """
    ledger = record.get("event_ledger")
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema_version") != EPISODE_EVENT_LEDGER_SCHEMA_VERSION
    ):
        record["event_ledger"] = build_event_ledger(record)
    return record["event_ledger"]


def reconcile_event_ledger(ledger: Mapping[str, Any]) -> list[str]:
    """Return reconciliation violations for one event ledger."""
    violations: list[str] = []
    exact = ledger.get("exact_events")
    exact = exact if isinstance(exact, Mapping) else {}
    reconciliation = ledger.get("reconciliation")
    reconciliation = reconciliation if isinstance(reconciliation, Mapping) else {}
    metric_definitions = ledger.get("metric_definitions")
    metric_definitions = metric_definitions if isinstance(metric_definitions, Mapping) else {}
    collision_events = ledger.get("collision_events")
    collision_events = collision_events if isinstance(collision_events, list) else []

    collision_metric = _finite_float(reconciliation.get("collision_metric_value"))
    if bool(exact.get("collision")) and (collision_metric is None or collision_metric <= 0.0):
        violations.append("exact collision event requires collision metric > 0")
    provenance = ledger.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    invalid_start_with_observed_goal = provenance.get(
        "spawn_validity_schema_version"
    ) == SPAWN_VALIDITY_SCHEMA_VERSION and provenance.get("invalid_reason") in {
        SPAWN_OVERLAP_INVALID_REASON,
        RESET_CLEARANCE_UNAVAILABLE_REASON,
    }
    if (
        bool(exact.get("goal_reached"))
        and bool(exact.get("invalid_run"))
        and not invalid_start_with_observed_goal
    ):
        violations.append("goal_reached and invalid_run are mutually exclusive")
    missing_definitions = [
        name
        for name in (
            "collision_count",
            "near_miss",
            "clearance_breach",
            "ttc_breach",
            "oscillation",
            "late_evasive",
            "occlusion_near_miss",
        )
        if not isinstance(metric_definitions.get(name), Mapping)
        or not metric_definitions[name].get("kind")
    ]
    if missing_definitions:
        violations.append("missing metric definitions: " + ", ".join(missing_definitions))
    for index, event in enumerate(collision_events):
        if not isinstance(event, Mapping):
            violations.append(f"collision_events[{index}] must be a mapping")
            continue
        partner_type = str(event.get("collision_partner_type") or "")
        if partner_type not in COLLISION_PARTNER_TYPES:
            violations.append(
                f"collision_events[{index}].collision_partner_type invalid: {partner_type!r}"
            )
        collision_time = _finite_float(event.get("collision_time"))
        if collision_time is None or collision_time < 0.0:
            violations.append(f"collision_events[{index}].collision_time must be finite and >= 0")
        if not str(event.get("clearance_series_source") or "").strip():
            violations.append(
                f"collision_events[{index}].clearance_series_source must be non-empty"
            )
        if not str(event.get("exact_event_source") or "").strip():
            violations.append(f"collision_events[{index}].exact_event_source must be non-empty")
    violations.extend(_contact_provenance_violations(ledger, collision_events))
    return violations


def _contact_provenance_violations(
    ledger: Mapping[str, Any], collision_events: list[Any]
) -> list[str]:
    """Check the opt-in block without altering legacy ledger reconciliation.

    Returns:
        Violations of the contact-provenance structural contract.
    """
    violations: list[str] = []
    contact_provenance = ledger.get("contact_provenance")
    if contact_provenance is not None:
        if (
            not isinstance(contact_provenance, Mapping)
            or contact_provenance.get("schema_version") != CONTACT_PROVENANCE_SCHEMA_VERSION
        ):
            violations.append("contact_provenance must use contact_provenance.v1")
        else:
            contacts = contact_provenance.get("contacts")
            if not isinstance(contacts, list) or len(contacts) != len(collision_events):
                violations.append("contact_provenance must cover every collision event")
            else:
                for index, contact in enumerate(contacts):
                    violations.extend(
                        _contact_entry_violations(index, contact, collision_events[index])
                    )
            if contact_provenance.get("planner_causation_admitted") is not False:
                violations.append("contact_provenance cannot admit planner causation")
            status = (
                _contact_classification_status(contact_provenance, contacts or [])
                if isinstance(contacts, list)
                else None
            )
            if status is not None and contact_provenance.get("classification_status") != status:
                violations.append("contact_provenance classification status is inconsistent")
    return violations


def _contact_entry_violations(index: int, contact: Any, event: Any) -> list[str]:
    """Check one classified contact against its exact event.

    Returns:
        Violations for this contact index.
    """
    prefix = f"contact_provenance.contacts[{index}]"
    if not isinstance(event, Mapping):
        return [f"{prefix} exact event is not a mapping"]
    if not isinstance(contact, Mapping) or contact.get("collision_event_index") != index:
        return [f"{prefix} index mismatch"]
    violations = _contact_source_violations(prefix, contact, event)
    speed = _finite_float(contact.get("robot_speed_at_contact_m_s"))
    if speed is None or speed < 0.0:
        if contact.get("contact_class") != "unresolved_missing_robot_speed":
            violations.append(f"{prefix} missing speed must be unresolved")
        return violations
    step_index = contact.get("contact_step_index")
    if not isinstance(step_index, int) or isinstance(step_index, bool) or step_index < 0:
        if contact.get("contact_class") != "unresolved_missing_step_index":
            violations.append(f"{prefix} missing step index must be unresolved")
        return violations
    expected_class = (
        "geometry_contact"
        if contact.get("collision_partner_type") != "pedestrian"
        else "reset_overlap_contact"
        if contact.get("reset_overlap_match") is True
        else "respawn_contact"
        if contact.get("respawn_match") is not None
        else "pedestrian_on_stationary_robot"
        if speed <= STATIONARY_ROBOT_SPEED_THRESHOLD_M_S
        else "moving_robot_pedestrian_contact"
    )
    if contact.get("contact_class") != expected_class:
        violations.append(f"{prefix} class inconsistent with speed")
    return violations


def _contact_source_violations(
    prefix: str, contact: Mapping[str, Any], event: Mapping[str, Any]
) -> list[str]:
    """Check classified fields against their saved exact-event source.

    Returns:
        Source/provenance disagreement messages.
    """
    violations: list[str] = []
    if contact.get("collision_time") != event.get("collision_time") or contact.get(
        "collision_partner_type"
    ) != event.get("collision_partner_type"):
        violations.append(f"{prefix} event identity mismatch")
    if contact.get("robot_speed_at_contact_m_s") != event.get("robot_speed_at_contact_m_s"):
        violations.append(f"{prefix} robot speed differs from saved collision source")
    if contact.get("contact_step_index") != event.get("contact_step_index"):
        violations.append(f"{prefix} step index differs from saved collision source")
    if contact.get("robot_speed_measurement_window") != "simulation_step":
        violations.append(f"{prefix} robot speed measurement window is missing")
    if "contact_partner_ids" in event and not isinstance(event["contact_partner_ids"], list):
        violations.append(f"{prefix} saved contact partner IDs must be a list")
    return violations


def validate_record_event_ledger(record: dict[str, Any]) -> list[str]:
    """Ensure and reconcile the event ledger on one episode record.

    Returns:
        Reconciliation violation messages, if any.
    """
    ledger = ensure_event_ledger(record)
    violations = reconcile_event_ledger(ledger)
    scenario_params = record.get("scenario_params")
    if isinstance(scenario_params, Mapping):
        version = scenario_params.get("collision_attribution_version")
        if version not in (None, "v1"):
            violations.append(f"unsupported collision_attribution_version: {version!r}")
        elif version == "v1" and "contact_provenance" not in ledger:
            violations.append("opt-in contact_provenance.v1 block is missing")
        elif version == "v1":
            source_events = ledger.get("collision_events")
            if not isinstance(source_events, list) or not all(
                isinstance(event, Mapping) for event in source_events
            ):
                violations.append("opt-in collision event sources are malformed")
            elif ledger.get("contact_provenance") != _contact_provenance(record, source_events):
                violations.append("opt-in contact provenance differs from saved event source")
    return violations


__all__ = [
    "COLLISION_PARTNER_TYPES",
    "EPISODE_EVENT_LEDGER_SCHEMA_VERSION",
    "SUPPORTED_EVENT_LEDGER_SCHEMA_VERSIONS",
    "CollisionEventRecord",
    "EpisodeEventLedger",
    "ExactEvents",
    "SurrogateEvents",
    "build_event_ledger",
    "ensure_event_ledger",
    "reconcile_event_ledger",
    "validate_record_event_ledger",
]
