"""Fail-closed validator for the ``collision_causal_report.v1`` contract.

This module owns the cross-field semantic rules that the JSON schema in
``schemas/collision_causal_report.v1.json`` cannot express. The report is a
*model-scoped* reconstruction of a single collision: it separates observed
facts, the proximate mechanism, and the intervention-supported causal
contribution, and it never assigns legal or moral fault
(``normative_fault`` is always ``"not_assessed"``).

Design source: ``docs/context/collision_causality_online_risk_scenario_discovery_2026-07-12.md``
section 3 and the field map in
``docs/context/collision_causal_report_field_map_2026-07-13.md``.

Enum vocabularies for ``mechanism_label`` and ``confidence.level`` are reused
verbatim from :mod:`robot_sf.benchmark.failure_mechanism_taxonomy` so this
contract never grows a competing aggregate taxonomy. ``cause_location`` is an
orthogonal axis (a node in the implemented temporal causal graph), not a second
mechanism vocabulary.

The contract is deliberately conservative: planner-internal reconstruction
fields (observations, predictions, generated/selected candidates, guard and
arbitration results, feasible/applied commands) are marked *unavailable* unless
a canonical trace owner actually produced them. First-unsafe-action
(``t_uca``) and last-avoidable-state (``t_inevitable``) timestamps are available
in a joined report only when the replay engine actually computed them;
planner-internal reconstruction elements remain unavailable because this join
does not own a canonical trace for them. Fabricating those elements from
report-wide replay coverage is not allowed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_CONFIDENCES,
    MECHANISM_LABELS,
)
from robot_sf.benchmark.last_avoidable_replay import (
    VERDICT_ALREADY_UNAVOIDABLE,
    VERDICT_AVOIDABLE,
    VERDICT_UNKNOWN,
    LastAvoidableReport,
)
from robot_sf.errors import RobotSfError

if TYPE_CHECKING:
    from collections.abc import Mapping

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
    raise RuntimeError(
        "jsonschema package required for collision causal report validation"
    ) from exc

COLLISION_CAUSAL_REPORT_SCHEMA_VERSION = "collision_causal_report.v1"
COLLISION_CAUSAL_REPORT_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "collision_causal_report.v1.json"
)

#: The four systems-theoretic (STPA) unsafe-control-action forms plus fail-closed values.
UNSAFE_CONTROL_ACTION_CLASSES = frozenset(
    {
        "action_not_provided",
        "unsafe_action_provided",
        "unsafe_timing_or_order",
        "unsafe_duration",
        "not_applicable",
        "unknown",
    }
)

#: Dataflow-stage cause locations (temporal causal-graph nodes). Orthogonal to
#: ``mechanism_label``; this is *where* in the implemented pipeline the cause sits.
CAUSE_LOCATIONS = frozenset(
    {
        "scenario_initialisation",
        "observation_perception",
        "prediction",
        "candidate_generation",
        "candidate_scoring_selection",
        "safety_guard_arbitration",
        "command_conversion_actuation",
        "robot_dynamics",
        "pedestrian_response",
        "collision_metrics",
        "scenario_infeasibility_or_unavoidable",
        "simulator_logging_or_metric_artifact",
        "unknown_or_interacting",
    }
)

CAUSAL_VERDICTS = frozenset({"avoidable", "unavoidable", "unknown"})
PEDESTRIAN_RESPONSE_ASSUMPTIONS = frozenset({"replayed", "closed_loop", "unknown"})

#: The four incident timestamps, ordered as they occur along one timeline.
CRITICAL_TIMESTAMP_KEYS = ("t_danger", "t_uca", "t_inevitable", "t_contact")

#: Reconstruction elements that must be present (each as available/unavailable).
RECONSTRUCTION_ELEMENT_KEYS = (
    "observations",
    "predictions",
    "generated_candidates",
    "selected_candidate",
    "guard_arbitration_result",
    "feasible_command",
    "applied_command",
    "actor_states",
    "geometry",
)


class CollisionCausalReportError(RobotSfError, ValueError):
    """Raised when a collision causal report violates the fail-closed contract."""


def load_collision_causal_report_schema() -> dict[str, Any]:
    """Load the ``collision_causal_report.v1`` JSON schema from disk.

    Returns:
        The parsed JSON schema dictionary.
    """

    with COLLISION_CAUSAL_REPORT_SCHEMA_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reconcile_collision_causal_report(record: Mapping[str, Any]) -> list[str]:
    """Return the list of fail-closed contract violations for ``record``.

    The check is non-raising so callers can accumulate violations (the house
    pattern used by :func:`robot_sf.benchmark.event_ledger.reconcile_event_ledger`).
    An empty list means the record satisfies both the JSON schema and every
    cross-field semantic rule.

    Returns:
        Human-readable violation strings; empty when the record is valid.
    """

    schema = load_collision_causal_report_schema()
    try:
        jsonschema.validate(instance=record, schema=schema)
    except jsonschema.ValidationError as error:
        return [f"schema: {error.message}"]

    violations: list[str] = []
    violations.extend(_normative_fault_violations(record))
    violations.extend(_vocabulary_violations(record))
    violations.extend(_timestamp_violations(record))
    violations.extend(_missing_field_violations(record))
    violations.extend(_abstention_violations(record))
    violations.extend(_actual_cause_violations(record))
    violations.extend(_inevitability_violations(record))
    return violations


def validate_collision_causal_report(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a collision causal report, raising on the first contract breach.

    Returns:
        A shallow ``dict`` copy of the validated record.

    Raises:
        CollisionCausalReportError: If any schema or semantic rule is violated.
    """

    violations = reconcile_collision_causal_report(record)
    if violations:
        raise CollisionCausalReportError("; ".join(violations))
    return dict(record)


def abstained_collision_causal_report(
    *,
    report_id: str,
    case_id: str,
    reason: str,
    source_kind: str = "unknown",
    missing_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Build a valid, fully abstaining report that fails closed.

    Every reconstruction element and timestamp is marked unavailable, the
    verdict is ``unknown``, and no actual cause is asserted. This is the correct
    artifact when replay is nondeterministic, required trace fields are absent,
    or competing explanations are observationally equivalent.

    Returns:
        A report dict that satisfies :func:`validate_collision_causal_report`.
    """

    clean_reason = reason.strip() or "not_derivable"
    unavailable_ts = {
        key: {"available": False, "step": None, "time_s": None, "source": None}
        for key in CRITICAL_TIMESTAMP_KEYS
    }
    unavailable_elements = {
        key: {"available": False, "source": None, "detail": None}
        for key in RECONSTRUCTION_ELEMENT_KEYS
    }
    declared_missing = list(missing_fields or [])
    for key in (*CRITICAL_TIMESTAMP_KEYS, *RECONSTRUCTION_ELEMENT_KEYS):
        if key not in declared_missing:
            declared_missing.append(key)
    return {
        "schema_version": COLLISION_CAUSAL_REPORT_SCHEMA_VERSION,
        "report_id": report_id,
        "case_id": case_id,
        "normative_fault": "not_assessed",
        "data_source": {
            "source_kind": source_kind,
            "provenance_uri": None,
            "software_commit": None,
            "replay_determinism": "unknown",
        },
        "abstained": True,
        "abstention_reason": clean_reason,
        "observed_reconstruction": {
            "critical_timestamps": unavailable_ts,
            "elements": unavailable_elements,
        },
        "proximate_mechanism": {
            "mechanism_label": "unknown",
            "cause_location": "unknown_or_interacting",
            "unsafe_control_action_class": "unknown",
            "rationale": clean_reason,
        },
        "causal_contribution": {
            "verdict": "unknown",
            "intervention_model": "",
            "pedestrian_response_assumption": "unknown",
            "supported_actual_cause": False,
            "interventions": [],
        },
        "confidence": {"level": "unknown", "rationale": clean_reason},
        "assumptions": [],
        "missing_fields": declared_missing,
        "competing_explanations": [],
    }


# ---------------------------------------------------------------------------
# Counterfactual-replay join (issue #5442 -> #5441)
# ---------------------------------------------------------------------------

#: Verdict translation from ``last_avoidable_replay.v1`` to the causal-report
#: ``causal_contribution.verdict`` enum (``avoidable``/``unavoidable``/``unknown``).
_LAST_AVOIDABLE_TO_CAUSAL_VERDICT = {
    VERDICT_AVOIDABLE: "avoidable",
    VERDICT_ALREADY_UNAVOIDABLE: "unavoidable",
    VERDICT_UNKNOWN: "unknown",
}


@dataclass(frozen=True)
class CausalJoinMetadata:
    """Shared causal-report provenance for a ``last_avoidable_replay.v1`` join.

    The replay engine proves avoidability under the named intervention model,
    but it does not localise a planner-internal mechanism. Callers must provide
    independent evidence for the mechanism fields; otherwise they must use the
    explicit ``unknown`` values.

    Attributes:
        intervention_model: Named intervention/model under which the causal
            contribution is asserted (e.g. ``frozen_state_brake_swap__replayed_pedestrians``).
        mechanism_label: Shared taxonomy mechanism label for the proximate mechanism.
        cause_location: Temporal causal-graph node where the cause sits.
        unsafe_control_action_class: STPA form of the first unsafe control action.
        confidence_level: Shared taxonomy confidence level.
    """

    intervention_model: str
    mechanism_label: str
    cause_location: str
    unsafe_control_action_class: str
    confidence_level: str


def collide_causal_report_from_last_avoidable(
    *,
    report_id: str,
    case_id: str,
    replay: LastAvoidableReport,
    metadata: CausalJoinMetadata,
    competing_explanations: list[dict[str, Any]] | None = None,
    assumptions: list[str] | None = None,
) -> dict[str, Any]:
    """Embed a ``last_avoidable_replay.v1`` result as a ``collision_causal_report.v1``.

    This is the join named as remaining by issue #5442 (and PR #5459) now that the
    #5441 report contract has merged: a frozen-state counterfactual replay engine
    produces a self-contained ``last_avoidable_replay.v1``; this function wraps it
    into the additive causal-report contract without re-running the engine.

    Failure semantics are preserved across the join. A nondeterministic baseline or
    an incomplete feasible-action set yields a replay verdict of ``unknown`` and
    therefore a *fully abstaining* causal report — this never becomes
    ``unavoidable``. An ``already_unavoidable`` replay maps to a non-abstaining
    report whose causal ``verdict`` is ``unavoidable`` and whose
    ``supported_actual_cause`` is ``False`` (no planner action is the cause when
    contact was already unavoidable at ``t_danger``). An ``avoidable`` replay maps to
    a non-abstaining report that names the intervention model, records the
    minimal-sufficient preventing interventions, and sets
    ``supported_actual_cause`` to ``True``. In every case ``normative_fault`` stays
    ``not_assessed``.

    Args:
        report_id: Identifier for the produced causal report.
        case_id: Identifier for the collision case being analysed.
        replay: A validated :class:`~robot_sf.benchmark.last_avoidable_replay.LastAvoidableReport`.
        metadata: Shared causal-report provenance (intervention model, mechanism
            label/location, STPA class, confidence level).
        competing_explanations: Optional list of competing-explanation objects.
        assumptions: Optional list of named assumptions.

    Returns:
        A report dict satisfying :func:`validate_collision_causal_report`.
    """

    _validate_join_metadata(metadata)

    unsupported_verdict = replay.verdict not in _LAST_AVOIDABLE_TO_CAUSAL_VERDICT
    abstained = replay.abstained or replay.verdict == VERDICT_UNKNOWN or unsupported_verdict
    causal_verdict = "unknown" if abstained else _LAST_AVOIDABLE_TO_CAUSAL_VERDICT[replay.verdict]
    supported_actual_cause = replay.verdict == VERDICT_AVOIDABLE and not abstained

    config = replay.config
    ped_response = config.pedestrian_response
    replay_determinism = "deterministic" if replay.determinism.deterministic else "nondeterministic"
    rationale = _join_rationale(replay, abstained)
    missing_fields = _join_missing_fields(replay)
    interventions = _join_interventions(replay, ped_response, supported_actual_cause)

    if unsupported_verdict:
        abstention_reason = "unsupported_replay_verdict"
    elif replay.abstained and not replay.abstain_reason:
        abstention_reason = "replay_marked_abstained"
    else:
        abstention_reason = replay.abstain_reason or ""

    # A replay that does not support an actual cause cannot support a mechanism,
    # unsafe-action, or confidence claim either. In particular, an
    # already-unavoidable result is not evidence for metadata a caller may have
    # supplied for an avoidable fixture.
    mechanism_metadata = metadata if supported_actual_cause else None

    def _ts(available: bool, step: int | None) -> dict[str, Any]:
        # The replay reports t_* as integer control steps; express them as steps
        # (time_s is unknown without a recorded dt in the replay contract).
        return {
            "available": available,
            "step": step,
            "time_s": None,
            "source": (
                "last_avoidable_replay.v1.frozen_state_counterfactual_branch" if available else None
            ),
        }

    report: dict[str, Any] = {
        "schema_version": COLLISION_CAUSAL_REPORT_SCHEMA_VERSION,
        "report_id": report_id,
        "case_id": case_id,
        "normative_fault": "not_assessed",
        "data_source": {
            "source_kind": "synthetic_fixture",
            "provenance_uri": "last_avoidable_replay.v1+frozen_state_counterfactual_branch",
            "software_commit": None,
            "replay_determinism": replay_determinism,
        },
        "abstained": abstained,
        "abstention_reason": abstention_reason if abstained else "",
        "observed_reconstruction": {
            "critical_timestamps": {
                "t_danger": _ts(True, config.t_danger),
                "t_uca": _ts(replay.t_uca is not None, replay.t_uca),
                "t_inevitable": _ts(replay.t_inevitable is not None, replay.t_inevitable),
                "t_contact": _ts(True, config.t_contact),
            },
            "elements": {
                key: {
                    "available": False,
                    "source": None,
                    "detail": None,
                }
                for key in RECONSTRUCTION_ELEMENT_KEYS
            },
        },
        "proximate_mechanism": {
            "mechanism_label": (
                mechanism_metadata.mechanism_label if mechanism_metadata else "unknown"
            ),
            "cause_location": (
                mechanism_metadata.cause_location
                if mechanism_metadata
                else "unknown_or_interacting"
            ),
            "unsafe_control_action_class": (
                mechanism_metadata.unsafe_control_action_class if mechanism_metadata else "unknown"
            ),
            "rationale": rationale,
        },
        "causal_contribution": {
            "verdict": causal_verdict,
            "intervention_model": metadata.intervention_model if supported_actual_cause else "",
            "pedestrian_response_assumption": ped_response,
            "supported_actual_cause": supported_actual_cause,
            "interventions": interventions,
        },
        "confidence": {
            "level": mechanism_metadata.confidence_level if mechanism_metadata else "unknown",
            "rationale": rationale,
        },
        "assumptions": list(assumptions or []),
        "missing_fields": missing_fields,
        "competing_explanations": list(competing_explanations or []),
    }
    return validate_collision_causal_report(report)


def _validate_join_metadata(metadata: CausalJoinMetadata) -> None:
    """Fail closed if any join metadata value is outside the shared vocabularies."""
    if metadata.mechanism_label not in MECHANISM_LABELS:
        raise CollisionCausalReportError(
            f"mechanism_label {metadata.mechanism_label!r} is not in "
            "failure_mechanism_taxonomy.MECHANISM_LABELS"
        )
    if metadata.confidence_level not in MECHANISM_CONFIDENCES:
        raise CollisionCausalReportError(
            f"confidence.level {metadata.confidence_level!r} is not in "
            "failure_mechanism_taxonomy.MECHANISM_CONFIDENCES"
        )
    if metadata.cause_location not in CAUSE_LOCATIONS:
        raise CollisionCausalReportError(
            f"cause_location {metadata.cause_location!r} is not in CAUSE_LOCATIONS"
        )


def _join_rationale(replay: LastAvoidableReport, abstained: bool) -> str:
    """Return the proximate-mechanism/confidence rationale for the join verdict."""
    if abstained:
        return replay.abstain_reason or "last-avoidable replay abstained before causal attribution"
    if replay.verdict == VERDICT_AVOIDABLE:
        return (
            "Frozen-state counterfactual replay found an admissible action that "
            "prevented contact; the earliest avoiding action is reported as t_uca."
        )
    if replay.verdict == VERDICT_ALREADY_UNAVOIDABLE:
        return (
            "Frozen-state counterfactual replay found no admissible action preventing "
            "contact under full feasible-action coverage; contact was already "
            "unavoidable at t_danger, so no planner action is the cause."
        )
    return replay.abstain_reason or "last-avoidable replay unable to test avoidability"


def _join_missing_fields(replay: LastAvoidableReport) -> list[str]:
    """Return the missing_fields list for the joined report.

    The replay summary has no per-element canonical trace, so every
    planner-internal reconstruction element is unavailable for every verdict.
    Unsupported evidence is explicit, never inferred from report-wide coverage.
    """
    missing: list[str] = []
    missing.extend(RECONSTRUCTION_ELEMENT_KEYS)
    for key in CRITICAL_TIMESTAMP_KEYS:
        is_available = (
            key in ("t_danger", "t_contact")
            or (key == "t_uca" and replay.t_uca is not None)
            or (key == "t_inevitable" and replay.t_inevitable is not None)
        )
        if not is_available and key not in missing:
            missing.append(key)
    return missing


def _join_interventions(
    replay: LastAvoidableReport, ped_response: str, supported_actual_cause: bool
) -> list[dict[str, Any]]:
    """Return preventing interventions when an actual cause is supported, else empty."""
    if not supported_actual_cause:
        return []
    return [
        {
            "name": f"{item['action_label']}_at_step_{item['step']}",
            "admissible": True,
            "prevented_contact": True,
            "assumptions": [
                f"pedestrian response assumption: {ped_response}",
                f"substitution_mode: {item['substitution_mode']}",
                f"horizon: {item['horizon']}",
            ],
        }
        for item in replay.minimal_sufficient_interventions
    ]


# ---------------------------------------------------------------------------
# Semantic rule helpers
# ---------------------------------------------------------------------------


def _normative_fault_violations(record: Mapping[str, Any]) -> list[str]:
    if record.get("normative_fault") != "not_assessed":
        return ["normative_fault must be 'not_assessed'"]
    return []


def _vocabulary_violations(record: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    mechanism = record.get("proximate_mechanism", {})
    label = mechanism.get("mechanism_label")
    if label not in MECHANISM_LABELS:
        violations.append(
            f"mechanism_label {label!r} is not in failure_mechanism_taxonomy.MECHANISM_LABELS"
        )
    confidence = record.get("confidence", {})
    level = confidence.get("level")
    if level not in MECHANISM_CONFIDENCES:
        violations.append(
            f"confidence.level {level!r} is not in failure_mechanism_taxonomy.MECHANISM_CONFIDENCES"
        )
    return violations


def _timestamp_violations(record: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    timestamps = record.get("observed_reconstruction", {}).get("critical_timestamps", {})
    for key in CRITICAL_TIMESTAMP_KEYS:
        entry = timestamps.get(key, {})
        available = entry.get("available")
        step = entry.get("step")
        time_s = entry.get("time_s")
        if available:
            if step is None and time_s is None:
                violations.append(f"{key} is available but has neither step nor time_s")
        elif step is not None or time_s is not None:
            violations.append(
                f"{key} is unavailable but carries an inferred step/time_s; leave both null"
            )
    return violations


def _missing_field_violations(record: Mapping[str, Any]) -> list[str]:
    # Every unavailable element/timestamp must be declared in missing_fields.
    violations: list[str] = []
    declared = set(record.get("missing_fields", []))
    reconstruction = record.get("observed_reconstruction", {})
    timestamps = reconstruction.get("critical_timestamps", {})
    elements = reconstruction.get("elements", {})
    for key in CRITICAL_TIMESTAMP_KEYS:
        if not timestamps.get(key, {}).get("available", False) and key not in declared:
            violations.append(f"unavailable timestamp {key!r} must be listed in missing_fields")
    for key in RECONSTRUCTION_ELEMENT_KEYS:
        element = elements.get(key, {})
        if not element.get("available", False):
            if element.get("source") is not None or element.get("detail") is not None:
                violations.append(
                    f"element {key!r} is unavailable but carries inferred source/detail"
                )
            if key not in declared:
                violations.append(f"unavailable element {key!r} must be listed in missing_fields")
    return violations


def _abstention_violations(record: Mapping[str, Any]) -> list[str]:
    if not record.get("abstained"):
        return []
    violations: list[str] = []
    if not str(record.get("abstention_reason", "")).strip():
        violations.append("abstained report must set a non-empty abstention_reason")
    contribution = record.get("causal_contribution", {})
    if contribution.get("verdict") != "unknown":
        violations.append("abstained report must set causal_contribution.verdict to 'unknown'")
    if contribution.get("supported_actual_cause"):
        violations.append("abstained report cannot set supported_actual_cause to true")
    return violations


def _actual_cause_violations(record: Mapping[str, Any]) -> list[str]:
    # An intervention-supported actual cause needs an intervention that prevented contact.
    contribution = record.get("causal_contribution", {})
    if not contribution.get("supported_actual_cause"):
        return []
    violations: list[str] = []
    if contribution.get("verdict") != "avoidable":
        violations.append("supported_actual_cause requires verdict 'avoidable'")
    if not str(contribution.get("intervention_model", "")).strip():
        violations.append("supported_actual_cause requires a named intervention_model")
    interventions = contribution.get("interventions", [])
    if not any(item.get("prevented_contact") is True for item in interventions):
        violations.append(
            "supported_actual_cause requires at least one intervention with prevented_contact true"
        )
    if record.get("confidence", {}).get("level") == "unknown":
        violations.append("supported_actual_cause cannot carry confidence.level 'unknown'")
    return violations


def _inevitability_violations(record: Mapping[str, Any]) -> list[str]:
    # If contact was already inevitable at/before the first unsafe action, no planner cause.
    # From the design doc: when t_inevitable <= t_uca the system must not assign a planner
    # action as the collision cause; contact was unavoidable under that model.
    timestamps = record.get("observed_reconstruction", {}).get("critical_timestamps", {})
    uca = timestamps.get("t_uca", {})
    inevitable = timestamps.get("t_inevitable", {})
    if not (uca.get("available") and inevitable.get("available")):
        return []
    # The ordering guard only bites when the report claims a planner action as the cause.
    if not record.get("causal_contribution", {}).get("supported_actual_cause"):
        return []
    # Compare in a single shared unit so a schema-legal representation cannot evade the
    # guard: prefer integer control steps when both timestamps carry one, else fall back to
    # ``time_s``. ``step`` and ``time_s`` are both nullable in the schema, so a caller could
    # otherwise mark both timestamps ``available`` with ``step: null`` and slip an
    # inevitable-before-uca contact past the guard.
    for key in ("step", "time_s"):
        uca_value = uca.get(key)
        inevitable_value = inevitable.get(key)
        if uca_value is not None and inevitable_value is not None:
            if inevitable_value <= uca_value:
                return [
                    "t_inevitable <= t_uca: contact was already unavoidable, so a planner "
                    "action cannot be reported as the supported actual cause"
                ]
            return []
    # Available and a supported cause is claimed, but the two timestamps share no comparable
    # step or time_s: the inevitability ordering is undecidable, so fail closed rather than
    # letting the unverifiable claim stand.
    return [
        "t_uca and t_inevitable are marked available but share no comparable step or time_s: "
        "the inevitability ordering is undecidable, so a planner action cannot be reported as "
        "the supported actual cause"
    ]
