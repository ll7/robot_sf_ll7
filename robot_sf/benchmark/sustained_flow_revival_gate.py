"""Revival gate for issue #3813 sustained-flow follow-up work.

The gate consumes h600 interaction-exposure evidence from issue #3810 and decides whether
sustained-flow work should be revived, stopped as not load-bearing, or deferred until the
evidence is complete enough to judge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUSTAINED_FLOW_REVIVAL_GATE_SCHEMA_VERSION = "issue_3813.sustained_flow_revival_gate.v1"

DECISION_REVIVE = "revive_sustained_flow"
DECISION_STOP = "stop_not_load_bearing"
DECISION_DEFER = "defer_needs_more_evidence"
REVIVAL_GATE_DECISIONS = (DECISION_REVIVE, DECISION_STOP, DECISION_DEFER)

DEFAULT_H600_INTERACTION_EXPOSURE_EVIDENCE = Path(
    "docs/context/evidence/issue_3810_h600_interpretation_2026-07/"
    "interaction_exposure_diagnostics.json"
)

REQUIRED_INTERACTION_EXPOSURE_FIELDS = (
    "interaction_exposure_share",
    "robot_motion_share_before_first_clearance",
    "first_clearance_step",
    "low_exposure_success",
)

_COMPLETE_EXPOSURE_STATUSES = {"computed", "ok", "complete"}


@dataclass(frozen=True, slots=True)
class SustainedFlowRevivalGateReport:
    """Machine-readable issue #3813 revival-gate decision."""

    schema_version: str
    issue: int
    source_issue: int
    decision: str
    evidence_status: str
    interaction_exposure_evidence_path: str
    claim_impact_evidence_path: str | None
    required_interaction_exposure_fields: tuple[str, ...]
    affected_rows: tuple[dict[str, Any], ...]
    claim_decisions_changed: bool | None
    blocking_reasons: tuple[str, ...]
    claim_boundary: str
    next_action: str

    @property
    def ready_for_revived_implementation(self) -> bool:
        """Return whether the gate authorizes new sustained-flow implementation work."""

        return self.decision == DECISION_REVIVE

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the report."""

        payload = asdict(self)
        payload["ready_for_revived_implementation"] = self.ready_for_revived_implementation
        return payload


class SustainedFlowRevivalGateError(ValueError):
    """Raised when the revival-gate inputs are structurally invalid."""


def build_sustained_flow_revival_gate_report(
    interaction_exposure: Mapping[str, Any],
    *,
    interaction_exposure_evidence_path: str | Path = DEFAULT_H600_INTERACTION_EXPOSURE_EVIDENCE,
    claim_impact: Mapping[str, Any] | None = None,
    claim_impact_evidence_path: str | Path | None = None,
) -> SustainedFlowRevivalGateReport:
    """Classify whether issue #3813 sustained-flow work should be revived.

    Args:
        interaction_exposure: Parsed h600 interaction-exposure diagnostic evidence.
        interaction_exposure_evidence_path: Provenance path for the diagnostic evidence.
        claim_impact: Optional parsed report describing whether wait-it-out exclusions or
            caveats change claim decisions.
        claim_impact_evidence_path: Provenance path for ``claim_impact`` when supplied.

    Returns:
        A stable report with one of the three issue-authorized gate decisions.
    """

    if not isinstance(interaction_exposure, Mapping):
        raise SustainedFlowRevivalGateError("interaction exposure evidence must be a mapping")
    if claim_impact is not None and not isinstance(claim_impact, Mapping):
        raise SustainedFlowRevivalGateError("claim impact evidence must be a mapping")

    blocking_reasons: list[str] = []
    if not _interaction_exposure_complete(interaction_exposure):
        blocking_reasons.append("interaction-exposure diagnostics missing computed required fields")

    affected_rows = _affected_rows(interaction_exposure, claim_impact)
    if not affected_rows:
        blocking_reasons.append("affected planner/scenario rows not supplied")

    claim_decisions_changed = _claim_decisions_changed(interaction_exposure, claim_impact)
    if claim_decisions_changed is None:
        blocking_reasons.append("claim-decision impact not supplied")

    if blocking_reasons:
        decision = DECISION_DEFER
        evidence_status = "diagnostic-only-incomplete"
        next_action = (
            "retain h600 rows with interaction-exposure fields plus claim-impact comparison"
        )
    elif claim_decisions_changed:
        decision = DECISION_REVIVE
        evidence_status = "diagnostic-gate-complete"
        next_action = "start opt-in sustained-flow scenario schema prototype"
    else:
        decision = DECISION_STOP
        evidence_status = "diagnostic-gate-complete"
        next_action = "record sustained-flow as not load-bearing for current h600 evidence"

    return SustainedFlowRevivalGateReport(
        schema_version=SUSTAINED_FLOW_REVIVAL_GATE_SCHEMA_VERSION,
        issue=3813,
        source_issue=3810,
        decision=decision,
        evidence_status=evidence_status,
        interaction_exposure_evidence_path=Path(interaction_exposure_evidence_path).as_posix(),
        claim_impact_evidence_path=(
            Path(claim_impact_evidence_path).as_posix()
            if claim_impact_evidence_path is not None
            else None
        ),
        required_interaction_exposure_fields=REQUIRED_INTERACTION_EXPOSURE_FIELDS,
        affected_rows=affected_rows,
        claim_decisions_changed=claim_decisions_changed,
        blocking_reasons=tuple(blocking_reasons),
        claim_boundary=(
            "Revival gate only; no sustained-flow runtime implementation, benchmark campaign, "
            "default benchmark conclusion, or paper-facing claim is changed."
        ),
        next_action=next_action,
    )


def _interaction_exposure_complete(report: Mapping[str, Any]) -> bool:
    missing_fields = set(_string_sequence(report.get("missing_required_fields")))
    if missing_fields:
        return False

    runs = _run_mappings(report)
    if runs is not None and not _runs_complete(runs):
        return False
    if runs is None and str(report.get("status", "")).strip() not in _COMPLETE_EXPOSURE_STATUSES:
        return False

    available_fields = set(_string_sequence(report.get("available_columns")))
    available_fields.update(_string_sequence(report.get("computed_fields")))
    for run in runs or ():
        available_fields.update(_string_sequence(run.get("available_columns")))
        available_fields.update(_string_sequence(run.get("computed_fields")))
    # Fail closed: a report that only asserts a "complete" status string without
    # positively declaring the required exposure fields (via available_columns/
    # computed_fields at top level or per run) is treated as INCOMPLETE, so a
    # degraded upstream artifact cannot pass the gate to revive/stop. Absent
    # field-listing is not proof of completeness.
    if not set(REQUIRED_INTERACTION_EXPOSURE_FIELDS) <= available_fields:
        return False

    return True


def _run_mappings(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...] | None:
    runs = report.get("runs")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        return None
    return tuple(run for run in runs if isinstance(run, Mapping))


def _runs_complete(runs: tuple[Mapping[str, Any], ...]) -> bool:
    if not runs:
        return False
    for run in runs:
        if set(_string_sequence(run.get("missing_required_fields"))):
            return False
        status = str(run.get("status", "")).strip()
        if status and status not in _COMPLETE_EXPOSURE_STATUSES:
            return False
    return True


def _affected_rows(
    interaction_exposure: Mapping[str, Any],
    claim_impact: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    for source in (claim_impact, interaction_exposure):
        if source is None:
            continue
        for key in (
            "affected_rows",
            "affected_planner_scenario_rows",
            "affected_planner_rows",
        ):
            rows = source.get(key)
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                return tuple(dict(row) for row in rows if isinstance(row, Mapping))
    return ()


def _claim_decisions_changed(
    interaction_exposure: Mapping[str, Any],
    claim_impact: Mapping[str, Any] | None,
) -> bool | None:
    for source in (claim_impact, interaction_exposure):
        if source is None:
            continue
        for key in ("claim_decisions_changed", "claim_decision_changed"):
            value = source.get(key)
            if isinstance(value, bool):
                return value
        decision_delta = source.get("claim_decision_delta")
        if isinstance(decision_delta, Mapping):
            changed = decision_delta.get("changed")
            if isinstance(changed, bool):
                return changed
    return None


def _string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value)
