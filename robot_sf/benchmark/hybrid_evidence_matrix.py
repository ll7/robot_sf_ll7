"""Machine-readable validation for hybrid-learning evidence matrix rows."""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from robot_sf.common.artifact_paths import get_repository_root

EVALUATION_SLICES = frozenset({"not_run", "smoke", "nominal_sanity", "stress_slice", "full_matrix"})
EVIDENCE_TIERS = frozenset(
    {
        "launch_packet",
        "smoke_only",
        "nominal_only",
        "stress",
        "full_matrix",
        "degraded",
        "fallback",
        "failed",
        "not_available",
    }
)
VERDICTS = frozenset({"continue", "revise", "stop", "insufficient_evidence", "pending"})
SUCCESS_LIKE_TIERS = frozenset({"smoke_only", "nominal_only", "stress", "full_matrix"})
SYNTHESIS_TIERS = frozenset({"stress", "full_matrix"})
SYNTHESIS_VERDICTS = frozenset({"continue", "revise"})
# Campaign-lifecycle states for the #1489 prerequisite/status matrix, ordered by
# escalating readiness. Only ``complete`` lanes count toward unblocking synthesis.
COMPONENT_CAMPAIGN_STATES = ("missing", "blocked", "ready", "complete")
# Evidence tiers that represent a pre-runtime or non-comparable lane: the campaign
# has not produced durable runtime evidence yet, so the lane stays ``blocked``.
PRE_RUNTIME_TIERS = frozenset({"launch_packet", "failed", "not_available"})
# Default number of ``complete`` (synthesis-eligible) lanes required before the
# #1489 synthesis gate opens. The umbrella contract requires at least two
# component campaigns with durable comparable outputs.
DEFAULT_SYNTHESIS_PREREQUISITE_COUNT = 2
ISSUE_1489_INTEGRATION_NEXT_ACTIONS = {
    "blocked": ("Keep #1489 blocked; finish component campaign evidence before synthesis."),
    "ready": ("Integrate ready lanes into comparable durable component rows before synthesis."),
    "ready_for_synthesis": (
        "Run the conservative #1489 synthesis over the complete durable lanes."
    ),
}
# Per-mechanism recommendations the #1489 synthesis report can emit. ``continue``
# and ``revise`` are promoted only from durable ``complete`` lanes when the gate
# is open; ``stop`` is a terminal negative decision from a full-slice lane; every
# other lane state maps to ``gather_more_evidence`` (fail-closed default).
SYNTHESIS_RECOMMENDATIONS = frozenset({"continue", "revise", "stop", "gather_more_evidence"})
# Human-readable basis for each per-mechanism recommendation, keyed by the
# campaign-lifecycle state from :func:`classify_component_campaign_state`.
_SYNTHESIS_BASIS_BY_STATE = {
    "complete": "durable_complete_lane",
    "ready": "executed_non_synthesis",
    "blocked": "pre_runtime_or_invalid",
    "missing": "no_campaign_row",
}
ROW_REQUIRED_FIELDS = frozenset(
    {
        "component",
        "source_issue",
        "commit_artifact",
        "evaluation_slice",
        "guard_authority",
        "learned_component_contribution",
        "intervention_fallback_rates",
        "outcomes",
        "evidence_tier",
        "verdict",
    }
)
ROW_OPTIONAL_FIELDS = frozenset(
    {
        "comfort_exposure",
        "min_pedestrian_distance",
        "force_exposure_rate",
        "path_efficiency",
        "mean_time_to_goal",
        "baseline_comparator",
        "seed_schedule",
        "scenario_manifest",
    }
)
GUARD_FIELDS = frozenset({"mechanism", "active", "veto_rate"})
CONTRIBUTION_FIELDS = frozenset({"contribution_type", "bound", "active_rate"})
INTERVENTION_FIELDS = frozenset({"guard_veto_rate", "fallback_rate", "degraded_rate"})
OUTCOME_FIELDS = frozenset(
    {"success_rate", "collision_rate", "near_miss_rate", "low_progress_rate", "timeout_rate"}
)
DURABLE_URI_PREFIXES = (
    "wandb://",
    "wandb-artifact://",
    "artifact://",
    "s3://",
    "gs://",
    "https://",
)
_ISSUE_RE = re.compile(r"^#\d+$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


class HybridEvidenceMatrixValidationError(ValueError):
    """Raised when the validator cannot parse the input payload."""


def load_hybrid_evidence_input(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Load a hybrid evidence matrix file as one or more row mappings.

    Returns:
        Tuple containing the input format label and the loaded row list.
    """
    if not path.is_file():
        raise HybridEvidenceMatrixValidationError(f"input file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
        input_format = "rows"
    elif isinstance(payload, dict) and "rows" in payload:
        rows = payload["rows"]
        input_format = "matrix"
    elif isinstance(payload, dict):
        rows = [payload]
        input_format = "row"
    else:
        raise HybridEvidenceMatrixValidationError(
            "input payload must be a mapping, a list of rows, or a mapping with a 'rows' list"
        )
    if not isinstance(rows, list):
        raise HybridEvidenceMatrixValidationError("rows must be a list of mappings")
    return input_format, rows


def validate_hybrid_evidence_file(
    path: Path,
    *,
    repo_root: Path | None = None,
    check_git_history: bool = False,
) -> dict[str, Any]:
    """Load and validate a hybrid evidence matrix file.

    Returns:
        Structured validation report with per-row status, errors, and warnings.
    """
    input_format, rows = load_hybrid_evidence_input(path)
    report = validate_hybrid_evidence_rows(
        rows,
        repo_root=repo_root,
        check_git_history=check_git_history,
    )
    report["input_format"] = input_format
    report["input_path"] = _repo_relative_or_absolute(
        path.resolve(), root=(repo_root or get_repository_root())
    )
    return report


def validate_hybrid_evidence_rows(
    rows: list[dict[str, Any]],
    *,
    repo_root: Path | None = None,
    check_git_history: bool = False,
) -> dict[str, Any]:
    """Validate one or more hybrid evidence matrix rows.

    Returns:
        Structured validation report with aggregate counts and per-row details.
    """
    root = (repo_root or get_repository_root()).resolve()
    provenance_validation = "git_history" if check_git_history else "format_only"
    git_commit_cache: dict[str, bool] = {}
    row_reports = [
        _validate_row(
            row,
            index=index,
            repo_root=root,
            provenance_validation=provenance_validation,
            git_commit_cache=git_commit_cache,
        )
        for index, row in enumerate(rows)
    ]
    invalid_row_count = sum(1 for row in row_reports if row["status"] == "invalid")
    return {
        "status": "valid" if invalid_row_count == 0 else "invalid",
        "provenance_validation": provenance_validation,
        "row_count": len(row_reports),
        "valid_row_count": len(row_reports) - invalid_row_count,
        "invalid_row_count": invalid_row_count,
        "rows": row_reports,
    }


def classify_component_campaign_state(row_report: dict[str, Any]) -> str:
    """Classify one validated evidence row into a campaign-lifecycle state.

    The lifecycle escalates ``missing -> blocked -> ready -> complete``:

    - ``complete``: the row is synthesis-eligible — a durable comparable output
      (``stress``/``full_matrix`` tier with a ``continue``/``revise`` verdict and
      no validation errors). Only ``complete`` lanes count toward the #1489
      synthesis prerequisite, so no single pre-result lane can unblock synthesis.
    - ``ready``: the campaign executed and produced runtime evidence that is not
      yet synthesis-grade (smoke/nominal/degraded/fallback tiers, or a terminal
      tier with a non-synthesis verdict such as ``stop``).
    - ``blocked``: the row is pre-runtime or non-comparable — a launch packet, a
      ``not_run`` slice, a failed/unavailable tier, or an invalid row.

    ``missing`` is assigned by :func:`build_hybrid_prerequisite_matrix` for
    expected components that have no row; it is never returned here.

    Args:
        row_report: A per-row entry produced by :func:`validate_hybrid_evidence_rows`.

    Returns:
        One of ``"blocked"``, ``"ready"``, or ``"complete"``.
    """
    if row_report.get("synthesis_eligible"):
        return "complete"
    if row_report.get("status") == "invalid":
        return "blocked"
    if row_report.get("evaluation_slice") == "not_run":
        return "blocked"
    if row_report.get("evidence_tier") in PRE_RUNTIME_TIERS:
        return "blocked"
    return "ready"


def build_hybrid_prerequisite_matrix(
    rows: list[dict[str, Any]],
    *,
    expected_components: list[str] | None = None,
    repo_root: Path | None = None,
    check_git_history: bool = False,
    prerequisite_count: int = DEFAULT_SYNTHESIS_PREREQUISITE_COUNT,
) -> dict[str, Any]:
    """Build the #1489 prerequisite/status matrix for hybrid-learning lanes.

    Validates every row, classifies each into a campaign-lifecycle state, marks
    expected-but-absent components as ``missing``, and decides whether the
    synthesis prerequisite (at least ``prerequisite_count`` ``complete`` lanes)
    is met. The gate is intentionally conservative: a lane counts only when the
    row validator already marked it synthesis-eligible, so launch packets, smoke
    runs, fallback/degraded rows, and invalid rows never open the gate.

    Args:
        rows: Evidence-matrix row mappings, as accepted by the row validator.
        expected_components: Optional component names that should have a row.
            Each name with no matching row becomes a ``missing`` lane.
        repo_root: Repository root for provenance-path resolution.
        check_git_history: Forwarded to the row validator to verify git SHAs.
        prerequisite_count: Minimum ``complete`` lanes that open the gate.

    Returns:
        Structured report with the gate decision, per-state counts, and per-lane
        classification details.
    """
    if prerequisite_count < 1:
        raise HybridEvidenceMatrixValidationError("prerequisite_count must be >= 1")
    validation = validate_hybrid_evidence_rows(
        rows, repo_root=repo_root, check_git_history=check_git_history
    )
    lanes: list[dict[str, Any]] = []
    seen_components: set[str] = set()
    for row_report in validation["rows"]:
        component = row_report.get("component")
        if isinstance(component, str):
            seen_components.add(component)
        lanes.append(
            {
                "component": component,
                "source_issue": row_report.get("source_issue"),
                "state": classify_component_campaign_state(row_report),
                "evaluation_slice": row_report.get("evaluation_slice"),
                "evidence_tier": row_report.get("evidence_tier"),
                "verdict": row_report.get("verdict"),
                "row_status": row_report.get("status"),
                "synthesis_eligible": bool(row_report.get("synthesis_eligible")),
                "row_index": row_report.get("index"),
            }
        )
    for component in _missing_components(expected_components, seen_components):
        lanes.append(
            {
                "component": component,
                "source_issue": None,
                "state": "missing",
                "evaluation_slice": None,
                "evidence_tier": None,
                "verdict": None,
                "row_status": None,
                "synthesis_eligible": False,
                "row_index": None,
            }
        )
    state_counts = dict.fromkeys(COMPONENT_CAMPAIGN_STATES, 0)
    for lane in lanes:
        state_counts[lane["state"]] += 1
    complete_count = state_counts["complete"]
    prerequisite_met = complete_count >= prerequisite_count
    integration_status = summarize_issue_1489_integration_status(
        state_counts=state_counts,
        prerequisite_count=prerequisite_count,
        rows_valid=validation["status"] == "valid",
        invalid_row_count=validation["invalid_row_count"],
    )
    return {
        "gate": "ready_for_synthesis" if prerequisite_met else "blocked",
        "prerequisite_met": prerequisite_met,
        "prerequisite_count": prerequisite_count,
        "complete_count": complete_count,
        "lane_count": len(lanes),
        "state_counts": state_counts,
        "rows_valid": validation["status"] == "valid",
        "invalid_row_count": validation["invalid_row_count"],
        "provenance_validation": validation["provenance_validation"],
        "integration_status": integration_status,
        "lanes": lanes,
    }


def summarize_issue_1489_integration_status(
    *,
    state_counts: dict[str, int],
    prerequisite_count: int,
    rows_valid: bool,
    invalid_row_count: int,
) -> dict[str, Any]:
    """Summarize #1489 synthesis readiness without promoting weak evidence.

    The report names the blocker class and next empirical action while keeping
    the prerequisite gate fail-closed: only complete durable lanes can open the
    synthesis gate.

    Returns:
        Compact integration report for issue/PR state propagation.
    """
    complete_count = state_counts.get("complete", 0)
    ready_count = state_counts.get("ready", 0)
    blocked_count = state_counts.get("blocked", 0)
    missing_count = state_counts.get("missing", 0)
    remaining_complete_count = max(prerequisite_count - complete_count, 0)

    if rows_valid and remaining_complete_count == 0:
        status = "ready_for_synthesis"
        blockers: list[str] = []
    else:
        status = "blocked"
        blockers = []
        if not rows_valid:
            blockers.append(f"{invalid_row_count} invalid row(s)")
        if remaining_complete_count:
            blockers.append(f"{remaining_complete_count} more complete lane(s) required")
        if blocked_count:
            blockers.append(f"{blocked_count} blocked lane(s)")
        if missing_count:
            blockers.append(f"{missing_count} missing expected lane(s)")
        if ready_count:
            blockers.append(f"{ready_count} ready but not synthesis-complete lane(s)")

    if status == "blocked" and ready_count and not blocked_count and not missing_count:
        next_action = ISSUE_1489_INTEGRATION_NEXT_ACTIONS["ready"]
    else:
        next_action = ISSUE_1489_INTEGRATION_NEXT_ACTIONS[status]

    return {
        "issue": "#1489",
        "status": status,
        "claim_boundary": "not benchmark evidence; prerequisite/status integration only",
        "complete_count": complete_count,
        "required_complete_count": prerequisite_count,
        "remaining_complete_count": remaining_complete_count,
        "blockers": blockers,
        "next_empirical_action": next_action,
    }


def build_hybrid_prerequisite_matrix_file(
    path: Path,
    *,
    expected_components: list[str] | None = None,
    repo_root: Path | None = None,
    check_git_history: bool = False,
    prerequisite_count: int = DEFAULT_SYNTHESIS_PREREQUISITE_COUNT,
) -> dict[str, Any]:
    """Load a matrix file and build its #1489 prerequisite/status report.

    Args:
        path: YAML/JSON evidence-matrix file accepted by
            :func:`load_hybrid_evidence_input`.
        expected_components: Optional expected component names (see
            :func:`build_hybrid_prerequisite_matrix`).
        repo_root: Repository root for provenance-path resolution.
        check_git_history: Forwarded to the row validator to verify git SHAs.
        prerequisite_count: Minimum ``complete`` lanes that open the gate.

    Returns:
        The prerequisite-matrix report with the input format and path attached.
    """
    input_format, rows = load_hybrid_evidence_input(path)
    report = build_hybrid_prerequisite_matrix(
        rows,
        expected_components=expected_components,
        repo_root=repo_root,
        check_git_history=check_git_history,
        prerequisite_count=prerequisite_count,
    )
    report["input_format"] = input_format
    report["input_path"] = _repo_relative_or_absolute(
        path.resolve(), root=(repo_root or get_repository_root())
    )
    return report


def build_hybrid_synthesis_report(matrix_report: dict[str, Any]) -> dict[str, Any]:
    """Assemble the #1489 per-mechanism synthesis recommendation from a matrix.

    This is the synthesis-deliverable half of the #1489 contract. It turns the
    campaign-lifecycle matrix produced by :func:`build_hybrid_prerequisite_matrix`
    into an explicit per-mechanism recommendation
    (``continue``/``revise``/``stop``/``gather_more_evidence``) while staying
    fail-closed:

    - A ``continue``/``revise`` verdict is *promoted* (marked authoritative) only
      when the prerequisite gate is open — at least ``prerequisite_count`` durable
      ``complete`` lanes and no invalid rows. No single pre-result lane, launch
      packet, smoke run, or fallback/degraded row can open the gate, so none can
      promote a synthesis verdict.
    - A ``stop`` recommendation is surfaced only for a lane that actually executed
      a synthesis-tier slice (``stress``/``full_matrix``) and concluded ``stop``.
      It is a terminal negative decision, never a promoted synthesis verdict.
    - Every other lane (missing, blocked, executed-but-not-synthesis-grade) maps to
      ``gather_more_evidence``.

    Args:
        matrix_report: The report returned by
            :func:`build_hybrid_prerequisite_matrix` or
            :func:`build_hybrid_prerequisite_matrix_file`.

    Returns:
        Compact synthesis report with the overall eligibility decision and a
        per-mechanism recommendation list. When the gate is blocked the report
        echoes the integration-status blockers and next empirical action so no
        weak lane is silently promoted.
    """
    integration = matrix_report.get("integration_status", {})
    prerequisite_met = bool(matrix_report.get("prerequisite_met"))
    rows_valid = bool(matrix_report.get("rows_valid", True))
    gate_open = prerequisite_met and rows_valid

    mechanisms: list[dict[str, Any]] = []
    promoted_verdict_count = 0
    for lane in matrix_report.get("lanes", []):
        state = lane.get("state")
        verdict = lane.get("verdict")
        if state == "complete" and verdict in SYNTHESIS_VERDICTS:
            recommendation = verdict
        elif verdict == "stop" and lane.get("evidence_tier") in SYNTHESIS_TIERS:
            recommendation = "stop"
        else:
            recommendation = "gather_more_evidence"
        promoted = gate_open and state == "complete" and recommendation in SYNTHESIS_VERDICTS
        if promoted:
            promoted_verdict_count += 1
        mechanisms.append(
            {
                "component": lane.get("component"),
                "source_issue": lane.get("source_issue"),
                "state": state,
                "recommendation": recommendation,
                "recommendation_basis": _SYNTHESIS_BASIS_BY_STATE.get(state, "unknown"),
                "synthesis_verdict_promoted": promoted,
            }
        )

    return {
        "issue": "#1489",
        "status": "ready_for_synthesis" if gate_open else "blocked",
        "eligible": gate_open,
        "claim_boundary": (
            "conservative per-mechanism synthesis recommendation; a verdict is "
            "authoritative only when synthesis_verdict_promoted is true (gate open). "
            "Not benchmark evidence on its own."
        ),
        "prerequisite_count": matrix_report.get("prerequisite_count"),
        "complete_count": matrix_report.get("complete_count"),
        "promoted_verdict_count": promoted_verdict_count,
        "blockers": list(integration.get("blockers", [])),
        "next_empirical_action": integration.get("next_empirical_action"),
        "mechanisms": mechanisms,
    }


def build_hybrid_synthesis_report_file(
    path: Path,
    *,
    expected_components: list[str] | None = None,
    repo_root: Path | None = None,
    check_git_history: bool = False,
    prerequisite_count: int = DEFAULT_SYNTHESIS_PREREQUISITE_COUNT,
) -> dict[str, Any]:
    """Load a matrix file and build its #1489 synthesis recommendation report.

    Args:
        path: YAML/JSON evidence-matrix file accepted by
            :func:`load_hybrid_evidence_input`.
        expected_components: Optional expected component names (see
            :func:`build_hybrid_prerequisite_matrix`).
        repo_root: Repository root for provenance-path resolution.
        check_git_history: Forwarded to the row validator to verify git SHAs.
        prerequisite_count: Minimum ``complete`` lanes that open the gate.

    Returns:
        The synthesis report with the input format and path attached.
    """
    matrix_report = build_hybrid_prerequisite_matrix_file(
        path,
        expected_components=expected_components,
        repo_root=repo_root,
        check_git_history=check_git_history,
        prerequisite_count=prerequisite_count,
    )
    report = build_hybrid_synthesis_report(matrix_report)
    report["input_format"] = matrix_report.get("input_format")
    report["input_path"] = matrix_report.get("input_path")
    return report


def _missing_components(
    expected_components: list[str] | None,
    seen_components: set[str],
) -> list[str]:
    """Return expected component names that have no row, preserving input order.

    Returns:
        The de-duplicated list of expected names absent from ``seen_components``.
    """
    if not expected_components:
        return []
    if isinstance(expected_components, str):
        raise HybridEvidenceMatrixValidationError(
            "expected_components must be a list of strings, not a single string"
        )
    seen_normalized = {
        component.strip() for component in seen_components if isinstance(component, str)
    }
    missing: list[str] = []
    for component in expected_components:
        if not isinstance(component, str) or not component.strip():
            raise HybridEvidenceMatrixValidationError(
                "expected_components entries must be non-empty strings"
            )
        normalized = component.strip()
        if normalized not in seen_normalized and normalized not in missing:
            missing.append(normalized)
    return missing


def _validate_row(
    row: object,
    *,
    index: int,
    repo_root: Path,
    provenance_validation: str,
    git_commit_cache: dict[str, bool],
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not isinstance(row, dict):
        _append_problem(errors, "row", "must be a mapping")
        return {
            "index": index,
            "component": None,
            "status": "invalid",
            "synthesis_candidate": False,
            "synthesis_eligible": False,
            "errors": errors,
            "warnings": warnings,
        }

    _check_expected_fields(
        row,
        required=ROW_REQUIRED_FIELDS,
        optional=ROW_OPTIONAL_FIELDS,
        prefix="row",
        errors=errors,
    )

    component = _require_non_empty_string(row, "component", errors)
    source_issue = _require_non_empty_string(row, "source_issue", errors)
    if source_issue is not None and not _ISSUE_RE.fullmatch(source_issue):
        _append_problem(errors, "source_issue", "must match '#<number>'")

    evaluation_slice = _require_enum(row, "evaluation_slice", EVALUATION_SLICES, errors)
    evidence_tier = _require_enum(row, "evidence_tier", EVIDENCE_TIERS, errors)
    verdict = _require_enum(row, "verdict", VERDICTS, errors)
    synthesis_candidate = bool(evidence_tier in SYNTHESIS_TIERS and verdict in SYNTHESIS_VERDICTS)

    _validate_commit_artifact(
        row.get("commit_artifact"),
        field="commit_artifact",
        repo_root=repo_root,
        synthesis_candidate=synthesis_candidate,
        provenance_validation=provenance_validation,
        git_commit_cache=git_commit_cache,
        errors=errors,
    )
    guard = _validate_guard_authority(row.get("guard_authority"), errors)
    contribution = _validate_learned_component_contribution(
        row.get("learned_component_contribution"),
        evaluation_slice=evaluation_slice,
        evidence_tier=evidence_tier,
        errors=errors,
    )
    intervention = _validate_intervention_rates(row.get("intervention_fallback_rates"), errors)
    _validate_outcomes(row.get("outcomes"), errors)
    _validate_optional_fields(row, repo_root=repo_root, errors=errors)
    _validate_semantics(
        evaluation_slice=evaluation_slice,
        evidence_tier=evidence_tier,
        verdict=verdict,
        guard=guard,
        contribution=contribution,
        intervention=intervention,
        synthesis_candidate=synthesis_candidate,
        errors=errors,
        warnings=warnings,
    )

    return {
        "index": index,
        "component": component,
        "source_issue": source_issue,
        "evaluation_slice": evaluation_slice,
        "evidence_tier": evidence_tier,
        "verdict": verdict,
        "status": "valid" if not errors else "invalid",
        "synthesis_candidate": synthesis_candidate,
        "synthesis_eligible": synthesis_candidate and not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _validate_guard_authority(
    guard_raw: object,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    if not isinstance(guard_raw, dict):
        _append_problem(errors, "guard_authority", "must be a mapping")
        return {}
    _check_expected_fields(
        guard_raw,
        required=GUARD_FIELDS,
        optional=frozenset(),
        prefix="guard_authority",
        errors=errors,
    )
    mechanism = _require_non_empty_string(
        guard_raw, "guard_authority.mechanism", errors, parent=guard_raw
    )
    active = guard_raw.get("active")
    if not isinstance(active, bool):
        _append_problem(errors, "guard_authority.active", "must be a boolean")
    veto_rate = _require_nullable_rate(
        guard_raw,
        "guard_authority.veto_rate",
        errors,
        parent=guard_raw,
    )
    if active is True and veto_rate is None:
        _append_problem(errors, "guard_authority.veto_rate", "must be a number when active is true")
    if active is False and veto_rate is not None:
        _append_problem(errors, "guard_authority.veto_rate", "must be null when active is false")
    return {"mechanism": mechanism, "active": active, "veto_rate": veto_rate}


def _validate_learned_component_contribution(
    contribution_raw: object,
    *,
    evaluation_slice: str | None,
    evidence_tier: str | None,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    if not isinstance(contribution_raw, dict):
        _append_problem(errors, "learned_component_contribution", "must be a mapping")
        return {}
    _check_expected_fields(
        contribution_raw,
        required=CONTRIBUTION_FIELDS,
        optional=frozenset(),
        prefix="learned_component_contribution",
        errors=errors,
    )
    contribution_type = _require_non_empty_string(
        contribution_raw,
        "learned_component_contribution.contribution_type",
        errors,
        parent=contribution_raw,
    )
    bound = _require_non_empty_string(
        contribution_raw,
        "learned_component_contribution.bound",
        errors,
        parent=contribution_raw,
    )
    active_rate = _require_nullable_rate(
        contribution_raw,
        "learned_component_contribution.active_rate",
        errors,
        parent=contribution_raw,
    )
    if (
        active_rate is None
        and evaluation_slice != "not_run"
        and evidence_tier not in {"launch_packet", "failed", "not_available"}
    ):
        _append_problem(
            errors,
            "learned_component_contribution.active_rate",
            "must be a number for executed rows; use 0.0 when the component was active but made no change",
        )
    return {"contribution_type": contribution_type, "bound": bound, "active_rate": active_rate}


def _validate_intervention_rates(
    intervention_raw: object,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    if not isinstance(intervention_raw, dict):
        _append_problem(errors, "intervention_fallback_rates", "must be a mapping")
        return {}
    _check_expected_fields(
        intervention_raw,
        required=INTERVENTION_FIELDS,
        optional=frozenset(),
        prefix="intervention_fallback_rates",
        errors=errors,
    )
    return {
        "guard_veto_rate": _require_nullable_rate(
            intervention_raw,
            "intervention_fallback_rates.guard_veto_rate",
            errors,
            parent=intervention_raw,
        ),
        "fallback_rate": _require_nullable_rate(
            intervention_raw,
            "intervention_fallback_rates.fallback_rate",
            errors,
            parent=intervention_raw,
        ),
        "degraded_rate": _require_nullable_rate(
            intervention_raw,
            "intervention_fallback_rates.degraded_rate",
            errors,
            parent=intervention_raw,
        ),
    }


def _validate_outcomes(outcomes_raw: object, errors: list[dict[str, str]]) -> None:
    if not isinstance(outcomes_raw, dict):
        _append_problem(errors, "outcomes", "must be a mapping")
        return
    _check_expected_fields(
        outcomes_raw,
        required=OUTCOME_FIELDS,
        optional=frozenset(),
        prefix="outcomes",
        errors=errors,
    )
    for key in sorted(OUTCOME_FIELDS):
        _require_nullable_rate(outcomes_raw, f"outcomes.{key}", errors, parent=outcomes_raw)


def _validate_optional_fields(
    row: dict[str, Any],
    *,
    repo_root: Path,
    errors: list[dict[str, str]],
) -> None:
    for field in (
        "comfort_exposure",
        "min_pedestrian_distance",
        "force_exposure_rate",
        "path_efficiency",
        "mean_time_to_goal",
    ):
        if field in row:
            _require_nullable_number(row, field, errors)
    if "baseline_comparator" in row and row["baseline_comparator"] is not None:
        _require_non_empty_string(row, "baseline_comparator", errors)
    if "seed_schedule" in row and row["seed_schedule"] is not None:
        _require_non_empty_string(row, "seed_schedule", errors)
    if "scenario_manifest" in row and row["scenario_manifest"] is not None:
        value = _require_non_empty_string(row, "scenario_manifest", errors)
        if value is not None:
            _validate_reference_token(
                value, "scenario_manifest", repo_root=repo_root, errors=errors
            )


def _validate_semantics(  # noqa: PLR0913
    *,
    evaluation_slice: str | None,
    evidence_tier: str | None,
    verdict: str | None,
    guard: dict[str, Any],
    contribution: dict[str, Any],
    intervention: dict[str, Any],
    synthesis_candidate: bool,
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> None:
    active = guard.get("active")
    veto_rate = guard.get("veto_rate")
    guard_veto_rate = intervention.get("guard_veto_rate")
    fallback_rate = intervention.get("fallback_rate")
    degraded_rate = intervention.get("degraded_rate")
    active_rate = contribution.get("active_rate")

    _validate_execution_state(
        evaluation_slice=evaluation_slice,
        evidence_tier=evidence_tier,
        active=active,
        errors=errors,
    )
    _validate_slice_tier_alignment(
        evaluation_slice=evaluation_slice,
        evidence_tier=evidence_tier,
        errors=errors,
    )
    _validate_fallback_degraded_semantics(
        evidence_tier=evidence_tier,
        fallback_rate=fallback_rate,
        degraded_rate=degraded_rate,
        errors=errors,
    )
    if synthesis_candidate:
        _validate_synthesis_candidate(
            active=active,
            veto_rate=veto_rate,
            guard_veto_rate=guard_veto_rate,
            fallback_rate=fallback_rate,
            degraded_rate=degraded_rate,
            active_rate=active_rate,
            errors=errors,
            warnings=warnings,
        )

    if (
        evidence_tier in {"launch_packet", "fallback", "degraded", "failed", "not_available"}
        and verdict in SYNTHESIS_VERDICTS
    ):
        _append_problem(
            warnings,
            "verdict",
            f"{verdict!r} does not make the row synthesis-eligible because evidence_tier is {evidence_tier!r}",
        )


def _validate_execution_state(
    *,
    evaluation_slice: str | None,
    evidence_tier: str | None,
    active: object,
    errors: list[dict[str, str]],
) -> None:
    if evaluation_slice == "not_run":
        if evidence_tier != "launch_packet":
            _append_problem(
                errors,
                "evidence_tier",
                "must be 'launch_packet' when evaluation_slice is 'not_run'",
            )
        if active is not False:
            _append_problem(
                errors,
                "guard_authority.active",
                "must be false for non-execution launch-packet rows",
            )
        return
    if active is not True:
        _append_problem(errors, "guard_authority.active", "must be true for executed rows")


def _validate_slice_tier_alignment(
    *,
    evaluation_slice: str | None,
    evidence_tier: str | None,
    errors: list[dict[str, str]],
) -> None:
    tier_to_slice = {
        "launch_packet": "not_run",
        "smoke_only": "smoke",
        "nominal_only": "nominal_sanity",
        "stress": "stress_slice",
        "full_matrix": "full_matrix",
    }
    expected_slice = tier_to_slice.get(evidence_tier)
    if expected_slice is None or evaluation_slice is None or evaluation_slice == expected_slice:
        return
    _append_problem(
        errors,
        "evaluation_slice",
        f"must be {expected_slice!r} when evidence_tier is {evidence_tier!r}",
    )


def _validate_fallback_degraded_semantics(
    *,
    evidence_tier: str | None,
    fallback_rate: float | None,
    degraded_rate: float | None,
    errors: list[dict[str, str]],
) -> None:
    if fallback_rate is not None and fallback_rate > 0 and evidence_tier in SUCCESS_LIKE_TIERS:
        _append_problem(
            errors,
            "intervention_fallback_rates.fallback_rate",
            "requires a non-success evidence_tier because fallback is diagnostic-only",
        )
    if degraded_rate is not None and degraded_rate > 0 and evidence_tier in SUCCESS_LIKE_TIERS:
        _append_problem(
            errors,
            "intervention_fallback_rates.degraded_rate",
            "requires a non-success evidence_tier because degraded rows are excluded from synthesis",
        )
    if evidence_tier == "fallback" and not _is_positive_rate(fallback_rate):
        _append_problem(
            errors,
            "intervention_fallback_rates.fallback_rate",
            "must be > 0 when evidence_tier is 'fallback'",
        )
    if evidence_tier == "degraded" and not _is_positive_rate(degraded_rate):
        _append_problem(
            errors,
            "intervention_fallback_rates.degraded_rate",
            "must be > 0 when evidence_tier is 'degraded'",
        )


def _validate_synthesis_candidate(
    *,
    active: object,
    veto_rate: float | None,
    guard_veto_rate: float | None,
    fallback_rate: float | None,
    degraded_rate: float | None,
    active_rate: float | None,
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> None:
    if active is not True:
        _append_problem(
            errors, "guard_authority.active", "must be true for synthesis-eligible rows"
        )
    if veto_rate is None:
        _append_problem(
            errors,
            "guard_authority.veto_rate",
            "must be a number for synthesis-eligible rows",
        )
    if guard_veto_rate is None:
        _append_problem(
            errors,
            "intervention_fallback_rates.guard_veto_rate",
            "must be a number for synthesis-eligible rows",
        )
    if (
        veto_rate is not None
        and guard_veto_rate is not None
        and not math.isclose(veto_rate, guard_veto_rate, rel_tol=0.0, abs_tol=1e-9)
    ):
        _append_problem(
            errors,
            "intervention_fallback_rates.guard_veto_rate",
            "must match guard_authority.veto_rate for synthesis-eligible rows",
        )
    if fallback_rate is not None and fallback_rate > 0:
        _append_problem(
            errors,
            "intervention_fallback_rates.fallback_rate",
            "must be 0.0 or null for synthesis-eligible rows",
        )
    if degraded_rate is not None and degraded_rate > 0:
        _append_problem(
            errors,
            "intervention_fallback_rates.degraded_rate",
            "must be 0.0 or null for synthesis-eligible rows",
        )
    if active_rate is not None and active_rate > 0 and veto_rate == 0:
        _append_problem(
            warnings,
            "guard_authority.veto_rate",
            "is 0.0 while the learned component was active; #1489 should preserve the guard-not-exercised caveat",
        )


def _validate_commit_artifact(
    value: object,
    *,
    field: str,
    repo_root: Path,
    synthesis_candidate: bool,
    provenance_validation: str,
    git_commit_cache: dict[str, bool],
    errors: list[dict[str, str]],
) -> None:
    if not isinstance(value, str) or not value.strip():
        _append_problem(errors, field, "must be a non-empty string")
        return
    tokens = [token for token in _split_reference_tokens(value) if token]
    if not tokens:
        _append_problem(
            errors,
            field,
            "must include a git SHA token plus one or more provenance tokens",
        )
        return
    has_git_sha = False
    has_provenance_token = False
    git_sha_tokens: list[str] = []
    for token in tokens:
        normalized = token.lower()
        if _GIT_SHA_RE.fullmatch(normalized):
            has_git_sha = True
            git_sha_tokens.append(normalized)
            continue
        has_provenance_token = True
        _validate_reference_token(token, field, repo_root=repo_root, errors=errors)
    if not has_git_sha:
        _append_problem(errors, field, "must include a 7-40 character git SHA token")
    if not has_provenance_token:
        _append_problem(errors, field, "must include at least one provenance pointer token")
    if provenance_validation == "git_history":
        for sha in git_sha_tokens:
            if _git_commit_exists(sha, repo_root=repo_root, cache=git_commit_cache):
                continue
            _append_problem(
                errors,
                field,
                f"references unknown git commit SHA in repository history: {sha!r}",
            )


def _git_commit_exists(sha: str, *, repo_root: Path, cache: dict[str, bool]) -> bool:
    cached = cache.get(sha)
    if cached is not None:
        return cached
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    exists = result.returncode == 0
    cache[sha] = exists
    return exists


def _validate_reference_token(
    token: str,
    field: str,
    *,
    repo_root: Path,
    errors: list[dict[str, str]],
) -> None:
    if token.startswith(DURABLE_URI_PREFIXES):
        return
    path = Path(token)
    if path.is_absolute():
        _append_problem(
            errors, field, f"must use repository-root-relative paths, not absolute path {token!r}"
        )
        return
    if ".." in path.parts:
        _append_problem(errors, field, f"must not escape the repository root: {token!r}")
        return
    resolved = (repo_root / path).resolve()
    output_dir = (repo_root / "output").resolve()
    if resolved == output_dir or output_dir in resolved.parents:
        _append_problem(errors, field, f"must not depend on worktree-local output paths: {token!r}")
        return
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        _append_problem(errors, field, f"must resolve inside the repository root: {token!r}")
        return
    if not resolved.exists():
        _append_problem(errors, field, f"references a missing repository path: {token!r}")


def _check_expected_fields(
    mapping: dict[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    prefix: str,
    errors: list[dict[str, str]],
) -> None:
    missing = sorted(required - mapping.keys())
    for field in missing:
        _append_problem(errors, f"{prefix}.{field}" if prefix != "row" else field, "is required")
    allowed = required | optional
    unexpected = sorted(set(mapping) - allowed)
    for field in unexpected:
        _append_problem(
            errors,
            f"{prefix}.{field}" if prefix != "row" else field,
            "is not part of the canonical schema",
        )


def _require_non_empty_string(
    mapping: dict[str, Any],
    field: str,
    errors: list[dict[str, str]],
    *,
    parent: dict[str, Any] | None = None,
) -> str | None:
    source = mapping if parent is None else parent
    key = field if parent is None else field.rsplit(".", maxsplit=1)[-1]
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        _append_problem(errors, field, "must be a non-empty string")
        return None
    return value.strip()


def _require_enum(
    mapping: dict[str, Any],
    field: str,
    allowed: frozenset[str],
    errors: list[dict[str, str]],
) -> str | None:
    value = mapping.get(field)
    if not isinstance(value, str) or value not in allowed:
        _append_problem(errors, field, f"must be one of {sorted(allowed)!r}")
        return None
    return value


def _require_nullable_rate(
    mapping: dict[str, Any],
    field: str,
    errors: list[dict[str, str]],
    *,
    parent: dict[str, Any] | None = None,
) -> float | None:
    value = _require_nullable_number(mapping, field, errors, parent=parent)
    if value is None:
        return None
    if not 0.0 <= value <= 1.0:
        _append_problem(errors, field, "must be between 0.0 and 1.0 inclusive")
    return value


def _require_nullable_number(
    mapping: dict[str, Any],
    field: str,
    errors: list[dict[str, str]],
    *,
    parent: dict[str, Any] | None = None,
) -> float | None:
    source = mapping if parent is None else parent
    key = field if parent is None else field.rsplit(".", maxsplit=1)[-1]
    value = source.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _append_problem(errors, field, "must be a finite number or null")
        return None
    return float(value)


def _split_reference_tokens(value: str) -> list[str]:
    return [token.strip() for token in re.split(r"[\n,]+", value) if token.strip()]


def _append_problem(problems: list[dict[str, str]], field: str, message: str) -> None:
    problems.append({"field": field, "message": message})


def _is_positive_rate(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0


def _repo_relative_or_absolute(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "COMPONENT_CAMPAIGN_STATES",
    "DEFAULT_SYNTHESIS_PREREQUISITE_COUNT",
    "PRE_RUNTIME_TIERS",
    "SYNTHESIS_RECOMMENDATIONS",
    "HybridEvidenceMatrixValidationError",
    "build_hybrid_prerequisite_matrix",
    "build_hybrid_prerequisite_matrix_file",
    "build_hybrid_synthesis_report",
    "build_hybrid_synthesis_report_file",
    "classify_component_campaign_state",
    "load_hybrid_evidence_input",
    "validate_hybrid_evidence_file",
    "validate_hybrid_evidence_rows",
]
