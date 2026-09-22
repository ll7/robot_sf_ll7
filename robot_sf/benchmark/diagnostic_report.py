"""Deterministic evidence-linked diagnostic report adapters (issue #7387).

First typed diagnostic component: thin adapters over existing planner traces,
a deterministic evidence-linked report, and injected-mechanism fixtures.

Reuse contract (no competing framework):

- :mod:`robot_sf.benchmark.mechanism_trace` owns typed mechanism-trace rows and
  their JSON schema; :func:`adapt_orca_residual_row` only reshapes those rows.
- :mod:`robot_sf.benchmark.failure_diagnosis` owns the failure-diagnosis
  analysis/report contract; this module reuses the taxonomy vocabularies
  (``MECHANISM_CONFIDENCES`` / ``MECHANISM_EVIDENCE_MODES``) for finding
  confidence and evidence mode instead of inventing a parallel scale.
- Outcome fields (``collision_event``) are post-hoc augmentation only and can
  never trigger a mechanism finding.

Boundaries: observations and symptoms only. ``hypotheses`` is always empty
because asserting a cause requires a separately controlled test, which is out
of scope. Missing optional planner fields are a supported result and never
invalidate unrelated findings. Text fields are data and are never executed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_CONFIDENCES,
    MECHANISM_EVIDENCE_MODES,
)
from robot_sf.errors import RobotSfError

#: Schema version of the diagnostic report emitted by this module.
SCHEMA_VERSION = "diagnostic_report.v1"
#: Provenance source marker for the deterministic adapter.
DIAGNOSTIC_SOURCE = "diagnostic_report.deterministic.v1"

#: Deterministic rule versions bound to every finding.
RULE_OBSERVATION_AGE = "obs_age_mismatch.v1"
RULE_COST_SCALE = "cost_scale_inconsistency.v1"
RULE_COMMAND_EXECUTION = "command_execution_mismatch.v1"
RULE_VERSIONS = {
    "observation_age": RULE_OBSERVATION_AGE,
    "cost_scale": RULE_COST_SCALE,
    "command_execution": RULE_COMMAND_EXECUTION,
}

#: Gap norm above which commanded-vs-executed controls count as mismatched.
COMMAND_GAP_TOLERANCE = 1e-9

#: Caveats attached to every report; causal and repair claims stay out of scope.
REPORT_CAVEATS = (
    "observations cite measured trace values only; they are not causal inference.",
    "hypotheses require a separately controlled test and are empty in this leaf.",
    "collision_event is post-hoc outcome augmentation and never triggers a mechanism.",
    "text fields are data and are never executed; no repair action is proposed.",
)

#: Two-planner field-availability inventory. ``support`` is one of
#: ``"supported"`` (planner-visible), ``"simulator_only"``,
#: ``"post_hoc"``, or ``"unsupported"`` (missing is a supported result).
TWO_PLANNER_FIELD_INVENTORY: dict[str, dict[str, Any]] = {
    "orca_residual": {
        "planner_owner": "robot_sf/planner/guarded_ppo.py (action_adaptation)",
        "trace_owner": "robot_sf/benchmark/mechanism_trace.py::emit_orca_residual_row",
        "emitter_owner": "scripts/tools/emit_orca_residual_mechanism_trace.py",
        "fields": {
            "observation_version": {
                "support": "supported",
                "source": "planner step diagnostics (track/observation version)",
            },
            "candidate_table": {
                "support": "supported",
                "source": "selected_score + residual adaptation metadata",
            },
            "cost_terms_and_scale": {
                "support": "supported",
                "source": "planner-visible cost terms with units/scales",
            },
            "commanded_control": {
                "support": "supported",
                "source": "selected_command (planner-visible)",
            },
            "executed_control": {
                "support": "simulator_only",
                "source": "simulator-applied control, not planner-visible",
            },
            "saturation_flag": {
                "support": "supported",
                "source": "residual_clipped / bounded_residual_action metadata",
            },
            "collision_event": {
                "support": "post_hoc",
                "source": "outcome augmentation, never a diagnostic input",
            },
        },
    },
    "nominal_social_force": {
        "planner_owner": "robot_sf/sim/fast_pysf_wrapper.py::diagnostics",
        "trace_owner": "force-kernel fallback diagnostics (no candidate table)",
        "emitter_owner": "none: reactive planner computes no candidate costs",
        "fields": {
            "observation_version": {
                "support": "unsupported",
                "source": "reactive force kernel keeps no versioned observation",
            },
            "candidate_table": {
                "support": "unsupported",
                "source": "reactive planner never computes candidates",
            },
            "cost_terms_and_scale": {
                "support": "unsupported",
                "source": "no candidate-cost table exists to scale",
            },
            "commanded_control": {
                "support": "supported",
                "source": "desired-force command (planner-visible)",
            },
            "executed_control": {
                "support": "simulator_only",
                "source": "simulator-applied control, not planner-visible",
            },
            "saturation_flag": {
                "support": "unsupported",
                "source": "no saturation metadata recorded by force kernel",
            },
            "collision_event": {
                "support": "post_hoc",
                "source": "outcome augmentation, never a diagnostic input",
            },
        },
    },
}


class DiagnosticReportError(RobotSfError, ValueError):
    """Raised when diagnostic inputs violate the read-only input contract."""


def canonical_digest(payload: Any) -> str:
    """Return the deterministic sha256 hex digest of a JSON-canonical payload.

    Returns:
        Hex digest string.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_finite_number(value: Any) -> bool:
    """Return True for finite int/float values (bools excluded).

    Returns:
        True for finite numbers.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_finite(mapping: Mapping[str, Any], key: str, row_id: str) -> float | None:
    """Return an optional finite timestamp, failing deterministically if malformed.

    Returns:
        Finite float or None.
    """
    value = mapping.get(key)
    if value is None:
        return None
    if not _is_finite_number(value):
        raise DiagnosticReportError(f"row {row_id!r}: {key} must be a finite number")
    return float(value)


def _check_id(row: Mapping[str, Any]) -> str:
    """Return the validated row id.

    Returns:
        The non-empty ``row_id`` string.
    """
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise DiagnosticReportError("each diagnostic row needs a non-empty row_id string")
    return row_id


def _check_step(row: Mapping[str, Any], row_id: str) -> None:
    """Validate ``sim_step`` and ``sim_time_s`` for one row."""
    sim_step = row.get("sim_step")
    if isinstance(sim_step, bool) or not isinstance(sim_step, int) or sim_step < 0:
        raise DiagnosticReportError(f"row {row_id!r}: sim_step must be a non-negative int")
    if not _is_finite_number(row.get("sim_time_s")):
        raise DiagnosticReportError(f"row {row_id!r}: sim_time_s must be a finite number")


def _check_provenance(row: Mapping[str, Any], row_id: str) -> None:
    """Validate the provenance block of one row."""
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise DiagnosticReportError(f"row {row_id!r}: corrupt provenance (not a mapping)")
    digest = provenance.get("artifact_digest")
    source_row = provenance.get("source_row")
    if not isinstance(digest, str) or not digest:
        raise DiagnosticReportError(f"row {row_id!r}: corrupt provenance (artifact_digest)")
    if not isinstance(source_row, str) or not source_row:
        raise DiagnosticReportError(f"row {row_id!r}: corrupt provenance (source_row)")


def _check_controls(row: Mapping[str, Any], row_id: str) -> None:
    """Validate optional commanded/executed control pairs of one row."""
    for key in ("commanded_control", "executed_control"):
        control = row.get(key)
        if control is None:
            continue
        values = list(control) if isinstance(control, (list, tuple)) else None
        if values is None or len(values) != 2 or not all(_is_finite_number(v) for v in values):
            raise DiagnosticReportError(f"row {row_id!r}: {key} must be two finite numbers or null")


def _validate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy and validate one diagnostic input row (inputs are never mutated).

    Returns:
        A normalized copy of the validated row.
    """
    if not isinstance(row, Mapping):
        raise DiagnosticReportError("each diagnostic row must be a mapping")
    row_id = _check_id(row)
    _check_step(row, row_id)
    _check_provenance(row, row_id)
    _check_controls(row, row_id)

    cost_units = row.get("cost_units")
    if cost_units is not None and (not isinstance(cost_units, str) or not cost_units):
        raise DiagnosticReportError(f"row {row_id!r}: malformed cost_units")

    cost_scale = row.get("cost_scale")
    if cost_scale is not None and not isinstance(cost_scale, Mapping):
        raise DiagnosticReportError(f"row {row_id!r}: malformed cost_scale")

    normalized = dict(row)
    provenance = row.get("provenance")
    normalized["provenance"] = dict(provenance) if isinstance(provenance, Mapping) else provenance
    return normalized


def _finding_base(row: dict[str, Any], rule_version: str) -> dict[str, Any]:
    """Build the provenance/time fields shared by every finding.

    Returns:
        Shared finding fields.
    """
    return {
        "row_id": row["row_id"],
        "rule_version": rule_version,
        "source_digest": row["provenance"]["artifact_digest"],
        "source_row": row["provenance"]["source_row"],
        "sim_step": row["sim_step"],
        "sim_time_s": float(row["sim_time_s"]),
        "planner_id": row.get("planner_id"),
    }


def _check_observation_age(row: dict[str, Any]) -> tuple[list, list, list, list]:
    """Apply the stale-observation rule (measured version/age mismatch only).

    Returns:
        Finding lists.
    """
    obs: list[dict[str, Any]] = []
    symptoms: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    observed = row.get("observation_version")
    consumed = row.get("consumed_observation_version")
    if observed is None or consumed is None:
        missing.append(
            {
                "row_id": row["row_id"],
                "rule_version": RULE_OBSERVATION_AGE,
                "missing_field": "observation_version/consumed_observation_version",
                "reason": "version_unavailable",
            }
        )
        abstentions.append(
            {
                "row_id": row["row_id"],
                "rule_version": RULE_OBSERVATION_AGE,
                "reason": "cannot_assess_staleness_without_versions",
            }
        )
        return obs, symptoms, missing, abstentions
    if isinstance(observed, bool) or not isinstance(observed, int):
        raise DiagnosticReportError(f"row {row['row_id']!r}: observation_version must be int")
    if isinstance(consumed, bool) or not isinstance(consumed, int):
        raise DiagnosticReportError(
            f"row {row['row_id']!r}: consumed_observation_version must be int"
        )
    if consumed < observed:
        base = _finding_base(row, RULE_OBSERVATION_AGE)
        obs.append(
            {
                **base,
                "observation_id": "stale_observation_version",
                "detail": f"consumed v{consumed} older than source v{observed}",
                "confidence": "observed_mechanism",
                "evidence_mode": "direct_probe",
            }
        )
        symptoms.append(
            {
                **base,
                "symptom_id": "stale_observation_consumed",
                "detail": f"planner consumed v{consumed} while source was v{observed}",
                "confidence": "supported_hypothesis",
                "evidence_mode": "direct_probe",
            }
        )
    available = _require_finite(row, "planner_available_time_s", str(row["row_id"]))
    sourced = _require_finite(row, "observation_source_time_s", str(row["row_id"]))
    if available is not None and sourced is not None and available < sourced:
        base = _finding_base(row, RULE_OBSERVATION_AGE)
        obs.append(
            {
                **base,
                "observation_id": "incompatible_time_grid",
                "detail": "planner_available_time precedes observation_source_time",
                "confidence": "observed_mechanism",
                "evidence_mode": "direct_probe",
            }
        )
    return obs, symptoms, missing, abstentions


def _rescaled_total(terms: Mapping[str, Any], scale: Mapping[str, Any], row_id: str) -> float:
    """Recompute one candidate total under the declared cost scales.

    Returns:
        Rescaled total.
    """
    total = 0.0
    for term, value in terms.items():
        if not _is_finite_number(value):
            raise DiagnosticReportError(f"row {row_id!r}: non-finite cost term {term!r}")
        factor = scale.get(term, 1.0)
        if not _is_finite_number(factor):
            raise DiagnosticReportError(f"row {row_id!r}: non-finite cost scale {term!r}")
        total += float(value) * float(factor)
    return total


def _check_cost_scale(row: dict[str, Any]) -> tuple[list, list, list, list]:
    """Apply the cost-scale rule (needs a demonstrable candidate-order change).

    Returns:
        The ``(observations, symptoms, missing, abstentions)`` finding lists.
    """
    candidates = row.get("candidates")
    scale = row.get("cost_scale")
    gated = _missing_cost_inputs(row, candidates, scale)
    if gated is not None:
        return gated
    if not isinstance(candidates, list) or not isinstance(scale, Mapping):
        raise DiagnosticReportError(f"row {row['row_id']!r}: malformed cost_scale")
    recorded, rescaled = _score_candidates(row, candidates, scale)
    recorded_best = min(recorded, key=lambda key: (recorded[key], key))
    rescaled_best = min(rescaled, key=lambda key: (rescaled[key], key))
    if recorded_best == rescaled_best:
        return [], [], [], []
    base = _finding_base(row, RULE_COST_SCALE)
    return (
        [
            {
                **base,
                "observation_id": "cost_scale_order_change",
                "detail": f"recorded best {recorded_best} differs from rescaled best {rescaled_best}",
                "confidence": "observed_mechanism",
                "evidence_mode": "direct_probe",
            }
        ],
        [
            {
                **base,
                "symptom_id": "cost_scale_inconsistency",
                "detail": f"declared scales reverse candidate order ({recorded_best}>{rescaled_best})",
                "confidence": "supported_hypothesis",
                "evidence_mode": "direct_probe",
            }
        ],
        [],
        [],
    )


def _missing_cost_inputs(
    row: dict[str, Any], candidates: Any, scale: Any
) -> tuple[list, list, list, list] | None:
    """Return missing/abstention records when cost inputs are unavailable.

    Returns:
        The four finding lists when inputs are missing, else None.
    """
    row_id = str(row["row_id"])
    if candidates is None:
        reason, field = "no_candidate_table_no_cost_assessment", "candidates"
    elif not isinstance(candidates, list) or len(candidates) < 2:
        reason, field = "ordering_needs_two_candidates", "candidates>=2"
    elif scale is None:
        reason, field = "cannot_assess_scale_without_declared_scales", "cost_scale"
    elif not isinstance(scale, Mapping):
        raise DiagnosticReportError(f"row {row_id!r}: malformed cost_scale")
    else:
        return None
    return (
        [],
        [],
        [
            {
                "row_id": row_id,
                "rule_version": RULE_COST_SCALE,
                "missing_field": field,
                "reason": "candidate_table_unsupported_by_planner"
                if "table" in reason
                else ("insufficient_candidates" if ">=" in field else "cost_scale_unavailable"),
            }
        ],
        [{"row_id": row_id, "rule_version": RULE_COST_SCALE, "reason": reason}],
    )


def _score_candidates(
    row: dict[str, Any], candidates: list, scale: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, float]]:
    """Score recorded vs rescaled candidate totals for one row.

    Returns:
        The ``(recorded, rescaled)`` best-cost maps keyed by candidate id.
    """
    recorded: dict[str, float] = {}
    rescaled: dict[str, float] = {}
    for entry in candidates:
        if not isinstance(entry, Mapping):
            raise DiagnosticReportError(f"row {row['row_id']!r}: candidate must be a mapping")
        candidate_id = entry.get("candidate_id")
        cost = entry.get("cost")
        terms = entry.get("cost_terms")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise DiagnosticReportError(f"row {row['row_id']!r}: candidate needs a string id")
        if not _is_finite_number(cost):
            raise DiagnosticReportError(f"row {row['row_id']!r}: non-finite candidate cost")
        if not isinstance(terms, Mapping) or not terms:
            raise DiagnosticReportError(f"row {row['row_id']!r}: candidate needs cost_terms")
        recorded[candidate_id] = float(cost)
        rescaled[candidate_id] = _rescaled_total(terms, scale, str(row["row_id"]))
    return recorded, rescaled


def _check_command_execution(row: dict[str, Any]) -> tuple[list, list, list, list]:
    """Apply the command-vs-executed rule (needs a recorded saturation flag).

    Returns:
        The ``(observations, symptoms, missing, abstentions)`` finding lists.
    """
    obs: list[dict[str, Any]] = []
    symptoms: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    commanded = row.get("commanded_control")
    executed = row.get("executed_control")
    if commanded is None or executed is None:
        missing.append(
            {
                "row_id": row["row_id"],
                "rule_version": RULE_COMMAND_EXECUTION,
                "missing_field": "commanded_control/executed_control",
                "reason": "control_pair_unavailable",
            }
        )
        abstentions.append(
            {
                "row_id": row["row_id"],
                "rule_version": RULE_COMMAND_EXECUTION,
                "reason": "cannot_compare_without_command_pair",
            }
        )
        return obs, symptoms, missing, abstentions
    gap = math.dist(
        (float(commanded[0]), float(commanded[1])), (float(executed[0]), float(executed[1]))
    )
    if gap <= COMMAND_GAP_TOLERANCE:
        return obs, symptoms, missing, abstentions
    saturated = row.get("saturation_flag")
    if saturated is True:
        base = _finding_base(row, RULE_COMMAND_EXECUTION)
        obs.append(
            {
                **base,
                "observation_id": "command_execution_gap",
                "detail": f"command/executed gap {gap:.6f} with saturation recorded",
                "confidence": "observed_mechanism",
                "evidence_mode": "direct_probe",
            }
        )
        symptoms.append(
            {
                **base,
                "symptom_id": "saturated_command_execution_gap",
                "detail": f"gap {gap:.6f} localized with recorded saturation flag",
                "confidence": "supported_hypothesis",
                "evidence_mode": "direct_probe",
            }
        )
    else:
        missing.append(
            {
                "row_id": row["row_id"],
                "rule_version": RULE_COMMAND_EXECUTION,
                "missing_field": "saturation_flag",
                "reason": "saturation_flag_unavailable",
            }
        )
        abstentions.append(
            {
                "row_id": row["row_id"],
                "rule_version": RULE_COMMAND_EXECUTION,
                "reason": "gap_without_saturation_flag_is_unsupported",
            }
        )
    return obs, symptoms, missing, abstentions


def _sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order findings by (sim_step, row_id, rule_version) for reorder invariance.

    Returns:
        Sorted findings.
    """
    return sorted(
        findings,
        key=lambda f: (
            f["sim_step"],
            f["row_id"],
            f["rule_version"],
            json.dumps(f, sort_keys=True),
        ),
    )


def diagnose_diagnostic_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a deterministic evidence-linked diagnostic report from input rows.

    Read-only: inputs are copied, never mutated, and source text is never
    executed. ``collision_event`` and ``unsupported_narrative`` can never
    create an observation, symptom, or hypothesis; a narrative only yields an
    abstention recording its rejection.

    Args:
        rows: Diagnostic input rows (see module docstring and adapters).

    Returns:
        Versioned report dict with observations, symptoms, hypotheses (always
        empty), missing_evidence, abstentions, provenance, and digests.

    Raises:
        DiagnosticReportError: On duplicate ids, corrupt provenance, malformed
            timestamps/units, or non-finite values, with deterministic reasons.
    """
    normalized = [_validate_row(row) for row in rows]
    seen: set[str] = set()
    for row in normalized:
        if row["row_id"] in seen:
            raise DiagnosticReportError(f"duplicate row_id {row['row_id']!r}")
        seen.add(row["row_id"])
    ordered = sorted(normalized, key=lambda r: str(r["row_id"]))
    if "observed_mechanism" not in MECHANISM_CONFIDENCES:
        raise DiagnosticReportError("taxonomy confidence vocabulary unavailable")
    if "direct_probe" not in MECHANISM_EVIDENCE_MODES:
        raise DiagnosticReportError("taxonomy evidence-mode vocabulary unavailable")

    observations: list[dict[str, Any]] = []
    symptoms: list[dict[str, Any]] = []
    missing_evidence: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    for row in ordered:
        for check in (_check_observation_age, _check_cost_scale, _check_command_execution):
            result = check(row)
            observations.extend(result[0])
            symptoms.extend(result[1])
            missing_evidence.extend(result[2])
            abstentions.extend(result[3])
        if isinstance(row.get("unsupported_narrative"), str) and row.get("unsupported_narrative"):
            abstentions.append(
                {
                    "row_id": row["row_id"],
                    "rule_version": "report",
                    "reason": "unsupported_narrative_not_admitted",
                }
            )
    missing_evidence = sorted(
        missing_evidence, key=lambda m: (m["row_id"], m["rule_version"], m["missing_field"])
    )
    abstentions = sorted(abstentions, key=lambda a: (a["row_id"], a["rule_version"], a["reason"]))
    input_digest = canonical_digest(ordered)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_source": DIAGNOSTIC_SOURCE,
        "rule_versions": dict(RULE_VERSIONS),
        "input_digest": input_digest,
        "observations": _sort_findings(observations),
        "symptoms": _sort_findings(symptoms),
        "hypotheses": [],
        "missing_evidence": missing_evidence,
        "abstentions": abstentions,
        "provenance": {
            "input_digest": input_digest,
            "row_count": len(ordered),
            "row_ids": sorted(seen),
        },
        "caveats": list(REPORT_CAVEATS),
    }
    unsigned = dict(report)
    report["report_digest"] = canonical_digest(unsigned)
    return report


def adapt_orca_residual_row(
    mechanism_row: Mapping[str, Any],
    *,
    source_digest: str,
    source_row: str,
    planner_id: str = "orca_residual",
    timing: Mapping[str, Any] | None = None,
    observations: Mapping[str, Any] | None = None,
    planner_extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapt one ``mechanism_trace.v1`` ORCA-residual row to a diagnostic row.

    Only reshapes fields owned by :mod:`robot_sf.benchmark.mechanism_trace`;
    timing/candidate fields stay missing unless the caller supplies them from
    planner-visible sources via the ``timing``, ``observations``, and
    ``planner_extras`` mappings. Missing fields are a supported result.

    Returns:
        Diagnostic input row.
    """
    if not isinstance(mechanism_row, Mapping):
        raise DiagnosticReportError("mechanism_row must be a mapping")
    step = mechanism_row.get("activation_step", 0)
    command = mechanism_row.get("selected_command")
    timing_map = dict(timing) if isinstance(timing, Mapping) else {}
    observations_map = dict(observations) if isinstance(observations, Mapping) else {}
    planner_extras_map = dict(planner_extras) if isinstance(planner_extras, Mapping) else {}
    sim_time_s = timing_map.get("sim_time_s")
    observation_source_time_s = timing_map.get("observation_source_time_s")
    planner_available_time_s = timing_map.get("planner_available_time_s")
    observation_version = observations_map.get("observation_version")
    consumed_observation_version = observations_map.get("consumed_observation_version")
    candidates = planner_extras_map.get("candidates")
    cost_scale = planner_extras_map.get("cost_scale")
    executed_control = planner_extras_map.get("executed_control")
    saturation_flag = planner_extras_map.get("saturation_flag")
    return {
        "row_id": f"{planner_id}:step:{int(step)}",
        "planner_id": planner_id,
        "sim_step": int(step),
        "sim_time_s": sim_time_s if sim_time_s is not None else float(step),
        "observation_source_time_s": observation_source_time_s,
        "planner_available_time_s": planner_available_time_s,
        "logging_time_s": None,
        "observation_version": observation_version,
        "consumed_observation_version": consumed_observation_version,
        "candidates": candidates,
        "cost_scale": cost_scale,
        "cost_units": "unitless" if cost_scale is not None else None,
        "commanded_control": list(command) if isinstance(command, (list, tuple)) else None,
        "executed_control": executed_control,
        "saturation_flag": saturation_flag,
        "collision_event": False,
        "unsupported_narrative": None,
        "provenance": {"artifact_digest": source_digest, "source_row": source_row},
    }


def adapt_nominal_social_force_diagnostics(
    sf_diagnostics: Mapping[str, Any],
    *,
    source_digest: str,
    source_row: str,
    sim_step: int = 0,
    sim_time_s: float = 0.0,
    commanded_control: list[float] | None = None,
    executed_control: list[float] | None = None,
) -> dict[str, Any]:
    """Adapt nominal Social Force diagnostics to a diagnostic input row.

    The reactive force kernel (``FastPysfWrapper.diagnostics``) records no
    candidate table, cost scales, observation versions, or saturation flags,
    so those fields stay missing by design; a candidate-cost table is never
    invented for this planner.

    Returns:
        Diagnostic input row.
    """
    if not isinstance(sf_diagnostics, Mapping):
        raise DiagnosticReportError("sf_diagnostics must be a mapping")
    return {
        "row_id": f"nominal_social_force:step:{int(sim_step)}",
        "planner_id": "nominal_social_force",
        "sim_step": int(sim_step),
        "sim_time_s": float(sim_time_s),
        "observation_source_time_s": None,
        "planner_available_time_s": None,
        "logging_time_s": None,
        "observation_version": None,
        "consumed_observation_version": None,
        "candidates": None,
        "cost_scale": None,
        "cost_units": None,
        "commanded_control": commanded_control,
        "executed_control": executed_control,
        "saturation_flag": None,
        "collision_event": False,
        "unsupported_narrative": None,
        "provenance": {"artifact_digest": source_digest, "source_row": source_row},
    }
