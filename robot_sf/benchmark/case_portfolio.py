"""Chapter 7 worked-example portfolio selection (issue #6789).

This module builds a versioned ``ch7_case_portfolio.v2`` manifest for the
Chapter 7 worked-example portfolio. It is a composition layer over the
existing case-capsule and Pareto/coverage-selection contracts, not a second
generic selector and not a benchmark metric. Evidence quality is treated as a
Boolean eligibility gate before Pareto membership or scientific-interest
dimensions are considered.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from robot_sf.benchmark.scenario_generation.portfolio_selector import (
    compute_pareto_front as compute_5601_pareto_front,
)
from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "ch7_case_portfolio.v2"
SELECTOR_VERSION = "ch7_case_portfolio.v2.0"

EligibilityStatus = Literal["pass", "fail", "unavailable", "not_applicable"]

GRAINS: frozenset[str] = frozenset(
    {"episode", "matched_planner_pair", "matched_seed_pair", "cell", "cross_cell"}
)
CONCEPTUAL_GRAINS: frozenset[str] = frozenset({"campaign", "cell", "matched_contrast", "trace"})
ROLES: frozenset[str] = frozenset(
    {
        "prototype",
        "criticism",
        "boundary",
        "planner_upset",
        "seed_sensitivity",
        "metric_disagreement",
        "process_contrast",
        "feasibility_criticism",
        "negative_control",
        "causal_abstention",
    }
)
PUBLIC_ROLE_ALIASES: dict[str, str] = {
    "prototype_or_medoid": "prototype",
    "criticism_or_outlier": "criticism",
    "boundary_case": "boundary",
    "same_outcome_different_process": "process_contrast",
    "negative_or_robust_control": "negative_control",
}

REQUIRED_ELIGIBILITY_BY_GRAIN: dict[str, tuple[str, ...]] = {
    "cross_cell": (
        "release_campaign_identity",
        "source_hashes",
        "exact_digest_human_review_admission",
        "durable_source_status",
        "typed_outcome_collision_semantics",
        "execution_status",
        "scenario_config_seed_provenance",
        "telemetry_sufficiency",
    ),
    "cell": (
        "release_campaign_identity",
        "source_hashes",
        "exact_digest_human_review_admission",
        "durable_source_status",
        "typed_outcome_collision_semantics",
        "execution_status",
        "scenario_config_seed_provenance",
        "route_feasibility",
        "release_vs_rerun_outcome_agreement",
        "telemetry_sufficiency",
    ),
    "matched_planner_pair": (
        "release_campaign_identity",
        "source_hashes",
        "exact_digest_human_review_admission",
        "durable_source_status",
        "typed_outcome_collision_semantics",
        "execution_status",
        "scenario_config_seed_provenance",
        "matched_initial_state_or_shared_prefix",
        "release_vs_rerun_outcome_agreement",
        "exact_repeat_or_context_sensitivity",
        "telemetry_sufficiency",
    ),
    "matched_seed_pair": (
        "release_campaign_identity",
        "source_hashes",
        "exact_digest_human_review_admission",
        "durable_source_status",
        "typed_outcome_collision_semantics",
        "execution_status",
        "scenario_config_seed_provenance",
        "matched_initial_state_or_shared_prefix",
        "release_vs_rerun_outcome_agreement",
        "exact_repeat_or_context_sensitivity",
        "telemetry_sufficiency",
    ),
    "episode": (
        "release_campaign_identity",
        "source_hashes",
        "exact_digest_human_review_admission",
        "durable_source_status",
        "typed_outcome_collision_semantics",
        "execution_status",
        "scenario_config_seed_provenance",
        "route_feasibility",
        "trace_resolution",
        "trace_schema",
        "release_vs_rerun_outcome_agreement",
        "visualization_only_status",
        "telemetry_sufficiency",
    ),
}
TRACE_PRESENTATION_CHECKS: tuple[str, ...] = (
    "trace_resolution",
    "trace_schema",
    "visualization_only_status",
)
ALLOWED_NOT_APPLICABLE_CHECKS_BY_GRAIN_ROLE: dict[tuple[str, str], frozenset[str]] = {
    ("cell", "feasibility_criticism"): frozenset({"release_vs_rerun_outcome_agreement"}),
}
ADMISSIBLE_EXECUTION_MODES: frozenset[str] = frozenset({"native", "adapter_disclosed"})
HEX64_ZERO_OR_LOWER = frozenset("0123456789abcdef")
EVENT_TYPES: frozenset[str] = frozenset(
    {"collision", "min_clearance", "first_gate_breach", "stall_onset", "terminal"}
)
PRESENTATION_VIEWS: tuple[str, ...] = (
    "world_xy",
    "route_sn",
    "time_space",
    "event_timeline",
    "cell_context",
)
TRACE_PRESENTATION_VIEWS: frozenset[str] = frozenset({"route_sn", "time_space", "event_timeline"})
PARETO_STATUSES: frozenset[str] = frozenset({"nondominated", "dominated", "not_applicable"})
INITIAL_OUTCOME_STATUSES: frozenset[str] = frozenset({"pass", "fail", "unavailable"})
TELEMETRY_GRADES: frozenset[str] = frozenset(
    {"geometry", "kinematics", "controller", "counterfactual"}
)

SCIENTIFIC_DIMENSIONS: tuple[str, ...] = (
    "evidence_grade",
    "provenance_completeness",
    "topology_mechanism",
    "terminal_outcome",
    "criticality_persistence",
    "entropy_bimodality",
    "paired_divergence",
    "metric_disagreement",
    "representativeness_or_outlier",
)
POST_SELECTION_DIMENSIONS: tuple[str, ...] = ("telemetry_visualizability", "page_cost")
PARETO_DIRECTIONS: dict[str, Literal["maximize", "minimize"]] = dict.fromkeys(
    SCIENTIFIC_DIMENSIONS, "maximize"
)
ROLE_DIMENSION_EXEMPTIONS: dict[str, frozenset[str]] = {
    "planner_upset": frozenset({"entropy_bimodality"}),
    "feasibility_criticism": frozenset(
        {"entropy_bimodality", "paired_divergence", "metric_disagreement"}
    ),
    "metric_disagreement": frozenset({"entropy_bimodality"}),
}
ROLE_CHECK_EXEMPTIONS: dict[tuple[str, str], frozenset[str]] = {
    ("matched_seed_pair", "seed_sensitivity"): frozenset(
        {"matched_initial_state_or_shared_prefix"}
    ),
}
CLAIM_GRADES: frozenset[str] = frozenset(
    {"descriptive", "proximate_mechanism", "model_relative_causal", "abstention"}
)
DIVERSITY_FIELDS: tuple[str, ...] = ("topology", "failure_class", "process_class")
FINITE_MISSING_PAGE_COST = 1_000_000_000.0


class CasePortfolioError(RobotSfError, ValueError):
    """Raised when a Chapter 7 portfolio input or manifest is malformed."""


ELIGIBILITY_STATUSES: frozenset[str] = frozenset({"pass", "fail", "unavailable", "not_applicable"})


@dataclass(frozen=True)
class PortfolioConstraints:
    """Hard coverage constraints for Chapter 7 portfolio selection."""

    required_roles: tuple[str, ...]
    required_grains: tuple[str, ...] = ("cross_cell", "cell", "matched_planner_pair", "episode")
    required_conceptual_grains: tuple[str, ...] = (
        "campaign",
        "cell",
        "matched_contrast",
        "trace",
    )
    required_topologies: tuple[str, ...] = ()
    required_failure_classes: tuple[str, ...] = ()
    required_process_classes: tuple[str, ...] = ()
    target_size: int = 4
    max_size: int = 4
    require_unique_primary_roles: bool = True
    frozen_role_targets: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioValidation:
    """Structural validation result for ``ch7_case_portfolio.v2``."""

    structural_violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the manifest is structurally valid."""
        return not self.structural_violations

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable validation summary."""
        return {"ok": self.ok, "structural_violations": list(self.structural_violations)}


def canonical_sha256(obj: Any) -> str:
    """Return a deterministic SHA-256 for a JSON-serialisable object."""
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json_or_gzip(path: Any) -> Any:
    """Read a plain JSON or ``.gz`` JSON file.

    Returns:
        Parsed JSON payload.
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    """Return the raw SHA-256 digest for a file.

    Returns:
        Hex digest of the raw bytes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_deterministic_json(payload: Mapping[str, Any], path: Any) -> None:
    """Write deterministic, byte-stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _required_checks(grain: str) -> tuple[str, ...]:
    if grain not in REQUIRED_ELIGIBILITY_BY_GRAIN:
        raise CasePortfolioError(f"unknown grain {grain!r}")
    return REQUIRED_ELIGIBILITY_BY_GRAIN[grain]


def _required_checks_for_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    checks = list(_required_checks(str(record.get("grain", ""))))
    if _declares_trace(record):
        checks.extend(check for check in TRACE_PRESENTATION_CHECKS if check not in checks)
    return tuple(checks)


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in HEX64_ZERO_OR_LOWER for char in value)
    )


def _is_nonzero_hex64(value: Any) -> bool:
    return _is_hex64(value) and value != "0" * 64


def _public_role(value: Any) -> str:
    role = str(value)
    return PUBLIC_ROLE_ALIASES.get(role, role)


def _declares_trace(record: Mapping[str, Any]) -> bool:
    presentation = record.get("presentation")
    required_views = (
        presentation.get("required_views", []) if isinstance(presentation, Mapping) else []
    )
    return "trace" in _conceptual_coverage(record) or any(
        view in TRACE_PRESENTATION_VIEWS for view in required_views
    )


def _declares_trace_from_entry(entry: Mapping[str, Any]) -> bool:
    presentation = entry.get("presentation")
    required_views = (
        presentation.get("required_views", []) if isinstance(presentation, Mapping) else []
    )
    return "trace" in entry.get("conceptual_coverage", []) or any(
        view in TRACE_PRESENTATION_VIEWS for view in required_views
    )


def _digest_binding_failure(
    source: Mapping[str, Any], declared_field: str, observed_field: str
) -> str:
    declared = source.get(declared_field)
    observed = source.get(observed_field)
    if not _is_nonzero_hex64(declared):
        return declared_field
    if not _is_nonzero_hex64(observed):
        return observed_field
    if declared != observed:
        return f"{declared_field}!={observed_field}"
    return ""


def _source_hash_failure_reason(record: Mapping[str, Any]) -> str:
    source = _source_block(record)
    missing: list[str] = []
    release_failure = _digest_binding_failure(
        source, "release_rows_sha256", "observed_release_rows_sha256"
    )
    if release_failure:
        missing.append(release_failure)
    if _declares_trace(record) or source.get("visualization_only_reexecution") is True:
        trace_failure = _digest_binding_failure(
            source, "trace_package_sha256", "observed_trace_package_sha256"
        )
        if trace_failure:
            missing.append(trace_failure)
    return ",".join(missing)


def _finite_time_or_unavailable(value: Any) -> bool:
    if value == "unavailable":
        return True
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _source_hash_failure_from_source(entry: Mapping[str, Any]) -> str:
    source = _source_block(entry, entry.get("source_boundary", {}), entry.get("source_refs", []))
    missing: list[str] = []
    release_failure = _digest_binding_failure(
        source, "release_rows_sha256", "observed_release_rows_sha256"
    )
    if release_failure:
        missing.append(release_failure)
    if _declares_trace_from_entry(entry) or source.get("visualization_only_reexecution") is True:
        trace_failure = _digest_binding_failure(
            source, "trace_package_sha256", "observed_trace_package_sha256"
        )
        if trace_failure:
            missing.append(trace_failure)
    return ",".join(missing)


def _status(record: Mapping[str, Any], check: str) -> str:
    eligibility = record.get("eligibility")
    if not isinstance(eligibility, Mapping):
        return "unavailable"
    value = eligibility.get(check)
    if isinstance(value, Mapping):
        value = value.get("status")
    if value in ELIGIBILITY_STATUSES:
        return str(value)
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "unavailable"


def _execution_mode(record: Mapping[str, Any]) -> str:
    eligibility = record.get("eligibility")
    if not isinstance(eligibility, Mapping):
        return "unavailable"
    execution = eligibility.get("execution_status")
    if isinstance(execution, Mapping):
        return str(
            execution.get("execution_mode")
            or execution.get("mode")
            or eligibility.get("execution_mode")
            or "unavailable"
        )
    mode = eligibility.get("execution_mode")
    if isinstance(mode, str) and mode:
        return mode
    return "unavailable"


def _public_check_status(checks: Mapping[str, Any], check: str) -> str:
    detail = checks.get(check)
    if not isinstance(detail, Mapping):
        return "unavailable"
    status = str(detail.get("status", "unavailable"))
    return "unavailable" if status == "not_applicable" else status


def _check_reason(checks: Mapping[str, Any], check: str) -> str:
    detail = checks.get(check)
    if not isinstance(detail, Mapping):
        return "check not required for this grain"
    return str(detail.get("reason") or "check not required for this grain")


def _declared_telemetry_grade(record: Mapping[str, Any], checks: Mapping[str, Any]) -> str:
    detail = checks.get("telemetry_sufficiency")
    if isinstance(detail, Mapping):
        grade = detail.get("telemetry_grade") or detail.get("grade")
        if grade in TELEMETRY_GRADES:
            return str(grade)
    eligibility = record.get("eligibility")
    raw = eligibility.get("telemetry_sufficiency") if isinstance(eligibility, Mapping) else None
    if isinstance(raw, Mapping):
        grade = raw.get("telemetry_grade") or raw.get("grade")
        if grade in TELEMETRY_GRADES:
            return str(grade)
    boundary = record.get("source_boundary")
    if isinstance(boundary, Mapping) and boundary.get("telemetry_grade") in TELEMETRY_GRADES:
        return str(boundary["telemetry_grade"])
    if record.get("telemetry_grade") in TELEMETRY_GRADES:
        return str(record["telemetry_grade"])
    return "geometry"


def _eligibility_check_report(
    record: Mapping[str, Any],
    eligibility_map: Mapping[str, Any],
    check: str,
    *,
    execution_mode: str,
    allowed_not_applicable: frozenset[str],
    exempt: frozenset[str],
) -> tuple[dict[str, Any], dict[str, str] | None]:
    status = _status(record, check)
    detail = eligibility_map.get(check)
    reason = ""
    if isinstance(detail, Mapping):
        reason = str(detail.get("reason") or detail.get("detail") or "")
    if check == "execution_status" and execution_mode not in ADMISSIBLE_EXECUTION_MODES:
        status = "fail"
        reason = (
            f"execution_mode={execution_mode} is not admissible for selection; "
            "expected native or adapter_disclosed"
        )
    if check == "source_hashes" and status == "pass":
        source_failure = _source_hash_failure_reason(record)
        if source_failure:
            status = "fail"
            reason = f"source_hashes pass requires concrete hash fields: {source_failure}"
    if status == "not_applicable" and check not in allowed_not_applicable:
        status = "fail"
        reason = f"not_applicable is not allowed for required check {check}"
    applicable = status != "not_applicable" and check not in exempt
    report = {"status": status, "reason": reason, "applicable": applicable}
    if check == "telemetry_sufficiency" and isinstance(detail, Mapping):
        telemetry_detail_grade = detail.get("telemetry_grade") or detail.get("grade")
        if telemetry_detail_grade in TELEMETRY_GRADES:
            report["telemetry_grade"] = str(telemetry_detail_grade)
    blocker = (
        {"check": check, "status": status, "reason": reason}
        if applicable and status != "pass"
        else None
    )
    return report, blocker


def _eligibility_report(record: Mapping[str, Any]) -> dict[str, Any]:
    grain = str(record.get("grain", ""))
    role = _public_role(record.get("primary_role", ""))
    required = _required_checks_for_record(record)
    checks: dict[str, Any] = {}
    blockers: list[dict[str, str]] = []
    eligibility = record.get("eligibility")
    eligibility_map = eligibility if isinstance(eligibility, Mapping) else {}
    exempt = ROLE_CHECK_EXEMPTIONS.get((grain, role), frozenset())
    allowed_not_applicable = ALLOWED_NOT_APPLICABLE_CHECKS_BY_GRAIN_ROLE.get(
        (grain, role), frozenset()
    )
    execution_mode = _execution_mode(record)
    for check in required:
        check_report, blocker = _eligibility_check_report(
            record,
            eligibility_map,
            check,
            execution_mode=execution_mode,
            allowed_not_applicable=allowed_not_applicable,
            exempt=exempt,
        )
        checks[check] = check_report
        if blocker is not None:
            blockers.append(blocker)
    eligibility_status = "admitted" if not blockers else "excluded"
    if any(blocker["status"] == "unavailable" for blocker in blockers):
        eligibility_status = "unavailable"
    initial_state_match = _public_check_status(checks, "matched_initial_state_or_shared_prefix")
    outcome_match = _public_check_status(checks, "release_vs_rerun_outcome_agreement")
    telemetry_status = checks.get("telemetry_sufficiency", {}).get("status")
    telemetry_grade = _declared_telemetry_grade(record, checks)
    telemetry_reason = (
        "telemetry_sufficiency:pass"
        if telemetry_status == "pass"
        else f"telemetry_sufficiency:{telemetry_status or 'unavailable'}"
    )
    return {
        "eligible": not blockers,
        "status": eligibility_status,
        "reasons": [f"{b['check']}:{b['status']}" for b in blockers],
        "execution_mode": execution_mode,
        "typed_outcome_semantics": checks.get("typed_outcome_collision_semantics", {}).get("status")
        == "pass",
        "initial_state_match": initial_state_match,
        "initial_state_match_reason": _check_reason(
            checks, "matched_initial_state_or_shared_prefix"
        ),
        "outcome_match": outcome_match,
        "outcome_match_reason": _check_reason(checks, "release_vs_rerun_outcome_agreement"),
        "telemetry_grade": telemetry_grade,
        "telemetry_grade_reason": telemetry_reason,
        "required_checks": list(required),
        "checks": checks,
        "blockers": blockers,
    }


def _case_id(record: Mapping[str, Any]) -> str:
    case_id = record.get("case_id") or record.get("candidate_id")
    if not isinstance(case_id, str) or not case_id:
        raise CasePortfolioError("each evidence unit needs a non-empty case_id")
    return case_id


def _validate_unit_shape(record: Mapping[str, Any]) -> None:
    case_id = _case_id(record)
    grain = record.get("grain")
    role = _public_role(record.get("primary_role"))
    if grain not in GRAINS:
        raise CasePortfolioError(f"{case_id}: unknown grain {grain!r}")
    if role not in ROLES:
        raise CasePortfolioError(f"{case_id}: unknown primary_role {role!r}")
    if not isinstance(record.get("allowed_claim"), str) or not record.get("allowed_claim"):
        raise CasePortfolioError(f"{case_id}: allowed_claim is required")
    conceptual = record.get("conceptual_grain")
    if conceptual not in CONCEPTUAL_GRAINS:
        raise CasePortfolioError(f"{case_id}: unknown conceptual_grain {conceptual!r}")
    for tag in _conceptual_coverage(record):
        if tag not in CONCEPTUAL_GRAINS:
            raise CasePortfolioError(f"{case_id}: unknown conceptual coverage tag {tag!r}")
    if record.get("claim_grade") not in CLAIM_GRADES:
        raise CasePortfolioError(f"{case_id}: unknown claim_grade {record.get('claim_grade')!r}")


def _numeric_dimensions(record: Mapping[str, Any]) -> dict[str, float | None]:
    raw = record.get("dimensions")
    raw_map = raw if isinstance(raw, Mapping) else {}
    values: dict[str, float | None] = {}
    for key in (*SCIENTIFIC_DIMENSIONS, *POST_SELECTION_DIMENSIONS):
        value = raw_map.get(key)
        if value in {None, "unavailable"}:
            values[key] = None
            continue
        try:
            values[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise CasePortfolioError(
                f"{_case_id(record)}: dimension {key!r} must be numeric"
            ) from exc
    return values


def _role_scientific_dimensions(role: str) -> tuple[str, ...]:
    exempt = ROLE_DIMENSION_EXEMPTIONS.get(_public_role(role), frozenset())
    return tuple(key for key in SCIENTIFIC_DIMENSIONS if key not in exempt)


def _scientific_vector(record: Mapping[str, Any]) -> list[float] | None:
    dimensions = _numeric_dimensions(record)
    role_dimensions = _role_scientific_dimensions(_public_role(record.get("primary_role")))
    if any(dimensions[key] is None for key in role_dimensions):
        return None
    return [float(dimensions[key] or 0.0) for key in role_dimensions]


def compute_pareto_membership(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute role-local Pareto membership through the #5601 primitive.

    Returns:
        Mapping with the front, dominance ledger, vectors, and directions.
    """
    ids = [_case_id(record) for record in records]
    roles = {
        case_id: _public_role(record.get("primary_role"))
        for case_id, record in zip(ids, records, strict=True)
    }
    vectors: dict[str, list[float]] = {}
    unavailable: dict[str, list[str]] = {}
    for case_id, record in zip(ids, records, strict=True):
        vector = _scientific_vector(record)
        role_dimensions = _role_scientific_dimensions(_public_role(record.get("primary_role")))
        if vector is None:
            unavailable[case_id] = [
                key for key in role_dimensions if _numeric_dimensions(record).get(key) is None
            ]
        else:
            vectors[case_id] = vector

    front: set[str] = set()
    dominated: dict[str, Any] = {}
    for role in sorted(set(roles.values())):
        role_ids = [case_id for case_id in ids if roles[case_id] == role and case_id in vectors]
        role_vectors = [vectors[case_id] for case_id in role_ids]
        role_dimensions = _role_scientific_dimensions(role)
        directions = {key: PARETO_DIRECTIONS[key] for key in role_dimensions}
        role_front, role_dominated = compute_5601_pareto_front(role_ids, role_vectors, directions)
        front.update(role_front)
        dominated.update(role_dominated)
    return {
        "pareto_front": sorted(front),
        "dominated": {case_id: dominated[case_id] for case_id in sorted(dominated)},
        "dimension_unavailable": dict(sorted(unavailable.items())),
        "vectors": {case_id: vectors[case_id] for case_id in sorted(vectors)},
        "directions": dict(PARETO_DIRECTIONS),
        "adapter": "role_local_adapter_over_issue_5601_compute_pareto_front",
        "adapter_semantics": (
            "Each primary role is filtered independently with the issue #5601 "
            "Pareto primitive over that role's applicable scientific dimensions; "
            "post-selection presentation cost and visualizability are excluded."
        ),
    }


def _coverage_values(record: Mapping[str, Any]) -> dict[str, str]:
    coverage = record.get("coverage")
    coverage_map = coverage if isinstance(coverage, Mapping) else {}
    topology = coverage_map.get("topology") or record.get("topology") or "unknown"
    mechanism = coverage_map.get("mechanism") or record.get("mechanism") or "unknown"
    failure_class = coverage_map.get("failure_class") or record.get("failure_class") or "unknown"
    process_class = coverage_map.get("process_class") or record.get("process_class") or "unknown"
    return {
        "role": _public_role(record.get("primary_role")),
        "grain": str(record.get("grain")),
        "conceptual_grain": str(record.get("conceptual_grain")),
        "topology": str(topology),
        "mechanism": str(mechanism),
        "failure_class": str(failure_class),
        "process_class": str(process_class),
    }


def _conceptual_coverage(record: Mapping[str, Any]) -> tuple[str, ...]:
    coverage = record.get("conceptual_coverage")
    if not isinstance(coverage, list | tuple):
        presentation = record.get("presentation")
        if isinstance(presentation, Mapping):
            coverage = presentation.get("conceptual_coverage") or presentation.get("coverage_tags")
    if not isinstance(coverage, list | tuple):
        coverage = [record.get("conceptual_grain")]
    return tuple(sorted({str(tag) for tag in coverage if tag}))


def _pairwise_descriptor_diversity(portfolio: Sequence[Mapping[str, Any]]) -> float:
    if len(portfolio) < 2:
        return 0.0
    distances: list[float] = []
    for left, right in itertools.combinations(portfolio, 2):
        left_cov = _coverage_values(left)
        right_cov = _coverage_values(right)
        mismatches = sum(1 for field in DIVERSITY_FIELDS if left_cov[field] != right_cov[field])
        distances.append(mismatches / len(DIVERSITY_FIELDS))
    return min(distances) if distances else 0.0


def _portfolio_score(
    portfolio: Sequence[Mapping[str, Any]],
    constraints: PortfolioConstraints,
) -> tuple[Any, ...]:
    roles = {_public_role(item.get("primary_role")) for item in portfolio}
    grains = {str(item.get("grain")) for item in portfolio}
    conceptual_grains = {tag for item in portfolio for tag in _conceptual_coverage(item)}
    topologies = {_coverage_values(item)["topology"] for item in portfolio}
    dims = [_numeric_dimensions(item) for item in portfolio]
    evidence_grades = [d["evidence_grade"] for d in dims if d["evidence_grade"] is not None]
    page_costs = [d["page_cost"] for d in dims if d["page_cost"] is not None]
    return (
        len(set(constraints.required_roles) - roles),
        len(set(constraints.required_grains) - grains),
        len(set(constraints.required_conceptual_grains) - conceptual_grains),
        len(set(constraints.required_topologies) - topologies),
        len(
            set(constraints.required_failure_classes)
            - {_coverage_values(item)["failure_class"] for item in portfolio}
        ),
        len(
            set(constraints.required_process_classes)
            - {_coverage_values(item)["process_class"] for item in portfolio}
        ),
        -_pairwise_descriptor_diversity(portfolio),
        -min(evidence_grades) if evidence_grades else 0.0,
        sum(page_costs) if page_costs else FINITE_MISSING_PAGE_COST,
        sorted(_case_id(item) for item in portfolio),
    )


def _satisfies_hard_constraints(
    portfolio: Sequence[Mapping[str, Any]],
    constraints: PortfolioConstraints,
) -> bool:
    roles = [_public_role(item.get("primary_role")) for item in portfolio]
    if constraints.require_unique_primary_roles and len(roles) != len(set(roles)):
        return False
    conceptual_coverage = {tag for item in portfolio for tag in _conceptual_coverage(item)}
    coverage_values = [_coverage_values(item) for item in portfolio]
    return (
        set(constraints.required_roles).issubset(roles)
        and set(constraints.required_grains).issubset(
            {str(item.get("grain")) for item in portfolio}
        )
        and set(constraints.required_conceptual_grains).issubset(conceptual_coverage)
        and set(constraints.required_topologies).issubset(
            {value["topology"] for value in coverage_values}
        )
        and set(constraints.required_failure_classes).issubset(
            {value["failure_class"] for value in coverage_values}
        )
        and set(constraints.required_process_classes).issubset(
            {value["process_class"] for value in coverage_values}
        )
    )


def _is_nonrelaxable_valid_subset(
    portfolio: Sequence[Mapping[str, Any]],
    constraints: PortfolioConstraints,
) -> bool:
    roles = [_public_role(item.get("primary_role")) for item in portfolio]
    return not constraints.require_unique_primary_roles or len(roles) == len(set(roles))


def _select_exact(
    candidates: Sequence[Mapping[str, Any]],
    constraints: PortfolioConstraints,
) -> tuple[list[str], dict[str, Any]]:
    max_size = min(constraints.max_size, len(candidates))
    best_ids: list[str] = []
    best_score: tuple[Any, ...] | None = None
    best_complete = False
    best_valid = False
    enumeration_count = 0
    ordered = sorted(candidates, key=_case_id)
    sizes = range(max_size, -1, -1)
    for size in sizes:
        for combo in itertools.combinations(ordered, size):
            enumeration_count += 1
            valid = _is_nonrelaxable_valid_subset(combo, constraints)
            if not valid:
                continue
            complete = _satisfies_hard_constraints(combo, constraints)
            score = _portfolio_score(combo, constraints)
            if best_score is None:
                best_score = score
                best_ids = [_case_id(item) for item in combo]
                best_complete = complete
                best_valid = valid
                continue
            if complete and not best_complete:
                best_score = score
                best_ids = [_case_id(item) for item in combo]
                best_complete = True
                best_valid = valid
                continue
            if complete == best_complete and score < best_score:
                best_score = score
                best_ids = [_case_id(item) for item in combo]
                best_complete = complete
                best_valid = valid
    return sorted(best_ids), {
        "enumeration_count": enumeration_count,
        "best_score": list(best_score) if best_score is not None else [],
        "complete_constraints_satisfied": best_complete,
        "nonrelaxable_valid_subset": best_valid,
        "constraint_classes": {
            "non_relaxable": [
                "unique_primary_roles",
                "required_roles",
                "required_grains",
                "required_conceptual_coverage",
                "required_topologies",
                "required_failure_classes",
                "required_process_classes",
            ],
            "relaxable_tie_breakers": [
                "minimum_pairwise_descriptor_diversity",
                "minimum_evidence_grade",
                "page_cost",
                "stable_case_id",
            ],
        },
    }


def _constraints_from_config(config: Mapping[str, Any]) -> PortfolioConstraints:
    raw = config.get("selection", {})
    selection = raw if isinstance(raw, Mapping) else {}
    required_roles = tuple(_public_role(role) for role in selection.get("required_roles", ()))
    if not required_roles:
        raise CasePortfolioError("selection.required_roles must not be empty")
    required_grains = tuple(str(grain) for grain in selection.get("required_grains", tuple(GRAINS)))
    required_conceptual_grains = tuple(
        str(grain)
        for grain in selection.get(
            "required_conceptual_grains", ("campaign", "cell", "matched_contrast", "trace")
        )
    )
    required_topologies = tuple(str(value) for value in selection.get("required_topologies", ()))
    required_failure_classes = tuple(
        str(value) for value in selection.get("required_failure_classes", ())
    )
    required_process_classes = tuple(
        str(value) for value in selection.get("required_process_classes", ())
    )
    frozen = selection.get("frozen_role_targets", {})
    frozen_map = (
        {_public_role(k): str(v) for k, v in frozen.items()} if isinstance(frozen, Mapping) else {}
    )
    return PortfolioConstraints(
        required_roles=required_roles,
        required_grains=required_grains,
        required_conceptual_grains=required_conceptual_grains,
        required_topologies=required_topologies,
        required_failure_classes=required_failure_classes,
        required_process_classes=required_process_classes,
        target_size=int(selection.get("target_size", len(required_roles))),
        max_size=int(selection.get("max_size", selection.get("target_size", len(required_roles)))),
        require_unique_primary_roles=bool(selection.get("require_unique_primary_roles", True)),
        frozen_role_targets=frozen_map,
    )


def _normalized_config_for_hash(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    units = normalized.get("evidence_units")
    if isinstance(units, list):
        normalized["evidence_units"] = sorted(
            [dict(unit) for unit in units if isinstance(unit, Mapping)],
            key=_case_id,
        )
    return normalized


def _load_inventory(
    config: Mapping[str, Any], candidate_manifest: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    inventory = config.get("evidence_units", [])
    if not isinstance(inventory, list):
        raise CasePortfolioError("evidence_units must be a list")
    loaded = [dict(item) for item in inventory if isinstance(item, Mapping)]
    for item in loaded:
        item["primary_role"] = _public_role(item.get("primary_role"))
        coverage = item.get("coverage")
        if isinstance(coverage, Mapping) and coverage.get("mechanism") in PUBLIC_ROLE_ALIASES:
            item["coverage"] = {**dict(coverage), "mechanism": _public_role(coverage["mechanism"])}

    if candidate_manifest:
        candidates = candidate_manifest.get("candidates", [])
        if not isinstance(candidates, list):
            raise CasePortfolioError("candidate manifest candidates must be a list")
        by_id = {str(c.get("candidate_id")): c for c in candidates if isinstance(c, Mapping)}
        for item in loaded:
            source_id = item.get("candidate_manifest_id")
            if source_id:
                if source_id not in by_id:
                    raise CasePortfolioError(
                        f"configured candidate_manifest_id not found: {source_id}"
                    )
                source = by_id[str(source_id)]
                item.setdefault("candidate_id", source.get("candidate_id"))
                item.setdefault("scenario_id", source.get("scenario_id"))
                item.setdefault("planner", source.get("planner"))
                item.setdefault("source_candidate", source)
                item.setdefault("source_candidate_sha256", canonical_sha256(source))
    return loaded


def _source_inventory_ledger(
    candidate_manifest: Mapping[str, Any] | None,
    consumed_by_source_id: Mapping[str, str],
) -> list[dict[str, Any]]:
    if not candidate_manifest:
        return []
    candidates = candidate_manifest.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    entries: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = str(candidate.get("candidate_id"))
        consumed_by = consumed_by_source_id.get(candidate_id)
        reason = (
            f"upstream_consumed_by_configured_case:{consumed_by}"
            if consumed_by
            else "upstream_not_configured_for_ch7_case_contract"
        )
        entries.append(
            {
                "case_id": f"upstream::{candidate_id}",
                "source_candidate_id": candidate_id,
                "grain": "upstream_inventory",
                "conceptual_grain": "upstream_inventory",
                "primary_role": "upstream_inventory",
                "disposition": "excluded",
                "exclusion_reasons": [reason],
                "eligibility": {
                    "eligible": False,
                    "status": "excluded",
                    "reasons": [reason],
                    "execution_mode": "unavailable",
                    "typed_outcome_semantics": False,
                    "initial_state_match": "unavailable",
                    "outcome_match": "unavailable",
                    "telemetry_grade": "geometry",
                    "telemetry_grade_reason": "upstream inventory is outside the Chapter 7 case contract",
                    "required_checks": [],
                    "checks": {},
                    "blockers": [
                        {
                            "check": "ch7_case_contract_applicability",
                            "status": "not_applicable",
                            "reason": (
                                "retained from upstream #5446 inventory; "
                                + (
                                    "mapped to configured case"
                                    if consumed_by
                                    else "not configured for the Chapter 7 case contract"
                                )
                            ),
                        }
                    ],
                },
                "pareto_member": False,
                "stable_tie_break": f"upstream::{candidate_id}",
                "source_candidate_sha256": canonical_sha256(candidate),
                "consumed_by_case_id": consumed_by,
            }
        )
    return sorted(entries, key=lambda item: item["case_id"])


def build_ch7_worked_example_portfolio(
    config: Mapping[str, Any],
    *,
    candidate_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic Chapter 7 worked-example portfolio manifest.

    Returns:
        The schema-versioned manifest as a dictionary.
    """
    constraints = _constraints_from_config(config)
    units = _load_inventory(config, candidate_manifest)
    if not units:
        raise CasePortfolioError("no evidence units supplied")
    for unit in units:
        _validate_unit_shape(unit)

    eligibility_by_id = {_case_id(unit): _eligibility_report(unit) for unit in units}
    eligible = [unit for unit in units if eligibility_by_id[_case_id(unit)]["eligible"]]

    if constraints.frozen_role_targets:
        frozen_ids = set(constraints.frozen_role_targets.values())
        eligible_for_pareto = [unit for unit in eligible if _case_id(unit) in frozen_ids]
    else:
        eligible_for_pareto = eligible

    pareto = compute_pareto_membership(eligible_for_pareto)
    pareto_ids = set(pareto["pareto_front"])
    pareto_candidates = [unit for unit in eligible_for_pareto if _case_id(unit) in pareto_ids]

    selected_ids, enumeration = _select_exact(pareto_candidates, constraints)
    enumeration["frozen_role_targets"] = dict(sorted(constraints.frozen_role_targets.items()))

    selected_set = set(selected_ids)
    by_id = {_case_id(unit): unit for unit in units}
    selected = [
        _selected_record(
            by_id[case_id], order=index + 1, dimensions=_numeric_dimensions(by_id[case_id])
        )
        for index, case_id in enumerate(selected_ids)
    ]
    selected_roles = {item["primary_role"] for item in selected}
    selected_grains = {item["grain"] for item in selected}
    selected_conceptual_grains = {
        tag for item in selected for tag in item.get("conceptual_coverage", [])
    }
    selected_coverage = [_coverage_values(item) for item in selected]
    selected_topologies = {item["topology"] for item in selected_coverage}
    selected_failure_classes = {item["failure_class"] for item in selected_coverage}
    selected_process_classes = {item["process_class"] for item in selected_coverage}
    uncovered_roles = [role for role in constraints.required_roles if role not in selected_roles]
    uncovered_grains = [
        grain for grain in constraints.required_grains if grain not in selected_grains
    ]
    uncovered_conceptual_grains = [
        grain
        for grain in constraints.required_conceptual_grains
        if grain not in selected_conceptual_grains
    ]
    uncovered_topologies = [
        value for value in constraints.required_topologies if value not in selected_topologies
    ]
    uncovered_failure_classes = [
        value
        for value in constraints.required_failure_classes
        if value not in selected_failure_classes
    ]
    uncovered_process_classes = [
        value
        for value in constraints.required_process_classes
        if value not in selected_process_classes
    ]
    unique_roles_satisfied = not constraints.require_unique_primary_roles or len(
        selected_roles
    ) == len(selected)
    status = "complete" if enumeration["complete_constraints_satisfied"] else "partial"

    ledger = _build_ledger(units, eligibility_by_id, pareto, selected_set, constraints)
    consumed_by_source_id = {
        str(unit.get("candidate_manifest_id")): _case_id(unit)
        for unit in units
        if unit.get("candidate_manifest_id")
    }
    source_ledger = _source_inventory_ledger(candidate_manifest, consumed_by_source_id)
    full_ledger = sorted([*ledger, *source_ledger], key=lambda item: item["case_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "status": status,
        "claim_boundary": (
            "Selection and exclusion record only. It does not establish causal evidence, "
            "real-world prevalence, planner ranking, or dissertation-ready figures."
        ),
        "inputs": {
            "config_sha256": canonical_sha256(_normalized_config_for_hash(config)),
            "candidate_manifest_sha256": canonical_sha256(candidate_manifest)
            if candidate_manifest
            else None,
            "candidate_manifest_schema": candidate_manifest.get("schema_version")
            if isinstance(candidate_manifest, Mapping)
            else None,
            "source_inventory_count": len(source_ledger),
        },
        "constraints": {
            "required_roles": list(constraints.required_roles),
            "required_grains": list(constraints.required_grains),
            "required_conceptual_grains": list(constraints.required_conceptual_grains),
            "required_topologies": list(constraints.required_topologies),
            "required_failure_classes": list(constraints.required_failure_classes),
            "required_process_classes": list(constraints.required_process_classes),
            "target_size": constraints.target_size,
            "max_size": constraints.max_size,
            "require_unique_primary_roles": constraints.require_unique_primary_roles,
            "frozen_role_targets": dict(sorted(constraints.frozen_role_targets.items())),
        },
        "selection_pipeline": [
            "source_inventories",
            "fail_closed_eligibility",
            "role_assignment",
            "pareto_filtering",
            "exact_enumeration",
            "hard_role_topology_grain_coverage",
            "lexicographic_selection",
            "stable_case_id_tie_break",
            "complete_selected_excluded_ledger",
        ],
        "summary": {
            "n_inventory": len(units),
            "n_source_inventory": len(source_ledger),
            "n_ledger": len(full_ledger),
            "n_eligible": len(eligible),
            "n_pareto": len(pareto_ids),
            "n_selected": len(selected),
            "uncovered_roles": uncovered_roles,
            "uncovered_grains": uncovered_grains,
            "uncovered_conceptual_grains": uncovered_conceptual_grains,
            "uncovered_topologies": uncovered_topologies,
            "uncovered_failure_classes": uncovered_failure_classes,
            "uncovered_process_classes": uncovered_process_classes,
            "unique_primary_roles_satisfied": unique_roles_satisfied,
        },
        "pareto_analysis": {
            "directions": pareto["directions"],
            "front": pareto["pareto_front"],
            "dominated": pareto["dominated"],
            "dimension_unavailable": pareto["dimension_unavailable"],
            "adapter": pareto["adapter"],
            "adapter_semantics": pareto["adapter_semantics"],
        },
        "exact_enumeration": enumeration,
        "selected": selected,
        "ledger": full_ledger,
        "content_sha256": "",
    }


def finalize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a manifest with its deterministic content hash filled in."""
    payload = dict(manifest)
    payload["content_sha256"] = ""
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def _schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "ch7_case_portfolio.schema.v2.json"


def _selected_record(
    unit: Mapping[str, Any],
    *,
    order: int,
    dimensions: Mapping[str, float | None],
) -> dict[str, Any]:
    allowed_claim = str(unit.get("allowed_claim"))
    eligibility = _eligibility_report(unit)
    source_boundary = unit.get("source_boundary", {})
    source_refs = unit.get("source_refs", [])
    pareto_status = "nondominated"
    source = _source_block(unit, source_boundary, source_refs)
    presentation = _presentation_block(unit)
    return {
        "selection_order": order,
        "case_id": _case_id(unit),
        "grain": str(unit.get("grain")),
        "conceptual_grain": str(unit.get("conceptual_grain")),
        "conceptual_coverage": list(_conceptual_coverage(unit)),
        "primary_role": _public_role(unit.get("primary_role")),
        "secondary_descriptors": list(unit.get("secondary_descriptors", [])),
        "claim": {
            "grade": unit.get("claim_grade", "candidate"),
            "allowed": [allowed_claim],
            "forbidden": list(unit.get("forbidden_claims", [])),
        },
        "allowed_claim": allowed_claim,
        "event_anchor": _event_anchor_block(unit),
        "presentation": presentation,
        "eligibility": eligibility,
        "selection": {
            "vector": {
                "scientific_pareto": {
                    key: dimensions[key]
                    for key in _role_scientific_dimensions(_public_role(unit.get("primary_role")))
                },
                "post_selection": {key: dimensions[key] for key in POST_SELECTION_DIMENSIONS},
            },
            "pareto_adapter": "role_local_adapter_over_issue_5601_compute_pareto_front",
            "pareto_status": pareto_status,
            "coverage_tags": list(_conceptual_coverage(unit)),
            "included_reason": "selected_by_exact_coverage",
            "excluded_reason": "unavailable",
            "scientific_dimensions_applied": list(
                _role_scientific_dimensions(_public_role(unit.get("primary_role")))
            ),
            "post_selection_dimensions": list(POST_SELECTION_DIMENSIONS),
        },
        "scenario_id": unit.get("scenario_id"),
        "planner": unit.get("planner"),
        "coverage": _coverage_values(unit),
        "selection_vector": {
            "scientific_pareto": {key: dimensions[key] for key in SCIENTIFIC_DIMENSIONS},
            "post_selection": {key: dimensions[key] for key in POST_SELECTION_DIMENSIONS},
        },
        "dimensions": dict(sorted(dimensions.items())),
        "source_refs": source_refs,
        "source_boundary": source_boundary,
        "source": source,
        "source_candidate_sha256": unit.get("source_candidate_sha256"),
    }


def _source_refs_list(source_refs: Any) -> list[str]:
    if not isinstance(source_refs, list):
        return []
    return [str(ref) for ref in source_refs if isinstance(ref, str) and ref]


def _safe_source_digest(ref: Any) -> tuple[str, str]:
    if not isinstance(ref, str) or not ref:
        return "unavailable", "missing source ref"
    try:
        path = Path(ref)
        if path.is_absolute() or ".." in path.parts:
            return "unavailable", f"unsafe source ref {ref!r}"
        root = _repo_root()
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            return "unavailable", f"source ref escapes repo {ref!r}"
        if not resolved.is_file():
            return "unavailable", f"source ref is unreadable {ref!r}"
        return file_sha256(resolved), ""
    except (OSError, RuntimeError, ValueError) as exc:
        return "unavailable", f"source ref error for {ref!r}: {type(exc).__name__}"


def _source_ref_for_kind(boundary: Mapping[str, Any], refs: Sequence[str], kind: str) -> Any:
    if kind == "release":
        explicit = boundary.get("release_rows_ref") or boundary.get("release_ref")
        if explicit:
            return explicit
        return refs[0] if refs else None
    explicit = boundary.get("trace_package_ref") or boundary.get("trace_ref")
    if explicit:
        return explicit
    return refs[1] if len(refs) > 1 else None


def _source_block(
    unit: Mapping[str, Any],
    source_boundary: Any | None = None,
    source_refs: Any | None = None,
) -> dict[str, Any]:
    boundary = (
        source_boundary if isinstance(source_boundary, Mapping) else unit.get("source_boundary", {})
    )
    refs = _source_refs_list(
        source_refs if isinstance(source_refs, list) else unit.get("source_refs", [])
    )
    release_id = unit.get("release_id")
    release_hash = unit.get("release_rows_sha256")
    trace_hash = unit.get("trace_package_sha256")
    expected_release_hash = unit.get("expected_release_rows_sha256")
    expected_trace_hash = unit.get("expected_trace_package_sha256")
    if isinstance(boundary, Mapping):
        release_id = release_id or boundary.get("release_id")
        release_hash = release_hash or boundary.get("release_rows_sha256")
        trace_hash = trace_hash or boundary.get("trace_package_sha256") or "unavailable"
        expected_release_hash = expected_release_hash or boundary.get(
            "expected_release_rows_sha256"
        )
        expected_trace_hash = (
            expected_trace_hash or boundary.get("expected_trace_package_sha256") or "unavailable"
        )
    boundary_map = boundary if isinstance(boundary, Mapping) else {}
    release_ref = _source_ref_for_kind(boundary_map, refs, "release")
    trace_ref = _source_ref_for_kind(boundary_map, refs, "trace")
    observed_release_hash, release_ref_error = _safe_source_digest(release_ref)
    observed_trace_hash, trace_ref_error = _safe_source_digest(trace_ref)
    return {
        "release_id": str(release_id or "unavailable"),
        "release_rows_sha256": str(release_hash or "unavailable"),
        "trace_package_sha256": str(trace_hash or "unavailable"),
        "expected_release_rows_sha256": str(expected_release_hash or "unavailable"),
        "expected_trace_package_sha256": str(expected_trace_hash or "unavailable"),
        "observed_release_rows_sha256": observed_release_hash,
        "observed_trace_package_sha256": observed_trace_hash,
        "release_rows_ref": str(release_ref or "unavailable"),
        "trace_package_ref": str(trace_ref or "unavailable"),
        "source_ref_errors": {
            "release_rows": release_ref_error,
            "trace_package": trace_ref_error,
        },
        "visualization_only_reexecution": bool(
            isinstance(boundary, Mapping)
            and boundary.get(
                "visualization_only_rerun", boundary.get("visualization_only_reexecution", False)
            )
        ),
        "refs": refs,
        "boundary": boundary,
        "candidate_manifest_id": unit.get("candidate_manifest_id"),
        "candidate_sha256": unit.get("source_candidate_sha256"),
    }


def _event_anchor_block(unit: Mapping[str, Any]) -> dict[str, Any]:
    anchor = unit.get("event_anchor")
    anchor_map = dict(anchor) if isinstance(anchor, Mapping) else {}
    return {
        **anchor_map,
        "type": str(anchor_map.get("type") or "terminal"),
        "time_s": anchor_map.get("time_s", "unavailable"),
        "source_field": str(
            anchor_map.get("source_field") or anchor_map.get("source") or "unavailable"
        ),
        "shared_between_cases": bool(anchor_map.get("shared_between_cases", False)),
    }


def _presentation_block(unit: Mapping[str, Any]) -> dict[str, Any]:
    presentation = unit.get("presentation")
    presentation_map = dict(presentation) if isinstance(presentation, Mapping) else {}
    shared = presentation_map.get("shared_axis_contract", presentation_map.get("shared_axes"))
    if isinstance(shared, Mapping):
        shared = shared.get("contract_id") or shared.get("status") or "shared_axis_mapping"
    keyframes = presentation_map.get("semantic_keyframes", presentation_map.get("keyframes", []))
    return {
        **presentation_map,
        "required_views": list(presentation_map.get("required_views", [])),
        "coverage_tags": list(_conceptual_coverage(unit)),
        "shared_axis_contract": str(shared) if shared is not None else "unavailable",
        "semantic_keyframes": keyframes if isinstance(keyframes, list) else [keyframes],
    }


def _build_ledger(
    units: Sequence[Mapping[str, Any]],
    eligibility_by_id: Mapping[str, Mapping[str, Any]],
    pareto: Mapping[str, Any],
    selected_ids: set[str],
    constraints: PortfolioConstraints,
) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    frozen_ids = set(constraints.frozen_role_targets.values())
    pareto_ids = set(pareto.get("pareto_front", []))
    dimension_unavailable = pareto.get("dimension_unavailable", {})
    dominated = pareto.get("dominated", {})
    selected_order = {case_id: index + 1 for index, case_id in enumerate(sorted(selected_ids))}
    for unit in sorted(units, key=_case_id):
        case_id = _case_id(unit)
        eligibility = eligibility_by_id[case_id]
        if case_id in selected_ids:
            disposition = "selected"
            reasons: list[str] = []
            pareto_status = "nondominated"
            included_reason = "selected_by_exact_coverage"
            excluded_reason = "unavailable"
        elif not eligibility["eligible"]:
            disposition = "excluded"
            reasons = [f"eligibility_{b['check']}:{b['status']}" for b in eligibility["blockers"]]
            pareto_status = "not_applicable"
            included_reason = "unavailable"
            excluded_reason = ",".join(reasons) if reasons else "unavailable"
        elif constraints.frozen_role_targets and case_id not in frozen_ids:
            disposition = "excluded"
            reasons = ["not_frozen_first_production_target"]
            pareto_status = "not_applicable"
            included_reason = "unavailable"
            excluded_reason = reasons[0]
        elif case_id in dimension_unavailable:
            disposition = "excluded"
            reasons = [
                "scientific_dimension_unavailable:" + ",".join(dimension_unavailable[case_id])
            ]
            pareto_status = "not_applicable"
            included_reason = "unavailable"
            excluded_reason = reasons[0]
        elif case_id not in pareto_ids:
            disposition = "excluded"
            reasons = ["dominated_pareto"]
            pareto_status = "dominated"
            included_reason = "unavailable"
            excluded_reason = reasons[0]
        else:
            disposition = "excluded"
            reasons = ["not_selected_exact_coverage"]
            pareto_status = "nondominated"
            included_reason = "unavailable"
            excluded_reason = reasons[0]
        ledger.append(
            {
                "case_id": case_id,
                "grain": unit.get("grain"),
                "conceptual_grain": unit.get("conceptual_grain"),
                "conceptual_coverage": list(_conceptual_coverage(unit)),
                "primary_role": _public_role(unit.get("primary_role")),
                "secondary_descriptors": list(unit.get("secondary_descriptors", [])),
                "allowed_claim": str(unit.get("allowed_claim", "")),
                "forbidden_claims": list(unit.get("forbidden_claims", [])),
                "claim_grade": unit.get("claim_grade", "candidate"),
                "event_anchor": _event_anchor_block(unit),
                "presentation": _presentation_block(unit),
                "disposition": disposition,
                "exclusion_reasons": reasons,
                "eligibility": eligibility,
                "pareto_member": case_id in pareto_ids,
                "pareto_reason": dominated.get(case_id) or {},
                "selection": {
                    "vector": {
                        "scientific_pareto": {
                            key: _numeric_dimensions(unit)[key]
                            for key in _role_scientific_dimensions(
                                _public_role(unit.get("primary_role"))
                            )
                        },
                        "post_selection": {
                            key: _numeric_dimensions(unit)[key] for key in POST_SELECTION_DIMENSIONS
                        },
                    },
                    "pareto_status": pareto_status,
                    "coverage_tags": list(_conceptual_coverage(unit)),
                    "included_reason": included_reason,
                    "excluded_reason": excluded_reason,
                },
                "coverage": _coverage_values(unit),
                "dimensions": dict(sorted(_numeric_dimensions(unit).items())),
                "source": _source_block(unit),
                "source_refs": unit.get("source_refs", []),
                "source_boundary": unit.get("source_boundary", {}),
                "candidate_manifest_id": unit.get("candidate_manifest_id"),
                "source_candidate_sha256": unit.get("source_candidate_sha256"),
                "scenario_id": unit.get("scenario_id"),
                "planner": unit.get("planner"),
                "selection_order": selected_order[case_id] if case_id in selected_ids else None,
                "stable_tie_break": case_id,
            }
        )
    return ledger


def _constraints_from_manifest_constraints(raw: Mapping[str, Any]) -> PortfolioConstraints:
    return PortfolioConstraints(
        required_roles=tuple(_public_role(value) for value in raw.get("required_roles", ())),
        required_grains=tuple(str(value) for value in raw.get("required_grains", ())),
        required_conceptual_grains=tuple(
            str(value) for value in raw.get("required_conceptual_grains", ())
        ),
        required_topologies=tuple(str(value) for value in raw.get("required_topologies", ())),
        required_failure_classes=tuple(
            str(value) for value in raw.get("required_failure_classes", ())
        ),
        required_process_classes=tuple(
            str(value) for value in raw.get("required_process_classes", ())
        ),
        target_size=int(raw.get("target_size", 0)),
        max_size=int(raw.get("max_size", raw.get("target_size", 0))),
        require_unique_primary_roles=bool(raw.get("require_unique_primary_roles", True)),
        frozen_role_targets={
            _public_role(k): str(v) for k, v in (raw.get("frozen_role_targets", {}) or {}).items()
        }
        if isinstance(raw.get("frozen_role_targets", {}), Mapping)
        else {},
    )


def _eligibility_input_from_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        return {}
    checks = report.get("checks")
    input_checks: dict[str, Any] = {}
    if isinstance(checks, Mapping):
        for check, detail in checks.items():
            if isinstance(detail, Mapping):
                copied = {
                    "status": detail.get("status", "unavailable"),
                    "reason": detail.get("reason", ""),
                }
                if check == "execution_status":
                    copied["execution_mode"] = report.get("execution_mode", "unavailable")
                if detail.get("telemetry_grade") in TELEMETRY_GRADES:
                    copied["telemetry_grade"] = detail["telemetry_grade"]
                input_checks[str(check)] = copied
            else:
                input_checks[str(check)] = detail
    input_checks["execution_mode"] = report.get("execution_mode", "unavailable")
    return input_checks


def _ledger_record_for_eligibility_replay(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": entry.get("case_id"),
        "grain": entry.get("grain"),
        "conceptual_grain": entry.get("conceptual_grain"),
        "conceptual_coverage": entry.get("conceptual_coverage", []),
        "primary_role": entry.get("primary_role"),
        "presentation": entry.get("presentation", {}),
        "source_boundary": entry.get("source_boundary", {}),
        "source_refs": entry.get("source_refs", []),
        "candidate_manifest_id": entry.get("candidate_manifest_id"),
        "source_candidate_sha256": entry.get("source_candidate_sha256"),
        "eligibility": _eligibility_input_from_report(entry.get("eligibility", {})),
    }


def _expected_eligibility_from_ledger(entry: Mapping[str, Any]) -> dict[str, Any]:
    return _eligibility_report(_ledger_record_for_eligibility_replay(entry))


def _selected_record_from_ledger(entry: Mapping[str, Any]) -> dict[str, Any]:
    role = _public_role(entry.get("primary_role"))
    dimensions = entry.get("dimensions") if isinstance(entry.get("dimensions"), Mapping) else {}
    source_refs = entry.get("source_refs", [])
    source_boundary = entry.get("source_boundary", {})
    allowed_claim = str(entry.get("allowed_claim", ""))
    source = _source_block(entry, source_boundary, source_refs)
    eligibility = _expected_eligibility_from_ledger(entry)
    return {
        "selection_order": entry.get("selection_order"),
        "case_id": entry.get("case_id"),
        "grain": str(entry.get("grain")),
        "conceptual_grain": str(entry.get("conceptual_grain")),
        "conceptual_coverage": list(entry.get("conceptual_coverage", [])),
        "primary_role": role,
        "secondary_descriptors": list(entry.get("secondary_descriptors", [])),
        "claim": {
            "grade": entry.get("claim_grade", "candidate"),
            "allowed": [allowed_claim],
            "forbidden": list(entry.get("forbidden_claims", [])),
        },
        "allowed_claim": allowed_claim,
        "event_anchor": entry.get("event_anchor", {}),
        "presentation": entry.get("presentation", {}),
        "eligibility": eligibility,
        "selection": {
            "vector": {
                "scientific_pareto": {
                    key: dimensions.get(key) for key in _role_scientific_dimensions(role)
                },
                "post_selection": {key: dimensions.get(key) for key in POST_SELECTION_DIMENSIONS},
            },
            "pareto_adapter": "role_local_adapter_over_issue_5601_compute_pareto_front",
            "pareto_status": entry.get("selection", {}).get("pareto_status")
            if isinstance(entry.get("selection"), Mapping)
            else "unavailable",
            "coverage_tags": list(entry.get("conceptual_coverage", [])),
            "included_reason": entry.get("selection", {}).get("included_reason")
            if isinstance(entry.get("selection"), Mapping)
            else "unavailable",
            "excluded_reason": entry.get("selection", {}).get("excluded_reason")
            if isinstance(entry.get("selection"), Mapping)
            else "unavailable",
            "scientific_dimensions_applied": list(_role_scientific_dimensions(role)),
            "post_selection_dimensions": list(POST_SELECTION_DIMENSIONS),
        },
        "scenario_id": entry.get("scenario_id"),
        "planner": entry.get("planner"),
        "coverage": entry.get("coverage", {}),
        "selection_vector": {
            "scientific_pareto": {key: dimensions.get(key) for key in SCIENTIFIC_DIMENSIONS},
            "post_selection": {key: dimensions.get(key) for key in POST_SELECTION_DIMENSIONS},
        },
        "dimensions": dict(sorted(dimensions.items())),
        "source_refs": source_refs,
        "source_boundary": source_boundary,
        "source": source,
        "source_candidate_sha256": entry.get("source_candidate_sha256"),
    }


def _ledger_replay_records(ledger: Sequence[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in ledger:
        if not isinstance(entry, Mapping) or str(entry.get("case_id", "")).startswith("upstream::"):
            continue
        records.append(
            {
                "case_id": entry.get("case_id"),
                "grain": entry.get("grain"),
                "conceptual_grain": entry.get("conceptual_grain"),
                "conceptual_coverage": entry.get("conceptual_coverage", []),
                "primary_role": entry.get("primary_role"),
                "coverage": entry.get("coverage", {}),
                "dimensions": entry.get("dimensions", {}),
                "eligibility": _expected_eligibility_from_ledger(entry),
            }
        )
    return records


def validate_ch7_worked_example_portfolio(  # noqa: C901, PLR0912, PLR0915
    manifest: Any,
) -> PortfolioValidation:
    """Structurally validate a Chapter 7 worked-example portfolio manifest.

    Returns:
        Validation result with structural violations, if any.
    """
    violations: list[str] = []
    if not isinstance(manifest, Mapping):
        return PortfolioValidation([f"manifest must be a dict, got {type(manifest).__name__}"])
    from robot_sf.common.optional_import import try_import  # noqa: PLC0415

    jsonschema = try_import("jsonschema")
    if jsonschema is None:
        violations.append("schema validation unavailable: jsonschema is not installed")
    else:
        try:
            schema = json.loads(_schema_path().read_text(encoding="utf-8"))
            errors = sorted(
                jsonschema.Draft202012Validator(schema).iter_errors(manifest),
                key=lambda error: list(error.absolute_path),
            )
            for error in errors:
                path = "/".join(str(part) for part in error.absolute_path)
                violations.append(f"schema /{path}: {error.message}")
        except (OSError, json.JSONDecodeError) as exc:
            violations.append(f"schema validation unavailable: {exc}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        violations.append(f"schema_version mismatch: {manifest.get('schema_version')!r}")
    expected = canonical_sha256({**dict(manifest), "content_sha256": ""})
    if manifest.get("content_sha256") != expected:
        violations.append("content_sha256 mismatch")
    selected = manifest.get("selected")
    ledger = manifest.get("ledger")
    if not isinstance(selected, list):
        violations.append("selected must be a list")
        selected = []
    if not isinstance(ledger, list) or not ledger:
        violations.append("ledger must be a non-empty list")
        ledger = []
    selected_roles: set[str] = set()
    selected_grains: set[str] = set()
    selected_conceptual: set[str] = set()
    selected_topologies: set[str] = set()
    selected_failure_classes: set[str] = set()
    selected_process_classes: set[str] = set()
    for item in selected:
        if not isinstance(item, Mapping):
            violations.append("selected contains non-dict item")
            continue
        role = item.get("primary_role")
        if role in selected_roles:
            violations.append(f"selected role repeated: {role}")
        selected_roles.add(str(role))
        if role not in ROLES:
            violations.append(f"selected item has unknown role: {role!r}")
        if item.get("grain") not in GRAINS:
            violations.append(f"selected item has unknown grain: {item.get('grain')!r}")
        else:
            selected_grains.add(str(item.get("grain")))
        if item.get("conceptual_grain") not in CONCEPTUAL_GRAINS:
            violations.append(
                f"selected item has unknown conceptual_grain: {item.get('conceptual_grain')!r}"
            )
        conceptual_coverage = item.get("conceptual_coverage")
        if not isinstance(conceptual_coverage, list) or not conceptual_coverage:
            violations.append(f"{item.get('case_id')}: missing conceptual_coverage")
        else:
            for tag in conceptual_coverage:
                if tag not in CONCEPTUAL_GRAINS:
                    violations.append(f"{item.get('case_id')}: unknown conceptual coverage {tag!r}")
                else:
                    selected_conceptual.add(str(tag))
        if not item.get("allowed_claim"):
            violations.append(f"{item.get('case_id')}: missing allowed_claim")
        vector = item.get("selection_vector")
        if not isinstance(vector, Mapping):
            violations.append(f"{item.get('case_id')}: missing selection_vector")
        claim = item.get("claim")
        if (
            not isinstance(claim, Mapping)
            or claim.get("grade") not in CLAIM_GRADES
            or not claim.get("allowed")
            or not isinstance(claim.get("forbidden"), list)
            or not claim.get("forbidden")
        ):
            violations.append(f"{item.get('case_id')}: malformed claim block")
        anchor = item.get("event_anchor")
        if not isinstance(anchor, Mapping) or not {
            "type",
            "time_s",
            "source_field",
            "shared_between_cases",
        }.issubset(anchor):
            violations.append(f"{item.get('case_id')}: malformed event_anchor")
        elif (
            anchor.get("type") not in EVENT_TYPES
            or not _finite_time_or_unavailable(anchor.get("time_s"))
            or not isinstance(anchor.get("source_field"), str)
            or not isinstance(anchor.get("shared_between_cases"), bool)
        ):
            violations.append(f"{item.get('case_id')}: malformed event_anchor")
        presentation = item.get("presentation")
        if (
            not isinstance(presentation, Mapping)
            or not isinstance(presentation.get("required_views"), list)
            or "shared_axis_contract" not in presentation
            or "semantic_keyframes" not in presentation
        ):
            violations.append(f"{item.get('case_id')}: malformed presentation contract")
        else:
            views = presentation.get("required_views", [])
            shared_axis = presentation.get("shared_axis_contract")
            if (
                any(view not in PRESENTATION_VIEWS for view in views)
                or not isinstance(shared_axis, str)
                or not shared_axis
                or not isinstance(presentation.get("semantic_keyframes"), list)
            ):
                violations.append(f"{item.get('case_id')}: malformed presentation contract")
        if not isinstance(item.get("source"), Mapping):
            violations.append(f"{item.get('case_id')}: missing source block")
        selection = item.get("selection")
        if not isinstance(selection, Mapping):
            violations.append(f"{item.get('case_id')}: missing selection block")
        elif (
            not isinstance(selection.get("vector"), Mapping)
            or selection.get("pareto_status") not in PARETO_STATUSES
            or not isinstance(selection.get("coverage_tags"), list)
            or not isinstance(selection.get("included_reason"), str)
            or not isinstance(selection.get("excluded_reason"), str)
        ):
            violations.append(f"{item.get('case_id')}: malformed selection block")
        coverage = item.get("coverage")
        if isinstance(coverage, Mapping):
            selected_topologies.add(str(coverage.get("topology")))
            selected_failure_classes.add(str(coverage.get("failure_class")))
            selected_process_classes.add(str(coverage.get("process_class")))
    ledger_ids = [entry.get("case_id") for entry in ledger if isinstance(entry, Mapping)]
    if len(ledger_ids) != len(set(ledger_ids)):
        violations.append("ledger case_id values must be unique")
    selected_ids = {item.get("case_id") for item in selected if isinstance(item, Mapping)}
    ledger_selected = {
        entry.get("case_id")
        for entry in ledger
        if isinstance(entry, Mapping) and entry.get("disposition") == "selected"
    }
    if selected_ids != ledger_selected:
        violations.append("selected list and selected ledger entries differ")
    ledger_by_id = {entry.get("case_id"): entry for entry in ledger if isinstance(entry, Mapping)}
    for selected_id in selected_ids:
        entry = ledger_by_id.get(selected_id)
        if not isinstance(entry, Mapping):
            continue
        eligibility = entry.get("eligibility")
        if not isinstance(eligibility, Mapping) or eligibility.get("eligible") is not True:
            violations.append(f"{selected_id}: selected record is not eligible")
        if entry.get("pareto_member") is not True:
            violations.append(f"{selected_id}: selected record is not Pareto member")
        selected_item = next(
            (
                item
                for item in selected
                if isinstance(item, Mapping) and item.get("case_id") == selected_id
            ),
            None,
        )
        if isinstance(selected_item, Mapping) and selected_item.get("eligibility") != eligibility:
            violations.append(f"{selected_id}: selected embedded eligibility differs from ledger")
        if isinstance(selected_item, Mapping):
            expected_selected = _selected_record_from_ledger(entry)
            if selected_item != expected_selected:
                violations.append(
                    f"{selected_id}: selected record differs from recomputed ledger source"
                )
    for entry in ledger:
        if not isinstance(entry, Mapping):
            violations.append("ledger contains non-dict item")
            continue
        if entry.get("disposition") == "excluded" and not entry.get("exclusion_reasons"):
            violations.append(f"{entry.get('case_id')}: excluded ledger entry needs a reason")
        eligibility = entry.get("eligibility")
        if not isinstance(eligibility, Mapping) or "eligible" not in eligibility:
            violations.append(f"{entry.get('case_id')}: missing eligibility report")
            continue
        if str(entry.get("case_id", "")).startswith("upstream::"):
            continue
        expected_source = _source_block(
            entry, entry.get("source_boundary", {}), entry.get("source_refs", [])
        )
        if entry.get("source") != expected_source:
            violations.append(f"{entry.get('case_id')}: stale source block")
        expected_eligibility = _expected_eligibility_from_ledger(entry)
        if (
            eligibility.get("status") not in {"admitted", "excluded", "unavailable"}
            or not isinstance(eligibility.get("reasons"), list)
            or eligibility.get("execution_mode") not in {*ADMISSIBLE_EXECUTION_MODES, "unavailable"}
            or eligibility.get("initial_state_match") not in INITIAL_OUTCOME_STATUSES
            or eligibility.get("outcome_match") not in INITIAL_OUTCOME_STATUSES
            or eligibility.get("telemetry_grade") not in TELEMETRY_GRADES
        ):
            violations.append(f"{entry.get('case_id')}: malformed eligibility report")
        blockers = eligibility.get("blockers")
        checks = eligibility.get("checks")
        if not isinstance(blockers, list) or not isinstance(checks, Mapping):
            violations.append(f"{entry.get('case_id')}: malformed eligibility report")
            continue
        if eligibility.get("required_checks") != expected_eligibility["required_checks"]:
            violations.append(f"{entry.get('case_id')}: stale eligibility required_checks")
        expected_checks = expected_eligibility["checks"]
        if set(checks) != set(expected_checks):
            violations.append(f"{entry.get('case_id')}: stale eligibility checks")
        for key in (
            "eligible",
            "status",
            "reasons",
            "execution_mode",
            "typed_outcome_semantics",
            "initial_state_match",
            "initial_state_match_reason",
            "outcome_match",
            "outcome_match_reason",
            "telemetry_grade",
            "telemetry_grade_reason",
            "required_checks",
            "checks",
            "blockers",
        ):
            if eligibility.get(key) != expected_eligibility.get(key):
                violations.append(f"{entry.get('case_id')}: stale eligibility {key}")
        recomputed_blockers = []
        for check, detail in checks.items():
            if not isinstance(detail, Mapping):
                violations.append(f"{entry.get('case_id')}: malformed eligibility check {check}")
                continue
            if detail.get("applicable") is not False and detail.get("status") != "pass":
                recomputed_blockers.append(str(check))
        if bool(recomputed_blockers) == bool(eligibility.get("eligible")):
            violations.append(f"{entry.get('case_id')}: stale eligibility eligible flag")
        blocker_checks = [
            str(blocker.get("check")) for blocker in blockers if isinstance(blocker, Mapping)
        ]
        if sorted(recomputed_blockers) != sorted(blocker_checks):
            violations.append(f"{entry.get('case_id')}: stale eligibility blockers")
        source_check = checks.get("source_hashes")
        if isinstance(source_check, Mapping) and source_check.get("status") == "pass":
            source_failure = _source_hash_failure_from_source(entry)
            if source_failure:
                violations.append(f"{entry.get('case_id')}: stale source_hashes pass")
            if eligibility.get("eligible") is True and source_failure:
                violations.append(f"{entry.get('case_id')}: eligible with unavailable source hash")
    summary = manifest.get("summary")
    constraints = manifest.get("constraints")
    if isinstance(summary, Mapping):
        if summary.get("n_selected") != len(selected):
            violations.append("summary.n_selected does not match selected length")
        if summary.get("n_ledger") != len(ledger):
            violations.append("summary.n_ledger does not match ledger length")
        if summary.get("n_eligible") != sum(
            1
            for entry in ledger
            if isinstance(entry, Mapping)
            and not str(entry.get("case_id", "")).startswith("upstream::")
            and isinstance(entry.get("eligibility"), Mapping)
            and entry["eligibility"].get("eligible") is True
        ):
            violations.append("summary.n_eligible does not match ledger eligibility")
        if summary.get("n_pareto") != sum(
            1
            for entry in ledger
            if isinstance(entry, Mapping)
            and not str(entry.get("case_id", "")).startswith("upstream::")
            and entry.get("pareto_member") is True
        ):
            violations.append("summary.n_pareto does not match ledger Pareto flags")
    if isinstance(constraints, Mapping):
        replay_constraints = _constraints_from_manifest_constraints(constraints)
        roles_unique = len(selected_roles) == len(selected)
        missing_roles = set(constraints.get("required_roles", [])) - selected_roles
        missing_grains = set(constraints.get("required_grains", [])) - selected_grains
        missing_conceptual = (
            set(constraints.get("required_conceptual_grains", [])) - selected_conceptual
        )
        missing_topologies = set(constraints.get("required_topologies", [])) - selected_topologies
        missing_failure = (
            set(constraints.get("required_failure_classes", [])) - selected_failure_classes
        )
        missing_process = (
            set(constraints.get("required_process_classes", [])) - selected_process_classes
        )
        complete_constraints = (
            (not constraints.get("require_unique_primary_roles", True) or roles_unique)
            and not missing_roles
            and not missing_grains
            and not missing_conceptual
            and not missing_topologies
            and not missing_failure
            and not missing_process
        )
        enumeration = manifest.get("exact_enumeration")
        if isinstance(enumeration, Mapping) and (
            enumeration.get("complete_constraints_satisfied") is not complete_constraints
        ):
            violations.append("exact_enumeration.complete_constraints_satisfied is stale")
        if manifest.get("status") == "complete" and not complete_constraints:
            violations.append("complete status missing non-relaxable coverage")
        if manifest.get("status") == "partial" and complete_constraints:
            violations.append("partial status despite complete non-relaxable coverage")
        replay_records = _ledger_replay_records(ledger)
        eligible_records = [
            record
            for record in replay_records
            if isinstance(record.get("eligibility"), Mapping)
            and record["eligibility"].get("eligible") is True
        ]
        if replay_constraints.frozen_role_targets:
            frozen_ids = set(replay_constraints.frozen_role_targets.values())
            eligible_for_pareto = [
                record for record in eligible_records if _case_id(record) in frozen_ids
            ]
        else:
            eligible_for_pareto = eligible_records
        replay_pareto = compute_pareto_membership(eligible_for_pareto)
        pareto_analysis = manifest.get("pareto_analysis")
        if isinstance(pareto_analysis, Mapping):
            if pareto_analysis.get("directions") != replay_pareto["directions"]:
                violations.append("pareto_analysis.directions is stale")
            if pareto_analysis.get("front") != replay_pareto["pareto_front"]:
                violations.append("pareto_analysis.front is stale")
            if pareto_analysis.get("dominated") != replay_pareto["dominated"]:
                violations.append("pareto_analysis.dominated is stale")
            if (
                pareto_analysis.get("dimension_unavailable")
                != replay_pareto["dimension_unavailable"]
            ):
                violations.append("pareto_analysis.dimension_unavailable is stale")
        replay_pareto_ids = set(replay_pareto["pareto_front"])
        for entry in ledger:
            if not isinstance(entry, Mapping) or str(entry.get("case_id", "")).startswith(
                "upstream::"
            ):
                continue
            if entry.get("pareto_member") is not (_case_id(entry) in replay_pareto_ids):
                violations.append(f"{entry.get('case_id')}: stale ledger Pareto membership")
        replay_candidates = [
            record for record in eligible_for_pareto if _case_id(record) in replay_pareto_ids
        ]
        replay_selected, replay_enumeration = _select_exact(replay_candidates, replay_constraints)
        if set(replay_selected) != selected_ids:
            violations.append("selected cases differ from replayed exact selector")
        enumeration = manifest.get("exact_enumeration")
        if isinstance(enumeration, Mapping):
            for key in (
                "enumeration_count",
                "best_score",
                "complete_constraints_satisfied",
                "nonrelaxable_valid_subset",
            ):
                if enumeration.get(key) != replay_enumeration.get(key):
                    violations.append(f"exact_enumeration.{key} is stale")
        if isinstance(summary, Mapping):
            replay_selected_records = [
                record for record in replay_records if _case_id(record) in selected_ids
            ]
            replay_roles = {str(record.get("primary_role")) for record in replay_selected_records}
            replay_grains = {str(record.get("grain")) for record in replay_selected_records}
            replay_conceptual = {
                tag for record in replay_selected_records for tag in _conceptual_coverage(record)
            }
            replay_coverage = [_coverage_values(record) for record in replay_selected_records]
            expected_uncovered = {
                "uncovered_roles": [
                    role for role in replay_constraints.required_roles if role not in replay_roles
                ],
                "uncovered_grains": [
                    grain
                    for grain in replay_constraints.required_grains
                    if grain not in replay_grains
                ],
                "uncovered_conceptual_grains": [
                    tag
                    for tag in replay_constraints.required_conceptual_grains
                    if tag not in replay_conceptual
                ],
                "uncovered_topologies": [
                    value
                    for value in replay_constraints.required_topologies
                    if value not in {row["topology"] for row in replay_coverage}
                ],
                "uncovered_failure_classes": [
                    value
                    for value in replay_constraints.required_failure_classes
                    if value not in {row["failure_class"] for row in replay_coverage}
                ],
                "uncovered_process_classes": [
                    value
                    for value in replay_constraints.required_process_classes
                    if value not in {row["process_class"] for row in replay_coverage}
                ],
            }
            for key, expected_value in expected_uncovered.items():
                if summary.get(key) != expected_value:
                    violations.append(f"summary.{key} is stale")
    if manifest.get("status") == "complete" and isinstance(constraints, Mapping):
        if not set(constraints.get("required_roles", [])).issubset(selected_roles):
            violations.append("complete status missing required role coverage")
        if not set(constraints.get("required_grains", [])).issubset(selected_grains):
            violations.append("complete status missing required grain coverage")
        if not set(constraints.get("required_conceptual_grains", [])).issubset(selected_conceptual):
            violations.append("complete status missing conceptual presentation coverage")
    return PortfolioValidation(violations)
