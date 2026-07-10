"""Typed loader and report generator for ``odd_hazard_coverage.v1`` matrices."""

# ruff: noqa: DOC201

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

ODD_HAZARD_COVERAGE_SCHEMA_VERSION = "odd_hazard_coverage.v1"
ODD_HAZARD_COVERAGE_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "odd_hazard_coverage.v1.json"
)


@dataclass(frozen=True, slots=True)
class ContractRef:
    """Reference to a checked-in contract file and contract id."""

    source: str
    contract_id: str
    required_for_benchmark_claim: bool = False


@dataclass(frozen=True, slots=True)
class CoverageRow:
    """One ODD condition x hazard class x scenario family coverage cell."""

    odd_condition: str
    hazard_class: str
    scenario_family: str
    metrics: list[str]
    included_planners: list[str]
    evidence_tier: str
    status: str
    gap_reason: str = ""
    source_configs: list[str] | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class GapRow:
    """Explicitly uncovered or blocked ODD/hazard coverage gap."""

    gap_id: str
    status: str
    reason: str
    affected_hazard_classes: list[str] | None = None
    affected_scenario_families: list[str] | None = None
    tracking_issue: str = ""


@dataclass(frozen=True, slots=True)
class Provenance:
    """Provenance metadata for a coverage matrix."""

    source_issue: str
    authored_by: str
    source_files: list[str]
    notes: str


@dataclass(frozen=True, slots=True)
class OddHazardCoverageMatrix:
    """Typed ``odd_hazard_coverage.v1`` payload."""

    schema_version: str
    id: str
    claim_boundary: str
    odd_contract_ref: ContractRef
    coverage_rows: list[CoverageRow]
    known_gaps: list[GapRow]
    provenance: Provenance
    hazard_traceability_ref: ContractRef | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the matrix to JSON-safe primitives."""

        payload = asdict(self)
        if self.hazard_traceability_ref is None:
            payload.pop("hazard_traceability_ref")
        return payload


class OddHazardCoverageValidationError(RobotSfError, ValueError):
    """Raised when an ODD hazard coverage matrix fails validation."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""

        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@lru_cache(maxsize=1)
def load_odd_hazard_coverage_schema() -> dict[str, Any]:
    """Load the public ``odd_hazard_coverage.v1`` JSON schema."""

    return json.loads(ODD_HAZARD_COVERAGE_SCHEMA_FILE.read_text(encoding="utf-8"))


def load_odd_hazard_coverage_matrix(path: Path) -> OddHazardCoverageMatrix:
    """Load and validate an ODD hazard coverage matrix from YAML or JSON."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise OddHazardCoverageValidationError(["expected a mapping payload"], source=path)
    return odd_hazard_coverage_matrix_from_dict(raw, source=path)


def odd_hazard_coverage_matrix_from_dict(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> OddHazardCoverageMatrix:
    """Validate and convert a mapping into a typed ODD hazard coverage matrix."""

    errors = _schema_validation_errors(payload)
    errors.extend(_semantic_validation_errors(payload))
    if errors:
        raise OddHazardCoverageValidationError(errors, source=source)
    return _matrix_from_payload(payload)


def validate_matrix_references(
    matrix: OddHazardCoverageMatrix,
    *,
    repo_root: Path = Path("."),
) -> list[str]:
    """Validate that referenced contracts and source configs exist and resolve.

    Returns:
        List of reference errors. An empty list means all references resolved.
    """

    root = repo_root.resolve()
    errors = _validate_contract_refs(matrix, root)
    errors.extend(_validate_odd_conditions(matrix, root))
    errors.extend(_validate_hazard_classes(matrix, root))
    errors.extend(_validate_source_configs(matrix, root))
    errors.extend(_validate_provenance_sources(matrix, root))
    return errors


def _validate_contract_refs(
    matrix: OddHazardCoverageMatrix,
    root: Path,
) -> list[str]:
    """Validate that contract references resolve to existing ids."""

    errors: list[str] = []
    for ref, name in (
        (matrix.odd_contract_ref, "odd_contract_ref"),
        (matrix.hazard_traceability_ref, "hazard_traceability_ref"),
    ):
        if ref is None:
            continue
        path = root / ref.source
        if not path.exists():
            errors.append(f"{name}.source '{ref.source}' does not exist")
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"{name}.source '{ref.source}' could not be loaded: {exc}")
            continue
        ids = _contract_ids(raw)
        if ref.contract_id not in ids:
            errors.append(f"{name}.contract_id '{ref.contract_id}' was not found in {ref.source}")
    return errors


def _validate_hazard_classes(
    matrix: OddHazardCoverageMatrix,
    root: Path,
) -> list[str]:
    """Validate that coverage hazard classes are declared in the traceability mapping."""

    errors: list[str] = []
    if matrix.hazard_traceability_ref is None or not matrix.hazard_traceability_ref.source:
        return errors
    declared_hazards = _hazard_ids_from_traceability(root / matrix.hazard_traceability_ref.source)
    for row in matrix.coverage_rows:
        if row.hazard_class not in declared_hazards:
            errors.append(
                f"coverage_rows hazard_class '{row.hazard_class}' is not declared in "
                f"{matrix.hazard_traceability_ref.source}"
            )
    return errors


def _validate_odd_conditions(
    matrix: OddHazardCoverageMatrix,
    root: Path,
) -> list[str]:
    """Validate that coverage rows use the referenced ODD contract id."""

    path = root / matrix.odd_contract_ref.source
    if not path.exists():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    declared_ids = _contract_ids(raw)
    if not declared_ids:
        return []
    return [
        f"coverage_rows odd_condition '{row.odd_condition}' is not declared in "
        f"{matrix.odd_contract_ref.source}"
        for row in matrix.coverage_rows
        if row.odd_condition not in declared_ids
    ]


def _validate_source_configs(
    matrix: OddHazardCoverageMatrix,
    root: Path,
) -> list[str]:
    """Validate that each coverage row's source config paths exist."""

    errors: list[str] = []
    for index, row in enumerate(matrix.coverage_rows):
        for config in row.source_configs or []:
            if not (root / config).exists():
                errors.append(f"coverage_rows[{index}].source_configs '{config}' does not exist")
    return errors


def _validate_provenance_sources(
    matrix: OddHazardCoverageMatrix,
    root: Path,
) -> list[str]:
    """Validate that provenance source files exist."""

    return [
        f"provenance.source_files '{source}' does not exist"
        for source in matrix.provenance.source_files
        if not (root / source).exists()
    ]


def generate_json_report(
    matrix: OddHazardCoverageMatrix,
    *,
    repo_root: Path = Path("."),
    command: str = "",
) -> dict[str, Any]:
    """Generate a JSON-safe coverage report from a validated matrix."""

    reference_errors = validate_matrix_references(matrix, repo_root=repo_root)
    status_counts = _status_counts(matrix.coverage_rows)
    gap_status_counts = _gap_status_counts(matrix.known_gaps)
    generation = _generation_provenance(repo_root=repo_root)
    generation["command"] = command
    return {
        "schema_version": ODD_HAZARD_COVERAGE_SCHEMA_VERSION,
        "matrix_id": matrix.id,
        "generated_at_utc": _utc_now(),
        "claim_boundary": matrix.claim_boundary,
        "odd_contract_ref": asdict(matrix.odd_contract_ref),
        "hazard_traceability_ref": (
            asdict(matrix.hazard_traceability_ref) if matrix.hazard_traceability_ref else None
        ),
        "summary": {
            "coverage_row_count": len(matrix.coverage_rows),
            "status_counts": status_counts,
            "known_gap_count": len(matrix.known_gaps),
            "gap_status_counts": gap_status_counts,
            "reference_errors": reference_errors,
            "reference_valid": not reference_errors,
        },
        "coverage_rows": [_coverage_row_to_dict(row) for row in matrix.coverage_rows],
        "known_gaps": [_gap_row_to_dict(row) for row in matrix.known_gaps],
        "provenance": asdict(matrix.provenance),
        "generation": generation,
    }


def generate_markdown_report(
    matrix: OddHazardCoverageMatrix,
    *,
    repo_root: Path = Path("."),
    command: str = "",
) -> str:
    """Generate a Markdown coverage report from a validated matrix."""

    reference_errors = validate_matrix_references(matrix, repo_root=repo_root)
    status_counts = _status_counts(matrix.coverage_rows)
    gap_status_counts = _gap_status_counts(matrix.known_gaps)
    generation = _generation_provenance(repo_root=repo_root)
    generation["command"] = command

    lines: list[str] = [
        "# ODD Hazard Coverage Matrix",
        "",
        f"- Matrix id: `{matrix.id}`",
        f"- Generated at: {_utc_now()}",
        f"- Generation command: `{command or 'not recorded'}`",
        f"- Source/base commit: `{generation['base_commit']}`",
        f"- Worktree state: `{generation['tree_state']}`",
        "",
        "## Claim Boundary",
        "",
        matrix.claim_boundary,
        "",
        "## Evidence Status Summary",
        "",
        f"- Coverage rows: {len(matrix.coverage_rows)}",
        f"- Known gaps: {len(matrix.known_gaps)}",
        f"- Coverage status counts: {_format_counts(status_counts)}",
        f"- Gap status counts: {_format_counts(gap_status_counts)}",
        f"- Reference validation: {'passed' if not reference_errors else 'failed'}",
        "",
    ]

    if reference_errors:
        lines.extend(
            [
                "### Reference Validation Errors",
                "",
                *["- " + error for error in reference_errors],
                "",
            ]
        )

    lines.extend(
        [
            "## Coverage Rows",
            "",
            "| ODD condition | Hazard class | Scenario family | Status | Evidence tier | Metrics | Planners | Gap reason |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in matrix.coverage_rows:
        metrics = "; ".join(row.metrics)
        planners = "; ".join(row.included_planners) if row.included_planners else "none"
        gap_reason = row.gap_reason.replace("\n", " ") if row.gap_reason else "—"
        lines.append(
            f"| {row.odd_condition} | {row.hazard_class} | {row.scenario_family} | "
            f"{row.status} | {row.evidence_tier} | {metrics} | {planners} | {gap_reason} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Known Gaps",
            "",
            "| Gap id | Status | Affected hazard classes | Affected scenario families | Reason | Tracking issue |",
            "|---|---|---|---|---|---|",
        ]
    )
    for gap in matrix.known_gaps:
        hazards = "; ".join(gap.affected_hazard_classes or ["—"])
        families = "; ".join(gap.affected_scenario_families or ["—"])
        reason = gap.reason.replace("\n", " ")
        issue = gap.tracking_issue or "—"
        lines.append(
            f"| {gap.gap_id} | {gap.status} | {hazards} | {families} | {reason} | {issue} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Benchmark Wording Guard",
            "",
            "- Rows with status `covered` may be described as represented by checked-in configs only.",
            "- Rows with status `weakly_covered` must be described as config-only, candidate, or diagnostic.",
            "- Rows with status `blocked` or `absent` must not be described as covered, benchmark-success, or paper-facing evidence.",
            "- This report is schema/proposal evidence unless a separate benchmark run updates the evidence tier.",
            "",
            "## Provenance",
            "",
            f"- Source issue: {matrix.provenance.source_issue}",
            f"- Authored by: {matrix.provenance.authored_by}",
            "- Source files:",
            *[f"  - {path}" for path in matrix.provenance.source_files],
            "",
            matrix.provenance.notes,
            "",
        ]
    )
    return "\n".join(lines)


def _matrix_from_payload(payload: Mapping[str, Any]) -> OddHazardCoverageMatrix:
    """Build a typed matrix from a schema-valid payload."""

    return OddHazardCoverageMatrix(
        schema_version=str(payload["schema_version"]),
        id=str(payload["id"]),
        claim_boundary=str(payload["claim_boundary"]),
        odd_contract_ref=_contract_ref_from_payload(payload["odd_contract_ref"]),
        hazard_traceability_ref=_optional_contract_ref(payload.get("hazard_traceability_ref")),
        coverage_rows=[_coverage_row_from_payload(row) for row in payload["coverage_rows"]],
        known_gaps=[_gap_row_from_payload(row) for row in payload["known_gaps"]],
        provenance=_provenance_from_payload(payload["provenance"]),
    )


def _contract_ref_from_payload(payload: Mapping[str, Any]) -> ContractRef:
    """Build a typed contract reference."""

    return ContractRef(
        source=str(payload["source"]),
        contract_id=str(payload["contract_id"]),
        required_for_benchmark_claim=bool(payload.get("required_for_benchmark_claim", False)),
    )


def _optional_contract_ref(payload: Any) -> ContractRef | None:
    """Build a typed contract reference when present."""

    if payload is None:
        return None
    return _contract_ref_from_payload(payload)


def _coverage_row_from_payload(payload: Mapping[str, Any]) -> CoverageRow:
    """Build a typed coverage row."""

    return CoverageRow(
        odd_condition=str(payload["odd_condition"]),
        hazard_class=str(payload["hazard_class"]),
        scenario_family=str(payload["scenario_family"]),
        metrics=list(payload["metrics"]),
        included_planners=list(payload.get("included_planners", [])),
        evidence_tier=str(payload["evidence_tier"]),
        status=str(payload["status"]),
        gap_reason=str(payload.get("gap_reason", "")),
        source_configs=list(payload.get("source_configs", [])) or None,
        notes=str(payload.get("notes", "")),
    )


def _gap_row_from_payload(payload: Mapping[str, Any]) -> GapRow:
    """Build a typed gap row."""

    return GapRow(
        gap_id=str(payload["gap_id"]),
        status=str(payload["status"]),
        reason=str(payload["reason"]),
        affected_hazard_classes=list(payload.get("affected_hazard_classes", [])) or None,
        affected_scenario_families=list(payload.get("affected_scenario_families", [])) or None,
        tracking_issue=str(payload.get("tracking_issue", "")),
    )


def _provenance_from_payload(payload: Mapping[str, Any]) -> Provenance:
    """Build typed provenance metadata."""

    return Provenance(
        source_issue=str(payload["source_issue"]),
        authored_by=str(payload["authored_by"]),
        source_files=list(payload["source_files"]),
        notes=str(payload["notes"]),
    )


def _schema_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return sorted JSON Schema validation errors for one matrix payload."""

    validator = Draft202012Validator(load_odd_hazard_coverage_schema())
    return [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]


def _semantic_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return cross-field validation errors not expressible in the JSON Schema."""

    errors: list[str] = []
    for index, row in enumerate(payload.get("coverage_rows", [])):
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status", ""))
        gap_reason = str(row.get("gap_reason", "")).strip()
        if status != "covered" and not gap_reason:
            errors.append(f"/coverage_rows/{index}/gap_reason: required when status is '{status}'")
    return errors


def _contract_ids(raw: Any) -> set[str]:
    """Extract contract ids from a contract file payload."""

    ids: set[str] = set()
    if not isinstance(raw, Mapping):
        return ids
    contracts = raw.get("contracts", raw)
    if isinstance(contracts, Mapping):
        contracts = [contracts]
    if not isinstance(contracts, list):
        return ids
    for contract in contracts:
        if isinstance(contract, Mapping):
            contract_id = contract.get("id")
            if isinstance(contract_id, str):
                ids.add(contract_id)
    return ids


def _hazard_ids_from_traceability(path: Path) -> set[str]:
    """Extract declared hazard ids from a hazard traceability file."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return set()
    if not isinstance(raw, Mapping):
        return set()
    hazards = raw.get("hazards", [])
    if not isinstance(hazards, list):
        return set()
    return {
        str(hazard["id"])
        for hazard in hazards
        if isinstance(hazard, Mapping) and isinstance(hazard.get("id"), str)
    }


def _coverage_row_to_dict(row: CoverageRow) -> dict[str, Any]:
    """Convert a coverage row to a JSON-safe dictionary."""

    payload = asdict(row)
    if not row.source_configs:
        payload.pop("source_configs")
    if not row.notes:
        payload.pop("notes")
    if not row.gap_reason:
        payload.pop("gap_reason")
    return payload


def _gap_row_to_dict(row: GapRow) -> dict[str, Any]:
    """Convert a gap row to a JSON-safe dictionary."""

    payload = asdict(row)
    if not row.affected_hazard_classes:
        payload.pop("affected_hazard_classes")
    if not row.affected_scenario_families:
        payload.pop("affected_scenario_families")
    if not row.tracking_issue:
        payload.pop("tracking_issue")
    return payload


def _status_counts(rows: Sequence[CoverageRow]) -> dict[str, int]:
    """Count coverage row statuses."""

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def _gap_status_counts(gaps: Sequence[GapRow]) -> dict[str, int]:
    """Count gap row statuses."""

    counts: dict[str, int] = {}
    for gap in gaps:
        counts[gap.status] = counts.get(gap.status, 0) + 1
    return counts


def _format_counts(counts: dict[str, int]) -> str:
    """Render a status count dictionary for Markdown."""

    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"


def _utc_now() -> str:
    """Return the current UTC timestamp as an ISO string."""

    return datetime.now(UTC).isoformat()


def _git_commit(*, repo_root: Path) -> str:
    """Return the current Git commit, or ``unknown`` outside Git."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _git_tree_state(*, repo_root: Path) -> str:
    """Return the current git tree state, or ``unknown`` outside Git."""

    try:
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--short", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return "dirty" if status.strip() else "clean"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _generation_provenance(*, repo_root: Path) -> dict[str, str]:
    """Return explicit generation provenance metadata."""

    return {
        "base_commit": _git_commit(repo_root=repo_root),
        "tree_state": _git_tree_state(repo_root=repo_root),
    }


__all__ = [
    "ODD_HAZARD_COVERAGE_SCHEMA_FILE",
    "ODD_HAZARD_COVERAGE_SCHEMA_VERSION",
    "ContractRef",
    "CoverageRow",
    "GapRow",
    "OddHazardCoverageMatrix",
    "OddHazardCoverageValidationError",
    "Provenance",
    "generate_json_report",
    "generate_markdown_report",
    "load_odd_hazard_coverage_matrix",
    "load_odd_hazard_coverage_schema",
    "odd_hazard_coverage_matrix_from_dict",
    "validate_matrix_references",
]
