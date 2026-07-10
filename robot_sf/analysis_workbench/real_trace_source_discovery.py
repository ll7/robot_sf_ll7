"""Public-source discovery ledger for issue #3278 real micromobility traces.

The checker records which public sources were reviewed for late-evasive and
oscillatory validation data. It does not download, copy, or inspect raw
external traces. Its only claim is whether the metadata ledger names at least
one usable source for each required validation target.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

REAL_TRACE_SOURCE_DISCOVERY_SCHEMA_VERSION = "real_trace_source_discovery.v1"
SOURCE_DISCOVERY_EVIDENCE_BOUNDARY = "public_source_discovery_only_no_real_world_validation"
SOURCE_DISCOVERY_STATUS_READY = "ready"
SOURCE_DISCOVERY_STATUS_BLOCKED = "blocked"

DEFAULT_REQUIRED_TARGETS = (
    "late_evasive_reaction",
    "oscillatory_local_control",
)

USABLE_ACCESS_STATUSES = {"available"}
USABLE_LICENSE_STATUSES = {"accepted", "permissive"}
USABLE_COVERAGE_STATUSES = {"direct"}

SCHEMA_FILE = Path(__file__).resolve().parent / "schemas" / "real_trace_source_discovery.v1.json"


class RealTraceSourceDiscoveryError(RobotSfError, ValueError):
    """Raised when a real-trace source discovery ledger is malformed."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""
        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class SourceCandidateReport:
    """Compatibility status for one public-source candidate."""

    source_id: str
    title: str
    status: str
    matched_targets: tuple[str, ...]
    blockers: tuple[str, ...] = field(default_factory=tuple)
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class RealTraceSourceDiscoveryReport:
    """Fail-closed report for issue #3278 public-source discovery."""

    schema_version: str
    discovery_status: str
    evidence_boundary: str
    required_targets: tuple[str, ...]
    ready_targets: tuple[str, ...]
    blocked_targets: tuple[str, ...]
    blockers: tuple[str, ...]
    source_reports: tuple[SourceCandidateReport, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize report as stable JSON.

        Returns:
            JSON string suitable for CLI output.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@lru_cache(maxsize=1)
def load_real_trace_source_discovery_schema() -> dict[str, Any]:
    """Load the JSON schema for ``real_trace_source_discovery.v1`` ledgers.

    Returns:
        Parsed JSON schema mapping.
    """
    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))


def load_real_trace_source_discovery(path: str | Path) -> dict[str, Any]:
    """Load and schema-check a YAML or JSON source-discovery ledger.

    Returns:
        Parsed ledger mapping.
    """
    ledger_path = Path(path)
    if not ledger_path.is_file():
        raise RealTraceSourceDiscoveryError(["ledger file not found"], source=ledger_path)

    try:
        payload = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise RealTraceSourceDiscoveryError(
            [f"invalid YAML/JSON: {exc}"], source=ledger_path
        ) from exc

    if not isinstance(payload, Mapping):
        raise RealTraceSourceDiscoveryError(["expected a mapping payload"], source=ledger_path)

    _raise_on_schema_errors(payload, source=ledger_path)
    return dict(payload)


def check_real_trace_source_discovery(
    ledger: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> RealTraceSourceDiscoveryReport:
    """Check whether public-source discovery has usable issue #3278 inputs.

    A target is ready only when at least one candidate source has accepted or
    permissive licensing, available access, and direct coverage for that target.
    All other states remain blocked external-input evidence.

    Returns:
        Structured fail-closed source discovery report.
    """
    _raise_on_schema_errors(ledger, source=source)

    required_targets = tuple(ledger.get("required_targets", DEFAULT_REQUIRED_TARGETS))
    source_reports = tuple(
        _candidate_report(candidate, required_targets)
        for candidate in ledger.get("candidate_sources", [])
    )
    ready_targets = tuple(
        target
        for target in required_targets
        if any(
            report.status == SOURCE_DISCOVERY_STATUS_READY and target in report.matched_targets
            for report in source_reports
        )
    )
    blocked_targets = tuple(
        target for target in required_targets if target not in set(ready_targets)
    )

    blockers = list(_ledger_blockers(ledger))
    blockers.extend(
        f"required target {target!r} has no usable public source" for target in blocked_targets
    )

    discovery_status = (
        SOURCE_DISCOVERY_STATUS_READY
        if not blockers and not blocked_targets
        else SOURCE_DISCOVERY_STATUS_BLOCKED
    )

    return RealTraceSourceDiscoveryReport(
        schema_version=REAL_TRACE_SOURCE_DISCOVERY_SCHEMA_VERSION,
        discovery_status=discovery_status,
        evidence_boundary=SOURCE_DISCOVERY_EVIDENCE_BOUNDARY,
        required_targets=required_targets,
        ready_targets=ready_targets,
        blocked_targets=blocked_targets,
        blockers=tuple(blockers),
        source_reports=source_reports,
    )


def _raise_on_schema_errors(payload: Mapping[str, Any], *, source: str | Path | None) -> None:
    """Raise when a ledger violates the JSON schema."""
    validator = Draft202012Validator(load_real_trace_source_discovery_schema())
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]
    if errors:
        raise RealTraceSourceDiscoveryError(errors, source=source)


def _candidate_report(
    candidate: Mapping[str, Any],
    required_targets: Sequence[str],
) -> SourceCandidateReport:
    """Classify one candidate source against required targets.

    Returns:
        Candidate-level status and blockers.
    """
    matched_targets = tuple(
        target for target in candidate.get("covered_targets", []) if target in required_targets
    )
    blockers = list(_candidate_blockers(candidate))
    missing_targets = [
        target for target in candidate.get("covered_targets", []) if target not in required_targets
    ]
    if missing_targets:
        blockers.append(
            "covered_targets includes unsupported target(s): " + ", ".join(sorted(missing_targets))
        )

    status = (
        SOURCE_DISCOVERY_STATUS_READY
        if matched_targets and not blockers
        else SOURCE_DISCOVERY_STATUS_BLOCKED
    )

    return SourceCandidateReport(
        source_id=str(candidate["source_id"]),
        title=str(candidate["title"]),
        status=status,
        matched_targets=matched_targets,
        blockers=tuple(blockers),
        notes=candidate.get("notes"),
    )


def _ledger_blockers(ledger: Mapping[str, Any]) -> tuple[str, ...]:
    """Return ledger-level blockers independent of individual sources.

    Returns:
        Human-readable blocker strings.
    """
    blockers: list[str] = []
    discovery_status = ledger.get("discovery_status")
    if discovery_status != "complete":
        blockers.append(f"discovery_status is {discovery_status!r}, expected 'complete'")
    if not ledger.get("candidate_sources"):
        blockers.append("candidate_sources is empty")
    return tuple(blockers)


def _candidate_blockers(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """Return candidate-level blockers for source usability.

    Returns:
        Human-readable blocker strings.
    """
    blockers: list[str] = []
    access_status = candidate.get("access_status")
    if access_status not in USABLE_ACCESS_STATUSES:
        blockers.append(f"access_status is {access_status!r}, expected 'available'")

    license_status = candidate.get("license_status")
    if license_status not in USABLE_LICENSE_STATUSES:
        blockers.append(f"license_status is {license_status!r}, expected accepted/permissive")

    coverage_status = candidate.get("coverage_status")
    if coverage_status not in USABLE_COVERAGE_STATUSES:
        blockers.append(f"coverage_status is {coverage_status!r}, expected 'direct'")

    if not candidate.get("covered_targets"):
        blockers.append("covered_targets is empty")
    return tuple(blockers)
