"""Staging-manifest preflight for real AMV command-response traces (issue #2415).

Issue #2415 asks to stage *real* AMV command-response actuation traces so the
synthetic actuation envelope (``robot_sf/benchmark/synthetic_actuation.py``) can
eventually be calibrated against measured command-response dynamics. That
calibration is **blocked on external data**: per the maintainer decision on
#2415 (2026-06-22) no realistic real-data source is currently available
(estimated < 5% feasibility), and Robot SF never ingests or redistributes raw
command-response traces.

This module fills the *local, buildable* half: a metadata-only manifest that
declares, per candidate trace bundle, what staging through the ``amv-calibration``
external-data path (``scripts/tools/manage_external_data.py``) would have to look
like before a command-response calibration can ingest it. It checks:

* **provenance / license** -- source URL, license, license status, citation;
* **command/response/timing channels** -- a command-response trace must declare
  at least one command channel, one response channel, and the timing/latency
  fields a calibration needs (schema-enforced ``minItems``);
* **calibration targets** -- the canonical synthetic-actuation envelope fields a
  calibrated trace would inform, validated against the synthetic-actuation
  vocabulary (``synthetic_actuation.actuation_variability_fields()``) so the
  manifest cannot silently drift from what calibration can actually consume;
* **explicit external-data blockers** -- a ``blocked-external-input`` trace must
  name the staging issue(s) blocking it;
* **declared-vs-live staging reconciliation** -- when a live staging probe is
  supplied (e.g. ``manage_external_data.check_asset``), a trace declared
  ``staged`` whose files are not actually present fails closed.

It deliberately does **not** ingest any trace bundle, read raw command-response
samples, run a calibration, or assert real-world / hardware-calibrated realism.
``evidence_boundary`` on every report is
:data:`AMV_TRACE_MANIFEST_EVIDENCE_BOUNDARY`, and a calibration-ingest is only
permitted once at least one trace bundle is staged and manifest-clean.

The canonical calibration-target vocabulary lives with the synthetic actuation
envelope (``robot_sf.benchmark.synthetic_actuation.actuation_variability_fields``).
Callers pass it in via ``allowed_calibration_targets`` so this module stays free
of a benchmark import at definition time; the CLI wires the two together.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

AMV_TRACE_MANIFEST_SCHEMA_VERSION = "amv_command_response_trace_manifest.v1"
AMV_TRACE_MANIFEST_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "amv_command_response_trace_manifest.v1.json"
)

#: Explicit boundary stamped on every report so a passing manifest preflight is
#: never mistaken for a staged trace bundle, an executed calibration, or a
#: hardware-calibrated realism claim.
AMV_TRACE_MANIFEST_EVIDENCE_BOUNDARY = (
    "manifest_contract_only_no_trace_ingest_no_calibration_run_no_calibrated_claim"
)

# Declared per-trace staging states (must match the JSON schema enum).
STAGING_STATUS_STAGED = "staged"
STAGING_STATUS_MISSING = "missing"
STAGING_STATUS_BLOCKED = "blocked-external-input"

# Resolved overall manifest states.
MANIFEST_STATUS_READY = "ready"
MANIFEST_STATUS_BLOCKED_EXTERNAL = "blocked-external-input"
MANIFEST_STATUS_INVALID = "invalid"

#: Live staging-probe statuses that count as "the declared files are actually
#: present" (matches ``manage_external_data.check_asset`` ``status`` values).
_LIVE_AVAILABLE_STATUSES = frozenset({"available", "staged"})


class AmvTraceManifestError(RobotSfError, ValueError):
    """Raised when an AMV command-response trace manifest fails schema checks."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error from schema messages."""
        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class TraceStagingReport:
    """Per-trace result of checking one command-response trace staging entry."""

    trace_id: str
    asset_id: str | None
    declared_staging_status: str
    live_staging_status: str | None
    effective_staged: bool
    calibration_ready: bool
    command_channels: list[str]
    response_channels: list[str]
    timing_fields: list[str]
    calibration_targets: list[str]
    unknown_calibration_targets: list[str]
    blockers: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AmvTraceManifestReport:
    """Aggregate result of checking an AMV command-response trace manifest."""

    schema_version: str
    manifest_id: str
    issue: int
    evidence_boundary: str
    manifest_status: str
    calibration_ingest_allowed: bool
    calibration_ready_traces: list[str]
    traces: list[TraceStagingReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "issue": self.issue,
            "evidence_boundary": self.evidence_boundary,
            "manifest_status": self.manifest_status,
            "calibration_ingest_allowed": self.calibration_ingest_allowed,
            "calibration_ready_traces": list(self.calibration_ready_traces),
            "traces": [trace.to_dict() for trace in self.traces],
        }

    @property
    def blockers(self) -> list[str]:
        """Return all per-trace blockers, trace-prefixed and sorted."""
        return sorted(
            f"{trace.trace_id}: {blocker}" for trace in self.traces for blocker in trace.blockers
        )


@lru_cache(maxsize=1)
def load_amv_trace_manifest_schema() -> dict[str, Any]:
    """Load the manifest JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """
    return json.loads(AMV_TRACE_MANIFEST_SCHEMA_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _manifest_validator() -> Draft202012Validator:
    """Return a cached schema validator (schema compilation is reused)."""
    return Draft202012Validator(load_amv_trace_manifest_schema())


def _raise_on_schema_errors(payload: Mapping[str, Any], *, source: str | Path | None) -> None:
    """Raise :class:`AmvTraceManifestError` if the payload is invalid."""
    validator = _manifest_validator()
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    ]
    if errors:
        raise AmvTraceManifestError(errors, source=source)


def load_amv_trace_manifest(path: str | Path) -> dict[str, Any]:
    """Load and schema-validate a trace manifest from JSON or YAML.

    Returns:
        The validated manifest mapping.

    Raises:
        AmvTraceManifestError: when the file is missing or invalid.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise AmvTraceManifestError(["manifest file not found"], source=manifest_path)
    text = manifest_path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise AmvTraceManifestError([f"invalid YAML/JSON: {exc}"], source=manifest_path) from exc
    if not isinstance(payload, Mapping):
        raise AmvTraceManifestError(["expected a mapping payload"], source=manifest_path)
    _raise_on_schema_errors(payload, source=manifest_path)
    return dict(payload)


def check_amv_trace_manifest(
    manifest: Mapping[str, Any],
    *,
    allowed_calibration_targets: set[str] | None = None,
    live_staging_status: Mapping[str, str] | None = None,
    source: str | Path | None = None,
) -> AmvTraceManifestReport:
    """Check an AMV command-response trace staging manifest.

    Args:
        manifest: An ``amv_command_response_trace_manifest.v1`` mapping. Schema-validated here.
        allowed_calibration_targets: Canonical synthetic-actuation envelope field names a
            calibrated trace would inform (e.g.
            ``set(synthetic_actuation.actuation_variability_fields())``). When provided, any
            declared calibration target outside this set is a fail-closed blocker so the manifest
            cannot drift from what calibration can actually consume. When ``None``, the
            calibration-target vocabulary check is skipped.
        live_staging_status: Optional mapping of ``asset_id`` to a live staging status (e.g.
            ``manage_external_data.check_asset(...)["status"]``). Used to reconcile a declared
            ``staged`` trace against whether its files are actually present.
        source: Optional source path for error messages.

    Returns:
        A structured manifest report. The report never asserts a staged trace bundle, an executed
        calibration, or real-world realism; ``calibration_ingest_allowed`` is only True when at
        least one trace is staged (declared and, if probed, live) and manifest-clean.

    Raises:
        AmvTraceManifestError: when the manifest violates the schema.
    """
    _raise_on_schema_errors(manifest, source=source)

    trace_reports = [
        _check_trace(
            trace,
            allowed_calibration_targets=allowed_calibration_targets,
            live_staging_status=live_staging_status,
        )
        for trace in manifest["traces"]
    ]

    calibration_ready = sorted(t.trace_id for t in trace_reports if t.calibration_ready)
    any_blockers = any(t.blockers for t in trace_reports)
    if any_blockers:
        manifest_status = MANIFEST_STATUS_INVALID
        ingest_allowed = False
    elif calibration_ready:
        manifest_status = MANIFEST_STATUS_READY
        ingest_allowed = True
    else:
        # A well-formed manifest with no staged trace is the expected state for
        # #2415 today: report blocked-external-input rather than substituting a
        # synthetic stand-in (acceptance / stop rule).
        manifest_status = MANIFEST_STATUS_BLOCKED_EXTERNAL
        ingest_allowed = False

    return AmvTraceManifestReport(
        schema_version=AMV_TRACE_MANIFEST_SCHEMA_VERSION,
        manifest_id=str(manifest["manifest_id"]),
        issue=int(manifest["issue"]),
        evidence_boundary=AMV_TRACE_MANIFEST_EVIDENCE_BOUNDARY,
        manifest_status=manifest_status,
        calibration_ingest_allowed=ingest_allowed,
        calibration_ready_traces=calibration_ready,
        traces=trace_reports,
    )


def _check_trace(
    trace: Mapping[str, Any],
    *,
    allowed_calibration_targets: set[str] | None,
    live_staging_status: Mapping[str, str] | None,
) -> TraceStagingReport:
    """Check one trace staging entry and return its report.

    Returns:
        A :class:`TraceStagingReport`. ``calibration_ready`` is True only for a blocker-free trace
        that is declared ``staged`` and (when probed) live present.
    """
    trace_id = str(trace["trace_id"])
    asset_id = trace.get("asset_id")
    declared_status = str(trace["staging_status"])
    command_channels = [str(value) for value in trace["command_channels"]]
    response_channels = [str(value) for value in trace["response_channels"]]
    timing_fields = [str(value) for value in trace["timing_fields"]]
    calibration_targets = [str(value) for value in trace["calibration_targets"]]
    blockers: list[str] = []

    unknown_targets = _unknown_calibration_targets(calibration_targets, allowed_calibration_targets)
    for unknown in unknown_targets:
        blockers.append(
            f"calibration target {unknown!r} is not a canonical synthetic-actuation envelope field"
        )

    # An explicit external-data blocker must name the staging issue(s) holding it.
    if declared_status == STAGING_STATUS_BLOCKED and not trace.get("blocker_issues"):
        blockers.append(
            "blocked-external-input trace must name at least one blocker issue in blocker_issues"
        )

    live_status = _resolve_live_status(asset_id, live_staging_status)
    effective_staged = declared_status == STAGING_STATUS_STAGED
    if effective_staged and live_status is not None and live_status not in _LIVE_AVAILABLE_STATUSES:
        # Fail closed: a manifest that claims staged but whose files are absent
        # would let a command-response calibration run on nothing.
        blockers.append(
            f"declared staging_status 'staged' but live probe reports {live_status!r}; "
            "stage and validate the trace bundle before claiming staged"
        )
        effective_staged = False

    calibration_ready = effective_staged and not blockers

    return TraceStagingReport(
        trace_id=trace_id,
        asset_id=str(asset_id) if asset_id is not None else None,
        declared_staging_status=declared_status,
        live_staging_status=live_status,
        effective_staged=effective_staged,
        calibration_ready=calibration_ready,
        command_channels=command_channels,
        response_channels=response_channels,
        timing_fields=timing_fields,
        calibration_targets=calibration_targets,
        unknown_calibration_targets=unknown_targets,
        blockers=sorted(blockers),
    )


def _unknown_calibration_targets(
    declared_targets: list[str], allowed_calibration_targets: set[str] | None
) -> list[str]:
    """Return declared calibration targets outside the canonical envelope vocabulary.

    Returns:
        Sorted unknown target names, or an empty list when no vocabulary is supplied.
    """
    if allowed_calibration_targets is None:
        return []
    return sorted({t for t in declared_targets if t not in allowed_calibration_targets})


def _resolve_live_status(
    asset_id: Any, live_staging_status: Mapping[str, str] | None
) -> str | None:
    """Return the live staging status for an asset id, or ``None`` when not probed."""
    if live_staging_status is None or asset_id is None:
        return None
    status = live_staging_status.get(str(asset_id))
    return str(status) if status is not None else None
