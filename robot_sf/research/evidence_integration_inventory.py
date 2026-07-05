#!/usr/bin/env python3
"""Evidence-stream integration contract inventory (design-stage, non-calibrated).

This module is the canonical owner for the *contract inventory* slice of issue #3293
("design evidence integration between simulation and real-world AMV data"). It enumerates
the evidence streams Robot SF may eventually integrate and the provenance + uncertainty
fields each stream must carry **before** it can be linked to a scenario, model, or claim.

Scope and boundaries (read before extending):

- This is a *presence-only* contract inventory. ``check_stream_metadata`` verifies that the
  required provenance/uncertainty keys are present on a synthetic metadata record. It does
  **not** ingest real data, validate field *values*, compute calibration, weight evidence,
  or make any safety claim.
- Per the maintainer decision on issue #3293 (2026-06-22), real AMV actuation data is not
  realistically available (<5% feasibility) and implementation that depends on it is
  hard-blocked. Streams that need external data are marked ``blocked_external`` with an
  explicit ``blocked_until`` unblock condition. Any synthetic envelope built on this
  inventory stays ``non_calibrated`` and must be labeled as such.
- Separating calibration, benchmark, and operational evidence (the ``EvidenceCategory``
  axis) is a hard requirement of the issue: these categories use different denominators and
  must not be mixed.

Companions: ``scripts/tools/check_evidence_integration_inventory.py`` (thin CLI) and
``docs/context/issue_3293_evidence_integration_contract_inventory.md`` (design note).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "BASE_PROVENANCE_FIELDS",
    "BASE_UNCERTAINTY_FIELDS",
    "EVIDENCE_STREAMS",
    "EvidenceCategory",
    "EvidenceStreamSpec",
    "FeasibilityStatus",
    "StreamCheckResult",
    "build_integration_report",
    "check_stream_metadata",
    "get_stream",
    "list_streams",
]


class EvidenceCategory(StrEnum):
    """Evidence category, defined by which denominator and claim the stream supports.

    These categories must never be merged into a single pooled denominator: a calibration
    residual is not a benchmark success rate and neither is an operational incident rate.
    """

    CALIBRATION = "calibration"
    """Evidence that adjusts or checks a model against measured reality (e.g. AMV actuation
    command-response, sim-vs-real trajectory comparison). Denominator = matched samples."""

    BENCHMARK = "benchmark"
    """Evidence from controlled, reproducible simulation campaigns under a frozen contract.
    Denominator = scenario/seed episodes."""

    OPERATIONAL = "operational"
    """Evidence from real deployment / pilot / fleet operation. Denominator = operating
    time, missions, or interventions; rarely directly comparable to benchmark episodes."""


class FeasibilityStatus(StrEnum):
    """Whether the stream can be integrated now or is blocked on external access."""

    FEASIBLE_NOW = "feasible_now"
    """Can be produced from in-repo simulation/tooling without external data access."""

    PARTIAL_EXTERNAL = "partial_external"
    """A non-calibrated synthetic or already-public-dataset slice is feasible now, but the
    calibration-grade integration still needs external access."""

    BLOCKED_EXTERNAL = "blocked_external"
    """Requires external data, asset, license, or field-measurement access not in repo."""


# Base provenance fields required of *every* evidence stream before it can be linked to a
# scenario, model, or claim. Stream-specific extras are appended in each EvidenceStreamSpec.
BASE_PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_id",  # stable identifier for the originating dataset/run/measurement
    "collection_method",  # how the evidence was produced (sim run, sensor log, manual, ...)
    "license_or_access",  # license / access note; required even for in-repo simulation
    "commit_or_version",  # repo commit, dataset version, or firmware/vehicle version
    "denominator",  # what the rate/metric is over (episodes, samples, op-hours, ...)
    "scenario_link",  # how the stream maps to scenarios/models/claims (id or "unmapped")
)
"""Provenance keys every stream must carry. See module docstring for the no-mixing rule."""

# Base uncertainty fields required of every stream. ``calibration_status`` is mandatory so a
# synthetic or proxy envelope can never silently pass as calibrated evidence.
BASE_UNCERTAINTY_FIELDS: tuple[str, ...] = (
    "uncertainty_basis",  # how uncertainty is characterized (CI, bootstrap, none, ...)
    "sample_size",  # number of independent units behind the estimate
    "calibration_status",  # one of: calibrated | non_calibrated | not_applicable
)
"""Uncertainty keys every stream must carry. Presence-only here; values are not judged."""


@dataclass(frozen=True)
class EvidenceStreamSpec:
    """Contract for a single integratable evidence stream.

    The field tuples are *requirements*, not data. This class describes what a record from
    the stream must declare; it does not hold or validate the record's values.
    """

    stream_id: str
    title: str
    category: EvidenceCategory
    feasibility: FeasibilityStatus
    description: str
    extra_provenance_fields: tuple[str, ...] = ()
    extra_uncertainty_fields: tuple[str, ...] = ()
    blocked_until: str | None = None
    related_issues: tuple[int, ...] = ()

    @property
    def required_provenance_fields(self) -> tuple[str, ...]:
        """Full provenance contract: base fields plus stream-specific extras."""
        return BASE_PROVENANCE_FIELDS + self.extra_provenance_fields

    @property
    def required_uncertainty_fields(self) -> tuple[str, ...]:
        """Full uncertainty contract: base fields plus stream-specific extras."""
        return BASE_UNCERTAINTY_FIELDS + self.extra_uncertainty_fields

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the contract for CLI/report output."""
        return {
            "stream_id": self.stream_id,
            "title": self.title,
            "category": self.category.value,
            "feasibility": self.feasibility.value,
            "description": self.description,
            "required_provenance_fields": list(self.required_provenance_fields),
            "required_uncertainty_fields": list(self.required_uncertainty_fields),
            "blocked_until": self.blocked_until,
            "related_issues": list(self.related_issues),
        }


# Canonical, ordered inventory of integratable evidence streams for issue #3293.
# Keep this list conservative: add a stream only when its denominator and contract are
# distinct from an existing one, and mark external dependence honestly.
EVIDENCE_STREAMS: tuple[EvidenceStreamSpec, ...] = (
    EvidenceStreamSpec(
        stream_id="simulation_trace",
        title="Simulation campaign traces",
        category=EvidenceCategory.BENCHMARK,
        feasibility=FeasibilityStatus.FEASIBLE_NOW,
        description=(
            "Episode-level traces and summaries from reproducible Robot SF simulation "
            "campaigns under a frozen benchmark contract. Primary in-repo evidence source."
        ),
        extra_provenance_fields=("planner_mode", "seed_set", "config_path"),
        related_issues=(3065,),
    ),
    EvidenceStreamSpec(
        stream_id="amv_command_response",
        title="AMV command-response actuation measurements",
        category=EvidenceCategory.CALIBRATION,
        feasibility=FeasibilityStatus.BLOCKED_EXTERNAL,
        description=(
            "Measured AMV/micromobility command-vs-response (latency, rider coupling, "
            "actuation envelope) used to calibrate the actuation model. Maintainer estimate "
            "of obtaining a real source: <5%. Any synthetic envelope stays non_calibrated."
        ),
        extra_provenance_fields=("vehicle_id", "actuation_units", "measurement_rig"),
        extra_uncertainty_fields=("measurement_noise_model",),
        blocked_until=(
            "A real AMV command-response source or field-measurement method becomes "
            "available (issue #3293 maintainer decision 2026-06-22)."
        ),
        related_issues=(2230, 2531),
    ),
    EvidenceStreamSpec(
        stream_id="external_pedestrian_trajectory",
        title="External pedestrian trajectory data",
        category=EvidenceCategory.CALIBRATION,
        feasibility=FeasibilityStatus.PARTIAL_EXTERNAL,
        description=(
            "Public/external pedestrian trajectory datasets (e.g. for scenario priors and "
            "sim-vs-real motion comparison). A non-calibrated public-dataset slice may be "
            "feasible now via the ingestion contract; calibration-grade use needs staging."
        ),
        extra_provenance_fields=("dataset_name", "coordinate_frame", "frame_rate_hz"),
        blocked_until=(
            "Staged, license-cleared external trajectory dataset via the real-trajectory "
            "ingestion contract (issue #3065)."
        ),
        related_issues=(3065, 2479),
    ),
    EvidenceStreamSpec(
        stream_id="pilot_fleet_operational",
        title="Pilot / fleet operational data",
        category=EvidenceCategory.OPERATIONAL,
        feasibility=FeasibilityStatus.BLOCKED_EXTERNAL,
        description=(
            "Operational data from a pilot deployment or vehicle fleet (interventions, "
            "incidents, operating time). Operational denominators are not directly "
            "comparable to benchmark episodes and must be kept separate."
        ),
        extra_provenance_fields=("deployment_id", "operating_window"),
        blocked_until="A pilot deployment or fleet data-sharing arrangement exists.",
        related_issues=(),
    ),
)


@dataclass(frozen=True)
class StreamCheckResult:
    """Result of a presence-only metadata check against one stream's contract."""

    stream_id: str
    missing_provenance_fields: tuple[str, ...]
    missing_uncertainty_fields: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when every required provenance and uncertainty key is present."""
        return not self.missing_provenance_fields and not self.missing_uncertainty_fields

    @property
    def exit_code(self) -> int:
        """Shell-friendly exit code: 0 when complete, 1 when fields are missing."""
        return 0 if self.ok else 1

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the check result."""
        return {
            "stream_id": self.stream_id,
            "ok": self.ok,
            "missing_provenance_fields": list(self.missing_provenance_fields),
            "missing_uncertainty_fields": list(self.missing_uncertainty_fields),
        }


def list_streams() -> tuple[EvidenceStreamSpec, ...]:
    """Return the canonical inventory of evidence-stream contracts."""
    return EVIDENCE_STREAMS


def build_integration_report() -> dict[str, object]:
    """Build issue #3293 consolidation report from the stream inventory.

    The report is intentionally derived from static contracts only: it does not ingest external
    data, run campaigns, compute calibration, or combine denominators. It gives downstream agents
    one compact contract view of blockers, invalid evidence combinations, and the next empirical
    action that remains non-calibrated.

    Returns:
        JSON-serializable integration report dictionary.
    """
    streams = list_streams()
    categories: dict[str, list[str]] = {category.value: [] for category in EvidenceCategory}
    for spec in streams:
        categories[spec.category.value].append(spec.stream_id)

    blockers = [
        {
            "stream_id": spec.stream_id,
            "feasibility": spec.feasibility.value,
            "blocked_until": spec.blocked_until,
        }
        for spec in streams
        if spec.feasibility is not FeasibilityStatus.FEASIBLE_NOW
    ]

    return {
        "issue": 3293,
        "status": "design_stage_external_data_blocked",
        "claim_boundary": (
            "Inventory-derived integration report only; not benchmark evidence, not calibration "
            "evidence, not operational evidence, and not a paper-facing claim."
        ),
        "streams": [spec.to_dict() for spec in streams],
        "categories": categories,
        "blockers_remaining": blockers,
        "invalid_combinations": [
            {
                "rule": "do_not_pool_denominators",
                "reason": (
                    "Calibration residuals, benchmark episode rates, and operational incident "
                    "rates use incompatible denominators and must remain separate."
                ),
                "categories": [
                    EvidenceCategory.CALIBRATION.value,
                    EvidenceCategory.BENCHMARK.value,
                    EvidenceCategory.OPERATIONAL.value,
                ],
            },
            {
                "rule": "amv_command_response_required_for_calibration",
                "reason": (
                    "Synthetic AMV envelopes or public pedestrian trajectories cannot support "
                    "hardware-calibrated AMV actuation claims without real command-response data."
                ),
                "blocked_stream": "amv_command_response",
            },
        ],
        "next_empirical_action": {
            "status": "blocked_schema_or_data_prereq",
            "action": (
                "After the real-trajectory ingestion contract stages a license-cleared public "
                "trajectory dataset, run a non-calibrated bounded side-by-side comparison against "
                "simulation_trace metadata using this inventory as the presence gate."
            ),
            "depends_on": ["issue #3065 external trajectory staging contract"],
            "allowed_claim": (
                "Diagnostic non-calibrated comparison of distributions; no AMV calibration, "
                "operational safety, planner ranking, benchmark, or paper-facing claim."
            ),
        },
    }


def get_stream(stream_id: str) -> EvidenceStreamSpec:
    """Look up a single stream contract by id.

    Returns:
        The :class:`EvidenceStreamSpec` whose ``stream_id`` matches.

    Raises:
        KeyError: if ``stream_id`` is not a known stream.
    """
    for spec in EVIDENCE_STREAMS:
        if spec.stream_id == stream_id:
            return spec
    known = ", ".join(spec.stream_id for spec in EVIDENCE_STREAMS)
    raise KeyError(f"unknown evidence stream '{stream_id}'; known streams: {known}")


def check_stream_metadata(stream_id: str, metadata: dict[str, object]) -> StreamCheckResult:
    """Check that a synthetic metadata record carries the required contract fields.

    This is a *presence-only* structural check on a caller-supplied (synthetic) record.
    It does not inspect field values, ingest data, or make any calibration/safety claim.

    Args:
        stream_id: Stream to check against (see :data:`EVIDENCE_STREAMS`).
        metadata: A metadata record (e.g. loaded from synthetic JSON). Provenance keys are
            read from ``metadata`` directly or from a nested ``provenance`` mapping, and
            uncertainty keys from ``metadata`` directly or a nested ``uncertainty`` mapping.

    Returns:
        A :class:`StreamCheckResult` listing any missing required fields.

    Raises:
        KeyError: if ``stream_id`` is unknown.
    """
    spec = get_stream(stream_id)
    # Guard against a non-dict record (e.g. ``None``) so a malformed caller cannot crash the
    # presence check; a non-dict is treated as "no fields present".
    record = metadata if isinstance(metadata, dict) else {}
    provenance = record.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    uncertainty = record.get("uncertainty")
    uncertainty = uncertainty if isinstance(uncertainty, dict) else {}

    def _present(key: str, nested: dict[str, object]) -> bool:
        """Return True if ``key`` appears at the top level or in its nested block."""
        return key in record or key in nested

    missing_prov = tuple(f for f in spec.required_provenance_fields if not _present(f, provenance))
    missing_unc = tuple(f for f in spec.required_uncertainty_fields if not _present(f, uncertainty))
    return StreamCheckResult(
        stream_id=stream_id,
        missing_provenance_fields=missing_prov,
        missing_uncertainty_fields=missing_unc,
    )
