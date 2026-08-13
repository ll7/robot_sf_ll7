"""Cell-binding metadata for trace dossier manifests.

This module builds a pure metadata block that a future dossier renderer can
embed in its artifact manifest. It names the campaign cell, terminal verdict
counts, and selected trace identity without running simulations, reading trace
files, or making benchmark claims.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

TRACE_DOSSIER_CELL_BINDING_SCHEMA_VERSION = "trace_dossier_cell_binding.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TraceDossierCellBindingError(ValueError):
    """Raised when a trace dossier cannot be bound to one campaign cell."""


@dataclass(frozen=True, slots=True)
class TraceDossierCellIdentity:
    """Stable identity for one campaign cell."""

    campaign_id: str
    cell_id: str
    scenario_id: str
    planner_id: str
    release_arm_id: str | None = None
    scenario_family: str | None = None


@dataclass(frozen=True, slots=True)
class TraceDossierSelectedTrace:
    """Selected trace identity and checksum metadata."""

    cell_id: str
    episode_id: str
    seed: int
    trace_artifact_uri: str
    trace_sha256: str
    terminal_verdict: str


@dataclass(frozen=True, slots=True)
class TraceDossierCellBinding:
    """Versioned manifest block that binds one dossier to one campaign cell."""

    schema_version: str
    cell: TraceDossierCellIdentity
    selected_trace: TraceDossierSelectedTrace
    terminal_verdict_counts: dict[str, int]
    cell_episode_count: int
    selected_verdict_count: int
    evidence_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-shaped metadata block."""

        payload = asdict(self)
        payload["terminal_verdict_counts"] = dict(sorted(self.terminal_verdict_counts.items()))
        return payload


def build_trace_dossier_cell_binding(
    *,
    cell: Mapping[str, Any] | TraceDossierCellIdentity,
    selected_trace: Mapping[str, Any] | TraceDossierSelectedTrace,
    terminal_verdict_counts: Mapping[str, Any],
) -> TraceDossierCellBinding:
    """Build validated metadata that binds one trace dossier to one campaign cell.

    The returned block is provenance metadata only. It does not infer a terminal
    verdict, rank planners, validate trace file contents, or admit the selected
    trace as benchmark or paper-facing evidence.

    Returns:
        A frozen ``trace_dossier_cell_binding.v1`` metadata block.
    """

    normalized_cell = _normalize_cell(cell)
    normalized_trace = _normalize_trace(selected_trace)
    counts = _normalize_counts(terminal_verdict_counts)
    selected_count = counts.get(normalized_trace.terminal_verdict)
    if normalized_trace.cell_id != normalized_cell.cell_id:
        raise TraceDossierCellBindingError("selected_trace.cell_id must match cell.cell_id")
    if selected_count is None or selected_count <= 0:
        raise TraceDossierCellBindingError(
            "selected_trace terminal_verdict must have a positive count in terminal_verdict_counts"
        )
    return TraceDossierCellBinding(
        schema_version=TRACE_DOSSIER_CELL_BINDING_SCHEMA_VERSION,
        cell=normalized_cell,
        selected_trace=normalized_trace,
        terminal_verdict_counts=counts,
        cell_episode_count=sum(counts.values()),
        selected_verdict_count=selected_count,
        evidence_boundary=(
            "metadata_only_not_benchmark_evidence_until_trace_export_and_renderer_provenance_pass"
        ),
    )


def _normalize_cell(cell: Mapping[str, Any] | TraceDossierCellIdentity) -> TraceDossierCellIdentity:
    """Validate one campaign-cell identity.

    Returns:
        Normalized campaign-cell identity.
    """

    values = asdict(cell) if isinstance(cell, TraceDossierCellIdentity) else dict(cell)
    _reject_unknown_keys(
        values,
        {
            "campaign_id",
            "cell_id",
            "scenario_id",
            "planner_id",
            "release_arm_id",
            "scenario_family",
        },
        "cell",
    )
    return TraceDossierCellIdentity(
        campaign_id=_required_text(values.get("campaign_id"), "cell.campaign_id"),
        cell_id=_required_text(values.get("cell_id"), "cell.cell_id"),
        scenario_id=_required_text(values.get("scenario_id"), "cell.scenario_id"),
        planner_id=_required_text(values.get("planner_id"), "cell.planner_id"),
        release_arm_id=_optional_text(values.get("release_arm_id"), "cell.release_arm_id"),
        scenario_family=_optional_text(values.get("scenario_family"), "cell.scenario_family"),
    )


def _normalize_trace(
    selected_trace: Mapping[str, Any] | TraceDossierSelectedTrace,
) -> TraceDossierSelectedTrace:
    """Validate the selected trace identity without reading the artifact.

    Returns:
        Normalized selected trace identity.
    """

    values = (
        asdict(selected_trace)
        if isinstance(selected_trace, TraceDossierSelectedTrace)
        else dict(selected_trace)
    )
    _reject_unknown_keys(
        values,
        {"cell_id", "episode_id", "seed", "trace_artifact_uri", "trace_sha256", "terminal_verdict"},
        "selected_trace",
    )
    seed = values.get("seed")
    if type(seed) is not int or seed < 0:
        raise TraceDossierCellBindingError("selected_trace.seed must be a non-negative integer")
    trace_sha256 = _required_text(values.get("trace_sha256"), "selected_trace.trace_sha256")
    if _SHA256_RE.fullmatch(trace_sha256) is None:
        raise TraceDossierCellBindingError(
            "selected_trace.trace_sha256 must be a lowercase SHA-256 digest"
        )
    return TraceDossierSelectedTrace(
        cell_id=_required_text(values.get("cell_id"), "selected_trace.cell_id"),
        episode_id=_required_text(values.get("episode_id"), "selected_trace.episode_id"),
        seed=seed,
        trace_artifact_uri=_required_text(
            values.get("trace_artifact_uri"), "selected_trace.trace_artifact_uri"
        ),
        trace_sha256=trace_sha256,
        terminal_verdict=_required_text(
            values.get("terminal_verdict"), "selected_trace.terminal_verdict"
        ),
    )


def _normalize_counts(raw_counts: Mapping[str, Any]) -> dict[str, int]:
    """Validate and sort terminal verdict counts.

    Returns:
        Sorted terminal verdict counts.
    """

    if not raw_counts:
        raise TraceDossierCellBindingError("terminal_verdict_counts must be non-empty")
    counts: dict[str, int] = {}
    for raw_label, raw_count in raw_counts.items():
        label = _required_text(raw_label, "terminal_verdict_counts label")
        if type(raw_count) is not int or raw_count < 0:
            raise TraceDossierCellBindingError(
                f"terminal_verdict_counts[{label!r}] must be a non-negative integer"
            )
        counts[label] = raw_count
    if sum(counts.values()) <= 0:
        raise TraceDossierCellBindingError("terminal_verdict_counts total must be positive")
    return dict(sorted(counts.items()))


def _required_text(value: Any, field: str) -> str:
    """Return a non-empty text value or raise a cell-binding error."""

    if not isinstance(value, str) or not value.strip():
        raise TraceDossierCellBindingError(f"{field} must be non-empty text")
    return value


def _reject_unknown_keys(values: Mapping[str, Any], allowed: set[str], field: str) -> None:
    """Reject mapping fields that the versioned metadata contract cannot preserve."""

    unknown = sorted(set(values) - allowed)
    if unknown:
        raise TraceDossierCellBindingError(f"{field} contains unknown fields: {', '.join(unknown)}")


def _optional_text(value: Any, field: str) -> str | None:
    """Return optional non-empty text, preserving missing values."""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TraceDossierCellBindingError(f"{field} must be non-empty text when provided")
    return value
