"""Readiness and leakage checks for proposal-model failure-archive reruns.

This module answers one bounded issue #3275 question without running a
benchmark campaign:

    Can a proposal model trained or selected from one certified failure archive
    be rerun against a separate certified failure archive without archive-ID
    leakage or missing certification metadata?

The verdict is intentionally fail-closed. Overlap or missing required metadata
blocks readiness. Diagnostic-only rerun outputs are preserved as diagnostics
and never promoted to benchmark evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.adversarial.disjoint_evaluation import (
    ARCHIVE_SCHEMA_VERSION,
    archive_sha256,
    compute_overlap_provenance,
)

READY = "ready"
DIAGNOSTIC_ONLY = "diagnostic_only"
BLOCKED = "blocked"

SCHEMA_VERSION = "failure_archive_rerun_readiness.v1"
CLAIM_BOUNDARY = (
    "readiness/leakage check only; no benchmark campaign run, no adversarial-improvement claim, "
    "no dissertation or paper-facing claim promotion"
)

_CERTIFICATION_KEYS = (
    "certification_metadata",
    "certification",
    "certification_status",
    "candidate_certification",
    "scenario_certification",
    "scenario_certificate",
)
_PASSING_CERTIFICATION_STATUSES = {
    "accepted",
    "certified",
    "certified_valid_failure",
    "pass",
    "passed",
    "ready",
    "valid",
}
_DIAGNOSTIC_ONLY_VALUES = {
    "diagnostic",
    "diagnostic-only",
    "diagnostic_only",
    "held_out_diagnostic_only",
    "not_benchmark_evidence",
    "plumbing_validation_only",
}
_DIAGNOSTIC_STATUS_KEYS = (
    "claim_boundary",
    "diagnostic_only",
    "evidence_status",
    "held_out_evidence_status",
    "interpretation",
    "result_classification",
    "status",
)


@dataclass(frozen=True)
class FailureArchiveRerunReadiness:
    """Fail-closed readiness verdict for a disjoint failure-archive rerun."""

    status: str
    source_archive: str
    rerun_archive: str
    source_entry_count: int = 0
    rerun_entry_count: int = 0
    source_archive_sha256: str | None = None
    rerun_archive_sha256: str | None = None
    overlap_provenance: dict[str, Any] = field(default_factory=dict)
    archive_id_overlap: list[str] = field(default_factory=list)
    missing_certification_archive_ids: list[str] = field(default_factory=list)
    invalid_certification_archive_ids: list[str] = field(default_factory=list)
    diagnostic_only_outputs: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """Return whether this rerun input is ready for a diagnostic rerun."""

        return self.status == READY

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable readiness payload."""

        return {
            "schema_version": SCHEMA_VERSION,
            "claim_boundary": CLAIM_BOUNDARY,
            "status": self.status,
            "ready": self.ready,
            "source_archive": self.source_archive,
            "rerun_archive": self.rerun_archive,
            "source_entry_count": self.source_entry_count,
            "rerun_entry_count": self.rerun_entry_count,
            "source_archive_sha256": self.source_archive_sha256,
            "rerun_archive_sha256": self.rerun_archive_sha256,
            "overlap_provenance": dict(self.overlap_provenance),
            "archive_id_overlap": list(self.archive_id_overlap),
            "missing_certification_archive_ids": list(self.missing_certification_archive_ids),
            "invalid_certification_archive_ids": list(self.invalid_certification_archive_ids),
            "diagnostic_only_outputs": list(self.diagnostic_only_outputs),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


def classify_failure_archive_rerun_readiness(
    source_archive: str | Path,
    rerun_archive: str | Path,
    *,
    rerun_output: str | Path | None = None,
) -> FailureArchiveRerunReadiness:
    """Classify whether a proposal-model rerun archive is disjoint and certified.

    Args:
        source_archive: Archive used to fit/select/rank proposal-model candidates.
        rerun_archive: Separate archive intended for the rerun/evaluation slice.
        rerun_output: Optional rerun report JSON. Diagnostic-only markers cap the
            verdict at ``diagnostic_only`` when inputs otherwise pass.

    Returns:
        A fail-closed readiness verdict. The function never runs planners,
        benchmarks, proposal sampling, or artifact publication.
    """

    source_path = Path(source_archive)
    rerun_path = Path(rerun_archive)
    source_status, source_payload, source_reason = _load_archive(source_path)
    rerun_status, rerun_payload, rerun_reason = _load_archive(rerun_path)

    blockers: list[str] = []
    warnings: list[str] = []
    if source_status != READY:
        blockers.append(f"source_archive_{source_status}:{source_reason}")
    if rerun_status != READY:
        blockers.append(f"rerun_archive_{rerun_status}:{rerun_reason}")

    source_entries = _archive_entries(source_payload)
    rerun_entries = _archive_entries(rerun_payload)
    source_hash = archive_sha256(source_payload) if isinstance(source_payload, dict) else None
    rerun_hash = archive_sha256(rerun_payload) if isinstance(rerun_payload, dict) else None

    overlap = compute_overlap_provenance(source_entries, rerun_entries)
    archive_id_overlap = list(overlap.get("archive_id_overlap", []))
    if archive_id_overlap:
        blockers.append(f"archive_id_overlap:{len(archive_id_overlap)}")
    if overlap.get("scenario_family_overlap_count", 0):
        blockers.append(
            f"scenario_family_overlap:{overlap['scenario_family_overlap_count']}"
        )
    if overlap.get("seed_overlap_count", 0):
        blockers.append(f"seed_overlap:{overlap['seed_overlap_count']}")

    missing_certification, invalid_certification = _certification_gaps(rerun_entries)
    if missing_certification:
        blockers.append(f"missing_certification_metadata:{len(missing_certification)}")
    if invalid_certification:
        blockers.append(f"invalid_certification_status:{len(invalid_certification)}")

    diagnostic_outputs = _diagnostic_only_outputs(Path(rerun_output) if rerun_output else None)
    status = BLOCKED if blockers else READY
    if status == READY and diagnostic_outputs:
        status = DIAGNOSTIC_ONLY

    return FailureArchiveRerunReadiness(
        status=status,
        source_archive=str(source_archive),
        rerun_archive=str(rerun_archive),
        source_entry_count=len(source_entries),
        rerun_entry_count=len(rerun_entries),
        source_archive_sha256=source_hash,
        rerun_archive_sha256=rerun_hash,
        overlap_provenance=overlap,
        archive_id_overlap=archive_id_overlap,
        missing_certification_archive_ids=missing_certification,
        invalid_certification_archive_ids=invalid_certification,
        diagnostic_only_outputs=diagnostic_outputs,
        blockers=blockers,
        warnings=warnings,
    )


def _load_archive(path: Path) -> tuple[str, dict[str, Any] | None, str]:
    """Load one archive JSON and validate its top-level shape.

    Returns:
        ``(status, payload, reason)`` with ``payload`` present only when JSON
        parsing succeeded.
    """

    if not path.exists():
        return BLOCKED, None, f"path_missing:{path}"
    if path.stat().st_size == 0:
        return BLOCKED, None, f"file_empty:{path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return BLOCKED, None, f"unreadable:{exc}"
    if not isinstance(payload, dict):
        return BLOCKED, None, "payload_not_object"
    schema_version = payload.get("schema_version")
    if schema_version != ARCHIVE_SCHEMA_VERSION:
        return BLOCKED, payload, f"unexpected_schema_version:{schema_version!r}"
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        return BLOCKED, payload, "archive_has_no_entries"
    return READY, payload, "ok"


def _archive_entries(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return dictionary entries from a parsed archive payload."""

    if not isinstance(payload, dict):
        return []
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _certification_gaps(entries: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Return archive IDs missing or failing rerun certification metadata."""

    missing: list[str] = []
    invalid: list[str] = []
    for index, entry in enumerate(entries):
        archive_id = str(entry.get("archive_id") or f"<entry:{index}>")
        certification = _first_certification_value(entry)
        if certification is None:
            missing.append(archive_id)
            continue
        status = _certification_status(certification)
        if status and status not in _PASSING_CERTIFICATION_STATUSES:
            invalid.append(archive_id)
    return missing, invalid


def _first_certification_value(entry: dict[str, Any]) -> Any:
    """Return the first recognized certification metadata field."""

    for key in _CERTIFICATION_KEYS:
        value = entry.get(key)
        if value is not None:
            return value
    return None


def _certification_status(value: Any) -> str | None:
    """Normalize a certification metadata value into a status token.

    Returns:
        Lowercase certification status when available, otherwise ``None``.
    """

    if isinstance(value, str):
        return value.strip().lower() or None
    if isinstance(value, dict):
        for key in ("status", "verdict", "classification", "result"):
            raw_status = value.get(key)
            if isinstance(raw_status, str) and raw_status.strip():
                return raw_status.strip().lower()
        return "certified" if value else None
    if isinstance(value, bool):
        return "passed" if value else "failed"
    return None


def _diagnostic_only_outputs(path: Path | None) -> list[str]:
    """Return diagnostic-only markers from an optional rerun output JSON."""

    if path is None:
        return []
    if not path.exists() or path.stat().st_size == 0:
        return [f"rerun_output_unavailable:{path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"rerun_output_unreadable:{exc}"]
    if not isinstance(payload, dict):
        return ["rerun_output_not_object"]
    markers: list[str] = []
    _collect_diagnostic_markers(payload, markers, prefix="")
    return sorted(set(markers))


def _collect_diagnostic_markers(payload: Any, markers: list[str], *, prefix: str) -> None:
    """Collect diagnostic-only status markers from nested report payloads."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key in _DIAGNOSTIC_STATUS_KEYS and _is_diagnostic_only_value(value):
                markers.append(f"{child_prefix}:{value}")
            _collect_diagnostic_markers(value, markers, prefix=child_prefix)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _collect_diagnostic_markers(value, markers, prefix=f"{prefix}[{index}]")


def _is_diagnostic_only_value(value: Any) -> bool:
    """Return whether a report value explicitly marks diagnostic-only evidence."""

    if isinstance(value, bool):
        return value is True
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().replace(" ", "_")
    return normalized in _DIAGNOSTIC_ONLY_VALUES or "diagnostic_only" in normalized


__all__ = [
    "BLOCKED",
    "DIAGNOSTIC_ONLY",
    "READY",
    "FailureArchiveRerunReadiness",
    "classify_failure_archive_rerun_readiness",
]
