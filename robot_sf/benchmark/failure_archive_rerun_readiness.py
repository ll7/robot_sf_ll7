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
    scenario_family_key,
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
_NULL_TEST_REQUIRED_KEYS = (
    "null_tests_reject_null",
    "shuffled_outcome_null_test",
    "ranking_permutation_test",
)
_NULL_TEST_CONTAINER_KEYS = (
    "independent_evaluation",
    "null_test_prerequisites",
    "null_tests",
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
    missing_overlap_metadata_archive_ids: list[str] = field(default_factory=list)
    missing_certification_archive_ids: list[str] = field(default_factory=list)
    invalid_certification_archive_ids: list[str] = field(default_factory=list)
    diagnostic_only_outputs: list[str] = field(default_factory=list)
    null_test_prerequisite_source: str | None = None
    null_test_prerequisite_status: str = "not_checked"
    missing_null_test_prerequisites: list[str] = field(default_factory=list)
    invalid_null_test_prerequisites: list[str] = field(default_factory=list)
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
            "missing_overlap_metadata_archive_ids": list(self.missing_overlap_metadata_archive_ids),
            "missing_certification_archive_ids": list(self.missing_certification_archive_ids),
            "invalid_certification_archive_ids": list(self.invalid_certification_archive_ids),
            "diagnostic_only_outputs": list(self.diagnostic_only_outputs),
            "null_test_prerequisite_source": self.null_test_prerequisite_source,
            "null_test_prerequisite_status": self.null_test_prerequisite_status,
            "missing_null_test_prerequisites": list(self.missing_null_test_prerequisites),
            "invalid_null_test_prerequisites": list(self.invalid_null_test_prerequisites),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


def classify_failure_archive_rerun_readiness(
    source_archive: str | Path,
    rerun_archive: str | Path,
    *,
    rerun_output: str | Path | None = None,
    null_test_prerequisites: str | Path | dict[str, Any] | None = None,
) -> FailureArchiveRerunReadiness:
    """Classify whether a proposal-model rerun archive is disjoint and certified.

    Args:
        source_archive: Archive used to fit/select/rank proposal-model candidates.
        rerun_archive: Separate archive intended for the rerun/evaluation slice.
        rerun_output: Optional rerun report JSON. Diagnostic-only markers cap the
            verdict at ``diagnostic_only`` when inputs otherwise pass.
        null_test_prerequisites: Optional JSON path or payload containing the
            null-test readiness fields expected before any held-out claim.

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
    blockers.extend(_overlap_blockers(overlap))
    missing_overlap_metadata = _overlap_metadata_gaps(source_entries, rerun_entries)
    blockers.extend(_count_blockers("missing_overlap_metadata", missing_overlap_metadata))

    missing_certification, invalid_certification = _certification_gaps(rerun_entries)
    blockers.extend(_count_blockers("missing_certification_metadata", missing_certification))
    blockers.extend(_count_blockers("invalid_certification_status", invalid_certification))

    diagnostic_outputs = _diagnostic_only_outputs(Path(rerun_output) if rerun_output else None)
    null_source, null_status, missing_nulls, invalid_nulls = _null_test_prerequisite_gaps(
        null_test_prerequisites
    )
    blockers.extend(_count_blockers("missing_null_test_prerequisites", missing_nulls))
    blockers.extend(_count_blockers("invalid_null_test_prerequisites", invalid_nulls))
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
        missing_overlap_metadata_archive_ids=missing_overlap_metadata,
        missing_certification_archive_ids=missing_certification,
        invalid_certification_archive_ids=invalid_certification,
        diagnostic_only_outputs=diagnostic_outputs,
        null_test_prerequisite_source=null_source,
        null_test_prerequisite_status=null_status,
        missing_null_test_prerequisites=missing_nulls,
        invalid_null_test_prerequisites=invalid_nulls,
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


def _overlap_blockers(overlap: dict[str, Any]) -> list[str]:
    """Return blocker tokens from overlap provenance."""

    blockers: list[str] = []
    for key, blocker in (
        ("archive_id_overlap_count", "archive_id_overlap"),
        ("scenario_family_overlap_count", "scenario_family_overlap"),
        ("seed_overlap_count", "seed_overlap"),
    ):
        count = overlap.get(key, 0)
        if count:
            blockers.append(f"{blocker}:{count}")
    return blockers


def _count_blockers(blocker: str, values: list[str]) -> list[str]:
    """Return a single count blocker when values are present."""

    return [f"{blocker}:{len(values)}"] if values else []


def _overlap_metadata_gaps(
    source_entries: list[dict[str, Any]],
    rerun_entries: list[dict[str, Any]],
) -> list[str]:
    """Return archive IDs whose metadata cannot prove disjointness."""

    gaps: list[str] = []
    for side, entries in (("source", source_entries), ("rerun", rerun_entries)):
        for index, entry in enumerate(entries):
            raw_archive_id = entry.get("archive_id")
            archive_id = str(raw_archive_id or f"{side}:<entry:{index}>")
            entry_gaps: list[str] = []
            if raw_archive_id is None:
                entry_gaps.append("archive_id")
            if scenario_family_key(entry) == "unknown_family":
                entry_gaps.append("scenario_family")
            candidate = entry.get("candidate")
            if not isinstance(candidate, dict) or candidate.get("scenario_seed") is None:
                entry_gaps.append("scenario_seed")
            if entry_gaps:
                gaps.append(f"{side}:{archive_id}:{','.join(entry_gaps)}")
    return gaps


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
        if status is None or status not in _PASSING_CERTIFICATION_STATUSES:
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

    if isinstance(value, bool):
        return "passed" if value else "failed"
    if isinstance(value, str):
        return value.strip().lower() or None
    if isinstance(value, dict):
        for key in ("status", "verdict", "classification", "result"):
            if key not in value:
                continue
            raw_status = value[key]
            if isinstance(raw_status, bool):
                return "passed" if raw_status else "failed"
            if isinstance(raw_status, str) and raw_status.strip():
                return raw_status.strip().lower()
            if raw_status is None or (isinstance(raw_status, str) and not raw_status.strip()):
                return "failed"
        return "certified" if value else None
    return None


def _null_test_prerequisite_gaps(
    payload_or_path: str | Path | dict[str, Any] | None,
) -> tuple[str | None, str, list[str], list[str]]:
    """Return fail-closed gaps for optional null-test prerequisite metadata."""

    if payload_or_path is None:
        return None, "not_checked", [], []
    if isinstance(payload_or_path, dict):
        source = "inline_payload"
        payload: dict[str, Any] | None = payload_or_path
        load_gap = None
    else:
        path = Path(payload_or_path)
        source = str(path)
        payload, load_gap = _load_json_object(path)
    if load_gap is not None:
        return source, "blocked", [], [load_gap]

    prerequisites = _null_test_container(payload)
    missing = [
        key
        for key in _NULL_TEST_REQUIRED_KEYS
        if key not in prerequisites or prerequisites.get(key) in (None, "", [])
    ]
    invalid: list[str] = []
    if prerequisites.get("null_tests_reject_null") is not True:
        invalid.append("null_tests_reject_null_not_true")
    for key in ("shuffled_outcome_null_test", "ranking_permutation_test"):
        value = prerequisites.get(key)
        if not isinstance(value, dict):
            continue
        if value.get("status") != "complete":
            invalid.append(f"{key}_status_not_complete")
        if "p_value" not in value:
            invalid.append(f"{key}_missing_p_value")
    status = "ready" if not missing and not invalid else "blocked"
    return source, status, missing, invalid


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load a JSON object, returning a compact fail-closed reason on failure.

    Returns:
        ``(payload, reason)`` with ``reason`` populated only when loading fails.
    """

    if not path.exists():
        return None, f"null_test_prerequisites_missing:{path}"
    if path.stat().st_size == 0:
        return None, f"null_test_prerequisites_empty:{path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"null_test_prerequisites_unreadable:{exc}"
    if not isinstance(payload, dict):
        return None, "null_test_prerequisites_not_object"
    return payload, None


def _null_test_container(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return the nested null-test prerequisite object from a report payload."""

    if not isinstance(payload, dict):
        return {}
    for key in _NULL_TEST_CONTAINER_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


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
