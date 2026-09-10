#!/usr/bin/env python3
"""Fail-closed cleanup-eligibility guard for one artifact or output identity (#8906).

The guard answers one question deterministically: may the bytes behind an explicit artifact/output
identity be deleted without risking durable evidence? It composes lifecycle records owned
elsewhere -- output-root ownership, preservation/verification receipts, active writers/leases, and
consumer references -- and never deletes or mutates anything. Anything other than ``eligible``
fails closed. Reason codes and blocking owner identities are stable and public-safe: logical IDs
only, never locators, hostnames, mount paths, or URLs. Record schema:
``cleanup_eligibility_record.v1`` (JSON); see ``docs/context/cleanup_eligibility.md`` for the gate
contract and fixture bundle. Exit codes: 0 eligible, 2 protected/unknown/conflict or usage error,
1 unreadable record.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "cleanup_eligibility_record.v1"
REPORT_SCHEMA = "cleanup_eligibility_report.v1"
EXIT_ELIGIBLE, EXIT_BLOCKED, EXIT_UNREADABLE = 0, 2, 1

DURABLE_STORAGE = ("public_release", "cloud_durable", "personal_durable")
EXPIRING_STORAGE = ("institutional_durable", "institutional_cache")
STORAGE_CLASSES = DURABLE_STORAGE + EXPIRING_STORAGE + ("local_scratch", "unknown")
RETENTION_CLASSES = (
    "durable_required",
    "release_facing",
    "historical",
    "diagnostic",
    "superseded",
    "disposable",
)
REDUNDANT_RETENTION = frozenset({"durable_required", "release_facing"})
OWNER_STATES = ("active", "closed", "retired", "unknown")
WRITER_STATES = ("active", "released", "unknown")
CONSUMER_STATES = ("active", "resolved", "unknown")
PROJECTION_STATES = ("public_safe", "unresolved", "missing")
VERIFICATIONS = ("verified", "unverified", "failed")
EVIDENCE_BASES = frozenset({"checksum_receipt", "manifest_verification"})
NON_EVIDENCE_BASES = frozenset("directory_name age git_ignore scheduler_completion missing".split())
VERIFICATION_BASES = EVIDENCE_BASES | NON_EVIDENCE_BASES
_ONLY_COPY_CODES = frozenset(
    "no_verified_durable_copy only_expiring_host_copy insufficient_failure_domains".split()
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RecordError(Exception):
    """Raised when the record file cannot be read or parsed as a JSON object."""


@dataclass(frozen=True, slots=True)
class EligibilityReport:
    """Deterministic, public-safe check result for one artifact identity."""

    artifact_id: str
    outcome: str
    semantic_digest: str | None
    reason_codes: tuple[str, ...]
    blocking_owners: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the byte-stable JSON payload for the report."""
        return {
            "schema": REPORT_SCHEMA,
            "artifact_id": self.artifact_id,
            "outcome": self.outcome,
            "semantic_digest": self.semantic_digest,
            "reason_codes": list(self.reason_codes),
            "blocking_owners": list(self.blocking_owners),
        }


@dataclass(frozen=True, slots=True)
class _Copy:
    copy_id: str | None
    storage_class: str | None
    failure_domain: str | None
    byte_digest: str | None
    verification: str | None
    verification_basis: str | None


@dataclass(frozen=True, slots=True)
class _Actor:
    actor_id: str | None
    state: str | None


@dataclass(frozen=True, slots=True)
class _Facts:
    semantic_digest: str | None
    retention_class: str | None
    owner_id: str | None
    owner_state: str | None
    pointer_digest: str | None
    projection_state: str | None
    retention_hold: bool
    copies: tuple[_Copy, ...]
    writers: tuple[_Actor, ...]
    consumers: tuple[_Actor, ...]
    problems: tuple[str, ...]


def _enum(value: object, allowed: Sequence[str]) -> str | None:
    return value if isinstance(value, str) and value in allowed else None


def _safe_id(value: object) -> str | None:
    return value if isinstance(value, str) and _SAFE_ID_RE.fullmatch(value) else None


def _sha256(value: object) -> str | None:
    return value if isinstance(value, str) and _SHA256_RE.fullmatch(value) else None


def _safe_or_reject(value: object, problems: list[str], missing_code: str) -> str | None:
    """Return a public-safe logical ID or record why it cannot be used."""
    parsed = _safe_id(value)
    if parsed is None:
        problems.append(
            "private_locator_rejected" if isinstance(value, str) and value else missing_code
        )
    return parsed


def _read_copies(raw: object, problems: list[str]) -> tuple[_Copy, ...]:
    if not isinstance(raw, list):
        problems.append("record_schema_error")
        return ()
    copies: list[_Copy] = []
    for item in raw:
        if not isinstance(item, Mapping):
            problems.append("record_schema_error")
            continue
        domain = None
        if "failure_domain" in item:
            domain = _safe_or_reject(item["failure_domain"], problems, "missing_failure_domain")
        copies.append(
            _Copy(
                _safe_or_reject(item.get("copy_id"), problems, "record_schema_error"),
                _enum(item.get("storage_class"), STORAGE_CLASSES),
                domain,
                _sha256(item.get("byte_digest")),
                _enum(item.get("verification"), VERIFICATIONS),
                _enum(item.get("verification_basis"), VERIFICATION_BASES),
            )
        )
    return tuple(copies)


def _read_actors(
    raw: object, problems: list[str], id_field: str, states: Sequence[str]
) -> tuple[_Actor, ...]:
    if not isinstance(raw, list):
        problems.append("record_schema_error")
        return ()
    actors: list[_Actor] = []
    for item in raw:
        if not isinstance(item, Mapping):
            problems.append("record_schema_error")
            continue
        actors.append(
            _Actor(
                _safe_or_reject(item.get(id_field), problems, "record_schema_error"),
                _enum(item.get("state"), states),
            )
        )
    return tuple(actors)


def _read_facts(entry: Mapping[str, Any]) -> _Facts:
    problems: list[str] = []
    semantic = _sha256(entry.get("semantic_digest"))
    retention = _enum(entry.get("retention_class"), RETENTION_CLASSES)
    owner = _safe_or_reject(entry.get("output_owner"), problems, "record_schema_error")
    owner_state = _enum(entry.get("output_owner_state"), OWNER_STATES)
    projection = _enum(entry.get("private_projection"), PROJECTION_STATES)
    pointer_raw = entry.get("pointer_digest")
    pointer = None if pointer_raw is None else _sha256(pointer_raw)
    hold = entry.get("retention_hold", False)
    if (
        semantic is None
        or retention is None
        or owner_state is None
        or projection is None
        or (pointer_raw is not None and pointer is None)
        or not isinstance(hold, bool)
    ):
        problems.append("record_schema_error")
    return _Facts(
        semantic,
        retention,
        owner,
        owner_state,
        pointer,
        projection,
        hold if isinstance(hold, bool) else False,
        _read_copies(entry.get("copies"), problems),
        _read_actors(entry.get("writers"), problems, "writer_id", WRITER_STATES),
        _read_actors(entry.get("consumers"), problems, "consumer_id", CONSUMER_STATES),
        tuple(problems),
    )


def _copy_rejection(copy: _Copy, semantic_digest: str | None) -> str:
    if copy.verification == "verified" and copy.verification_basis in NON_EVIDENCE_BASES:
        return "non_evidence_basis"
    if (
        copy.verification == "verified"
        and copy.verification_basis in EVIDENCE_BASES
        and copy.byte_digest is not None
        and semantic_digest is not None
        and copy.byte_digest != semantic_digest
    ):
        return "destination_digest_mismatch"
    return "destination_bytes_unverified"


def _assess_copies(  # noqa: C901 - one fail-closed classification over copy evidence
    facts: _Facts, problems: list[str]
) -> tuple[list[str], list[str], list[str]]:
    if not facts.copies:
        problems.append("no_copies_recorded")
        return [], [], []
    required = 2 if facts.retention_class in REDUNDANT_RETENTION else 1
    seen: set[str] = set()
    conflicts: list[str] = []
    exact: list[str] = []
    durable_exact: list[str] = []
    domains: set[str] = set()
    durable_rejections: list[str] = []
    durable_labels: list[str] = []
    fallback_rejections: list[str] = []
    fallback_labels: list[str] = []
    expiring = False
    for copy in facts.copies:
        if copy.copy_id is None:
            continue
        label = f"copy:{copy.copy_id}"
        if copy.copy_id in seen:
            conflicts.append("duplicate_copy")
            continue
        seen.add(copy.copy_id)
        if (
            copy.verification == "verified"
            and copy.verification_basis in EVIDENCE_BASES
            and copy.byte_digest is not None
            and copy.byte_digest == facts.semantic_digest
        ):
            exact.append(label)
            expiring |= copy.storage_class in EXPIRING_STORAGE
            if copy.storage_class in DURABLE_STORAGE:
                if copy.failure_domain is None:
                    problems.append("missing_failure_domain")
                else:
                    durable_exact.append(label)
                    domains.add(copy.failure_domain)
            continue
        if copy.verification == "verified" and copy.byte_digest is None:
            problems.append("missing_digest")
            continue
        rejection = _copy_rejection(copy, facts.semantic_digest)
        fallback_rejections.append(rejection)
        fallback_labels.append(label)
        if copy.storage_class in DURABLE_STORAGE:
            durable_rejections.append(rejection)
            durable_labels.append(label)
    if len(domains) >= required:
        return [], [], conflicts
    if domains:
        return ["insufficient_failure_domains"], durable_exact, conflicts
    if exact:
        codes = ["no_verified_durable_copy"]
        if expiring:
            codes.append("only_expiring_host_copy")
        return codes, exact, conflicts
    if durable_rejections:
        return sorted(set(durable_rejections)), durable_labels, conflicts
    return (
        sorted(set(fallback_rejections) or {"destination_bytes_unverified"}),
        fallback_labels,
        conflicts,
    )


def _evaluate(  # noqa: C901, PLR0912 - one fail-closed decision over independent lifecycle gates
    entry: Mapping[str, Any], artifact_id: str
) -> EligibilityReport:
    facts = _read_facts(entry)
    problems = list(facts.problems)
    conflicts: list[str] = []
    active: list[str] = []
    referenced: list[str] = []
    owners: list[str] = []

    if facts.owner_state == "active":
        active.append("active_output_owner")
        owners.append(f"owner:{facts.owner_id or artifact_id}")
    elif facts.owner_state == "unknown":
        problems.append("owner_state_unknown")
    if (
        facts.pointer_digest
        and facts.semantic_digest
        and facts.pointer_digest != facts.semantic_digest
    ):
        conflicts.append("stale_pointer")
        owners.append(f"pointer:{artifact_id}")
    if facts.projection_state and facts.projection_state != "public_safe":
        problems.append("private_projection_unresolved")
        owners.append(f"projection:{artifact_id}")
    if facts.retention_hold:
        referenced.append("retention_hold")
        owners.append(f"retention:{artifact_id}")

    active_writers = [writer for writer in facts.writers if writer.state == "active"]
    if len(active_writers) > 1:
        conflicts.append("concurrent_writer")
    elif active_writers:
        active.append("active_writer_or_lease")
    owners.extend(f"writer:{writer.actor_id}" for writer in active_writers)
    if any(writer.state == "unknown" for writer in facts.writers):
        problems.append("writer_state_unknown")

    for consumer in facts.consumers:
        if consumer.state == "active":
            referenced.append("unresolved_consumer")
            owners.append(f"consumer:{consumer.actor_id}")
        elif consumer.state == "unknown":
            problems.append("consumer_state_unknown")

    copy_codes, copy_owners, copy_conflicts = _assess_copies(facts, problems)
    conflicts.extend(copy_conflicts)
    if conflicts:
        outcome = "conflict"
    elif active:
        outcome = "protected_active"
    elif referenced:
        outcome = "protected_referenced"
    elif copy_codes and "private_locator_rejected" not in problems:
        outcome = (
            "protected_only_copy" if set(copy_codes) & _ONLY_COPY_CODES else "protected_unverified"
        )
    elif problems:
        outcome = "unknown"
    else:
        outcome = "eligible"
    reason_codes = tuple(sorted(set(problems + conflicts + active + referenced + copy_codes)))
    return EligibilityReport(
        artifact_id=artifact_id,
        outcome=outcome,
        semantic_digest=facts.semantic_digest,
        reason_codes=reason_codes,
        blocking_owners=tuple(sorted(set(owners + copy_owners))),
    )


def _load_payload(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecordError(f"cannot read record file {path.name!r}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecordError(f"record file {path.name!r} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RecordError(f"record file {path.name!r} must contain a JSON object")
    return payload


def check_artifact_file(record_path: Path, artifact_id: str) -> EligibilityReport:
    """Evaluate one artifact identity; ``artifact_id`` must be a public-safe logical ID.

    Raises:
        RecordError: If the file is unreadable, not JSON, or the ID is not public-safe.
    """
    if _SAFE_ID_RE.fullmatch(artifact_id) is None:
        raise RecordError("artifact id must be a public-safe logical id")
    payload = _load_payload(record_path)
    if payload.get("schema") != SCHEMA or not isinstance(payload.get("artifacts"), list):
        return EligibilityReport(artifact_id, "unknown", None, ("record_schema_error",), ())
    matches = [
        entry
        for entry in payload["artifacts"]
        if isinstance(entry, Mapping) and entry.get("artifact_id") == artifact_id
    ]
    if not matches:
        return EligibilityReport(artifact_id, "unknown", None, ("artifact_not_recorded",), ())
    if len(matches) > 1:
        return EligibilityReport(
            artifact_id,
            "unknown",
            None,
            ("duplicate_artifact_record",),
            (f"artifact:{artifact_id}",),
        )
    return _evaluate(matches[0], artifact_id)


def render_text(report: EligibilityReport) -> str:
    """Return the one-line public-safe text report."""
    codes = ",".join(report.reason_codes) or "-"
    owners = ",".join(report.blocking_owners) or "-"
    return (
        f"artifact={report.artifact_id} outcome={report.outcome} "
        f"reason_codes={codes} blocking_owners={owners}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check-only cleanup-eligibility guard; never deletes."
    )
    parser.add_argument("--record", type=Path, required=True, help="Record JSON path.")
    parser.add_argument("--artifact", required=True, help="Artifact identity to evaluate.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Required explicit check-only gate flag; the guard never deletes.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the JSON report to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cleanup-eligibility guard CLI and return a shell exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("--check is required; this guard is check-only and never deletes")
    try:
        report = check_artifact_file(args.record, args.artifact)
    except RecordError as exc:
        print(f"cleanup-eligibility: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return EXIT_ELIGIBLE if report.outcome == "eligible" else EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
