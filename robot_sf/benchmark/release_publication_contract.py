"""Fail-closed checks for camera-ready release publication metadata.

The contract is deliberately limited to publication integrity.  It does not run a
benchmark campaign or promote a paper-facing claim.  It prevents a release upload
when the release result, rebuilt campaign summary, signed bundle, or episode
provenance describe different evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_sf.benchmark.release_protocol import (
    resolve_campaign_artifact_path,
    resolve_regular_directory_path,
)
from robot_sf.benchmark.release_tag_identity import (
    HISTORICAL_RELEASE_SOURCE_SHA,
    HISTORICAL_RELEASE_TAG,
    check_tag_source_consistency,
    extract_tag_sha_component,
)

CONTRACT_SCHEMA_VERSION = "benchmark-release-publication-contract.v1"

_CONSISTENCY_FIELDS = (
    "campaign_id",
    "status",
    "evidence_status",
    "benchmark_success",
    "campaign_execution_status",
    "total_episodes",
    "total_runs",
    "successful_runs",
    "non_success_runs",
    "accepted_unavailable_runs",
    "unexpected_failed_runs",
    "core_total_runs",
    "core_successful_runs",
    "row_status_summary",
)
_CHECKSUM_LINE = re.compile(r"^(?P<digest>[0-9a-fA-F]{64})\s+(?P<path>\*?.+)$")
_PLACEHOLDER_TOKENS = ("{release_tag}", "<record-id>", "<record_id>", "00000")
_REQUIRED_NON_EMPTY_FIELDS = frozenset(
    {
        "publication_channels.release_tag",
        "publication_channels.doi",
        "publication_channels.release_url",
        "campaign.release_tag",
        "campaign.doi",
        "campaign.doi_url",
        "campaign.release_url",
        "release_tag",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object or raise a contract-shaped error.

    Returns:
        Parsed JSON object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file, reading it in chunks.

    Returns:
        The SHA-256 hex digest of the file contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checksums(path: Path) -> dict[str, str]:
    """Parse a root-relative SHA-256 manifest.

    Returns:
        Mapping from bundle-root-relative payload path to digest.
    """
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"{path}:{line_number}: malformed checksum entry")
        relative = match.group("path").lstrip("*")
        candidate = Path(relative)
        if (
            not relative.startswith("payload/")
            or candidate.is_absolute()
            or ".." in candidate.parts
        ):
            raise ValueError(
                f"{path}:{line_number}: checksum path must be payload-relative from bundle root: "
                f"{relative!r}"
            )
        if relative in entries:
            raise ValueError(f"{path}:{line_number}: duplicate checksum path {relative!r}")
        entries[relative] = match.group("digest").lower()
    if not entries:
        raise ValueError(f"{path}: checksum manifest contains no entries")
    return entries


def _campaign_block(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``campaign`` object from a campaign summary, failing closed when absent."""
    campaign = summary.get("campaign")
    if not isinstance(campaign, Mapping):
        raise ValueError("campaign_summary.json: missing object field 'campaign'")
    return campaign


def _add_metadata_consistency_blockers(
    blockers: list[str],
    release_result: Mapping[str, Any],
    campaign: Mapping[str, Any],
) -> None:
    """Require release-result and rebuilt-summary fields to agree."""
    for field in _CONSISTENCY_FIELDS:
        release_value = release_result.get(field)
        campaign_value = campaign.get(field)
        if field not in release_result or field not in campaign:
            blockers.append(f"release metadata field missing from both surfaces: {field}")
        elif release_value != campaign_value:
            blockers.append(
                f"release_result.json disagrees with campaign_summary.json for {field}: "
                f"{release_value!r} != {campaign_value!r}"
            )


def _add_placeholder_blockers(
    blockers: list[str], *, expected_release_tag: str, values: Mapping[str, Any]
) -> None:
    """Reject unresolved DOI, release URL, and release-tag templates."""
    for field, value in values.items():
        if field in _REQUIRED_NON_EMPTY_FIELDS and (
            not isinstance(value, str) or not value.strip()
        ):
            blockers.append(f"publication metadata field {field} must be a non-empty string")
            continue
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            blockers.append(f"publication metadata field {field} must be a non-empty string")
            continue
        if any(token in value for token in _PLACEHOLDER_TOKENS):
            blockers.append(f"unresolved publication placeholder in {field}: {value!r}")
    release_tag = values.get("release_tag")
    if release_tag != expected_release_tag:
        blockers.append(
            f"publication release tag {release_tag!r} does not match requested tag "
            f"{expected_release_tag!r}"
        )


def _episode_provenance(
    campaign_root: Path,
) -> tuple[set[str], int]:
    """Return episode software commits and unresolved goal+timeout row count."""
    commits: set[str] = set()
    goal_timeout_rows = 0
    runs_path = campaign_root / "runs"
    if runs_path.is_symlink():
        raise ValueError("campaign runs directory must not contain symlink components")
    if not runs_path.exists():
        return commits, goal_timeout_rows
    runs_path = resolve_regular_directory_path(runs_path, field_name="campaign runs directory")
    for episodes_path in sorted(runs_path.glob("*/episodes.jsonl")):
        episodes_path = resolve_campaign_artifact_path(
            campaign_root,
            episodes_path.relative_to(campaign_root).as_posix(),
        )
        with episodes_path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{episodes_path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, Mapping):
                    continue
                commit = row.get("git_hash")
                if isinstance(commit, str) and commit:
                    commits.add(commit)
                ledger = row.get("event_ledger")
                exact = ledger.get("exact_events") if isinstance(ledger, Mapping) else None
                if (
                    isinstance(exact, Mapping)
                    and exact.get("goal_reached")
                    and exact.get("timeout")
                ):
                    goal_timeout_rows += 1
    return commits, goal_timeout_rows


def _add_commit_blockers(
    blockers: list[str],
    *,
    commits: set[str],
    campaign: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Require a machine-readable explanation for runtime/publication drift."""
    publication = provenance.get("repository")
    publication = publication if isinstance(publication, Mapping) else {}
    publication_commit = publication.get("commit")
    if not isinstance(publication_commit, str) or not publication_commit:
        blockers.append("publication manifest is missing provenance.repository.commit")
        return

    declared_runtime = campaign.get("git_hash")
    runtime_commits = set(commits)
    if isinstance(declared_runtime, str) and declared_runtime:
        runtime_commits.add(declared_runtime)
    if runtime_commits == {publication_commit}:
        return

    explanation = provenance.get("commit_reconciliation")
    if not isinstance(explanation, Mapping):
        blockers.append(
            "runtime and publication commits differ without provenance.commit_reconciliation"
        )
        return
    if explanation.get("status") != "explained":
        blockers.append("provenance.commit_reconciliation.status must be 'explained'")
    if explanation.get("publication_commit") != publication_commit:
        blockers.append("provenance.commit_reconciliation.publication_commit is incorrect")
    declared = explanation.get("runtime_commits")
    if not isinstance(declared, list) or set(declared) != runtime_commits:
        blockers.append("provenance.commit_reconciliation.runtime_commits is incorrect")
    if (
        not isinstance(explanation.get("explanation"), str)
        or not explanation["explanation"].strip()
    ):
        blockers.append("provenance.commit_reconciliation.explanation is required")


def _collect_source_sha_fields(  # noqa: C901
    *,
    summary: Mapping[str, Any],
    campaign: Mapping[str, Any],
    release_result: Mapping[str, Any],
    publication: Mapping[str, Any],
    release_manifest: Mapping[str, Any],
    receipt: Mapping[str, Any] | None,
) -> list[tuple[str, Any]]:
    """Collect explicit final-source fields from every publication surface.

    Returns:
        Labeled source identity values found in the publication surfaces.
    """
    fields: list[tuple[str, Any]] = []

    def add_mapping(label: str, payload: Mapping[str, Any]) -> None:
        for key in ("source_sha", "source_commit", "git_hash"):
            if key in payload:
                fields.append((f"{label}.{key}", payload.get(key)))

    add_mapping("campaign", campaign)
    summary_release = summary.get("benchmark_release")
    if isinstance(summary_release, Mapping):
        add_mapping("summary.benchmark_release", summary_release)
    add_mapping("release_result", release_result)
    result_release = release_result.get("benchmark_release")
    if isinstance(result_release, Mapping):
        add_mapping("release_result.benchmark_release", result_release)
    acceptance = release_result.get("release_acceptance")
    if isinstance(acceptance, Mapping) and "source_commits" in acceptance:
        fields.append(
            ("release_result.release_acceptance.source_commits", acceptance["source_commits"])
        )

    manifest_provenance = release_manifest.get("provenance")
    add_mapping("release_manifest", release_manifest)
    if isinstance(manifest_provenance, Mapping):
        add_mapping("release_manifest.provenance", manifest_provenance)

    publication_provenance = publication.get("provenance")
    if isinstance(publication_provenance, Mapping):
        add_mapping("publication.provenance", publication_provenance)
        repository = publication_provenance.get("repository")
        if isinstance(repository, Mapping) and "commit" in repository:
            fields.append(("publication.provenance.repository.commit", repository["commit"]))

    if isinstance(receipt, Mapping):
        add_mapping("receipt", receipt)
        receipt_source = receipt.get("source")
        if isinstance(receipt_source, Mapping):
            add_mapping("receipt.source", receipt_source)
            if "execution_commit" in receipt_source:
                fields.append(
                    ("receipt.source.execution_commit", receipt_source["execution_commit"])
                )
        if "execution_commit" in receipt:
            fields.append(("receipt.execution_commit", receipt["execution_commit"]))
    return fields


def _add_source_identity_blockers(  # noqa: C901
    blockers: list[str],
    *,
    expected_release_tag: str,
    summary: Mapping[str, Any],
    campaign: Mapping[str, Any],
    release_result: Mapping[str, Any],
    publication: Mapping[str, Any],
    release_manifest: Mapping[str, Any],
    receipt: Mapping[str, Any] | None,
) -> None:
    """Cross-check final source SHA fields and SHA-bearing tag identity."""
    identity = extract_tag_sha_component(expected_release_tag)
    fields = _collect_source_sha_fields(
        summary=summary,
        campaign=campaign,
        release_result=release_result,
        publication=publication,
        release_manifest=release_manifest,
        receipt=receipt,
    )
    explicit_fields = [(label, value) for label, value in fields if value not in (None, "", [])]
    source_bound = identity.scheme != "semantic" or bool(
        any(label.rsplit(".", 1)[-1] in {"source_sha", "source_commit"} for label, _ in fields)
    )
    if not source_bound:
        return

    source_values: list[tuple[str, str]] = []
    for label, value in explicit_fields:
        values = value if label.endswith("source_commits") else [value]
        if not isinstance(values, list):
            values = [values]
        if not values:
            blockers.append(f"{label} must contain a final source SHA")
            continue
        for candidate in values:
            normalized = str(candidate).strip().lower() if isinstance(candidate, str) else ""
            if re.fullmatch(r"[0-9a-f]{40}", normalized) is None:
                blockers.append(f"{label} must contain an exact 40-character source SHA")
                continue
            source_values.append((label, normalized))

    distinct = {value for _, value in source_values}
    if not distinct:
        blockers.append("publication surfaces are missing the final source SHA")
        return
    if len(distinct) > 1:
        details = ", ".join(f"{label}={value}" for label, value in source_values)
        blockers.append(f"publication source SHA fields disagree: {details}")
        return

    source_sha = next(iter(distinct))
    if expected_release_tag == HISTORICAL_RELEASE_TAG:
        if source_sha != HISTORICAL_RELEASE_SOURCE_SHA:
            blockers.append(
                "historical release source SHA does not match its immutable published source"
            )
        return

    blockers.extend(check_tag_source_consistency(expected_release_tag, source_sha))
    required_surfaces = {
        "campaign": any(label.startswith("campaign.") for label, _ in source_values),
        "release_result": any(label.startswith("release_result.") for label, _ in source_values),
        "release_manifest": any(label.startswith("release_manifest") for label, _ in source_values),
        "publication": any(label.startswith("publication.") for label, _ in source_values),
    }
    for surface, present in required_surfaces.items():
        if not present:
            blockers.append(f"publication source SHA is missing from {surface}")


def _add_goal_timeout_blockers(
    blockers: list[str], *, goal_timeout_rows: int, provenance: Mapping[str, Any]
) -> None:
    """Require timing evidence or an explicit exclusion for goal+timeout rows."""
    if not goal_timeout_rows:
        return
    boundary = provenance.get("goal_timeout_boundary")
    if not isinstance(boundary, Mapping):
        blockers.append(
            f"{goal_timeout_rows} goal+timeout row(s) lack provenance.goal_timeout_boundary"
        )
        return
    status = boundary.get("status")
    note = boundary.get("note")
    if not isinstance(note, str) or not note.strip():
        blockers.append("provenance.goal_timeout_boundary.note is required")
    if status == "resolved":
        timing = boundary.get("timing_evidence")
        if not isinstance(timing, str) or not timing.strip():
            blockers.append(
                "resolved goal+timeout boundary requires provenance.goal_timeout_boundary."
                "timing_evidence"
            )
    elif status != "excluded":
        blockers.append("provenance.goal_timeout_boundary.status must be 'resolved' or 'excluded'")


def validate_release_publication_contract(  # noqa: C901, PLR0912, PLR0915
    campaign_root: Path,
    bundle_dir: Path,
    *,
    expected_release_tag: str,
) -> dict[str, Any]:
    """Validate publication metadata and return a machine-readable gate report.

    Returns:
        Contract report with ``pass`` or ``blocked`` status and blockers.
    """
    blockers: list[str] = []
    try:
        campaign_root = resolve_regular_directory_path(campaign_root, field_name="campaign root")
        bundle_dir = resolve_regular_directory_path(
            bundle_dir, field_name="publication bundle directory"
        )
        summary_path = resolve_campaign_artifact_path(
            campaign_root, "reports/campaign_summary.json"
        )
        release_result_path = resolve_campaign_artifact_path(
            campaign_root, "release/release_result.json"
        )
        manifest_path = resolve_campaign_artifact_path(bundle_dir, "publication_manifest.json")
        checksums_path = resolve_campaign_artifact_path(bundle_dir, "checksums.sha256")
        summary = _read_json(summary_path)
        release_result = _read_json(release_result_path)
        publication = _read_json(manifest_path)
        campaign = _campaign_block(summary)
    except (OSError, ValueError) as exc:
        blockers.append(str(exc))
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": blockers,
        }
    try:
        checksums = _parse_checksums(checksums_path)
    except (OSError, ValueError) as exc:
        blockers.append(str(exc))
        checksums = {}

    for relative, expected_digest in checksums.items():
        try:
            path = resolve_campaign_artifact_path(bundle_dir, relative)
        except (OSError, ValueError) as exc:
            blockers.append(f"checksum entry is unsafe: {relative}: {exc}")
            continue
        if _sha256(path) != expected_digest:
            blockers.append(f"checksum mismatch: {relative}")

    manifest_files = publication.get("files")
    if not isinstance(manifest_files, list):
        blockers.append("publication_manifest.json: files must be a list")
    else:
        manifest_checksums = {
            f"payload/{entry.get('path')}": entry.get("sha256")
            for entry in manifest_files
            if isinstance(entry, Mapping)
        }
        if manifest_checksums != checksums:
            blockers.append("publication manifest file hashes disagree with checksums.sha256")

    _add_metadata_consistency_blockers(blockers, release_result, campaign)
    channels = publication.get("publication_channels")
    channels = channels if isinstance(channels, Mapping) else {}
    provenance = publication.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    release_manifest = {}
    receipt: Mapping[str, Any] | None = None
    release_manifest_candidate = (
        bundle_dir / "payload" / "release" / "release_manifest.resolved.json"
    )
    try:
        release_manifest_path = resolve_campaign_artifact_path(
            bundle_dir, "payload/release/release_manifest.resolved.json"
        )
    except (OSError, ValueError) as exc:
        optional_missing = (
            isinstance(exc, ValueError)
            and "not a regular file" in str(exc)
            and "symlink components" not in str(exc)
            and not release_manifest_candidate.exists()
            and not release_manifest_candidate.is_symlink()
        )
        if not optional_missing:
            blockers.append(f"release manifest path is unsafe: {exc}")
        release_manifest_path = None
    if release_manifest_path is not None:
        try:
            release_manifest = _read_json(release_manifest_path)
        except ValueError as exc:
            blockers.append(str(exc))

    receipt_candidate = bundle_dir / "payload" / "provenance" / "derived_revalidation_receipt.json"
    try:
        receipt_path = resolve_campaign_artifact_path(
            bundle_dir, "payload/provenance/derived_revalidation_receipt.json"
        )
    except (OSError, ValueError) as exc:
        optional_missing = (
            isinstance(exc, ValueError)
            and "not a regular file" in str(exc)
            and "symlink components" not in str(exc)
            and not receipt_candidate.exists()
            and not receipt_candidate.is_symlink()
        )
        if not optional_missing:
            blockers.append(f"derived receipt path is unsafe: {exc}")
        receipt_path = None
    if receipt_path is not None:
        try:
            receipt = _read_json(receipt_path)
        except ValueError as exc:
            blockers.append(str(exc))

    _add_placeholder_blockers(
        blockers,
        expected_release_tag=expected_release_tag,
        values={
            "publication_channels.release_tag": channels.get("release_tag"),
            "publication_channels.doi": channels.get("doi"),
            "publication_channels.release_url": channels.get("release_url"),
            "campaign.release_tag": campaign.get("release_tag"),
            "campaign.doi": campaign.get("doi"),
            "campaign.doi_url": campaign.get("doi_url"),
            "campaign.release_url": campaign.get("release_url"),
            "release_manifest.provenance.doi": (
                release_manifest.get("provenance", {}).get("doi")
                if isinstance(release_manifest.get("provenance"), Mapping)
                else None
            ),
            "release_tag": channels.get("release_tag"),
        },
    )
    _add_source_identity_blockers(
        blockers,
        expected_release_tag=expected_release_tag,
        summary=summary,
        campaign=campaign,
        release_result=release_result,
        publication=publication,
        release_manifest=release_manifest,
        receipt=receipt,
    )
    try:
        commits, goal_timeout_rows = _episode_provenance(campaign_root)
        _add_commit_blockers(
            blockers,
            commits=commits,
            campaign=campaign,
            provenance=provenance,
        )
        _add_goal_timeout_blockers(
            blockers,
            goal_timeout_rows=goal_timeout_rows,
            provenance=provenance,
        )
    except (OSError, ValueError) as exc:
        blockers.append(str(exc))

    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "expected_release_tag": expected_release_tag,
    }
