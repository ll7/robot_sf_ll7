#!/usr/bin/env python3
"""Build the thin, source-preserving dissertation-coverage aggregate.

The aggregate is a repository-side consumer contract.  It reads only the
versioned evidence packages named by issue #8201, verifies their pinned bytes
and release identity, and projects their existing status vocabulary into a
small capability table.  It does not enumerate the repository, access a
private dissertation checkout, run experiments, or promote scientific claims.

The default invocation writes the tracked manifest, readable summary, and
checksum inventory.  ``--check`` performs the same deterministic build in
memory and fails if any tracked output or input digest is stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = Path("configs/publication/dissertation_coverage_v1.yaml")
DEFAULT_SCHEMA = Path("robot_sf/benchmark/schemas/dissertation_coverage_manifest.v1.json")
DEFAULT_MANIFEST = Path("docs/context/dissertation_coverage/coverage_manifest.v1.json")
DEFAULT_SUMMARY = Path("docs/context/dissertation_coverage/coverage_summary.md")
DEFAULT_CHECKSUMS = Path("docs/context/dissertation_coverage/SHA256SUMS")

MANIFEST_SCHEMA_VERSION = "dissertation_coverage_manifest.v1"
PROFILE_SCHEMA_VERSION = "dissertation_coverage_profile.v1"
REVIEW_MARKER = "AI-GENERATED NEEDS-REVIEW"

ANCHOR_KEYS = (
    "source_commit",
    "release_tag",
    "release_doi",
    "campaign_id",
    "planner_count",
    "scenario_cell_count",
    "seed_count",
    "expected_episode_count",
)

PRESERVED_FIELDS = (
    "status",
    "relationship_to_release",
    "relationship_to_anchor",
    "release_relationship",
    "implementation_status",
    "evidence_status",
    "dissertation_relationship",
    "strongest_permitted_statement",
    "safe_sentence",
)

IMPLEMENTATION_PROJECTION = {
    "implemented_and_tested": "implemented",
    "partial_prototype": "partial",
    "proxy_baseline_only": "prototype",
    "schema_and_tooling_only": "implemented",
    "synthetic_fixture_only": "prototype",
}

PLANNER_ANCHOR_PROJECTION = {
    "included_exact_key": "present_at_anchor",
    "family_represented_by_successor": "predecessor_only",
    "diagnostic_only": "predecessor_only",
    "post_anchor": "introduced_after_anchor",
    "blocked_or_unavailable": "unknown",
    "not_relevant": "unknown",
}

DELTA_ANCHOR_PROJECTION = {
    "present_at_anchor_unchanged": "present_at_anchor",
    "introduced_after_anchor": "introduced_after_anchor",
    "materially_extended_after_anchor": "predecessor_only",
}

DISsertation_STATUS_BY_RELATION = {
    "post_anchor_candidate": "absent",
    "future_work_bridge": "future_work_mentioned",
    "repository_only": "intentionally_out_of_scope",
}

ACTION_PROJECTION = {
    "planner_development_disclosure": "planner_funnel_candidate",
    "capability_status_table": "capability_status_table_candidate",
    "outlook_status_alignment": "outlook_status_candidate",
    "repository_only_documentation": "repository_only",
}


class CoverageContractError(ValueError):
    """Raised when a pinned input or aggregate contract is not safe to use."""


def _resolve(root: Path, path: Path | str) -> Path:
    """Resolve a repository-relative path without allowing path ambiguity."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _repo_relative(root: Path, path: Path) -> str:
    """Return a portable repository-relative path for a tracked file."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise CoverageContractError(f"path is outside repository root: {path}") from exc


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CoverageContractError(f"cannot read required input {path}") from exc
    return digest.hexdigest()


def _sha256_bytes(value: str) -> str:
    """Return the SHA-256 digest of deterministic text bytes."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object or raise a contract error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageContractError(f"invalid JSON input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CoverageContractError(f"JSON input must be an object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML object or raise a contract error."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CoverageContractError(f"invalid YAML input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CoverageContractError(f"YAML input must be an object: {path}")
    return payload


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    """Return a required mapping value."""
    if key not in mapping:
        raise CoverageContractError(f"missing {key!r} in {context}")
    return mapping[key]


def _require_string(mapping: dict[str, Any], key: str, context: str) -> str:
    """Return a required non-empty string."""
    value = _required(mapping, key, context)
    if not isinstance(value, str) or not value.strip():
        raise CoverageContractError(f"{context}.{key} must be a non-empty string")
    return value


def _require_digest(value: Any, context: str) -> str:
    """Return a valid lowercase SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise CoverageContractError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_commit_sha(value: Any, context: str) -> str:
    """Return a valid 40-character lowercase Git commit SHA."""
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise CoverageContractError(f"{context} must be a lowercase 40-character commit SHA")
    return value


def _resolve_tag(root: Path, tag: str) -> str:
    """Resolve a Git tag to its peeled commit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{tag}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CoverageContractError(f"cannot resolve frozen release tag {tag!r}") from exc
    return result.stdout.strip()


def _verify_commit_exists(root: Path, commit: str, context: str) -> None:
    """Verify a provenance commit is present in the local repository object store."""
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CoverageContractError(
            f"{context} is not a resolvable repository commit: {commit}"
        ) from exc


def _record_list(payload: dict[str, Any], field: str, source_path: Path) -> list[dict[str, Any]]:
    """Return a required list of object records from a source package."""
    value = _required(payload, field, str(source_path))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise CoverageContractError(f"{source_path}.{field} must be a list of objects")
    return value


def _source_spec_by_id(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate and index source specifications from the profile."""
    specs = _required(profile, "source_packages", "profile")
    if not isinstance(specs, list) or not specs:
        raise CoverageContractError("profile.source_packages must be a non-empty list")
    indexed: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for index, raw_spec in enumerate(specs):
        if not isinstance(raw_spec, dict):
            raise CoverageContractError(f"profile.source_packages[{index}] must be an object")
        source_id = _require_string(raw_spec, "source_id", f"profile.source_packages[{index}]")
        path = _require_string(raw_spec, "path", f"profile.source_packages[{index}]")
        schema = _require_string(raw_spec, "schema", f"profile.source_packages[{index}]")
        _require_digest(
            _required(raw_spec, "sha256", f"profile.source_packages[{index}]"), source_id
        )
        _require_string(raw_spec, "record_field", f"profile.source_packages[{index}]")
        for lineage_key in ("producer_issue", "producer_pr"):
            lineage_value = _required(raw_spec, lineage_key, f"profile.source_packages[{index}]")
            if not isinstance(lineage_value, int) or lineage_value < 1:
                raise CoverageContractError(
                    f"profile.source_packages[{index}].{lineage_key} must be a positive integer"
                )
        _require_commit_sha(
            _required(raw_spec, "producer_merge_commit", f"profile.source_packages[{index}]"),
            f"{source_id}.producer_merge_commit",
        )
        if source_id in indexed:
            raise CoverageContractError(f"duplicate source_id in profile: {source_id}")
        if path in paths:
            raise CoverageContractError(f"duplicate source path in profile: {path}")
        indexed[source_id] = {**raw_spec, "schema": schema}
        paths.add(path)
    return indexed


def _verify_anchor_and_release(  # noqa: C901, PLR0912 - fail-closed contract checks stay together
    profile: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify profile anchor, tag, and repository-owned release metadata."""
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise CoverageContractError(
            "profile.schema_version is not dissertation_coverage_profile.v1"
        )
    anchor = _required(profile, "anchor", "profile")
    if not isinstance(anchor, dict):
        raise CoverageContractError("profile.anchor must be an object")
    for key in ANCHOR_KEYS:
        _required(anchor, key, "profile.anchor")
    source_commit = _require_string(anchor, "source_commit", "profile.anchor")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise CoverageContractError("profile.anchor.source_commit must be a 40-character SHA")
    release_tag = _require_string(anchor, "release_tag", "profile.anchor")
    resolved_tag = _resolve_tag(root, release_tag)
    if resolved_tag != source_commit:
        raise CoverageContractError(
            f"frozen tag {release_tag!r} resolves to {resolved_tag}, expected {source_commit}"
        )

    metadata = _required(profile, "release_metadata", "profile")
    if not isinstance(metadata, dict):
        raise CoverageContractError("profile.release_metadata must be an object")
    manifest_path = _resolve(
        root, _require_string(metadata, "manifest_path", "profile.release_metadata")
    )
    campaign_path = _resolve(
        root, _require_string(metadata, "campaign_config_path", "profile.release_metadata")
    )
    manifest_digest = _require_digest(
        _required(metadata, "manifest_sha256", "profile.release_metadata"),
        "profile.release_metadata.manifest_sha256",
    )
    campaign_digest = _require_digest(
        _required(metadata, "campaign_config_sha256", "profile.release_metadata"),
        "profile.release_metadata.campaign_config_sha256",
    )
    if _sha256(manifest_path) != manifest_digest:
        raise CoverageContractError(
            f"release manifest digest drift: {_repo_relative(root, manifest_path)}"
        )
    if _sha256(campaign_path) != campaign_digest:
        raise CoverageContractError(
            f"campaign config digest drift: {_repo_relative(root, campaign_path)}"
        )

    release_manifest = _load_yaml(manifest_path)
    campaign_config = _load_yaml(campaign_path)
    if release_manifest.get("release_id") != anchor["campaign_id"]:
        raise CoverageContractError("release manifest release_id disagrees with anchor campaign_id")
    if release_manifest.get("release_tag") != release_tag:
        raise CoverageContractError("release manifest release_tag disagrees with frozen anchor")
    publication = _required(release_manifest, "publication", "release manifest")
    if not isinstance(publication, dict) or publication.get("version_doi") != anchor["release_doi"]:
        raise CoverageContractError("release manifest version DOI disagrees with frozen anchor")
    matrix = _required(release_manifest, "matrix", "release manifest")
    if not isinstance(matrix, dict):
        raise CoverageContractError("release manifest matrix must be an object")
    expected_matrix = {
        "planner_arms": anchor["planner_count"],
        "scenarios": anchor["scenario_cell_count"],
        "seeds": anchor["seed_count"],
        "expected_episode_cells": anchor["expected_episode_count"],
    }
    for key, expected in expected_matrix.items():
        if matrix.get(key) != expected:
            raise CoverageContractError(f"release manifest matrix.{key} disagrees with profile")
    if (
        campaign_config.get("release_tag") != release_tag
        or campaign_config.get("doi") != anchor["release_doi"]
    ):
        raise CoverageContractError("campaign config release identity disagrees with frozen anchor")
    planners = campaign_config.get("planners")
    if not isinstance(planners, list) or len(planners) != anchor["planner_count"]:
        raise CoverageContractError("campaign config planner count disagrees with frozen anchor")
    if campaign_config.get("horizon") != 600:
        raise CoverageContractError("campaign config horizon disagrees with frozen H600 anchor")
    return anchor, release_manifest, campaign_config


def _source_records(  # noqa: C901 - source package validation is intentionally fail-closed
    profile: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load pinned source packages and return artifact and record views."""
    specs = _source_spec_by_id(profile)
    artifacts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for source_id, spec in specs.items():
        path = _resolve(root, spec["path"])
        actual_digest = _sha256(path)
        expected_digest = _require_digest(spec["sha256"], source_id)
        if actual_digest != expected_digest:
            raise CoverageContractError(f"source digest drift for {source_id}: {spec['path']}")
        producer_commit = _require_commit_sha(
            spec["producer_merge_commit"], f"{source_id}.producer_merge_commit"
        )
        _verify_commit_exists(root, producer_commit, f"{source_id}.producer_merge_commit")
        payload = _load_json(path)
        if payload.get("review_marker") != REVIEW_MARKER:
            raise CoverageContractError(f"source {source_id} lacks the required review marker")
        if payload.get("schema") != spec["schema"]:
            raise CoverageContractError(f"source {source_id} schema disagrees with profile")
        field = spec["record_field"]
        if field == "root":
            source_rows = [payload]
        else:
            source_rows = _record_list(payload, field, path)
        artifact = {
            "source_id": source_id,
            "path": spec["path"],
            "sha256": actual_digest,
            "schema": spec["schema"],
            "review_marker": payload["review_marker"],
            "record_field": field,
            "record_count": len(source_rows),
            "producer_issue": spec["producer_issue"],
            "producer_pr": spec["producer_pr"],
            "producer_merge_commit": spec["producer_merge_commit"],
        }
        artifacts.append(artifact)
        seen_ids: set[str] = set()
        for index, row in enumerate(source_rows):
            capability_id = row.get("candidate_id", row.get("capability_id", row.get("bridge_id")))
            if not isinstance(capability_id, str) or not capability_id.strip():
                raise CoverageContractError(
                    f"source {source_id} record {index} lacks capability ID"
                )
            if capability_id in seen_ids:
                raise CoverageContractError(
                    f"duplicate capability ID in {source_id}: {capability_id}"
                )
            seen_ids.add(capability_id)
            records.append(
                {
                    "source_id": source_id,
                    "path": spec["path"],
                    "sha256": actual_digest,
                    "record_key": capability_id,
                    "record": row,
                    "producer_issue": spec["producer_issue"],
                    "producer_pr": spec["producer_pr"],
                    "producer_merge_commit": spec["producer_merge_commit"],
                }
            )
        if source_id == "planner_development_funnel":
            _verify_planner_source(payload, root, path)
        if source_id == "post_anchor_capability_delta":
            if payload.get("row_count") != len(source_rows):
                raise CoverageContractError("post-anchor delta row_count disagrees with records")
    return artifacts, records, specs


def _verify_planner_source(payload: dict[str, Any], root: Path, path: Path) -> None:
    """Verify the planner source's own roster and evidence pointers."""
    roster = payload.get("release_roster_keys")
    if (
        payload.get("release_roster_count") != 14
        or not isinstance(roster, list)
        or len(roster) != 14
        or not all(isinstance(key, str) for key in roster)
        or len(set(roster)) != 14
    ):
        raise CoverageContractError("planner funnel does not contain the frozen 14-arm roster")
    candidates = _record_list(payload, "candidates", path)
    if payload.get("total_candidate_count") != len(candidates):
        raise CoverageContractError(
            "planner funnel total_candidate_count disagrees with candidates"
        )
    included_keys = [
        row.get("candidate_id")
        for row in candidates
        if row.get("relationship_to_release") == "included_exact_key"
    ]
    if included_keys != roster:
        raise CoverageContractError(
            "planner funnel release_roster_keys disagrees with included candidate records"
        )
    for row in candidates:
        pointer = row.get("evidence_pointer")
        if row.get("evidence_status") == "release_evaluated":
            if not isinstance(pointer, str) or not (_resolve(root, pointer).is_file()):
                raise CoverageContractError(
                    f"release-evaluated planner row has no resolvable evidence pointer: {row.get('candidate_id')}"
                )


def _verify_reconciliation(  # noqa: C901 - reconciliation has independent fail-closed checks
    profile: dict[str, Any],
    root: Path,
    release_manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify explicitly declared roster and owner-path discrepancies."""
    planner_payload_records = [
        item for item in records if item["source_id"] == "planner_development_funnel"
    ]
    source_roster = [
        item["record"]["candidate_id"]
        for item in planner_payload_records
        if item["record"].get("relationship_to_release") == "included_exact_key"
    ]
    # The source package is authoritative for its own frozen snapshot.  Do not
    # silently replace its roster with the newer release-manifest roster.
    manifest_planners = _required(release_manifest, "planners", "release manifest")
    if (
        not isinstance(manifest_planners, dict)
        or not isinstance(manifest_planners.get("keys"), list)
        or not all(isinstance(key, str) for key in manifest_planners["keys"])
    ):
        raise CoverageContractError("release manifest planners.keys must be a list")
    manifest_roster = manifest_planners["keys"]
    source_only = sorted(set(source_roster) - set(manifest_roster))
    manifest_only = sorted(set(manifest_roster) - set(source_roster))
    expected = _required(profile, "expected_source_reconciliation", "profile")
    if not isinstance(expected, dict):
        raise CoverageContractError("profile.expected_source_reconciliation must be an object")
    expected_roster = _required(
        expected, "planner_roster", "profile.expected_source_reconciliation"
    )
    if not isinstance(expected_roster, dict):
        raise CoverageContractError(
            "profile.expected_source_reconciliation.planner_roster must be an object"
        )
    if expected_roster.get("source_only") != source_only:
        raise CoverageContractError("declared planner source-only reconciliation is stale")
    if expected_roster.get("release_manifest_only") != manifest_only:
        raise CoverageContractError("declared planner release-manifest reconciliation is stale")

    observed_missing: set[tuple[str, str, str]] = set()
    for item in records:
        owner_paths = item["record"].get("owner_paths", [])
        if owner_paths is None:
            owner_paths = []
        if not isinstance(owner_paths, list) or not all(
            isinstance(path, str) for path in owner_paths
        ):
            raise CoverageContractError(f"owner_paths is invalid for {item['record_key']}")
        for owner_path in owner_paths:
            if not _resolve(root, owner_path.rstrip("/")).exists():
                observed_missing.add((item["source_id"], item["record_key"], owner_path))
    declared_missing_raw = _required(
        expected, "stale_owner_paths", "profile.expected_source_reconciliation"
    )
    if not isinstance(declared_missing_raw, list):
        raise CoverageContractError("expected stale_owner_paths must be a list")
    declared_missing: set[tuple[str, str, str]] = set()
    for index, entry in enumerate(declared_missing_raw):
        if not isinstance(entry, dict):
            raise CoverageContractError(f"stale_owner_paths[{index}] must be an object")
        declared_missing.add(
            (
                _require_string(entry, "source_id", f"stale_owner_paths[{index}]"),
                _require_string(entry, "capability_id", f"stale_owner_paths[{index}]"),
                _require_string(entry, "path", f"stale_owner_paths[{index}]"),
            )
        )
    if observed_missing != declared_missing:
        raise CoverageContractError(
            "owner-path reconciliation is stale: observed missing paths differ from profile"
        )
    return {
        "planner_roster": {
            "status": "conflict_explicitly_recorded" if source_only or manifest_only else "aligned",
            "source_snapshot": source_roster,
            "release_manifest": manifest_roster,
            "source_only": source_only,
            "release_manifest_only": manifest_only,
        },
        "owner_paths": {
            "status": "known_stale_paths_explicitly_recorded"
            if declared_missing
            else "all_resolvable",
            "stale_paths": [
                {"source_id": source_id, "capability_id": capability_id, "path": path}
                for source_id, capability_id, path in sorted(declared_missing)
            ],
        },
    }


def _normalized_anchor_relation(item: dict[str, Any]) -> tuple[str, str, str]:
    """Return projected anchor relation, source field, and raw value."""
    record = item["record"]
    source_id = item["source_id"]
    if source_id == "planner_development_funnel":
        field = "relationship_to_release"
        raw = record.get(field)
        projected = PLANNER_ANCHOR_PROJECTION.get(raw, "unknown")
    elif source_id == "post_anchor_capability_delta":
        field = "status"
        raw = record.get(field)
        projected = DELTA_ANCHOR_PROJECTION.get(raw, "unknown")
        if record.get("dissertation_relationship") == "repository_only":
            projected = "operational_only"
    else:
        field = "relationship_to_anchor"
        raw = record.get(field)
        projected = (
            raw
            if raw
            in {
                "present_at_anchor",
                "predecessor_only",
                "introduced_after_anchor",
                "operational_only",
                "unknown",
            }
            else "unknown"
        )
    if not isinstance(raw, str) or not raw:
        raise CoverageContractError(
            f"{source_id}:{item['record_key']} lacks source anchor relation"
        )
    return projected, field, raw


def _source_priority(item: dict[str, Any]) -> tuple[int, str, str]:
    """Prefer detailed future-work cards, then delta rows, then funnel rows."""
    source_id = item["source_id"]
    priority = (
        0
        if source_id.startswith("future_work_card_")
        else 1
        if source_id == "post_anchor_capability_delta"
        else 2
    )
    return priority, source_id, item["record_key"]


def _raw_source_fields(item: dict[str, Any]) -> dict[str, Any]:
    """Select source fields that make the aggregate auditable without re-enumerating inputs."""
    return {field: item["record"][field] for field in PRESERVED_FIELDS if field in item["record"]}


def _mapped_implementation(record: dict[str, Any]) -> str:
    """Project implementation wording to the parent contract, retaining raw status separately."""
    raw = record.get("implementation_status")
    if raw is None:
        return "unknown"
    if not isinstance(raw, str):
        raise CoverageContractError("implementation_status must be a string")
    return IMPLEMENTATION_PROJECTION.get(raw, "unknown")


def _merge_status(items: list[dict[str, Any]], field: str, *, mapped: bool = False) -> str:
    """Merge a status only when all source-provided values agree."""
    values = []
    for item in items:
        if mapped and "implementation_status" not in item["record"]:
            continue
        value = _mapped_implementation(item["record"]) if mapped else item["record"].get(field)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise CoverageContractError(f"{field} must be a non-empty source string")
            values.append(value)
    unique = sorted(set(values))
    if len(unique) > 1:
        raise CoverageContractError(
            f"conflicting {field} values for {items[0]['record_key']}: {unique}"
        )
    return unique[0] if unique else "unknown"


def _dissertation_relationship(items: list[dict[str, Any]]) -> str:
    """Return source relationship wording or an explicit unavailable sentinel."""
    values = [
        item["record"].get("dissertation_relationship")
        for item in items
        if item["record"].get("dissertation_relationship") is not None
    ]
    if not values and any(item["source_id"].startswith("future_work_card_") for item in items):
        # The card package is explicitly a future-work bridge package.  This
        # names the package relationship, not a new dissertation finding.
        return "future_work_bridge"
    unique = sorted(set(values))
    if len(unique) > 1:
        raise CoverageContractError(
            f"conflicting dissertation_relationship values for {items[0]['record_key']}: {unique}"
        )
    return unique[0] if unique else "not_reported_by_source"


def _dissertation_status(relationship: str) -> str:
    """Project source relationship into the conservative summary vocabulary."""
    return DISsertation_STATUS_BY_RELATION.get(relationship, "unknown")


def _missing_proof(items: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Union source missing-proof statements while preserving each source list."""
    by_source: list[dict[str, Any]] = []
    values: set[str] = set()
    for item in sorted(items, key=_source_priority):
        raw = item["record"].get("missing_proof")
        if raw is None:
            raw_values = [
                "Not provided by this source package; consult the source before stronger wording."
            ]
            source_value: Any = None
        elif isinstance(raw, list) and all(isinstance(value, str) for value in raw):
            raw_values = list(raw)
            source_value = raw
        else:
            raise CoverageContractError(f"missing_proof is invalid for {item['record_key']}")
        values.update(raw_values)
        by_source.append(
            {
                "source_id": item["source_id"],
                "record_key": item["record_key"],
                "value": source_value,
                "effective_value": raw_values,
            }
        )
    return sorted(values), by_source


def _recommended_action(
    items: list[dict[str, Any]], dissertation_relationship: str
) -> tuple[str, list[str]]:
    """Project source downstream hints without using issue closure as evidence."""
    raw_actions: list[str] = []
    for item in items:
        source_id = item["source_id"]
        record = item["record"]
        if source_id == "planner_development_funnel":
            raw_actions.append("planner_funnel_candidate")
        for raw_action in record.get("candidates_for", []) or []:
            if not isinstance(raw_action, str):
                raise CoverageContractError(f"candidates_for is invalid for {item['record_key']}")
            if raw_action in ACTION_PROJECTION:
                raw_actions.append(ACTION_PROJECTION[raw_action])
        if source_id.startswith("future_work_card_"):
            raw_actions.append("outlook_status_candidate")
        if dissertation_relationship == "repository_only":
            raw_actions.append("repository_only")
    unique = sorted(set(raw_actions))
    priority = [
        "planner_funnel_candidate",
        "capability_status_table_candidate",
        "outlook_status_candidate",
        "repository_only",
    ]
    for candidate in priority:
        if candidate in unique:
            return candidate, unique
    return "none", unique


def _aggregate_row(items: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    """Build one thin aggregate row from one or more source records."""
    ordered = sorted(items, key=_source_priority)
    primary = ordered[0]["record"]
    capability_id = items[0]["record_key"]
    anchor_relations = []
    for item in sorted(items, key=lambda value: (value["source_id"], value["record_key"])):
        projected, field, raw = _normalized_anchor_relation(item)
        anchor_relations.append(
            {
                "source_id": item["source_id"],
                "field": field,
                "value": raw,
                "projected_value": projected,
            }
        )
    projected_relations = sorted({entry["projected_value"] for entry in anchor_relations})
    if len(projected_relations) != 1:
        raise CoverageContractError(
            f"conflicting anchor relations for {capability_id}: {projected_relations}"
        )
    implementation_status = _merge_status(items, "implementation_status", mapped=True)
    evidence_status = _merge_status(items, "evidence_status")
    dissertation_relationship = _dissertation_relationship(items)
    dissertation_status = _dissertation_status(dissertation_relationship)
    missing_proof, source_missing_proof = _missing_proof(items)
    claim_variants = []
    for item in ordered:
        record = item["record"]
        wording = record.get("safe_sentence", record.get("strongest_permitted_statement"))
        if isinstance(wording, str) and wording not in claim_variants:
            claim_variants.append(wording)
    if not claim_variants:
        raise CoverageContractError(f"no permitted wording for {capability_id}")
    source_issues = sorted(
        {
            issue
            for item in items
            for issue in [
                item["producer_issue"],
                *(item["record"].get("linked_issues", []) or []),
            ]
            if isinstance(issue, int)
        }
    )
    source_lineage = [
        {
            "source_id": item["source_id"],
            "issue": item["producer_issue"],
            "pr": item["producer_pr"],
            "merge_commit": item["producer_merge_commit"],
        }
        for item in sorted(items, key=lambda value: (value["source_id"], value["record_key"]))
    ]
    owner_paths = sorted(
        {
            path
            for item in items
            for path in item["record"].get("owner_paths", []) or []
            if isinstance(path, str)
        }
    )
    owner_path_status = []
    for owner_path in owner_paths:
        owner_path_status.append(
            {
                "path": owner_path,
                "status": "present"
                if _resolve(root, owner_path.rstrip("/")).exists()
                else "missing_expected_source_path",
                "source_ids": sorted(
                    {
                        item["source_id"]
                        for item in items
                        if owner_path in (item["record"].get("owner_paths", []) or [])
                    }
                ),
            }
        )
    first_commit_items = [
        item for item in ordered if isinstance(item["record"].get("first_commit"), str)
    ]
    if first_commit_items:
        first_commit = {
            "value": first_commit_items[0]["record"]["first_commit"],
            "availability": "source_reported",
            "source_id": first_commit_items[0]["source_id"],
            "field": "first_commit",
        }
    else:
        first_commit = {
            "value": None,
            "availability": "unavailable_in_source",
            "source_id": None,
            "field": None,
            "reason": "No first-known commit field is present in the named source packages.",
        }
    downstream_action, source_actions = _recommended_action(items, dissertation_relationship)
    source_records = []
    for item in sorted(items, key=lambda value: (value["source_id"], value["record_key"])):
        source_records.append(
            {
                "source_id": item["source_id"],
                "path": item["path"],
                "record_key": item["record_key"],
                "fields": _raw_source_fields(item),
            }
        )
    return {
        "capability_id": capability_id,
        "title": primary.get("title", primary.get("display_name", capability_id)),
        "category": primary.get("category", primary.get("family", "future_work_bridge")),
        "owner_paths": owner_paths,
        "owner_path_status": owner_path_status,
        "source_issues": source_issues,
        "source_lineage": source_lineage,
        "first_known_commit": first_commit,
        "anchor_relation": projected_relations[0],
        "source_anchor_relations": anchor_relations,
        "implementation_status": implementation_status,
        "source_implementation_statuses": [
            {
                "source_id": item["source_id"],
                "value": item["record"].get("implementation_status"),
            }
            for item in source_records_for(items)
        ],
        "evidence_status": evidence_status,
        "source_evidence_statuses": [
            {"source_id": item["source_id"], "value": item["record"].get("evidence_status")}
            for item in source_records_for(items)
        ],
        "dissertation_relationship": dissertation_relationship,
        "dissertation_status": dissertation_status,
        "strongest_permitted_wording": claim_variants[0],
        "claim_boundary_variants": claim_variants,
        "missing_proof": missing_proof,
        "source_missing_proof": source_missing_proof,
        "recommended_downstream_action": downstream_action,
        "source_recommended_actions": source_actions,
        "source_paths": sorted({item["path"] for item in items}),
        "source_digests": [
            {"source_id": item["source_id"], "path": item["path"], "sha256": item["sha256"]}
            for item in sorted(items, key=lambda value: (value["source_id"], value["record_key"]))
        ],
        "source_records": source_records,
    }


def source_records_for(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return source record items in deterministic order."""
    return sorted(items, key=lambda value: (value["source_id"], value["record_key"]))


def _build_payload(profile_path: Path, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the deterministic manifest payload and its complete input inventory."""
    profile = _load_yaml(profile_path)
    anchor, release_manifest, _campaign_config = _verify_anchor_and_release(profile, root)
    artifacts, records, _ = _source_records(profile, root)
    reconciliation = _verify_reconciliation(profile, root, release_manifest, records)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault(item["record_key"], []).append(item)
    capabilities = [_aggregate_row(grouped[key], root) for key in sorted(grouped)]
    if len({row["capability_id"] for row in capabilities}) != len(capabilities):
        raise CoverageContractError("duplicate capability IDs remain after aggregation")
    counts = {
        "capabilities": len(capabilities),
        "by_anchor_relation": dict(
            sorted(Counter(row["anchor_relation"] for row in capabilities).items())
        ),
        "by_evidence_status": dict(
            sorted(Counter(row["evidence_status"] for row in capabilities).items())
        ),
        "by_dissertation_status": dict(
            sorted(Counter(row["dissertation_status"] for row in capabilities).items())
        ),
    }
    consumer_profile = {
        "consumer_id": profile["consumer_id"],
        "repository": profile["repository"],
        **{key: anchor[key] for key in ANCHOR_KEYS},
    }
    source_references = sorted(artifacts, key=lambda item: item["source_id"])
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "review_marker": REVIEW_MARKER,
        "consumer_profile": consumer_profile,
        "source_packages": source_references,
        "source_reconciliation": reconciliation,
        "summary_counts": counts,
        "projection_policy": {
            "status": "source_values_preserved; projections_are_mechanical_and_non_promotional",
            "anchor_relation": "The aggregate maps source release/anchor fields to the parent relation vocabulary and retains every raw value in source_anchor_relations.",
            "implementation_status": "The aggregate maps source implementation wording to the parent vocabulary and retains every raw value in source_implementation_statuses.",
            "evidence_status": "Evidence status is copied from source packages; disagreements fail closed.",
            "strongest_permitted_wording": "The first wording follows detailed future-work card, post-anchor delta, then planner-funnel source priority; all variants remain listed.",
        },
        "claim_boundary": "This is repository-side provenance metadata only. It does not establish new benchmark performance, planner superiority, physical transfer, dissertation coverage, or a manuscript claim.",
        "capabilities": capabilities,
    }
    checksum_inputs = [
        _repo_relative(root, profile_path),
        _repo_relative(root, _resolve(root, profile["release_metadata"]["manifest_path"])),
        _repo_relative(root, _resolve(root, profile["release_metadata"]["campaign_config_path"])),
        *(item["path"] for item in source_references),
    ]
    return payload, sorted(set(checksum_inputs))


def _markdown_value(value: Any) -> str:
    """Render a compact, pipe-safe Markdown cell."""
    if value is None:
        return "not provided by source"
    if isinstance(value, list):
        return "<br>".join(_markdown_value(item) for item in value)
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _summary(payload: dict[str, Any], profile_path: Path) -> str:
    """Render the generated claim-neutral readable summary."""
    profile = payload["consumer_profile"]
    lines = [
        "<!-- AI-GENERATED NEEDS-REVIEW -->",
        "# Dissertation Coverage Aggregate",
        "",
        "This is a repository-side provenance map for downstream review. It preserves the named source packages and their status wording; it is not benchmark evidence and does not edit or read a dissertation repository.",
        "",
        "## Frozen consumer profile",
        "",
        f"- Profile: `{profile_path.as_posix()}`",
        f"- Consumer: `{profile['consumer_id']}`",
        f"- Repository: `{profile['repository']}`",
        f"- Anchor source commit: `{profile['source_commit']}`",
        f"- Release tag: `{profile['release_tag']}`",
        f"- DOI: `{profile['release_doi']}`",
        f"- Campaign identity: `{profile['campaign_id']}`",
        f"- Frozen matrix identity: {profile['planner_count']} planners × {profile['scenario_cell_count']} scenario cells × {profile['seed_count']} seeds = {profile['expected_episode_count']} expected episodes.",
        "",
        "## Source packages",
        "",
        "| Source | Producer | Schema | Records | SHA-256 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for source in payload["source_packages"]:
        lines.append(
            f"| `{source['path']}` | issue #{source['producer_issue']} / PR #{source['producer_pr']} "
            f"({source['producer_merge_commit'][:12]}) | `{source['schema']}` | "
            f"{source['record_count']} | `{source['sha256']}` |"
        )
    reconciliation = payload["source_reconciliation"]
    roster = reconciliation["planner_roster"]
    owners = reconciliation["owner_paths"]
    lines.extend(
        [
            "",
            "## Explicit source reconciliation",
            "",
            f"- Planner roster: `{roster['status']}`. Source-only keys: `{', '.join(roster['source_only']) or 'none'}`; current release-manifest-only keys: `{', '.join(roster['release_manifest_only']) or 'none'}`.",
            f"- Owner paths: `{owners['status']}`. Missing paths named by the source: `{', '.join(item['path'] for item in owners['stale_paths']) or 'none'}`.",
            "- These discrepancies are retained as source-accounting facts. They are not repaired, inferred away, or used to promote evidence.",
            "",
            "## Coverage counts",
            "",
        ]
    )
    for label, values in (
        ("anchor relation", payload["summary_counts"]["by_anchor_relation"]),
        ("evidence status", payload["summary_counts"]["by_evidence_status"]),
        ("dissertation status", payload["summary_counts"]["by_dissertation_status"]),
    ):
        lines.append(
            f"- By {label}: " + ", ".join(f"`{key}`={value}" for key, value in values.items()) + "."
        )
    lines.extend(
        [
            "",
            "## Capability rows",
            "",
            "The row contract is: capability → relation to frozen dissertation anchor → implementation status → evidence status → dissertation relationship → strongest permitted wording → exact missing proof.",
            "",
            "| Capability | Anchor relation | Implementation | Evidence | Dissertation relationship | Strongest permitted wording | Exact missing proof |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["capabilities"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{row['capability_id']}`<br>{_markdown_value(row['title'])}",
                    f"`{row['anchor_relation']}`",
                    f"`{row['implementation_status']}`",
                    f"`{row['evidence_status']}`",
                    f"`{row['dissertation_relationship']}`",
                    _markdown_value(row["strongest_permitted_wording"]),
                    _markdown_value(row["missing_proof"]),
                )
            )
            + " |"
        )
    repository_only = [
        row
        for row in payload["capabilities"]
        if row["dissertation_status"] == "intentionally_out_of_scope"
    ]
    lines.extend(
        [
            "",
            "## Repository-only capabilities",
            "",
            "Operational rows remain separate from dissertation scientific findings:",
            "",
            "| Capability | Evidence | Permitted wording |",
            "| --- | --- | --- |",
        ]
    )
    for row in repository_only:
        lines.append(
            f"| `{row['capability_id']}` | `{row['evidence_status']}` | {_markdown_value(row['strongest_permitted_wording'])} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary and rebuild",
            "",
            f"{payload['claim_boundary']}",
            "",
            "```text",
            "uv run python scripts/analysis/build_dissertation_coverage_manifest.py --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _checksum_text(
    root: Path,
    inventory: list[str],
    manifest_path: Path,
    manifest_bytes: str,
    summary_path: Path,
    summary_bytes: str,
) -> str:
    """Return a deterministic checksum inventory for inputs and generated outputs."""
    paths = [*inventory, _repo_relative(root, manifest_path), _repo_relative(root, summary_path)]
    unique_paths = sorted(set(paths))
    lines = ["# AI-GENERATED NEEDS-REVIEW"]
    for relative_path in unique_paths:
        if relative_path == _repo_relative(root, manifest_path):
            digest = _sha256_bytes(manifest_bytes)
        elif relative_path == _repo_relative(root, summary_path):
            digest = _sha256_bytes(summary_bytes)
        else:
            digest = _sha256(root / relative_path)
        lines.append(f"{digest}  {relative_path}")
    return "\n".join(lines) + "\n"


def _validate_payload(payload: dict[str, Any], schema_path: Path) -> None:
    """Validate the generated payload against its tracked JSON schema."""
    schema = _load_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)
    except Exception as exc:  # jsonschema exposes several concrete exception types.
        raise CoverageContractError(f"manifest schema validation failed: {exc}") from exc


def build_outputs(
    *,
    root: Path = REPO_ROOT,
    profile_path: Path = DEFAULT_PROFILE,
    schema_path: Path = DEFAULT_SCHEMA,
    manifest_path: Path = DEFAULT_MANIFEST,
    summary_path: Path = DEFAULT_SUMMARY,
    checksums_path: Path = DEFAULT_CHECKSUMS,
) -> tuple[dict[str, Any], str, str]:
    """Build expected manifest, summary, and checksum bytes without writing."""
    profile_file = _resolve(root, profile_path)
    schema_file = _resolve(root, schema_path)
    manifest_file = _resolve(root, manifest_path)
    summary_file = _resolve(root, summary_path)
    payload, inventory = _build_payload(profile_file, root)
    _validate_payload(payload, schema_file)
    summary = _summary(payload, Path(_repo_relative(root, profile_file)))
    # The checksum inventory includes profile/source/release inputs and both
    # generated outputs.  It deliberately excludes itself to avoid recursion.
    manifest_bytes = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    checksums = _checksum_text(
        root,
        inventory,
        manifest_file,
        manifest_bytes,
        summary_file,
        summary,
    )
    return payload, summary, checksums


def write_outputs(
    payload: dict[str, Any],
    summary: str,
    checksums: str,
    *,
    root: Path,
    manifest_path: Path,
    summary_path: Path,
    checksums_path: Path,
) -> None:
    """Write deterministic generated outputs."""
    manifest_file = _resolve(root, manifest_path)
    summary_file = _resolve(root, summary_path)
    checksums_file = _resolve(root, checksums_path)
    for path in (manifest_file, summary_file, checksums_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_file.write_text(summary, encoding="utf-8")
    checksums_file.write_text(checksums, encoding="utf-8")


def check_outputs(
    payload: dict[str, Any],
    summary: str,
    checksums: str,
    *,
    root: Path,
    manifest_path: Path,
    summary_path: Path,
    checksums_path: Path,
) -> None:
    """Fail if tracked generated outputs are absent or stale."""
    manifest_file = _resolve(root, manifest_path)
    summary_file = _resolve(root, summary_path)
    checksums_file = _resolve(root, checksums_path)
    expected_manifest = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    for path, expected in (
        (manifest_file, expected_manifest),
        (summary_file, summary),
        (checksums_file, checksums),
    ):
        try:
            actual = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CoverageContractError(f"generated output is missing: {path}") from exc
        if actual != expected:
            raise CoverageContractError(f"generated output is stale: {_repo_relative(root, path)}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    parser.add_argument("--check", action="store_true", help="check outputs without writing")
    args = parser.parse_args(argv)
    try:
        payload, summary, checksums = build_outputs(
            root=REPO_ROOT,
            profile_path=args.profile,
            schema_path=args.schema,
            manifest_path=args.manifest,
            summary_path=args.summary,
            checksums_path=args.checksums,
        )
        if args.check:
            check_outputs(
                payload,
                summary,
                checksums,
                root=REPO_ROOT,
                manifest_path=args.manifest,
                summary_path=args.summary,
                checksums_path=args.checksums,
            )
            print(
                f"dissertation coverage outputs are current ({payload['summary_counts']['capabilities']} capabilities)"
            )
        else:
            write_outputs(
                payload,
                summary,
                checksums,
                root=REPO_ROOT,
                manifest_path=args.manifest,
                summary_path=args.summary,
                checksums_path=args.checksums,
            )
            print(
                f"wrote dissertation coverage outputs ({payload['summary_counts']['capabilities']} capabilities)"
            )
        return 0
    except CoverageContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
