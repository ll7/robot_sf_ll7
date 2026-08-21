#!/usr/bin/env python3
"""Materialize the analysis-only collision-pressure slice from release 0.0.3.post1.

This script reads the published release archive without running a campaign.  It
verifies the outer archive and the archive's own payload checksums, adapts only
the two locators required by the generic collision-pressure report, and writes a
deterministic evidence packet for RobotSF issue #7724 / dissertation issue #613.

The output remains ``analysis_only``.  It does not admit evidence, alter the
dissertation, or define collision semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from robot_sf.benchmark.collision.collision_pressure_report import (
    CollisionPressureReportError,
    build_collision_pressure_report,
    write_collision_pressure_report,
)
from robot_sf.evidence.writers import write_json, write_review_sidecar, write_sha256sums, write_text

EXPECTED_RELEASE_TAG = "0.0.3.post1"
EXPECTED_RELEASE_ID = "paper_experiment_matrix_v2_h600_s30_v0_0_3_post1"
EXPECTED_PUBLICATION_COMMIT = "ded9027d2928512c14bc241397e0ab1d8f586654"
EXPECTED_ROW_PRODUCTION_COMMIT = "a307ef276d701f8d14dead1aa0513f44ee97c0b0"
EXPECTED_BUNDLE_SHA256 = "9bf6ea35a17ce812f0a9c841c3681bc072dcf7ba8c121cbcf05113b8514f4de1"
EXPECTED_BUNDLE_ASSET_NAME = (
    "paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_post1_corrected_"
    "publication_bundle.tar.gz"
)
EXPECTED_BUNDLE_ROOT = EXPECTED_BUNDLE_ASSET_NAME.removesuffix(".tar.gz")

EXPECTED_TOTAL_ROWS = 20_160
EXPECTED_RUN_COUNT = 14
EXPECTED_ROWS_PER_RUN = 1_440
ELIGIBLE_FAMILIES = ("doorway", "narrow_doorway", "robot_crowding")
EXPECTED_ELIGIBLE_ROWS = 2_100
EXPECTED_DENOMINATOR_DIGEST = "32821e179f55f5876fa25753ffa0a04a071fc76c84815372581a3a9d710365d5"
EXPECTED_CONTACT_EPISODES = 1_545
EXPECTED_COLLISION_EVENTS = 1_546
EXPECTED_PARTNER_EPISODE_COUNTS = {
    "pedestrian": 520,
    "static_geometry": 1_026,
    "boundary": 0,
    "goal_artifact": 0,
}
EXPECTED_OBSTACLE_EPISODES = 1_026
EXPECTED_OVERLAP_COUNTS = {
    "pedestrian_only": 519,
    "obstacle_only": 1_025,
    "pedestrian_and_obstacle": 1,
}
EXPECTED_FAMILY_COUNTS = {
    "doorway": {"eligible_episode_count": 1_260, "contact_episode_count": 948},
    "narrow_doorway": {"eligible_episode_count": 420, "contact_episode_count": 353},
    "robot_crowding": {"eligible_episode_count": 420, "contact_episode_count": 244},
}
EXPECTED_MISSING_OPTIONAL_FIELDS = {
    "collision_partner_id": 1_026,
    "relative_speed_at_contact": 0,
}
EXPECTED_MISSING_PARTNER_TYPES = {"static_geometry": 1_026}
EXPECTED_PAYLOAD_CHECKSUM_ENTRIES = 85
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/diss_613_collision_pressure_release_0_0_3_post1")
REPORT_SCHEMA_VERSION = "diss_613_collision_pressure_release_0_0_3_post1.v1"
_SHA256_HEX = set("0123456789abcdef")


class CollisionPressureReleaseError(ValueError):
    """Raised when the immutable release input cannot satisfy the packet contract."""


@dataclass(frozen=True)
class VerifiedArchive:
    """Verified archive identity and source-file checksums."""

    root_name: str
    archive_sha256: str
    checksums: dict[str, str]
    episode_members: tuple[str, ...]
    breakdown_member: str
    release_id: str
    publication_commit: str
    row_production_commit: str
    planner_keys: tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of bytes."""

    return hashlib.sha256(data).hexdigest()


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    """Require an immutable release field to match its declared identity."""

    if actual != expected:
        raise CollisionPressureReleaseError(
            f"{field} mismatch: expected {expected!r}, got {actual!r}"
        )


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    """Require a JSON object at a named release path."""

    if not isinstance(value, Mapping):
        raise CollisionPressureReleaseError(f"{field} must be a JSON object")
    return value


def _safe_member_name(name: str) -> str:
    """Normalize a tar member name while rejecting traversal."""

    normalized = name.rstrip("/")
    path = Path(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise CollisionPressureReleaseError(f"unsafe archive member path: {name!r}")
    return normalized


def _index_members(tar: tarfile.TarFile) -> tuple[str, dict[str, tarfile.TarInfo]]:
    """Index regular and directory members and require one safe archive root."""

    indexed: dict[str, tarfile.TarInfo] = {}
    for member in tar.getmembers():
        normalized = _safe_member_name(member.name)
        if normalized in indexed:
            raise CollisionPressureReleaseError(f"duplicate archive member: {normalized}")
        indexed[normalized] = member

    roots = {name.split("/", 1)[0] for name in indexed}
    if roots != {EXPECTED_BUNDLE_ROOT}:
        raise CollisionPressureReleaseError(
            f"archive root mismatch: expected {EXPECTED_BUNDLE_ROOT!r}, got {sorted(roots)!r}"
        )
    root_member = indexed.get(EXPECTED_BUNDLE_ROOT)
    if root_member is None or not root_member.isdir():
        raise CollisionPressureReleaseError("archive is missing its expected root directory")
    return EXPECTED_BUNDLE_ROOT, indexed


def _read_member(tar: tarfile.TarFile, member: tarfile.TarInfo, name: str) -> bytes:
    """Read a regular archive member without extracting it to disk."""

    if not member.isreg():
        raise CollisionPressureReleaseError(f"archive member is not a regular file: {name}")
    handle = tar.extractfile(member)
    if handle is None:
        raise CollisionPressureReleaseError(f"cannot read archive member: {name}")
    return handle.read()


def _read_json_member(tar: tarfile.TarFile, member: tarfile.TarInfo, name: str) -> dict[str, Any]:
    """Read one object-valued JSON archive member."""

    try:
        payload = json.loads(_read_member(tar, member, name).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollisionPressureReleaseError(
            f"invalid JSON in archive member {name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CollisionPressureReleaseError(f"archive member {name} must contain a JSON object")
    return payload


def _required_member(members: Mapping[str, tarfile.TarInfo], name: str) -> tarfile.TarInfo:
    """Return a required archive member or raise a packet-contract error."""

    member = members.get(name)
    if member is None:
        raise CollisionPressureReleaseError(f"release archive is missing {name}")
    return member


def _parse_checksums(raw: bytes, *, member_name: str) -> dict[str, str]:
    """Parse the release payload checksum list."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CollisionPressureReleaseError(
            f"checksum manifest is not UTF-8: {member_name}"
        ) from exc

    checksums: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            raise CollisionPressureReleaseError(
                f"malformed checksum line {line_number} in {member_name}"
            )
        digest, path = parts
        path = path.removeprefix("*")
        if len(digest) != 64 or any(char not in _SHA256_HEX for char in digest):
            raise CollisionPressureReleaseError(
                f"invalid SHA-256 on line {line_number} in {member_name}"
            )
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise CollisionPressureReleaseError(
                f"unsafe checksum path on line {line_number} in {member_name}: {path!r}"
            )
        if path in checksums:
            raise CollisionPressureReleaseError(f"duplicate checksum path: {path}")
        checksums[path] = digest
    if len(checksums) != EXPECTED_PAYLOAD_CHECKSUM_ENTRIES:
        raise CollisionPressureReleaseError(
            "release payload checksum count mismatch: "
            f"expected {EXPECTED_PAYLOAD_CHECKSUM_ENTRIES}, got {len(checksums)}"
        )
    return dict(sorted(checksums.items()))


def _verify_payload_checksums(
    tar: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    *,
    root_name: str,
    checksums: Mapping[str, str],
) -> None:
    """Verify every payload file named by the release checksum manifest."""

    for relative_name, expected in checksums.items():
        full_name = f"{root_name}/{relative_name}"
        member = members.get(full_name)
        if member is None:
            raise CollisionPressureReleaseError(
                f"checksum manifest names missing archive member: {relative_name}"
            )
        actual = _sha256_bytes(_read_member(tar, member, full_name))
        if actual != expected:
            raise CollisionPressureReleaseError(
                f"payload checksum mismatch for {relative_name}: expected {expected}, got {actual}"
            )


def verify_release_archive(bundle_path: Path) -> VerifiedArchive:
    """Verify archive identity, internal checksums, and release manifests."""

    if not bundle_path.is_file():
        raise CollisionPressureReleaseError(f"release archive is not a file: {bundle_path}")
    archive_sha256 = sha256_file(bundle_path)
    if archive_sha256 != EXPECTED_BUNDLE_SHA256:
        raise CollisionPressureReleaseError(
            f"release archive SHA-256 mismatch: expected {EXPECTED_BUNDLE_SHA256}, "
            f"got {archive_sha256}"
        )

    try:
        tar = tarfile.open(bundle_path, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise CollisionPressureReleaseError(f"cannot open release archive: {exc}") from exc

    with tar:
        root_name, members = _index_members(tar)
        checksums_name = f"{root_name}/checksums.sha256"
        checksums_member = members.get(checksums_name)
        if checksums_member is None:
            raise CollisionPressureReleaseError("release archive is missing checksums.sha256")
        checksums = _parse_checksums(
            _read_member(tar, checksums_member, checksums_name), member_name=checksums_name
        )
        _verify_payload_checksums(tar, members, root_name=root_name, checksums=checksums)

        publication_name = f"{root_name}/publication_manifest.json"
        publication = _read_json_member(
            tar, _required_member(members, publication_name), publication_name
        )
        payload_manifest_name = f"{root_name}/payload/manifest.json"
        payload_manifest = _read_json_member(
            tar, _required_member(members, payload_manifest_name), payload_manifest_name
        )
        release_manifest_name = f"{root_name}/payload/release/release_manifest.resolved.json"
        release_manifest = _read_json_member(
            tar, _required_member(members, release_manifest_name), release_manifest_name
        )

        _require_equal(
            publication.get("bundle_name"), root_name, "publication_manifest.bundle_name"
        )
        publication_channels = _require_mapping(
            publication.get("publication_channels"), "publication_manifest.publication_channels"
        )
        _require_equal(
            publication_channels.get("release_tag"), EXPECTED_RELEASE_TAG, "publication release tag"
        )
        publication_provenance = _require_mapping(
            publication.get("provenance"), "publication_manifest.provenance"
        )
        publication_repository = _require_mapping(
            publication_provenance.get("repository"), "publication_manifest.provenance.repository"
        )
        _require_equal(
            publication_repository.get("commit"),
            EXPECTED_PUBLICATION_COMMIT,
            "publication repository commit",
        )
        commit_reconciliation = _require_mapping(
            publication_provenance.get("commit_reconciliation"),
            "publication_manifest.provenance.commit_reconciliation",
        )
        _require_equal(
            commit_reconciliation.get("publication_commit"),
            EXPECTED_PUBLICATION_COMMIT,
            "publication commit reconciliation",
        )
        _require_equal(
            commit_reconciliation.get("execution_commit"),
            EXPECTED_ROW_PRODUCTION_COMMIT,
            "execution commit reconciliation",
        )
        _require_equal(
            payload_manifest.get("git_hash"), EXPECTED_PUBLICATION_COMMIT, "payload git_hash"
        )
        payload_release = _require_mapping(
            payload_manifest.get("benchmark_release"), "payload manifest benchmark_release"
        )
        _require_equal(
            payload_release.get("release_tag"), EXPECTED_RELEASE_TAG, "payload release tag"
        )
        _require_equal(payload_release.get("release_id"), EXPECTED_RELEASE_ID, "payload release id")
        _require_equal(
            release_manifest.get("release_tag"), EXPECTED_RELEASE_TAG, "release manifest tag"
        )
        _require_equal(
            release_manifest.get("release_id"), EXPECTED_RELEASE_ID, "release manifest id"
        )
        release_planners = _require_mapping(
            release_manifest.get("planners"), "release manifest planners"
        )
        planner_keys = release_planners.get("keys")
        if (
            not isinstance(planner_keys, list)
            or not planner_keys
            or any(not isinstance(key, str) or not key for key in planner_keys)
            or len(set(planner_keys)) != len(planner_keys)
        ):
            raise CollisionPressureReleaseError(
                "release manifest planners.keys must be a unique non-empty string list"
            )
        kinematics = _require_mapping(
            release_manifest.get("kinematics"), "release manifest kinematics"
        )
        _require_equal(
            kinematics.get("matrix"), ["differential_drive"], "release kinematics matrix"
        )

        episode_members = tuple(
            sorted(
                name[len(root_name) + 1 :]
                for name, member in members.items()
                if member.isreg()
                and name.startswith(f"{root_name}/payload/runs/")
                and Path(name).parts[-1] == "episodes.jsonl"
                and len(Path(name).parts) == len(Path(root_name).parts) + 4
            )
        )
        if len(episode_members) != EXPECTED_RUN_COUNT:
            raise CollisionPressureReleaseError(
                f"episode run-file count mismatch: expected {EXPECTED_RUN_COUNT}, "
                f"got {len(episode_members)}"
            )
        breakdown_member = "payload/reports/scenario_family_breakdown.csv"
        if f"{root_name}/{breakdown_member}" not in members:
            raise CollisionPressureReleaseError(
                "release archive is missing scenario_family_breakdown.csv"
            )
        run_arms = tuple(
            sorted(
                Path(member).parts[-2].removesuffix("__differential_drive")
                for member in episode_members
            )
        )
        _require_equal(
            run_arms,
            tuple(sorted(planner_keys)),
            "release planner/run-arm keys",
        )
        return VerifiedArchive(
            root_name=root_name,
            archive_sha256=archive_sha256,
            checksums=checksums,
            episode_members=episode_members,
            breakdown_member=breakdown_member,
            release_id=EXPECTED_RELEASE_ID,
            publication_commit=EXPECTED_PUBLICATION_COMMIT,
            row_production_commit=EXPECTED_ROW_PRODUCTION_COMMIT,
            planner_keys=tuple(sorted(planner_keys)),
        )


def _adapt_row(
    row: Mapping[str, Any],
    *,
    release_arm: str,
    source_member: str,
    source_line: int,
) -> tuple[dict[str, Any], str]:
    """Adapt one source row using only the declared family and identity locators."""

    episode_id = row.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise CollisionPressureReleaseError(
            f"{source_member}:{source_line} has no non-empty string episode_id"
        )
    scenario_params = _require_mapping(row.get("scenario_params"), "scenario_params")
    metadata = _require_mapping(scenario_params.get("metadata"), "scenario_params.metadata")
    scenario_family = metadata.get("archetype")
    if not isinstance(scenario_family, str) or not scenario_family:
        raise CollisionPressureReleaseError(
            f"{source_member}:{source_line} has no non-empty scenario_params.metadata.archetype"
        )
    ledger = _require_mapping(row.get("event_ledger"), "event_ledger")
    _require_equal(
        ledger.get("software_commit"),
        EXPECTED_ROW_PRODUCTION_COMMIT,
        "event ledger software commit",
    )
    return (
        {
            "episode_key": f"{release_arm}::{episode_id}",
            "episode_id": episode_id,
            "release_arm": release_arm,
            "scenario_family": scenario_family,
            "source_line": source_line,
            "source_member": source_member,
            "event_ledger": dict(ledger),
        },
        scenario_family,
    )


def _read_release_rows(  # noqa: C901, PLR0912
    bundle_path: Path, verified: VerifiedArchive
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read all release rows and retain only the declared family slice."""

    selected: list[dict[str, Any]] = []
    run_counts: dict[str, int] = {}
    family_counts: Counter[str] = Counter()
    seen_ids_by_arm: dict[str, set[str]] = defaultdict(set)
    try:
        tar = tarfile.open(bundle_path, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise CollisionPressureReleaseError(f"cannot reopen release archive: {exc}") from exc

    with tar:
        _, members = _index_members(tar)
        for relative_member in verified.episode_members:
            full_name = f"{verified.root_name}/{relative_member}"
            member = members[full_name]
            release_arm_path = Path(relative_member).parts[-2]
            if not release_arm_path.endswith("__differential_drive"):
                raise CollisionPressureReleaseError(
                    f"unexpected release-arm path (missing kinematics suffix): {relative_member}"
                )
            release_arm = release_arm_path.removesuffix("__differential_drive")
            if release_arm not in verified.planner_keys:
                raise CollisionPressureReleaseError(
                    f"release arm is not declared by the release manifest: {release_arm!r}"
                )
            handle = tar.extractfile(member)
            if handle is None:
                raise CollisionPressureReleaseError(f"cannot read source rows: {relative_member}")
            arm_count = 0
            for source_line, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise CollisionPressureReleaseError(
                        f"invalid JSON at {relative_member}:{source_line}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise CollisionPressureReleaseError(
                        f"source row at {relative_member}:{source_line} is not an object"
                    )
                arm_count += 1
                adapted, family = _adapt_row(
                    row,
                    release_arm=release_arm,
                    source_member=relative_member,
                    source_line=source_line,
                )
                episode_id = adapted["episode_id"]
                if episode_id in seen_ids_by_arm[release_arm]:
                    raise CollisionPressureReleaseError(
                        f"duplicate episode_id within release arm {release_arm!r}: {episode_id!r}"
                    )
                seen_ids_by_arm[release_arm].add(episode_id)
                if family in ELIGIBLE_FAMILIES:
                    selected.append(adapted)
                    family_counts[family] += 1
            run_counts[release_arm] = arm_count

    if any(count != EXPECTED_ROWS_PER_RUN for count in run_counts.values()):
        raise CollisionPressureReleaseError(
            f"per-arm row count mismatch: expected {EXPECTED_ROWS_PER_RUN}, got {run_counts}"
        )
    total_rows = sum(run_counts.values())
    if total_rows != EXPECTED_TOTAL_ROWS:
        raise CollisionPressureReleaseError(
            f"total release row count mismatch: expected {EXPECTED_TOTAL_ROWS}, got {total_rows}"
        )
    if len(selected) != EXPECTED_ELIGIBLE_ROWS:
        raise CollisionPressureReleaseError(
            f"eligible row count mismatch: expected {EXPECTED_ELIGIBLE_ROWS}, got {len(selected)}"
        )
    if dict(sorted(family_counts.items())) != {
        family: values["eligible_episode_count"]
        for family, values in EXPECTED_FAMILY_COUNTS.items()
    }:
        raise CollisionPressureReleaseError(
            f"eligible family counts mismatch: expected {EXPECTED_FAMILY_COUNTS}, got {dict(family_counts)}"
        )
    selected.sort(key=lambda row: row["episode_key"])
    duplicate_keys = [
        key for key, count in Counter(row["episode_key"] for row in selected).items() if count > 1
    ]
    if duplicate_keys:
        raise CollisionPressureReleaseError(
            "duplicate eligible arm-qualified episode keys: " + ", ".join(sorted(duplicate_keys))
        )
    return selected, {
        "total_release_rows": total_rows,
        "run_count": len(run_counts),
        "rows_per_run": dict(sorted(run_counts.items())),
        "eligible_rows": len(selected),
        "eligible_family_counts": dict(sorted(family_counts.items())),
        "source_member_count": len(verified.episode_members),
    }


def _parse_decimal(value: str, *, field: str) -> Decimal:
    """Parse a finite decimal from the release breakdown."""

    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise CollisionPressureReleaseError(f"invalid decimal {field}: {value!r}") from exc
    if not parsed.is_finite():
        raise CollisionPressureReleaseError(f"non-finite decimal {field}: {value!r}")
    return parsed


def _rounded_count(value: Decimal) -> int:
    """Round a release breakdown weighted count to the nearest integer."""

    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _aggregate_breakdown(  # noqa: C901
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    name: str,
) -> dict[str, Any]:
    """Independently aggregate the published family-breakdown CSV."""

    try:
        text = _read_member(tar, member, name).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CollisionPressureReleaseError(f"breakdown is not UTF-8: {name}") from exc
    rows = list(csv.DictReader(text.splitlines()))
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        family = row.get("scenario_family")
        if family in ELIGIBLE_FAMILIES:
            by_family[family].append(row)
    if any(len(by_family[family]) != EXPECTED_RUN_COUNT for family in ELIGIBLE_FAMILIES):
        raise CollisionPressureReleaseError(
            "scenario_family_breakdown.csv does not contain one row per release arm for each "
            f"eligible family: { {family: len(by_family[family]) for family in ELIGIBLE_FAMILIES} }"
        )

    totals: Counter[str] = Counter()
    family_counts: dict[str, dict[str, int]] = {}
    for family in ELIGIBLE_FAMILIES:
        family_rows = by_family[family]
        episodes = 0
        weighted: Counter[str] = Counter()
        for row in family_rows:
            try:
                row_episodes = int(row["episodes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CollisionPressureReleaseError(
                    f"invalid episodes value for breakdown family {family!r}"
                ) from exc
            if row_episodes <= 0:
                raise CollisionPressureReleaseError(
                    f"non-positive episodes value for breakdown family {family!r}"
                )
            episodes += row_episodes
            for field in (
                "collisions_mean",
                "total_collision_count_mean",
                "ped_collision_count_mean",
                "obstacle_collision_count_mean",
            ):
                weighted[field] += (
                    _parse_decimal(row[field], field=f"{family}.{field}") * row_episodes
                )
        family_counts[family] = {
            "eligible_episode_count": episodes,
            "contact_episode_count": _rounded_count(weighted["collisions_mean"]),
            "collision_event_count": _rounded_count(weighted["total_collision_count_mean"]),
            "pedestrian_contact_episode_count": _rounded_count(
                weighted["ped_collision_count_mean"]
            ),
            "obstacle_contact_episode_count": _rounded_count(
                weighted["obstacle_collision_count_mean"]
            ),
        }
        totals["eligible_episode_count"] += episodes
        for field in (
            "contact_episode_count",
            "collision_event_count",
            "pedestrian_contact_episode_count",
            "obstacle_contact_episode_count",
        ):
            totals[field] += family_counts[family][field]
    return {"eligible_episode_count": dict(totals), "family_counts": family_counts}


def _reconcile_report_and_breakdown(
    report: Mapping[str, Any], breakdown: Mapping[str, Any]
) -> None:
    """Require the exact typed report and independent published breakdown to agree."""

    selection = _require_mapping(report.get("selection"), "report.selection")
    denominator = _require_mapping(report.get("denominator"), "report.denominator")
    counts = _require_mapping(report.get("counts"), "report.counts")
    family_counts = _require_mapping(report.get("family_counts"), "report.family_counts")
    missingness = _require_mapping(report.get("missingness"), "report.missingness")
    if selection.get("excluded_row_count") != 0:
        raise CollisionPressureReleaseError(
            f"nonzero report exclusions are not allowed: {selection.get('exclusion_counts')}"
        )
    _require_equal(
        denominator.get("eligible_episode_count"), EXPECTED_ELIGIBLE_ROWS, "eligible episode count"
    )
    _require_equal(
        denominator.get("eligible_episode_key_sha256"),
        EXPECTED_DENOMINATOR_DIGEST,
        "eligible episode-key digest",
    )
    _require_equal(
        counts.get("contact_episode_count"), EXPECTED_CONTACT_EPISODES, "contact episodes"
    )
    _require_equal(
        counts.get("collision_event_count"), EXPECTED_COLLISION_EVENTS, "collision events"
    )
    _require_equal(
        counts.get("partner_type_episode_counts"),
        EXPECTED_PARTNER_EPISODE_COUNTS,
        "partner episode counts",
    )
    _require_equal(
        counts.get("obstacle_rollup_episode_count"), EXPECTED_OBSTACLE_EPISODES, "obstacle episodes"
    )
    _require_equal(
        counts.get("pedestrian_obstacle_overlap_episode_counts"),
        EXPECTED_OVERLAP_COUNTS,
        "pedestrian/obstacle overlap counts",
    )
    _require_equal(family_counts, EXPECTED_FAMILY_COUNTS, "family counts")
    actual_missingness = dict(missingness.get("optional_collision_event_fields", {}))
    for field in EXPECTED_MISSING_OPTIONAL_FIELDS:
        actual_missingness.setdefault(field, 0)
    _require_equal(
        actual_missingness, EXPECTED_MISSING_OPTIONAL_FIELDS, "optional-field missingness"
    )

    breakdown_totals = _require_mapping(breakdown.get("eligible_episode_count"), "breakdown totals")
    _require_equal(
        breakdown_totals.get("eligible_episode_count"),
        EXPECTED_ELIGIBLE_ROWS,
        "breakdown eligible rows",
    )
    _require_equal(
        breakdown_totals.get("contact_episode_count"),
        EXPECTED_CONTACT_EPISODES,
        "breakdown contact episodes",
    )
    _require_equal(
        breakdown_totals.get("collision_event_count"),
        EXPECTED_COLLISION_EVENTS,
        "breakdown collision events",
    )
    _require_equal(
        breakdown_totals.get("pedestrian_contact_episode_count"),
        EXPECTED_PARTNER_EPISODE_COUNTS["pedestrian"],
        "breakdown pedestrian contacts",
    )
    _require_equal(
        breakdown_totals.get("obstacle_contact_episode_count"),
        EXPECTED_OBSTACLE_EPISODES,
        "breakdown obstacle contacts",
    )
    breakdown_family_counts = {
        family: {
            "eligible_episode_count": values["eligible_episode_count"],
            "contact_episode_count": values["contact_episode_count"],
        }
        for family, values in _require_mapping(
            breakdown.get("family_counts"), "breakdown families"
        ).items()
    }
    _require_equal(
        breakdown_family_counts,
        {
            family: {
                "eligible_episode_count": values["eligible_episode_count"],
                "contact_episode_count": values["contact_episode_count"],
            }
            for family, values in EXPECTED_FAMILY_COUNTS.items()
        },
        "breakdown family counts",
    )


def _check_missingness_by_partner_type(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    """Require missing partner identifiers to remain explicit static-geometry data."""

    missing_ids_by_partner: Counter[str] = Counter()
    for row in rows:
        ledger = _require_mapping(row.get("event_ledger"), "adapted event_ledger")
        events = ledger.get("collision_events")
        if not isinstance(events, list):
            raise CollisionPressureReleaseError("event_ledger.collision_events must be a list")
        for event in events:
            event_mapping = _require_mapping(event, "collision event")
            if event_mapping.get("collision_partner_id") is None:
                partner_type = event_mapping.get("collision_partner_type")
                if not isinstance(partner_type, str) or not partner_type:
                    raise CollisionPressureReleaseError(
                        "missing collision_partner_id has no typed collision_partner_type"
                    )
                missing_ids_by_partner[partner_type] += 1
    expected_missing_ids = Counter(EXPECTED_MISSING_PARTNER_TYPES)
    if missing_ids_by_partner != expected_missing_ids:
        raise CollisionPressureReleaseError(
            "missing collision_partner_id values do not match the release contract: "
            f"expected {dict(expected_missing_ids)}, got {dict(missing_ids_by_partner)}"
        )
    return dict(sorted(missing_ids_by_partner.items()))


def _source_manifest(
    *,
    verified: VerifiedArchive,
    report: Mapping[str, Any],
    row_summary: Mapping[str, Any],
    breakdown: Mapping[str, Any],
    missing_partner_types: Mapping[str, int],
    generated_outputs: list[str],
) -> dict[str, Any]:
    """Build the deterministic source manifest for the packet."""

    source_checksums = {
        "release_archive": verified.archive_sha256,
        verified.breakdown_member: verified.checksums[verified.breakdown_member],
    }
    source_checksums.update(
        {member: verified.checksums[member] for member in verified.episode_members}
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "evidence_status": "analysis_only",
        "evidence_grade": "diagnostic-only",
        "analysis_only": True,
        "benchmark_promotion": False,
        "paper_facing": False,
        "claim_boundary": (
            "Analysis-only materialization of descriptive exact collision-event counts from "
            "immutable release rows. This packet does not admit evidence, restore a withdrawn "
            "dissertation row, or establish probability, causality, severity, physical risk, "
            "or real-world safety."
        ),
        "source": {
            "release_tag": EXPECTED_RELEASE_TAG,
            "release_id": verified.release_id,
            "publication_commit": verified.publication_commit,
            "row_production_commit": verified.row_production_commit,
            "bundle_asset_name": EXPECTED_BUNDLE_ASSET_NAME,
            "bundle_sha256": verified.archive_sha256,
            "payload_checksum_manifest_entries": len(verified.checksums),
            "source_checksums": dict(sorted(source_checksums.items())),
            "source_members": list(verified.episode_members) + [verified.breakdown_member],
        },
        "adapter": {
            "scenario_family": {
                "source_locator": "scenario_params.metadata.archetype",
                "target_field": "scenario_family",
                "operation": "copy_exact",
            },
            "episode_key": {
                "source_locators": {
                    "release_arm": (
                        "archive path payload/runs/<release-arm>__differential_drive/episodes.jsonl; "
                        "the fixed kinematics suffix maps to the manifest planner key"
                    ),
                    "episode_id": "episode_id",
                },
                "target_field": "episode_key",
                "operation": "format_exact",
                "format": "<release-arm>::<episode_id>",
            },
            "scientific_category_inference": "none",
        },
        "row_summary": dict(row_summary),
        "breakdown_reconciliation": dict(breakdown),
        "missing_collision_partner_id_by_type": dict(missing_partner_types),
        "missing_optional_fields": dict(EXPECTED_MISSING_OPTIONAL_FIELDS),
        "report_summary": {
            "schema_version": report["schema_version"],
            "eligible_episode_key_sha256": report["denominator"]["eligible_episode_key_sha256"],
            "eligible_episode_count": report["denominator"]["eligible_episode_count"],
            "contact_episode_count": report["counts"]["contact_episode_count"],
            "collision_event_count": report["counts"]["collision_event_count"],
            "excluded_row_count": report["selection"]["excluded_row_count"],
        },
        "generated_outputs": generated_outputs,
    }


def _render_readme(report: Mapping[str, Any], verified: VerifiedArchive) -> str:
    """Render a plain-language, claim-bounded packet README."""

    counts = report["counts"]
    denominator = report["denominator"]
    return f"""<!-- AI-GENERATED (robot_sf#7724) - NEEDS-REVIEW -->
# Issue #7724 collision-pressure release packet

This packet materializes a deterministic descriptive collision-pressure slice from the published
RobotSF release `{EXPECTED_RELEASE_TAG}`. It reads immutable typed event-ledger rows and runs no
simulation, campaign, GPU, SLURM job, or private-operations workflow.

## Claim boundary

- **Evidence status:** `analysis_only` / `diagnostic-only` candidate artifact.
- **Admission:** this packet does not admit evidence, change an evidence tier, or automatically
  restore the withdrawn dissertation Table 7.5 row in [ll7/diss#613](https://github.com/ll7/diss/issues/613).
- **Forbidden inference:** the counts are not a probability, ranking, causal mechanism, severity,
  physical-risk, deployment-safety, or real-world safety result.
- **Scope:** exact descriptive counts over `doorway`, `narrow_doorway`, and `robot_crowding`.

## Immutable input

- Release: `{EXPECTED_RELEASE_TAG}`
- Publication commit: `{verified.publication_commit}`
- Row-production commit: `{verified.row_production_commit}`
- Archive: `{EXPECTED_BUNDLE_ASSET_NAME}`
- Archive SHA-256: `{verified.archive_sha256}`
- Internal payload checksums: `{len(verified.checksums)}` entries verified before reading rows

## Reproduction

```bash
uv run python scripts/analysis/generate_collision_pressure_release_0_0_3_post1.py \\
  --bundle <path-to-{EXPECTED_BUNDLE_ASSET_NAME}> \\
  --output-dir docs/context/evidence/diss_613_collision_pressure_release_0_0_3_post1
```

Verify the packet bytes from the repository root with:

```bash
shasum -a 256 -c docs/context/evidence/diss_613_collision_pressure_release_0_0_3_post1/SHA256SUMS
```

The adapter records the only two transformations: `scenario_family` is copied from
`scenario_params.metadata.archetype`, and `episode_key` is formatted as
`<release-arm>::<episode_id>`. No scientific category is inferred or normalized.

## Reconciled result

| Quantity | Count |
| --- | ---: |
| Release rows across {EXPECTED_RUN_COUNT} run files | {EXPECTED_TOTAL_ROWS:,} |
| Eligible arm-qualified episodes | {denominator["eligible_episode_count"]:,} |
| Contact episodes | {counts["contact_episode_count"]:,} |
| Typed collision events | {counts["collision_event_count"]:,} |
| Pedestrian-contact episodes | {counts["partner_type_episode_counts"]["pedestrian"]:,} |
| Obstacle-contact episodes | {counts["obstacle_rollup_episode_count"]:,} |
| Pedestrian/obstacle overlap episodes | {counts["pedestrian_obstacle_overlap_episode_counts"]["pedestrian_and_obstacle"]:,} |
| Unexplained exclusions | {report["selection"]["excluded_row_count"]:,} |

The report's exact counts reconcile independently with the published
`payload/reports/scenario_family_breakdown.csv`. Missing `collision_partner_id` values remain
explicit: {EXPECTED_MISSING_OPTIONAL_FIELDS["collision_partner_id"]:,} static-geometry contacts;
relative-speed missingness is {EXPECTED_MISSING_OPTIONAL_FIELDS["relative_speed_at_contact"]}.

## Packet files

- `source_manifest.json`: immutable input identity, adapter locators, row totals, and reconciliation.
- `normalized_typed_ledger_slice.jsonl`: sorted, arm-qualified EpisodeEventLedger.v2 slice.
- `collision_pressure_report.json` / `.csv`: generic report outputs.
- `SHA256SUMS`: checksums for every packet file except the checksum manifest itself.

This packet is linked to [RobotSF issue #7724](https://github.com/ll7/robot_sf_ll7/issues/7724)
and requires a separate dissertation-side evidence-pin/card and author decision before any
manuscript use.
"""


def build_packet(bundle_path: Path, output_dir: Path) -> dict[str, Any]:
    """Verify the release and write the deterministic collision-pressure packet."""

    verified = verify_release_archive(bundle_path)
    selected_rows, row_summary = _read_release_rows(bundle_path, verified)
    missing_partner_types = _check_missingness_by_partner_type(selected_rows)
    input_checksums = {
        "release_archive": verified.archive_sha256,
        verified.breakdown_member: verified.checksums[verified.breakdown_member],
    }
    input_checksums.update(
        {member: verified.checksums[member] for member in verified.episode_members}
    )
    try:
        report = build_collision_pressure_report(
            selected_rows,
            eligible_families=ELIGIBLE_FAMILIES,
            source_commit=verified.row_production_commit,
            release_id=EXPECTED_RELEASE_TAG,
            bundle_id=verified.publication_commit,
            input_checksums=input_checksums,
        )
    except CollisionPressureReportError as exc:
        raise CollisionPressureReleaseError(f"collision-pressure report blocked: {exc}") from exc

    try:
        tar = tarfile.open(bundle_path, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise CollisionPressureReleaseError(
            f"cannot reopen release archive for breakdown: {exc}"
        ) from exc
    with tar:
        _, members = _index_members(tar)
        full_breakdown_name = f"{verified.root_name}/{verified.breakdown_member}"
        breakdown = _aggregate_breakdown(
            tar,
            members[full_breakdown_name],
            name=verified.breakdown_member,
        )
    _reconcile_report_and_breakdown(report, breakdown)

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "normalized_typed_ledger_slice.jsonl"
    normalized_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in selected_rows
    )
    normalized_path.write_text(normalized_text, encoding="utf-8")

    report_paths = write_collision_pressure_report(
        report,
        json_path=output_dir / "collision_pressure_report.json",
        csv_path=output_dir / "collision_pressure_report.csv",
    )
    # Exact-byte artifacts use review sidecars so the report schema and JSONL contract remain
    # unchanged by an inline marker.
    write_review_sidecar(normalized_path)
    write_review_sidecar(report_paths["json"])
    write_review_sidecar(report_paths["csv"])

    generated_outputs = [
        "README.md",
        "source_manifest.json",
        "normalized_typed_ledger_slice.jsonl",
        "normalized_typed_ledger_slice.jsonl.review.json",
        "collision_pressure_report.json",
        "collision_pressure_report.json.review.json",
        "collision_pressure_report.csv",
        "collision_pressure_report.csv.review.json",
        "SHA256SUMS",
    ]
    manifest = _source_manifest(
        verified=verified,
        report=report,
        row_summary=row_summary,
        breakdown=breakdown,
        missing_partner_types=missing_partner_types,
        generated_outputs=generated_outputs,
    )
    write_json(output_dir / "source_manifest.json", manifest)
    write_text(output_dir / "README.md", _render_readme(report, verified))
    write_sha256sums(output_dir)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "report": report,
        "row_summary": row_summary,
        "generated_outputs": generated_outputs,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the release packet generator and return a shell-friendly status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Verified release archive path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Evidence packet directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args(argv)
    try:
        result = build_packet(args.bundle, args.output_dir)
    except (
        CollisionPressureReleaseError,
        CollisionPressureReportError,
        OSError,
        tarfile.TarError,
    ) as exc:
        parser.exit(2, f"collision-pressure release packet blocked: {exc}\n")
    print(
        f"collision-pressure release packet written: {result['output_dir']} "
        f"({result['row_summary']['eligible_rows']} eligible rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
