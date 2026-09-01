#!/usr/bin/env python3
"""Fail-closed license-payload and release-member gate for Robot SF archives.

The gate checks every Robot SF wheel and source distribution found in a distribution directory.
At least one of each archive type is required. Each archive must carry the root GPL text, the
fast-pysf MIT text, the python-rvo2 Apache text, the SocNavBench MIT text, and the third-party
notice manifest. Source distributions must not carry the top-level ``model/`` artifact tree.
CI can additionally require the vendored ``pyrvo2`` companion wheel. Release validation can opt
into the strict member contract, which cross-checks archive payload paths against the tracked
asset-rights inventory and a proposed Git tree. This optional mode is deliberately separate from
the ordinary CI classification check: the current repository still contains known rights rows
that must be explicitly resolved or externalized before a software release.

Examples:
    python scripts/tools/check_distribution_licenses.py dist
    python scripts/tools/check_distribution_licenses.py dist --require-pyrvo2
    python scripts/tools/check_distribution_licenses.py dist --strict-asset-rights \
        --repo-root . --source-tree-ref HEAD
"""

from __future__ import annotations

import argparse
import stat
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from packaging.utils import InvalidWheelFilename, parse_wheel_filename

# ``uv run --no-project`` executes this file with ``scripts/tools`` as
# ``sys.path[0]``.  Add the repository root for the direct-script path while
# keeping normal package imports unchanged.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.tools import check_asset_rights_inventory as asset_inventory


class DistributionLicenseError(ValueError):
    """Raised when a distribution directory fails the license-payload contract."""


@dataclass(frozen=True)
class ArchiveRequirement:
    """Text markers and archive path suffixes required for one payload file."""

    label: str
    path_suffix: str
    markers: tuple[str, ...]


@dataclass(frozen=True)
class DistributionCheckResult:
    """Summary of a successful distribution-directory check."""

    wheels: tuple[Path, ...]
    sdists: tuple[Path, ...]
    pyrvo2_wheels: tuple[Path, ...]


REQUIREMENTS = (
    ArchiveRequirement(
        label="root GPL license",
        path_suffix="LICENSE",
        markers=(
            "GNU GENERAL PUBLIC LICENSE",
            "Version 3",
            "Copyright (C) 2007 Free Software Foundation",
        ),
    ),
    ArchiveRequirement(
        label="fast-pysf MIT license",
        path_suffix="fast-pysf/LICENSE",
        markers=(
            "MIT License",
            "Copyright (c) 2020 Yuxiang Gao",
            "Permission is hereby granted",
        ),
    ),
    ArchiveRequirement(
        label="python-rvo2 Apache license",
        path_suffix="third_party/python-rvo2/LICENSE",
        markers=(
            "Apache License",
            "Version 2.0",
            "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
        ),
    ),
    ArchiveRequirement(
        label="SocNavBench MIT license",
        path_suffix="third_party/socnavbench/LICENSE",
        markers=(
            "MIT License",
            "Copyright (c) 2020 Transportation, Bots, and Disability (TBD) Lab",
            "Permission is hereby granted",
        ),
    ),
    ArchiveRequirement(
        label="SocNavBench Apache license",
        path_suffix="third_party/socnavbench/LICENSES/Apache-2.0.txt",
        markers=(
            "Apache License",
            "Version 2.0",
            "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
        ),
    ),
    ArchiveRequirement(
        label="SocNavBench licensing manifest",
        path_suffix="third_party/socnavbench/LICENSING.yaml",
        markers=("schema_version: robot_sf.third_party_licensing.v1", "default_license_spdx: MIT"),
    ),
    ArchiveRequirement(
        label="SocNavBench provenance record",
        path_suffix="third_party/socnavbench/UPSTREAM.md",
        markers=("https://github.com/CMU-TBD/SocNavBench", "Commit:", "License: MIT"),
    ),
    ArchiveRequirement(
        label="third-party notice manifest",
        path_suffix="THIRD_PARTY_NOTICES.md",
        markers=(
            "Third-party notices",
            "fast-pysf",
            "MIT License",
            "Yuxiang Gao",
            "python-rvo2",
            "Apache License, Version 2.0",
            "SocNavBench",
            "TBD) Lab",
            "does not include model weights",
        ),
    ),
    ArchiveRequirement(
        label="python-rvo2 provenance record",
        path_suffix="third_party/python-rvo2/UPSTREAM.md",
        markers=("upstream_repository", "source_archive_sha256", "LOCAL_CHANGES.patch"),
    ),
    ArchiveRequirement(
        label="python-rvo2 local-change patch",
        path_suffix="third_party/python-rvo2/LOCAL_CHANGES.patch",
        markers=("diff -ruN", "third_party/python-rvo2"),
    ),
)

PYRVO2_REQUIREMENTS = (
    ArchiveRequirement(
        label="pyrvo2 Apache license",
        path_suffix="LICENSE",
        markers=(
            "Apache License",
            "Version 2.0",
            "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
        ),
    ),
    ArchiveRequirement(
        label="pyrvo2 provenance record",
        path_suffix="UPSTREAM.md",
        markers=("upstream_repository", "source_archive_sha256", "LOCAL_CHANGES.patch"),
    ),
    ArchiveRequirement(
        label="pyrvo2 local-change patch",
        path_suffix="LOCAL_CHANGES.patch",
        markers=("diff -ruN", "third_party/python-rvo2"),
    ),
)

SDIST_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip")
SOCNAV_EXCLUDED_MEMBERS = {"LICENSE", "LICENSING.yaml", "UPSTREAM.md"}
FORBIDDEN_SDIST_ROOTS = frozenset({"model"})
RELEASE_ALLOWED_ASSET_STATUSES = frozenset({"cleared", "project-authored"})


def _normalise_archive_member_name(name: str) -> str:
    """Return a safe POSIX archive member name or raise a contract error."""
    if not isinstance(name, str):
        raise DistributionLicenseError(f"unsafe archive member path: {name!r}")
    normalised = name.replace("\\", "/")
    raw_parts = normalised.split("/")
    if not normalised or normalised.startswith("/") or "\x00" in normalised:
        raise DistributionLicenseError(f"unsafe archive member path: {name!r}")
    for index, part in enumerate(raw_parts):
        if not part and index != len(raw_parts) - 1:
            raise DistributionLicenseError(f"unsafe archive member path: {name!r}")
        if part in {".", ".."} or ":" in part or part.endswith((".", " ")):
            raise DistributionLicenseError(f"unsafe archive member path: {name!r}")
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in part):
            raise DistributionLicenseError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(normalised)
    return path.as_posix()


def _validate_unique_archive_names(names: list[str], archive: Path) -> None:
    """Reject duplicate or unsafe names before archive contents are normalised to a mapping."""
    normalised = [_normalise_archive_member_name(name) for name in names]
    if len(normalised) != len(set(normalised)):
        duplicates = sorted(name for name in set(normalised) if normalised.count(name) > 1)
        raise DistributionLicenseError(
            f"{archive.name}: duplicate archive member names: {', '.join(duplicates)}"
        )


def _archive_members(archive: Path) -> dict[str, str]:
    """Read regular text members from a wheel or source-distribution archive."""
    try:
        if archive.suffix in {".whl", ".zip"}:
            with zipfile.ZipFile(archive) as source:
                infos = source.infolist()
                if any(stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK for info in infos):
                    links = sorted(
                        info.filename
                        for info in infos
                        if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK
                    )
                    raise DistributionLicenseError(
                        f"{archive.name}: symbolic-link archive members are forbidden: "
                        f"{', '.join(links)}"
                    )
                names = [info.filename for info in infos]
                _validate_unique_archive_names(names, archive)
                return {
                    _normalise_archive_member_name(name): source.read(name).decode(
                        "utf-8", errors="replace"
                    )
                    for name in names
                    if not name.endswith("/")
                }
        if archive.name.endswith(SDIST_SUFFIXES):
            with tarfile.open(archive, mode="r:*") as source:
                tar_members = source.getmembers()
                non_regular = sorted(
                    member.name
                    for member in tar_members
                    if not member.isfile() and not member.isdir()
                )
                if non_regular:
                    raise DistributionLicenseError(
                        f"{archive.name}: non-regular archive members are forbidden: "
                        f"{', '.join(non_regular)}"
                    )
                names = [member.name for member in tar_members]
                _validate_unique_archive_names(names, archive)
                members: dict[str, str] = {}
                for member in source.getmembers():
                    if not member.isfile():
                        continue
                    handle = source.extractfile(member)
                    if handle is None:
                        continue
                    members[_normalise_archive_member_name(member.name)] = handle.read().decode(
                        "utf-8", errors="replace"
                    )
                return members
    except (OSError, UnicodeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise DistributionLicenseError(f"cannot read archive {archive.name}: {exc}") from exc
    raise DistributionLicenseError(f"unsupported archive type: {archive.name}")


def _matching_members(archive: Path, members: dict[str, str], suffix: str) -> dict[str, str]:
    """Return members at the canonical wheel-license or source-root path."""
    required_parts = PurePosixPath(suffix).parts
    matching: dict[str, str] = {}
    for name, content in members.items():
        parts = PurePosixPath(name).parts
        if archive.suffix == ".whl":
            try:
                licenses_index = parts.index("licenses")
            except ValueError:
                continue
            if licenses_index == 0 or parts[licenses_index - 1] not in _wheel_metadata_roots(
                archive
            ):
                continue
            payload_parts = parts[licenses_index + 1 :]
        else:
            if len(parts) < 2:
                continue
            payload_parts = parts[1:]
        if payload_parts == required_parts:
            matching[name] = content
    return matching


def _source_payload_members(members: dict[str, str]) -> dict[str, str]:
    """Return source-distribution paths relative to the archive root."""
    payload: dict[str, str] = {}
    for name, content in members.items():
        parts = PurePosixPath(name).parts
        if len(parts) < 2:
            continue
        payload[PurePosixPath(*parts[1:]).as_posix()] = content
    return payload


def _socnav_manifest_payload(
    archive: Path, members: dict[str, str]
) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Load SocNavBench's manifest and upstream record from an sdist."""
    manifest_matches = _matching_members(archive, members, "third_party/socnavbench/LICENSING.yaml")
    upstream_matches = _matching_members(archive, members, "third_party/socnavbench/UPSTREAM.md")
    if not manifest_matches or not upstream_matches:
        return None, "", []

    manifest_text = next(iter(manifest_matches.values()))
    upstream_text = next(iter(upstream_matches.values()))
    try:
        manifest = yaml.safe_load(manifest_text)
    except yaml.YAMLError as exc:
        return None, "", [f"{archive.name}: SocNavBench licensing manifest is invalid YAML: {exc}"]
    if not isinstance(manifest, dict):
        return None, "", [f"{archive.name}: SocNavBench licensing manifest must be a mapping"]
    return manifest, upstream_text, []


def _check_socnav_identity(
    archive: Path, manifest: dict[str, Any], upstream_text: str
) -> list[str]:
    """Check the SocNavBench source identity and default license."""
    errors: list[str] = []
    if manifest.get("schema_version") != "robot_sf.third_party_licensing.v1":
        errors.append(f"{archive.name}: SocNavBench licensing schema version is invalid")
    for field, label in (
        ("source_repository", "source repository"),
        ("source_revision", "source revision"),
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or value not in upstream_text:
            errors.append(f"{archive.name}: SocNavBench {label} disagrees with UPSTREAM.md")
    if manifest.get("default_license_spdx") != "MIT":
        errors.append(f"{archive.name}: SocNavBench default license must remain MIT")
    return errors


def _socnav_classifications(archive: Path, manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate SocNavBench's upstream/local partition and its Apache override refinement.

    ``license_overrides`` refines the license of files that remain upstream files, so an
    override entry must also appear in ``upstream_files`` and is not a third partition.
    """
    errors: list[str] = []
    upstream_files = manifest.get("upstream_files")
    overrides = manifest.get("license_overrides")
    local_files = manifest.get("local_files")
    if not isinstance(upstream_files, list) or not all(
        isinstance(item, str) for item in upstream_files
    ):
        return [], [f"{archive.name}: SocNavBench upstream_files must be a list of paths"]
    if not isinstance(overrides, list) or not isinstance(local_files, list):
        return [], [f"{archive.name}: SocNavBench license classifications are malformed"]

    override_files: list[str] = []
    for override in overrides:
        if not isinstance(override, dict) or not isinstance(override.get("files"), list):
            errors.append(f"{archive.name}: SocNavBench license override is malformed")
            continue
        override_files.extend(item for item in override["files"] if isinstance(item, str))
        if override.get("license_spdx") != "Apache-2.0":
            errors.append(f"{archive.name}: SocNavBench overrides must use Apache-2.0")
    local_paths = [
        item.get("path")
        for item in local_files
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    ]
    stray_overrides = sorted(set(override_files) - set(upstream_files))
    if stray_overrides:
        errors.append(
            f"{archive.name}: SocNavBench license overrides must also be listed as "
            f"upstream files: {stray_overrides}"
        )
    classified = list(upstream_files) + local_paths
    if len(classified) != len(set(classified)):
        errors.append(f"{archive.name}: SocNavBench file classifications overlap")
    return classified, errors


def _socnav_source_files(members: dict[str, str]) -> set[str]:
    """Return vendored SocNavBench source paths that require classification."""
    payload = _source_payload_members(members)
    return {
        path.removeprefix("third_party/socnavbench/")
        for path in payload
        if path.startswith("third_party/socnavbench/")
        and path.removeprefix("third_party/socnavbench/") not in SOCNAV_EXCLUDED_MEMBERS
        and not path.removeprefix("third_party/socnavbench/").startswith("LICENSES/")
    }


def _check_socnav_file_partition(
    archive: Path, members: dict[str, str], classified: list[str]
) -> list[str]:
    """Require every archived SocNavBench source file to have one classification."""
    source_files = _socnav_source_files(members)
    if not source_files:
        return []
    declared = set(classified)
    errors: list[str] = []
    missing = sorted(declared - source_files)
    unclassified = sorted(source_files - declared)
    if missing:
        errors.append(f"{archive.name}: SocNavBench manifest lists missing files: {missing}")
    if unclassified:
        errors.append(f"{archive.name}: SocNavBench source files are unclassified: {unclassified}")
    return errors


def _check_socnav_override_evidence(
    archive: Path, members: dict[str, str], override_files: list[str]
) -> list[str]:
    """Require Apache evidence for every explicitly overridden source file."""
    payload = _source_payload_members(members)
    return [
        f"{archive.name}: SocNavBench override lacks Apache evidence: {path}"
        for path in override_files
        if "Apache License" not in payload.get(f"third_party/socnavbench/{path}", "")
        and "Apache-2.0" not in payload.get(f"third_party/socnavbench/{path}", "")
    ]


def _check_socnavbench_provenance(archive: Path, members: dict[str, str]) -> list[str]:
    """Validate the mixed-license manifest against the vendored source tree."""
    if not archive.name.endswith(SDIST_SUFFIXES):
        return []
    manifest, upstream_text, errors = _socnav_manifest_payload(archive, members)
    if manifest is None:
        return errors

    errors.extend(_check_socnav_identity(archive, manifest, upstream_text))
    # A rights-clean software candidate retains only SocNavBench's legal
    # notices and provenance record. In that external-source-checkout mode
    # there are no vendored source files whose upstream/local partition or
    # per-file override evidence can be checked in the archive.
    if not _socnav_source_files(members):
        return errors
    classified, classification_errors = _socnav_classifications(archive, manifest)
    errors.extend(classification_errors)
    if classification_errors:
        return errors
    override_files = [
        path
        for override in manifest["license_overrides"]
        if isinstance(override, dict)
        for path in override.get("files", [])
        if isinstance(path, str)
    ]
    errors.extend(_check_socnav_file_partition(archive, members, classified))
    errors.extend(_check_socnav_override_evidence(archive, members, override_files))
    return errors


def _forbidden_sdist_members(archive: Path, members: dict[str, str]) -> tuple[str, ...]:
    """Return source-archive members under a forbidden top-level artifact tree."""
    if not archive.name.endswith(SDIST_SUFFIXES):
        return ()
    forbidden = sorted(
        name
        for name in members
        if FORBIDDEN_SDIST_ROOTS.intersection(PurePosixPath(name).parts[:2])
    )
    return tuple(forbidden)


def _wheel_metadata_roots(archive: Path) -> frozenset[str]:
    """Return the exact generated metadata/data roots named by a wheel filename."""
    try:
        distribution, version, _build, _tags = parse_wheel_filename(archive.name)
    except InvalidWheelFilename:
        return frozenset()
    base = f"{str(distribution).replace('-', '_')}-{version}"
    return frozenset({f"{base}.dist-info", f"{base}.data"})


def _archive_source_path(
    archive: Path,
    member_name: str,
    *,
    source_root: str | None = None,
) -> str | None:
    """Map one archive member to its repository path, ignoring only exact metadata roots."""
    parts = PurePosixPath(member_name).parts
    if archive.suffix == ".whl":
        if parts and parts[0].endswith(".dist-info") and parts[0] in _wheel_metadata_roots(archive):
            return None
        mapped = PurePosixPath(*parts).as_posix() if parts else None
        if source_root is not None and mapped is not None:
            return PurePosixPath(source_root, mapped).as_posix()
        return mapped
    if archive.name.endswith(SDIST_SUFFIXES) and len(parts) > 1:
        return PurePosixPath(*parts[1:]).as_posix()
    return None


def _asset_inventory_report(repo_root: Path, inventory_path: Path | None) -> dict[str, Any]:
    """Load the tracked-path inventory used by strict archive checks."""
    return asset_inventory.build_report(repo_root.resolve(), inventory_path)


def _archive_member_contract_error(
    archive: Path,
    member_name: str,
    path_statuses: dict[str, Any],
    source_root: str | None,
) -> str | None:
    """Return one strict violation for a non-metadata archive member, if any."""
    member_parts = PurePosixPath(member_name).parts
    if "model" in member_parts:
        return (
            f"{archive.name}: model artifact member is forbidden in a software distribution: "
            f"{member_name}"
        )
    mapped_source_root = source_root
    if mapped_source_root is None and archive.name.startswith("pyrvo2-"):
        mapped_source_root = "third_party/python-rvo2"
    source_path = _archive_source_path(
        archive,
        member_name,
        source_root=mapped_source_root,
    )
    if source_path is None:
        return None
    if "model" in PurePosixPath(source_path).parts:
        return (
            f"{archive.name}: model artifact member is forbidden in a software distribution: "
            f"{member_name}"
        )
    if not asset_inventory._looks_like_asset(source_path):
        return None
    status = path_statuses.get(source_path)
    if status is None:
        return (
            f"{archive.name}: asset member is not covered by the tracked rights inventory: "
            f"{member_name} (source path {source_path})"
        )
    if status not in RELEASE_ALLOWED_ASSET_STATUSES:
        return (
            f"{archive.name}: asset member has non-release inventory status {status!r}: "
            f"{member_name} (source path {source_path})"
        )
    return None


def check_archive_member_contract(
    archive: Path,
    *,
    repo_root: Path,
    inventory_path: Path | None = None,
    source_root: str | None = None,
) -> tuple[str, ...]:
    """Return strict rights/member violations for one built distribution archive.

    Archive metadata is ignored, while package/source payload paths are mapped back to the
    repository. Asset-like members must be covered by the tracked inventory and use a release-safe
    status. Model artifacts are never accepted in a software distribution by this gate. The
    function is read-only and does not decide whether an unresolved row should be relicensed.
    """
    try:
        members = _archive_members(archive)
    except DistributionLicenseError as exc:
        return (str(exc),)

    report = _asset_inventory_report(repo_root, inventory_path)
    errors = [
        f"{archive.name}: asset inventory is structurally invalid: {issue['message']}"
        for issue in report.get("issues", [])
    ]
    path_statuses = report.get("path_statuses", {})
    if not isinstance(path_statuses, dict):
        return tuple(errors + [f"{archive.name}: asset inventory did not provide path statuses"])

    for member_name in sorted(members):
        violation = _archive_member_contract_error(
            archive,
            member_name,
            path_statuses,
            source_root,
        )
        if violation is not None:
            errors.append(violation)
    return tuple(sorted(set(errors)))


def _git_tree_entries(repo_root: Path, source_ref: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return regular Git paths and errors for non-regular tree members."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-tree",
                "-r",
                "-z",
                source_ref,
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DistributionLicenseError(
            f"could not enumerate source tree {source_ref!r}: {exc}"
        ) from exc
    paths: list[str] = []
    non_regular: list[str] = []
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        try:
            header, raw_path = entry.split(b"\t", 1)
            mode, object_type, _object_id = header.split(b" ", 2)
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise DistributionLicenseError(
                f"source tree {source_ref!r} contains a malformed Git tree entry"
            ) from exc
        if mode not in {b"100644", b"100755"} or object_type != b"blob":
            mode_text = mode.decode("ascii", errors="replace")
            type_text = object_type.decode("ascii", errors="replace")
            non_regular.append(
                f"source tree {source_ref!r} contains non-regular Git member {path!r} "
                f"(mode {mode_text}, type {type_text})"
            )
            continue
        paths.append(path)
    try:
        normalised_paths = tuple(
            sorted(asset_inventory._normalise_path(path) for path in paths if path)
        )
    except ValueError as exc:
        raise DistributionLicenseError(
            f"source tree {source_ref!r} contains an unsafe path: {exc}"
        ) from exc
    return normalised_paths, tuple(sorted(non_regular))


def _git_tree_paths(repo_root: Path, source_ref: str) -> tuple[str, ...]:
    """Return the exact regular-file paths contained in one Git tree."""
    paths, non_regular = _git_tree_entries(repo_root, source_ref)
    if non_regular:
        raise DistributionLicenseError("; ".join(non_regular))
    return paths


def check_source_tree_member_contract(
    repo_root: Path,
    *,
    source_ref: str = "HEAD",
    inventory_path: Path | None = None,
) -> tuple[str, ...]:
    """Return strict rights/member violations for a proposed Git source tree."""
    try:
        paths, non_regular = _git_tree_entries(repo_root.resolve(), source_ref)
    except DistributionLicenseError as exc:
        return (str(exc),)

    report = asset_inventory.build_report(
        repo_root.resolve(), inventory_path, tracked_paths=list(paths)
    )
    errors = list(non_regular)
    errors.extend(
        f"source tree {source_ref!r}: asset inventory is structurally invalid: {issue['message']}"
        for issue in report.get("issues", [])
    )
    errors.extend(
        f"source tree {source_ref!r}: unresolved asset-rights path remains tracked: {path}"
        for path in report.get("known_blocker_paths", [])
    )
    errors.extend(
        f"source tree {source_ref!r}: model artifact path is forbidden: {path}"
        for path in paths
        if "model" in PurePosixPath(path).parts
    )
    return tuple(sorted(set(errors)))


def _strict_member_contract_errors(
    wheels: tuple[Path, ...],
    sdists: tuple[Path, ...],
    *,
    pyrvo2_wheels: tuple[Path, ...] = (),
    repo_root: Path,
    inventory_path: Path | None,
    source_tree_ref: str | None,
) -> list[str]:
    """Collect strict archive/source-tree contract errors for a distribution."""
    errors: list[str] = []
    for archive in (*wheels, *sdists):
        errors.extend(
            check_archive_member_contract(
                archive,
                repo_root=repo_root,
                inventory_path=inventory_path,
            )
        )
    for archive in pyrvo2_wheels:
        errors.extend(
            check_archive_member_contract(
                archive,
                repo_root=repo_root,
                inventory_path=inventory_path,
                source_root="third_party/python-rvo2",
            )
        )
    if source_tree_ref is not None:
        errors.extend(
            check_source_tree_member_contract(
                repo_root,
                source_ref=source_tree_ref,
                inventory_path=inventory_path,
            )
        )
    return errors


def _check_archive(
    archive: Path, requirements: tuple[ArchiveRequirement, ...] = REQUIREMENTS
) -> list[str]:
    """Return all contract violations found in one archive."""
    try:
        members = _archive_members(archive)
    except DistributionLicenseError as exc:
        return [str(exc)]

    errors: list[str] = []
    forbidden_members = _forbidden_sdist_members(archive, members)
    if forbidden_members:
        errors.append(
            f"{archive.name}: forbidden source-distribution model artifact members "
            f"(top-level model/): {', '.join(forbidden_members)}"
        )
    for requirement in requirements:
        matches = _matching_members(archive, members, requirement.path_suffix)
        if not matches:
            errors.append(
                f"{archive.name}: missing {requirement.label} payload "
                f"(expected path ending in {requirement.path_suffix!r})"
            )
            continue
        valid = [
            name
            for name, content in matches.items()
            if all(marker in content for marker in requirement.markers)
        ]
        if not valid:
            errors.append(
                f"{archive.name}: {requirement.label} has wrong or incomplete content "
                f"(found {', '.join(sorted(matches))})"
            )
    errors.extend(_check_socnavbench_provenance(archive, members))
    return errors


def check_distribution(
    dist_dir: Path,
    *,
    require_pyrvo2: bool = False,
    strict_asset_rights: bool = False,
    repo_root: Path | None = None,
    inventory_path: Path | None = None,
    source_tree_ref: str | None = None,
) -> DistributionCheckResult:
    """Validate Robot SF archives in ``dist_dir`` or raise a clear error."""
    if not dist_dir.is_dir():
        raise DistributionLicenseError(f"distribution directory does not exist: {dist_dir}")

    wheels = tuple(sorted(dist_dir.glob("robot_sf-*.whl")))
    sdists = tuple(
        sorted(
            archive
            for archive in dist_dir.iterdir()
            if archive.name.startswith("robot_sf-") and archive.name.endswith(SDIST_SUFFIXES)
        )
    )
    pyrvo2_wheels = tuple(sorted(dist_dir.glob("pyrvo2-*.whl")))

    errors: list[str] = []
    if not wheels:
        errors.append("missing at least one Robot SF wheel (robot_sf-*.whl)")
    if not sdists:
        errors.append("missing at least one Robot SF sdist (robot_sf-* source archive)")
    if require_pyrvo2 and not pyrvo2_wheels:
        errors.append("missing required pyrvo2 companion wheel (pyrvo2-*.whl)")

    for archive in (*wheels, *sdists):
        errors.extend(_check_archive(archive))
    for archive in pyrvo2_wheels:
        errors.extend(_check_archive(archive, PYRVO2_REQUIREMENTS))

    if strict_asset_rights:
        errors.extend(
            _strict_member_contract_errors(
                wheels,
                sdists,
                repo_root=(repo_root or Path.cwd()).resolve(),
                inventory_path=inventory_path,
                pyrvo2_wheels=pyrvo2_wheels,
                source_tree_ref=source_tree_ref,
            )
        )

    if errors:
        details = "\n".join(f"  - {error}" for error in dict.fromkeys(errors))
        raise DistributionLicenseError(f"distribution license gate failed:\n{details}")

    return DistributionCheckResult(wheels=wheels, sdists=sdists, pyrvo2_wheels=pyrvo2_wheels)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist_dir", type=Path, help="Directory containing built distribution archives."
    )
    parser.add_argument(
        "--require-pyrvo2",
        action="store_true",
        help="Require at least one pyrvo2 companion wheel in the distribution directory.",
    )
    parser.add_argument(
        "--strict-asset-rights",
        action="store_true",
        help=(
            "Cross-check actual archive payload members against the tracked rights inventory and "
            "reject unresolved asset/model members."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root for --strict-asset-rights (defaults to the current directory).",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help=(
            "Tracked asset-rights inventory for --strict-asset-rights "
            "(defaults to <repo-root>/scripts/validation/asset_rights_inventory.v1.yaml)."
        ),
    )
    parser.add_argument(
        "--source-tree-ref",
        default=None,
        help="Also inspect this Git tree/ref for blocked assets and model artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the gate and return a CI-friendly status code."""
    args = _parse_args(argv)
    try:
        result = check_distribution(
            args.dist_dir,
            require_pyrvo2=args.require_pyrvo2,
            strict_asset_rights=args.strict_asset_rights,
            repo_root=args.repo_root,
            inventory_path=args.inventory,
            source_tree_ref=args.source_tree_ref,
        )
    except DistributionLicenseError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        "PASS: distribution license gate "
        f"({len(result.wheels)} wheel(s), {len(result.sdists)} sdist(s), "
        f"{len(result.pyrvo2_wheels)} pyrvo2 wheel(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
