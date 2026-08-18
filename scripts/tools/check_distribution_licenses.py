#!/usr/bin/env python3
"""Fail-closed license-payload gate for Robot SF distribution archives.

The gate checks every Robot SF wheel and source distribution found in a distribution directory.
At least one of each archive type is required. Each archive must carry the root GPL text, the
fast-pysf MIT text, the python-rvo2 Apache text, the SocNavBench MIT text, and the third-party
notice manifest. Source distributions must not carry the top-level ``model/`` artifact tree.
CI can additionally require the vendored ``pyrvo2`` companion wheel.

Examples:
    python scripts/tools/check_distribution_licenses.py dist
    python scripts/tools/check_distribution_licenses.py dist --require-pyrvo2
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


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


def _archive_members(archive: Path) -> dict[str, str]:
    """Read regular text members from a wheel or source-distribution archive."""
    try:
        if archive.suffix in {".whl", ".zip"}:
            with zipfile.ZipFile(archive) as source:
                return {
                    name: source.read(name).decode("utf-8", errors="replace")
                    for name in source.namelist()
                    if not name.endswith("/")
                }
        if archive.name.endswith(SDIST_SUFFIXES):
            with tarfile.open(archive, mode="r:*") as source:
                members: dict[str, str] = {}
                for member in source.getmembers():
                    if not member.isfile():
                        continue
                    handle = source.extractfile(member)
                    if handle is None:
                        continue
                    members[member.name] = handle.read().decode("utf-8", errors="replace")
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
            if licenses_index == 0 or parts[licenses_index - 1].find(".dist-info") == -1:
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


def check_distribution(dist_dir: Path, *, require_pyrvo2: bool = False) -> DistributionCheckResult:
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

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the gate and return a CI-friendly status code."""
    args = _parse_args(argv)
    try:
        result = check_distribution(args.dist_dir, require_pyrvo2=args.require_pyrvo2)
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
