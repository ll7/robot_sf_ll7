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
