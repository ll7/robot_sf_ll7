#!/usr/bin/env python3
"""Guard against version drift by making the git tag the single source of truth.

Three version axes historically drifted apart in this repository:

* the git tag / release line (canonical: external evidence provenance pins
  resolve by these tags, so the tag line must never be renumbered),
* the packaged version in ``pyproject.toml`` (now derived from the tag by
  hatch-vcs, no longer hardcoded),
* ``CITATION.cff`` (now kept in lockstep with the latest full release tag).

This checker verifies that, when HEAD is on a version tag, the built/installed
package version derives from that tag, and that ``CITATION.cff`` matches the
latest full release tag. It is wired into the release workflow (gating) and
into ``scripts/dev/ci_driver.sh`` (advisory).

Release-line tag convention:

* full release tags are plain ``X.Y.Z`` (e.g. ``0.0.2``) or ``vX.Y.Z``,
* release-candidate tags are ``rcX.Y.Z`` (e.g. ``rc0.0.3``); both map to the
  numeric ``X.Y.Z`` for package-version derivation, but only full releases set
  ``CITATION.cff``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CITATION = REPO_ROOT / "CITATION.cff"

# Release-line version tags only: plain X.Y.Z, vX.Y.Z, or rcX.Y.Z. This is
# deliberately stricter than the hatch-vcs tag_regex so that unrelated tags
# (artifact/*, camera-ready-*, ...) are not misread as release versions.
_VERSION_TAG_RE = re.compile(r"^(?:rc|v)?(?P<num>\d+\.\d+\.\d+)$")
_BASE_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")


def numeric_version_from_tag(tag: str) -> str | None:
    """Return the numeric ``X.Y.Z`` for a release-line tag, else ``None``.

    Examples: ``0.0.2`` -> ``0.0.2``, ``rc0.0.3`` -> ``0.0.3``,
    ``v1.2.3`` -> ``1.2.3``, ``artifact/foo`` -> ``None``.

    Returns:
        The numeric version string, or ``None`` when ``tag`` is not a
        release-line version tag.
    """
    match = _VERSION_TAG_RE.match(tag.strip())
    return match.group("num") if match else None


def is_release_candidate(tag: str) -> bool:
    """Return ``True`` for release-candidate tags (``rcX.Y.Z``).

    Returns:
        Whether the tag denotes a release candidate rather than a full release.
    """
    return tag.strip().startswith("rc")


def base_version(version: str) -> str:
    """Return the ``X.Y.Z`` release component of a (possibly dev) version.

    ``0.0.3.dev4+g1234`` -> ``0.0.3``; ``0.0.2`` -> ``0.0.2``.

    Returns:
        The leading numeric release component, or ``version`` unchanged when it
        does not start with an ``X.Y.Z`` triple.
    """
    match = _BASE_VERSION_RE.match(version.strip())
    return match.group(1) if match else version.strip()


def full_release_tags(all_tags: list[str]) -> list[str]:
    """Return the subset of tags that are full releases (not candidates).

    Returns:
        Tags matching the release-line convention with the ``rc`` prefix
        excluded.
    """
    return [
        tag for tag in all_tags if numeric_version_from_tag(tag) and not is_release_candidate(tag)
    ]


def latest_release_tag(all_tags: list[str]) -> str | None:
    """Return the highest full release tag by numeric version, else ``None``.

    Returns:
        The full release tag with the greatest ``X.Y.Z`` value, or ``None`` when
        no full release tag exists.
    """
    releases = full_release_tags(all_tags)
    if not releases:
        return None

    def sort_key(tag: str) -> tuple[int, ...]:
        num = numeric_version_from_tag(tag) or "0.0.0"
        return tuple(int(part) for part in num.split("."))

    return max(releases, key=sort_key)


def evaluate(
    head_tags: list[str],
    all_tags: list[str],
    package_version: str | None,
    citation_version: str,
) -> list[str]:
    """Return a list of alignment problems (empty means aligned).

    Args:
        head_tags: Tags pointing at the current HEAD commit.
        all_tags: All tags in the repository.
        package_version: The built/installed package version, or ``None`` when
            the package is not installed.
        citation_version: The ``version:`` field from ``CITATION.cff``.

    Returns:
        Human-readable problem strings; an empty list means everything aligns.
    """
    problems: list[str] = []

    head_version_tags = {
        numeric_version_from_tag(tag) for tag in head_tags if numeric_version_from_tag(tag)
    }
    if head_version_tags:
        if package_version is None:
            problems.append(
                "HEAD is on release tag(s) "
                f"{sorted(head_version_tags)} but robot_sf is not installed, so "
                "the tag-derived package version cannot be verified"
            )
        else:
            derived = base_version(package_version)
            if derived not in head_version_tags:
                problems.append(
                    f"package version {package_version!r} (base {derived}) does "
                    f"not derive from HEAD release tag(s) {sorted(head_version_tags)}"
                )

    latest = latest_release_tag(all_tags)
    if latest is None:
        problems.append("no full release tag (X.Y.Z) found to align CITATION.cff against")
    else:
        latest_num = numeric_version_from_tag(latest)
        if base_version(citation_version) != latest_num:
            problems.append(
                f"CITATION.cff version {citation_version!r} does not match the "
                f"latest full release tag {latest!r} ({latest_num})"
            )

    return problems


def _git(args: list[str]) -> list[str]:
    """Run a git command in the repo and return non-empty output lines.

    Returns:
        The command's stdout split into stripped, non-empty lines (empty on
        failure).
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def git_tags_at_head() -> list[str]:
    """Return tags pointing at the current HEAD.

    Returns:
        Tag names on HEAD, or an empty list when none/unavailable.
    """
    return _git(["tag", "--points-at", "HEAD"])


def git_all_tags() -> list[str]:
    """Return all tags in the repository.

    Returns:
        All tag names, or an empty list when unavailable.
    """
    return _git(["tag"])


def installed_package_version() -> str | None:
    """Return the installed ``robot_sf`` version, or ``None`` if not installed.

    Returns:
        The version string reported by installed package metadata, or ``None``.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        return pkg_version("robot_sf")
    except PackageNotFoundError:
        return None


def load_citation_version(path: Path) -> str:
    """Return the ``version:`` field from a ``CITATION.cff`` file.

    Returns:
        The citation version as a string.

    Raises:
        KeyError: If the file has no ``version`` field.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(data["version"])


def main(argv: list[str] | None = None) -> int:
    """Run the alignment check and report problems.

    Returns:
        Process exit code (0 when aligned or in advisory mode, 1 otherwise).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--citation",
        type=Path,
        default=DEFAULT_CITATION,
        help="Path to CITATION.cff (default: repo root CITATION.cff)",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Report problems but always exit 0 (non-gating CI usage).",
    )
    args = parser.parse_args(argv)

    head_tags = git_tags_at_head()
    all_tags = git_all_tags()
    package_version = installed_package_version()
    citation_version = load_citation_version(args.citation)

    problems = evaluate(head_tags, all_tags, package_version, citation_version)

    print(f"HEAD tags: {head_tags or '(none)'}")
    print(f"installed robot_sf version: {package_version or '(not installed)'}")
    print(f"CITATION.cff version: {citation_version}")
    latest = latest_release_tag(all_tags)
    print(f"latest full release tag: {latest or '(none)'}")

    if not problems:
        print("OK: all version axes align with the git tag / release line.")
        return 0

    print("\nVersion drift detected:")
    for problem in problems:
        print(f"  - {problem}")
    if args.advisory:
        print("\n(advisory mode: not failing this step)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
