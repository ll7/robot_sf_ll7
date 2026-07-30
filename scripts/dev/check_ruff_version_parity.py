#!/usr/bin/env python3
"""Guard against Ruff version drift across three independent version axes.

Checks that the Ruff version is consistent across:

1. The ``astral-sh/ruff-pre-commit`` ``rev:`` in ``.pre-commit-config.yaml``
2. ``[tool.ruff]`` ``required-version`` in ``pyproject.toml``
3. The ``ruff==`` pin in a ``[dependency-groups]`` group in ``pyproject.toml``

Exits non-zero when the versions disagree.  Intended to be wired into the
dev-validation surface as a gating local pre-commit hook and a non-gating
advisory CI check so that future drift fails fast.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"
RUFF_PRE_COMMIT_REPO = "https://github.com/astral-sh/ruff-pre-commit"
RUFF_PIN_PATTERN = re.compile(r"^ruff\s*==\s*(?P<version>\S+)$")


def normalize_version(raw: str) -> str:
    """Normalize a version string by stripping common prefixes.

    Strips, in order:
    * surrounding whitespace and quote characters,
    * a ``ruff==`` prefix,
    * a ``==`` prefix,
    * a leading ``v``.

    Examples:
        ``v0.16.0`` -> ``0.16.0``
        ``==0.16.0`` -> ``0.16.0``
        ``ruff==0.16.0`` -> ``0.16.0``
    """
    raw = raw.strip().strip("\"'")
    raw = re.sub(r"^ruff\s*==\s*", "", raw)
    raw = re.sub(r"^==\s*", "", raw)
    raw = re.sub(r"^v", "", raw)
    return raw


def read_pre_commit_version(path: Path) -> str:
    """Return the Ruff rev from ``.pre-commit-config.yaml``.

    Raises:
        ValueError: If the ruff-pre-commit repo or its rev is not found.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    repos = data.get("repos", [])
    for repo in repos:
        repo_url = str(repo.get("repo", "")).removesuffix(".git")
        if repo_url == RUFF_PRE_COMMIT_REPO:
            rev = repo.get("rev")
            if rev is not None:
                return str(rev)
            raise ValueError(f"ruff-pre-commit entry in {path} has no 'rev'")
    raise ValueError(f"no astral-sh/ruff-pre-commit repo found in {path}")


def read_pyproject_required_version(path: Path) -> str:
    """Return ``required-version`` from ``[tool.ruff]`` in ``pyproject.toml``.

    Raises:
        ValueError: If the field is missing or not found.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    tool = data.get("tool", {})
    ruff = tool.get("ruff", {})
    version = ruff.get("required-version")
    if version is not None:
        return str(version)
    raise ValueError(f"no [tool.ruff] required-version found in {path}")


def read_pyproject_dev_dep_pin(path: Path) -> str:
    """Return the ``ruff==X.Y.Z`` pin from ``[dependency-groups]``.

    Raises:
        ValueError: If no ruff pin is found in any dependency group.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    dep_groups = data.get("dependency-groups", {})
    dev_dependencies = dep_groups.get("dev", [])
    if not isinstance(dev_dependencies, list):
        raise ValueError(f"[dependency-groups].dev in {path} must be a list")
    for dependency in dev_dependencies:
        if isinstance(dependency, str) and RUFF_PIN_PATTERN.fullmatch(dependency.strip()):
            return dependency.strip()
    raise ValueError(f"no exact ruff== pin found in [dependency-groups].dev in {path}")


def evaluate(
    pre_commit_version: str,
    required_version: str,
    dev_pin: str,
) -> list[str]:
    """Compare three Ruff version axes and return a list of mismatches.

    Returns:
        Human-readable problem strings; empty means all three agree.
    """
    problems: list[str] = []

    norm_pre = normalize_version(pre_commit_version)
    norm_req = normalize_version(required_version)
    norm_dev = normalize_version(dev_pin)

    if norm_pre != norm_req:
        problems.append(
            f".pre-commit-config.yaml rev {pre_commit_version!r} "
            f"(normalized {norm_pre!r}) != "
            f"pyproject.toml [tool.ruff] required-version "
            f"{required_version!r} (normalized {norm_req!r})"
        )
    if norm_pre != norm_dev:
        problems.append(
            f".pre-commit-config.yaml rev {pre_commit_version!r} "
            f"(normalized {norm_pre!r}) != "
            f"pyproject.toml [dependency-groups] ruff pin "
            f"{dev_pin!r} (normalized {norm_dev!r})"
        )
    if norm_req != norm_dev:
        problems.append(
            f"pyproject.toml [tool.ruff] required-version "
            f"{required_version!r} (normalized {norm_req!r}) != "
            f"pyproject.toml [dependency-groups] ruff pin "
            f"{dev_pin!r} (normalized {norm_dev!r})"
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the Ruff version parity check.

    Returns:
        Exit code 0 when all versions agree, 1 on mismatch (or 0 in advisory).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pre-commit-config",
        type=Path,
        default=DEFAULT_PRE_COMMIT,
        help="Path to .pre-commit-config.yaml (default: repo root)",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help="Path to pyproject.toml (default: repo root)",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Report problems but always exit 0 (non-gating CI usage).",
    )
    args = parser.parse_args(argv)

    pre_commit_version = read_pre_commit_version(args.pre_commit_config)
    required_version = read_pyproject_required_version(args.pyproject)
    dev_pin = read_pyproject_dev_dep_pin(args.pyproject)

    problems = evaluate(pre_commit_version, required_version, dev_pin)

    print(f".pre-commit-config.yaml rev:       {pre_commit_version!r}")
    print(f"pyproject.toml required-version:   {required_version!r}")
    print(f"pyproject.toml dev ruff pin:       {dev_pin!r}")

    if not problems:
        print("OK: all three Ruff version axes agree.")
        return 0

    print("\nRuff version drift detected:")
    for problem in problems:
        print(f"  - {problem}")
    if args.advisory:
        print("\n(advisory mode: not failing this step)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
