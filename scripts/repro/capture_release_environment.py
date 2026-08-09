#!/usr/bin/env python3
"""Capture tag-side and verification-environment evidence for a release target."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, args: Sequence[str]) -> str:
    """Run a read-only Git query and return its stripped standard output."""
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no stderr"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _project_constraints(path: Path) -> dict[str, Any]:
    """Extract dependency declarations without resolving or changing them."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml is missing its [project] table")
    return {
        "requires_python": project.get("requires-python"),
        "license": project.get("license"),
        "dependencies": sorted(project.get("dependencies", [])),
        "optional_dependencies": {
            name: sorted(requirements)
            for name, requirements in sorted(project.get("optional-dependencies", {}).items())
        },
        "dependency_groups": {
            name: sorted(requirements)
            for name, requirements in sorted(document.get("dependency-groups", {}).items())
        },
    }


def _lock_resolved_packages(path: Path) -> list[dict[str, Any]]:
    """Extract the name/version resolution from a uv lock file."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    packages = document.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("uv.lock is missing non-empty [[package]] entries")
    resolved: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("uv.lock contains a non-mapping [[package]] entry")
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str):
            raise ValueError("uv.lock package entries require a string name field")
        if version is not None and not isinstance(version, str):
            raise ValueError("uv.lock package versions must be strings when present")
        entry: dict[str, Any] = {"name": name, "version": version}
        if version is None:
            source = package.get("source")
            if not isinstance(source, dict) or not isinstance(source.get("editable"), str):
                raise ValueError(
                    "uv.lock package entries without a version require an editable source"
                )
            entry["source"] = {"editable": source["editable"]}
        resolved.append(entry)
    return sorted(resolved, key=lambda item: (item["name"].lower(), item["version"] or ""))


def _uv_version() -> str | None:
    """Return the installed uv version when uv is available."""
    completed = subprocess.run(["uv", "--version"], capture_output=True, check=False, text=True)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _verification_environment() -> dict[str, Any]:
    """Capture the current interpreter and installed distribution inventory."""
    packages = sorted(
        {
            (distribution.metadata.get("Name", distribution.name), distribution.version)
            for distribution in importlib.metadata.distributions()
        },
        key=lambda item: (item[0].lower(), item[1]),
    )
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "uv_version": _uv_version(),
        "packages": [{"name": name, "version": version} for name, version in packages],
    }


def _runtime_record(path: Path) -> dict[str, Any]:
    """Validate and summarize one historical campaign runtime record."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    run = payload.get("run") if isinstance(payload.get("run"), dict) else payload
    python_version = run.get("python_version")
    packages = run.get("packages") or run.get("resolved_packages") or run.get("package_versions")
    if not isinstance(python_version, str) or not python_version:
        raise ValueError(f"{path} is missing an exact runtime python_version")
    if not isinstance(packages, list) or not packages:
        raise ValueError(f"{path} is missing an exact runtime package inventory")
    normalized = []
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError(f"{path} contains a non-mapping package entry")
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise ValueError(f"{path} contains a package without name/version")
        normalized.append({"name": name, "version": version})
    return {
        "artifact_name": path.name,
        "sha256": _sha256_file(path),
        "python_version": python_version,
        "packages": sorted(normalized, key=lambda item: (item["name"].lower(), item["version"])),
    }


def build_packet(
    repo_root: Path,
    release_tag: str,
    campaign_runtime_records: Sequence[Path] = (),
    require_clean: bool = False,
) -> dict[str, Any]:
    """Build a release environment packet from one checked-out tree."""
    status_lines = _git(repo_root, ["status", "--porcelain=v1"]).splitlines()
    if require_clean and status_lines:
        raise RuntimeError("worktree is not clean; refusing exact-head capture")

    pyproject = repo_root / "pyproject.toml"
    lockfile = repo_root / "uv.lock"
    resolved_packages = _lock_resolved_packages(lockfile)
    records = [_runtime_record(path) for path in campaign_runtime_records]
    tag_ref = f"refs/tags/{release_tag}"
    tag_result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", tag_ref],
        capture_output=True,
        check=False,
        text=True,
    )
    resolved_tag = tag_result.stdout.strip() if tag_result.returncode == 0 else None

    return {
        "schema_version": "release_environment_packet.v1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "release_tag": release_tag,
        "candidate_commit": _git(repo_root, ["rev-parse", "HEAD"]),
        "worktree": {
            "clean": not status_lines,
            "status_porcelain_v1": status_lines,
        },
        "tag": {
            "exists_in_checkout": resolved_tag is not None,
            "resolved_commit": resolved_tag,
            "publication_authorization": "not_recorded",
        },
        "repository_inputs": {
            "pyproject.toml": {
                "sha256": _sha256_file(pyproject),
                "constraints": _project_constraints(pyproject),
            },
            "uv.lock": {
                "sha256": _sha256_file(lockfile),
                "resolved_package_count": len(resolved_packages),
                "resolved_packages": resolved_packages,
            },
        },
        "verification_environment": _verification_environment(),
        "campaign_runtime_records": {
            "status": "complete" if records else "missing",
            "records": records,
            "note": (
                "Historical campaign runtime records must be supplied separately; "
                "the lockfile and verification environment are not substitutes."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the release environment capture CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-tag", required=True, help="Release tag being prepared.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON output path, or '-' to write the packet to stdout.",
    )
    parser.add_argument(
        "--campaign-runtime-record",
        action="append",
        type=Path,
        default=[],
        help="Exact historical campaign runtime JSON record; repeat for each campaign.",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Fail unless the checkout has no porcelain status lines.",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    packet = build_packet(
        repo_root,
        args.release_tag,
        campaign_runtime_records=args.campaign_runtime_record,
        require_clean=args.require_clean,
    )
    rendered = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if str(args.output) == "-":
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
