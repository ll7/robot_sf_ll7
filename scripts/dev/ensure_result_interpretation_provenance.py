#!/usr/bin/env python3
"""Hydrate Git objects required by committed result-interpretation fixtures.

The result-interpretation validator intentionally fails closed when a fixture's
source or artifact-catalog commit is unavailable.  A full-depth checkout of the
current branch does not necessarily include commits that are no longer reachable
from a branch ref, so CI fetches the exact commits declared by the fixtures before
running the tests.  An unavailable commit remains a hard failure.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parents[2]
PACKET_FIXTURE_DIR = ROOT / "tests/fixtures/result_interpretation_packet/v1"
ARTIFACT_CATALOG_FIXTURE = ROOT / "tests/fixtures/artifact_catalog/v1/valid_catalog.yaml"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _validated_commit(value: object, *, field: str, path: Path) -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise ValueError(f"{path}: {field} is not a full hexadecimal commit SHA")
    return value


def _load_packet(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: packet must be a JSON object")
    return payload


def collect_required_commits(
    packet_fixture_dir: Path = PACKET_FIXTURE_DIR,
    artifact_catalog_fixture: Path = ARTIFACT_CATALOG_FIXTURE,
) -> tuple[str, ...]:
    """Return source and catalog-generation commits required by the fixtures."""
    commits: set[str] = set()
    packet_paths = sorted(packet_fixture_dir.glob("*.json"))
    if not packet_paths:
        raise ValueError(f"no result-interpretation packet fixtures found in {packet_fixture_dir}")
    for packet_path in packet_paths:
        sources = _load_packet(packet_path).get("sources")
        if not isinstance(sources, list):
            raise ValueError(f"{packet_path}: sources must be a list")
        for index, source in enumerate(sources):
            if not isinstance(source, Mapping):
                raise ValueError(f"{packet_path}: source {index} must be an object")
            for field in ("commit", "tracked_commit"):
                commits.add(_validated_commit(source.get(field), field=field, path=packet_path))

    catalog = yaml.safe_load(artifact_catalog_fixture.read_text(encoding="utf-8"))
    if not isinstance(catalog, Mapping) or not isinstance(catalog.get("artifacts"), list):
        raise ValueError(f"{artifact_catalog_fixture}: artifacts must be a list")
    for index, artifact in enumerate(catalog["artifacts"]):
        if not isinstance(artifact, Mapping):
            raise ValueError(f"{artifact_catalog_fixture}: artifact {index} must be an object")
        commits.add(
            _validated_commit(
                artifact.get("generation_commit"),
                field="generation_commit",
                path=artifact_catalog_fixture,
            )
        )
    return tuple(sorted(commits))


def _commit_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _fetch_commits(commits: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), "fetch", "--no-tags", "origin", *commits],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    """Hydrate and verify every declared fixture provenance commit."""
    try:
        required = collect_required_commits()
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Result-interpretation provenance manifest is invalid: {exc}", file=sys.stderr)
        return 2

    missing = [commit for commit in required if not _commit_exists(commit)]
    if missing:
        print(f"Fetching {len(missing)} missing result-interpretation provenance commits.")
        result = _fetch_commits(missing)
        if result.returncode != 0:
            diagnostic = result.stderr.strip() or result.stdout.strip() or "no git diagnostic"
            print(
                "Could not hydrate result-interpretation provenance commits: " + diagnostic,
                file=sys.stderr,
            )
            return 2

    unresolved = [commit for commit in required if not _commit_exists(commit)]
    if unresolved:
        print(
            "Result-interpretation provenance remains unavailable after fetch: "
            + ", ".join(unresolved),
            file=sys.stderr,
        )
        return 2

    print(
        "Result-interpretation provenance ready: "
        f"{len(required)} declared commits, {len(missing)} fetched."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
