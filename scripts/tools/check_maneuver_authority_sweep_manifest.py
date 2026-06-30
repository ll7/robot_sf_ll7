#!/usr/bin/env python3
"""CLI for the issue #3213 maneuver-authority sweep manifest checker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.maneuver_authority_sweep_manifest import (
    MANIFEST_STATUS_READY,
    check_maneuver_authority_sweep_manifest,
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "maneuver_authority_sweep_manifest_issue_3213.yaml"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to maneuver-authority-sweep-manifest.v1 YAML.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve relative manifest paths.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the checker and print a structured JSON report."""
    args = _parse_args(argv)
    report = check_maneuver_authority_sweep_manifest(
        args.manifest,
        repo_root=args.repo_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == MANIFEST_STATUS_READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
