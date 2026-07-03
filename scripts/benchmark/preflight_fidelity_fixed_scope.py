#!/usr/bin/env python3
"""Preflight the full fixed-scope #3207 fidelity-sensitivity campaign (no execution).

Materializes the full fixed-scope run plan from the study config, resolves planner
availability against the algorithm-readiness catalog, and validates that the
primary (ranking) metric is identifiable by contract. Emits a launch/readiness
packet only; it fails closed and runs no benchmark episodes.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from robot_sf.benchmark.fidelity_fixed_scope_preflight import (
    DECISION_READY,
    build_fixed_scope_preflight,
    write_fixed_scope_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/research/fidelity_sensitivity_v1.yaml"


def _git_head() -> str:
    """Return the current git head, or ``unknown`` when unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _repo_rel(path: Path) -> str:
    """Return a repo-relative path string when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Fidelity-sensitivity study config (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output directory for the preflight packet JSON.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero when the preflight is not ready.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    packet = build_fixed_scope_preflight(
        config,
        config_path=_repo_rel(config_path),
        git_head=_git_head(),
    )

    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
        packet_path = write_fixed_scope_preflight(packet, out_dir)
        print(f"wrote fidelity fixed-scope preflight packet: {_repo_rel(packet_path)}")

    print(f"decision: {packet['decision']}")
    if packet["blockers"]:
        print("blockers:")
        for blocker in packet["blockers"]:
            print(f"  - {blocker}")

    if args.require_ready and packet["decision"] != DECISION_READY:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
