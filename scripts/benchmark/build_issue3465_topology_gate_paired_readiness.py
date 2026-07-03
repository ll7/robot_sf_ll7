#!/usr/bin/env python3
"""Build issue #3465 topology-gate paired preregistration readiness packet."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from robot_sf.benchmark.topology_gate_paired_preregistration import (
    build_topology_gate_readiness_packet,
    load_topology_gate_paired_config,
    write_readiness_packet,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/benchmarks/issue_3465_topology_gate_paired.yaml"


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Tracked preregistration YAML.")
    parser.add_argument(
        "--out",
        default="output/issue_3465_topology_gate_paired/readiness",
        help="Output deterministic readiness JSON.",
    )
    return parser.parse_args()


def main() -> int:
    """Build readiness packet and return process exit code."""

    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_topology_gate_paired_config(config_path)
    packet = build_topology_gate_readiness_packet(config)
    packet["git_head"] = _git_head()
    out_path = write_readiness_packet(packet, REPO_ROOT / args.out)
    try:
        display_path = out_path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = out_path
    print(display_path)
    print(f"status={packet['status']}")
    print(f"row_check.complete={packet['row_check']['complete']}")
    print(f"arm_check.complete={packet['arm_check']['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
