#!/usr/bin/env python3
"""Build the issue #3207 simulation-fidelity sensitivity launch packet."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from robot_sf.benchmark.fidelity_sensitivity import (
    build_launch_packet,
    load_fidelity_sensitivity_config,
    write_launch_packet,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/research/fidelity_sensitivity_v1.yaml",
        help="Tracked fidelity-sensitivity config path.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/context/evidence/issue_3207_fidelity_sensitivity_launch_packet_2026-06-20",
        help="Directory for compact launch-packet JSON and Markdown.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    config = load_fidelity_sensitivity_config(REPO_ROOT / args.config)
    packet = build_launch_packet(config, config_path=args.config, git_head=_git_head())
    write_launch_packet(packet, REPO_ROOT / args.output_dir)
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
