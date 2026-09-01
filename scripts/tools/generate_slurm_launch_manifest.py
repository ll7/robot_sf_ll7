#!/usr/bin/env python3
"""Generate the public no-submit Slurm launch manifest for a frozen release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.slurm_launch_manifest import (
    SCHEMA_VERSION,
    generate_slurm_launch_manifest,
    sha256_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-identity", type=Path, required=True)
    parser.add_argument("--runner-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="repository containing the identity, preflight artifacts, and output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate a launch manifest and print a compact receipt."""
    args = _parser().parse_args(argv)
    try:
        payload = generate_slurm_launch_manifest(
            resolved_identity_path=args.resolved_identity,
            runner_preflight_path=args.runner_preflight,
            output_path=args.output,
            repository_root=args.repository_root,
        )
        output_path = (
            args.output if args.output.is_absolute() else args.repository_root / args.output
        )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "generated",
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path.resolve()),
            "campaign_id": payload["campaign_id"],
            "planner_arms": payload["matrix"]["planner_arms"],
            "expected_episode_cells": payload["matrix"]["expected_episode_cells"],
            "no_submit": payload["no_submit"],
        }
    except (OSError, ValueError, TypeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "rejected",
                    "reason": str(exc),
                    "no_submit": True,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
