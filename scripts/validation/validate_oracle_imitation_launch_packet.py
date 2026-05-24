"""Validate oracle-imitation dataset launch packets before Slurm collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.training.oracle_imitation_launch_packet import (
    LaunchPacketError,
    validate_launch_packet,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate a pre-Slurm oracle-imitation dataset launch packet."
    )
    parser.add_argument("--config", required=True, type=Path, help="Launch-packet YAML path.")
    parser.add_argument(
        "--repo-root",
        default=Path.cwd(),
        type=Path,
        help="Repository root used to resolve relative paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON validation report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate a launch packet and return a shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)
    try:
        report = validate_launch_packet(args.config, repo_root=args.repo_root)
    except LaunchPacketError as exc:
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(str(exc))
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "oracle-imitation launch packet valid: "
            f"{report['dataset_id']} ({report['episode_count']} planned episodes)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
