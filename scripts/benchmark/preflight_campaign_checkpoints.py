"""Pre-``sbatch`` gate that verifies (and optionally stages) campaign arm checkpoints.

Run this before submitting a camera-ready benchmark campaign so a missing or corrupt arm
checkpoint fails in seconds on the submit node instead of ~14h into compute (issue #4613: the S30
campaign jobs 13296 and 13301 both failed identically on a missing PPO ``model_cache`` checkpoint).

Modes:

- default (``--check``): confirm every enabled arm's ``model_id`` / ``model_path`` checkpoint is
  present locally or has a durable remote source to stage from. Network-free.
- ``--stage``: enforced pre-submit staging -- actually download and checksum-verify each registry
  checkpoint into the durable cache so the compute node loads a validated file.

Exit codes are distinct so an sbatch wrapper can branch mechanically:

- ``0`` -- all arm checkpoints resolvable (``--stage`` also means staged + verified).
- ``2`` -- the campaign config file is missing or unreadable (cannot be evaluated).
- ``3`` -- one or more arm checkpoints are unresolvable (fail-closed; do not submit).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.campaign_checkpoint_preflight import (
    CampaignCheckpointPreflightError,
    check_campaign_arm_checkpoints_preflight_from_config,
)

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_BLOCKED = 3


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Verify (and optionally stage) camera-ready campaign arm checkpoints "
        "before sbatch (issue #4613).",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a camera-ready campaign config YAML.",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Download and checksum-verify each registry checkpoint into the durable cache "
        "(enforced pre-submit staging) instead of the cheap network-free resolvability check.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Optional model-registry path override.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory override for staged downloads.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the preflight summary as JSON on stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the campaign checkpoint preflight CLI.

    Returns:
        int: Process exit code (see module docstring).
    """
    args = build_arg_parser().parse_args(argv)
    if not args.config.is_file():
        print(f"error: campaign config not found: {args.config}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    try:
        summary = check_campaign_arm_checkpoints_preflight_from_config(
            args.config,
            stage=bool(args.stage),
            registry_path=args.registry_path,
            cache_dir=args.cache_dir,
        )
    except CampaignCheckpointPreflightError as exc:
        print(str(exc), file=sys.stderr)
        if args.json:
            print(json.dumps({"status": "blocked", "arms": list(exc.arms)}, indent=2))
        return EXIT_BLOCKED
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"error: could not evaluate campaign config: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    mode = "staged" if args.stage else "resolvable"
    if args.json:
        print(json.dumps({"status": "ok", **summary}, indent=2))
    else:
        print(
            f"campaign checkpoint preflight passed: {summary['resolved']}/{summary['checked']} "
            f"arm checkpoint reference(s) {mode}."
        )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
