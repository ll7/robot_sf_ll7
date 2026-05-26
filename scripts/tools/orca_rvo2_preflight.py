#!/usr/bin/env python3
"""Standalone ORCA-rvo2 preflight guard for camera-ready benchmark configs.

Use this script before submitting a SLURM camera-ready benchmark to fail fast when
the config includes ORCA planners but rvo2 is not installed.

Usage:
  python scripts/tools/orca_rvo2_preflight.py --config path/to/campaign.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.benchmark.orca_preflight import check_orca_rvo2_preflight_from_config

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to camera-ready campaign config YAML.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ORCA-rvo2 preflight check from CLI arguments."""
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parser = _build_parser()
    args = parser.parse_args(raw_argv)

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1

    try:
        check_orca_rvo2_preflight_from_config(args.config.resolve())
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    logger.info("ORCA-rvo2 preflight passed; config is safe to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
