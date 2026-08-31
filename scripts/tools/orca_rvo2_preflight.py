#!/usr/bin/env python3
"""Standalone ORCA-rvo2 preflight guard for camera-ready benchmark configs.

Use this script before submitting a SLURM camera-ready benchmark to fail fast when
the config includes ORCA planners but rvo2 is not installed.

Usage:
  python scripts/tools/orca_rvo2_preflight.py --config path/to/campaign.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.benchmark.orca_preflight import (
    OrcaRvo2PreflightError,
    check_orca_rvo2_preflight_from_config,
)

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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable preflight report on stdout.",
    )
    return parser


def _emit_report(report: dict[str, object], *, as_json: bool) -> None:
    """Emit a structured report when requested, leaving the default CLI unchanged."""
    if as_json:
        print(json.dumps(report, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ORCA-rvo2 preflight check from CLI arguments."""
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    config_path = args.config.resolve()
    report: dict[str, object] = {
        "schema_version": "orca-rvo2-preflight.v1",
        "config": str(config_path),
        "status": "blocked",
        "submit_safe": False,
        "no_submit": True,
        "blockers": [],
        "remediation": [],
        "planner_keys": [],
    }

    if not config_path.is_file():
        report["blockers"] = ["config_not_found"]
        report["remediation"] = [
            f"provide an existing campaign config: {args.config}",
        ]
        _emit_report(report, as_json=args.json)
        logger.error(f"Config file not found or not a regular file: {args.config}")
        return 1

    try:
        check_orca_rvo2_preflight_from_config(config_path)
    except OrcaRvo2PreflightError as exc:
        report["blockers"] = ["rvo2_unavailable"]
        report["remediation"] = [
            "pip install third_party/python-rvo2",
            "pip install git+https://github.com/mit-acl/Python-RVO2",
        ]
        report["planner_keys"] = list(exc.planner_keys)
        report["detail"] = str(exc)
        _emit_report(report, as_json=args.json)
        return 1
    except Exception as exc:
        report["status"] = "input_error"
        report["blockers"] = ["config_invalid_or_unreadable"]
        report["detail"] = str(exc)
        _emit_report(report, as_json=args.json)
        logger.error(f"Failed to load or validate campaign config: {exc}")
        return 1

    report["status"] = "ready"
    report["submit_safe"] = True
    _emit_report(report, as_json=args.json)
    logger.info("ORCA-rvo2 preflight passed; config is safe to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
