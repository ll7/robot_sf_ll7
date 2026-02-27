#!/usr/bin/env python3
"""Run a config-driven camera-ready benchmark campaign."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.benchmark.camera_ready_campaign import (
    load_campaign_config,
    prepare_campaign_preflight,
    run_campaign,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for camera-ready campaign execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to camera-ready campaign config YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional campaign base output directory. Defaults to output/benchmarks/camera_ready"
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label suffix embedded into campaign_id.",
    )
    parser.add_argument(
        "--skip-publication-bundle",
        action="store_true",
        help="Skip publication bundle export even if enabled in config.",
    )
    parser.add_argument(
        "--mode",
        choices=("run", "preflight"),
        default="run",
        help="Execution mode: full run or preflight-only artifact generation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"),
        help="Log level for campaign execution.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute camera-ready benchmark campaign from CLI arguments."""
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parser = _build_parser()
    args = parser.parse_args(raw_argv)

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    cfg = load_campaign_config(args.config)
    invoked_command = shlex.join([sys.executable, str(Path(__file__)), *raw_argv])
    if args.mode == "preflight":
        prepared = prepare_campaign_preflight(
            cfg,
            output_root=args.output_root,
            label=args.label,
            invoked_command=invoked_command,
        )
        result = {
            "campaign_id": prepared["campaign_id"],
            "campaign_root": str(prepared["campaign_root"]),
            "validate_config_path": str(prepared["validate_config_path"]),
            "preview_scenarios_path": str(prepared["preview_scenarios_path"]),
            "matrix_summary_json": str(prepared["matrix_summary_json_path"]),
            "matrix_summary_csv": str(prepared["matrix_summary_csv_path"]),
            "amv_coverage_json": str(prepared["amv_coverage_json_path"]),
            "amv_coverage_md": str(prepared["amv_coverage_md_path"]),
            "comparability_json": (
                str(prepared["comparability_json_path"])
                if prepared.get("comparability_json_path") is not None
                else None
            ),
            "comparability_md": (
                str(prepared["comparability_md_path"])
                if prepared.get("comparability_md_path") is not None
                else None
            ),
        }
    else:
        result = run_campaign(
            cfg,
            output_root=args.output_root,
            label=args.label,
            skip_publication_bundle=bool(args.skip_publication_bundle),
            invoked_command=invoked_command,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
