#!/usr/bin/env python3
"""Run a config-driven camera-ready benchmark campaign.

Exit codes preserve fail-closed campaign semantics for non-success outcomes:
- 0: benchmark-success campaign
- 2: unexpected failure, malformed result, or mixed failed/partial-failure outcome
- 3: accepted-unavailable-only campaign outcome (non-success, fail-closed)
"""

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
from robot_sf.benchmark.fallback_policy import campaign_exit_code
from robot_sf.benchmark.orca_preflight import OrcaRvo2PreflightError

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
        "--campaign-id",
        type=str,
        default=None,
        help=(
            "Optional exact campaign directory id. Use with resume-enabled configs to continue "
            "an interrupted campaign root."
        ),
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
        "--checkpoint-preflight-mode",
        choices=("metadata_only", "enforced_staged"),
        default="metadata_only",
        help=(
            "Arm-checkpoint preflight mode (issue #4613/#4663). 'metadata_only' (default) is the "
            "cheap network-free guard and is NOT submit-safe when any arm is only "
            "stageable_remote. 'enforced_staged' actually downloads and checksum-verifies each "
            "registry checkpoint into the durable cache before continuing; the submit/sbatch "
            "wrapper must use this mode (or run the public "
            "scripts/benchmark/submit_camera_ready_checkpoint_gate.sh) before requeueing. Only "
            "applied to the preflight-only mode path; 'run' mode keeps the cheap guard and "
            "expects checkpoints to be already staged on the compute node."
        ),
    )
    parser.add_argument(
        "--checkpoint-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory override for staged downloads "
        "(used with --checkpoint-preflight-mode=enforced_staged).",
    )
    parser.add_argument(
        "--checkpoint-registry-path",
        type=Path,
        default=None,
        help="Optional model-registry path override for the arm-checkpoint preflight.",
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
    try:
        if args.mode == "preflight":
            prepared = prepare_campaign_preflight(
                cfg,
                output_root=args.output_root,
                label=args.label,
                campaign_id=args.campaign_id,
                invoked_command=invoked_command,
                checkpoint_preflight_mode=args.checkpoint_preflight_mode,
                checkpoint_cache_dir=args.checkpoint_cache_dir,
                checkpoint_registry_path=args.checkpoint_registry_path,
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
                campaign_id=args.campaign_id,
                skip_publication_bundle=bool(args.skip_publication_bundle),
                invoked_command=invoked_command,
            )
    except OrcaRvo2PreflightError as exc:
        result = {
            "mode": args.mode,
            "status": "orca_preflight_failed",
            "status_reason": str(exc),
            "benchmark_success": False,
            "exit_code": 2,
            "campaign_execution_status": "failed",
            "evidence_status": "blocked",
            "row_status_summary": {
                "successful_evidence_rows": 0,
                "accepted_unavailable_rows": 0,
                "unexpected_failed_rows": 0,
                "fallback_or_degraded_rows": 0,
            },
        }
    print(json.dumps(result, indent=2))
    if args.mode == "preflight" and result.get("status") != "orca_preflight_failed":
        return 0
    return campaign_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
