#!/usr/bin/env python3
"""Validate the live same-seed forecast replay gate (issue #2944).

This is a fail-closed smoke gate for the none+cv forecast variants.  It does
not train models or run expensive campaigns.  It loads a simulation trace
export, builds ForecastBatch.v1 artifacts for each requested variant, computes
baseline closed-loop metrics from the trace, and reports a per-run
classification (native, blocked, degraded, diagnostic_only) that gates whether
the full forecast variant matrix should be expanded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from robot_sf.benchmark.live_forecast_replay_gate import (
    DEFAULT_HORIZONS_S,
    FORECAST_VARIANTS,
    SMOKE_FORECAST_VARIANTS,
    LiveForecastReplayGateConfig,
    LiveForecastReplayGateError,
    load_trace_tolerant,
    run_live_forecast_replay_gate,
    write_live_forecast_replay_gate_report,
)
from robot_sf.common.artifact_paths import get_repository_root

_DEFAULT_TRACE = (
    get_repository_root()
    / "tests"
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "dense_pedestrian_stress_episode_0000.json"
)


def _git_head() -> str | None:
    """Return the current repository HEAD short sha, or None when not available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10.0,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    default_horizons = " ".join(f"{horizon:g}" for horizon in DEFAULT_HORIZONS_S)
    parser = argparse.ArgumentParser(
        description="Native CV-only closed-loop replay smoke (issue #2944)."
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=_DEFAULT_TRACE,
        help="Path to a simulation_trace_export.v1 JSON trace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write gate_report.json and gate_report.md.",
    )
    parser.add_argument(
        "--horizon-s",
        type=float,
        action="append",
        help=f"Forecast horizon in seconds. Repeatable. Defaults to {default_horizons}.",
    )
    parser.add_argument(
        "--collision-distance-m",
        type=float,
        default=LiveForecastReplayGateConfig.collision_distance_m,
        help="Robot-pedestrian collision distance threshold in meters.",
    )
    parser.add_argument(
        "--near-miss-distance-m",
        type=float,
        default=LiveForecastReplayGateConfig.near_miss_distance_m,
        help="Robot-pedestrian near-miss distance threshold in meters.",
    )
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Evaluate all forecast variants (none, cv, semantic, interaction_aware, risk_filtered).",
    )
    parser.add_argument(
        "--generated-at-utc",
        help="Optional deterministic ISO-8601 generation timestamp.",
    )
    return parser


def _build_config_from_args(args: argparse.Namespace) -> LiveForecastReplayGateConfig:
    """Build a gate config from CLI arguments."""

    kwargs: dict[str, Any] = {
        "collision_distance_m": args.collision_distance_m,
        "near_miss_distance_m": args.near_miss_distance_m,
    }
    if args.horizon_s:
        kwargs["horizons_s"] = tuple(args.horizon_s)
    return LiveForecastReplayGateConfig(**kwargs)


def _variants_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the forecast variant set requested by the CLI."""

    return FORECAST_VARIANTS if args.full_matrix else SMOKE_FORECAST_VARIANTS


def main(argv: list[str] | None = None) -> int:
    """Run the gate and return a shell-friendly exit code."""

    args = build_arg_parser().parse_args(argv)

    if not args.trace.exists():
        print(
            json.dumps(
                {"status": "error", "error": f"trace not found: {args.trace}"},
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    try:
        trace = load_trace_tolerant(args.trace)
        config = _build_config_from_args(args)
        variants = _variants_from_args(args)
        report = run_live_forecast_replay_gate(
            trace,
            config=config,
            variants=variants,
            repo_head=_git_head(),
            generated_at_utc=args.generated_at_utc,
        )
    except (LiveForecastReplayGateError, OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "error", "error": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    if args.output_dir:
        json_path, md_path = write_live_forecast_replay_gate_report(report, args.output_dir)
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
