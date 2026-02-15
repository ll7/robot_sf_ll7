#!/usr/bin/env python3
"""Analyze threshold sensitivity (near-miss/comfort) by scenario family.

Input records must include replay payloads (`replay_steps`, `replay_peds`), and
comfort sweeps require `replay_ped_forces` (written when capture-replay is enabled
in the full benchmark orchestrator).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.constants import COLLISION_DIST, COMFORT_FORCE_THRESHOLD, NEAR_MISS_DIST
from robot_sf.benchmark.threshold_sensitivity import (
    analyze_threshold_sensitivity,
    sensitivity_episodes_from_replay_records,
)


def _float_grid(raw: str) -> list[float]:
    """Parse comma-separated numeric grid values."""
    vals = [v.strip() for v in str(raw).split(",") if v.strip()]
    return [float(v) for v in vals]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_jsonl", required=True, help="Input episodes JSONL path")
    parser.add_argument("--out", dest="output_json", required=True, help="Output report JSON path")
    parser.add_argument(
        "--collision-grid",
        default=_default_collision_grid(),
        help="Collision distance thresholds in meters (comma-separated).",
    )
    parser.add_argument(
        "--near-grid",
        default=_default_near_grid(),
        help="Near-miss distance thresholds in meters (comma-separated).",
    )
    parser.add_argument(
        "--comfort-grid",
        default=_default_comfort_grid(),
        help="Comfort force thresholds (comma-separated).",
    )
    parser.add_argument(
        "--ttc-grid",
        default="1.0,1.5,2.0",
        help="TTC horizons in seconds for TTC-gated near-miss analysis.",
    )
    parser.add_argument(
        "--relative-speed-reference",
        type=float,
        default=1.0,
        help="Reference relative speed (m/s) for speed-weighted near-miss scoring.",
    )
    return parser


def _default_collision_grid() -> str:
    """Return safe default collision-threshold grid with non-negative lower bound."""
    lo = max(0.0, COLLISION_DIST - 0.05)
    hi = max(0.0, COLLISION_DIST + 0.05)
    return f"{lo:.2f},{COLLISION_DIST:.2f},{hi:.2f}"


def _default_near_grid() -> str:
    """Return safe default near-miss-threshold grid with non-negative lower bound."""
    lo = max(0.0, NEAR_MISS_DIST - 0.1)
    hi = max(0.0, NEAR_MISS_DIST + 0.1)
    return f"{lo:.2f},{NEAR_MISS_DIST:.2f},{hi:.2f}"


def _default_comfort_grid() -> str:
    """Return safe default comfort-force-threshold grid with non-negative lower bound."""
    lo = max(0.0, COMFORT_FORCE_THRESHOLD - 0.5)
    hi = max(0.0, COMFORT_FORCE_THRESHOLD + 0.5)
    return f"{lo:.2f},{COMFORT_FORCE_THRESHOLD:.2f},{hi:.2f}"


def run(argv: list[str] | None = None) -> int:
    """Execute the threshold sensitivity analysis CLI.

    Returns:
        Process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    records = read_jsonl(args.input_jsonl)
    episodes = sensitivity_episodes_from_replay_records(records)
    if not episodes:
        parser.error(
            "No replay episodes found. Generate records with replay payloads "
            "(capture-replay enabled) and rerun.",
        )
    report: dict[str, Any] = analyze_threshold_sensitivity(
        episodes,
        collision_grid=_float_grid(args.collision_grid),
        near_miss_grid=_float_grid(args.near_grid),
        comfort_grid=_float_grid(args.comfort_grid),
        ttc_horizons_sec=_float_grid(args.ttc_grid),
        relative_speed_reference=float(args.relative_speed_reference),
    )
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
