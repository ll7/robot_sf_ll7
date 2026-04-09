#!/usr/bin/env python3
"""Evaluate a SAC checkpoint using map-runner scenarios and benchmark metrics.

Mirrors the structure of ``evaluate_predictive_planner.py``.  Runs
``run_map_batch`` with ``algo=sac`` and an algo config derived from
``configs/baselines/sac_gate_socnav_struct.yaml`` (or a user override),
reports per-scenario success rates, and enforces an optional quality gate.

Usage::

    uv run python scripts/validation/evaluate_sac.py \\
        --checkpoint output/models/sac/sac_gate_socnav_struct_v1.zip

Custom scenario matrix::

    uv run python scripts/validation/evaluate_sac.py \\
        --checkpoint output/models/sac/sac_gate_socnav_struct_v1.zip \\
        --scenario-matrix configs/scenarios/sets/classic_crossing_subset.yaml

Increase workers for parallel episode evaluation (read-only envs are safe)::

    uv run python scripts/validation/evaluate_sac.py \\
        --checkpoint output/models/sac/sac_gate_socnav_struct_v1.zip \\
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import yaml
from loguru import logger

from robot_sf.benchmark.map_runner import run_map_batch

_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
_DEFAULT_ALGO_CONFIG = Path("configs/baselines/sac_gate_socnav_struct.yaml")
_DEFAULT_SCENARIO_MATRIX = Path("configs/scenarios/classic_interactions.yaml")


def parse_args() -> argparse.Namespace:
    """Build CLI argument parser for SAC evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained SAC checkpoint (.zip).",
    )
    parser.add_argument(
        "--scenario-matrix",
        type=Path,
        default=_DEFAULT_SCENARIO_MATRIX,
        help=(
            "YAML scenario matrix (default: configs/scenarios/classic_interactions.yaml)."
        ),
    )
    parser.add_argument(
        "--algo-config",
        type=Path,
        default=None,
        help=(
            "Optional YAML algo config override "
            "(default: configs/baselines/sac_gate_socnav_struct.yaml)."
        ),
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=_SCHEMA_PATH,
        help="Path to episode JSON schema for validation.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=120,
        help="Maximum steps per episode (default: 120).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Simulation time-step in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel episode workers (default: 1).",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.3,
        help="Minimum success rate for quality gate (default: 0.3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tmp/sac_eval/latest"),
        help="Directory to write JSONL episodes and summary (default: output/tmp/sac_eval/latest).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="sac_eval",
        help="Run tag used for output file names (default: sac_eval).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device for the SAC model: auto, cpu, cuda (default: auto).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Episode metric helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    """Return arithmetic mean with zero fallback for empty lists."""
    return float(sum(values) / max(len(values), 1))


def _episode_success(row: dict) -> bool:
    """Resolve episode success from a map-runner episode record."""
    metrics = row.get("metrics", {})
    if "success_rate" in metrics:
        v = metrics.get("success_rate")
        if v is None or v == "":
            return False
        return float(v) >= 0.5
    success_val = metrics.get("success", False)
    if isinstance(success_val, bool):
        return success_val
    if success_val is None or success_val == "":
        return False
    return float(success_val) >= 0.5


def _nan_to_none(value: object) -> object:
    """Recursively replace NaN floats with None for JSON compatibility."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run SAC benchmark evaluation and enforce the quality gate.

    Returns:
        int: 0 on success, 1 if the quality gate fails.
    """
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.scenario_matrix.exists():
        raise FileNotFoundError(f"Scenario matrix not found: {args.scenario_matrix}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    algo_cfg_path = args.output_dir / f"{args.tag}_algo_config.yaml"
    jsonl_path = args.output_dir / f"{args.tag}.jsonl"
    summary_path = args.output_dir / f"{args.tag}_summary.json"

    if jsonl_path.exists():
        jsonl_path.unlink()

    # Build algo config from the baseline template or user override.
    base_cfg_path = args.algo_config or _DEFAULT_ALGO_CONFIG
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Algo config not found: {base_cfg_path}")
    base_cfg: dict = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}
    base_cfg["model_path"] = str(args.checkpoint.resolve())
    base_cfg["device"] = args.device
    algo_cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    logger.info("SAC checkpoint: {}", args.checkpoint)
    logger.info("Scenario matrix: {}", args.scenario_matrix)
    logger.info("Output dir: {}", args.output_dir)

    summary = run_map_batch(
        args.scenario_matrix,
        jsonl_path,
        schema_path=args.schema_path,
        algo="sac",
        algo_config_path=str(algo_cfg_path),
        horizon=int(args.horizon),
        dt=float(args.dt),
        workers=int(args.workers),
        resume=False,
        benchmark_profile="experimental",
    )

    lines = [
        line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise RuntimeError("No episodes written by map-runner evaluation.")
    rows = [json.loads(line) for line in lines]

    success_vals = [1.0 if _episode_success(row) else 0.0 for row in rows]
    min_dist_vals = [
        float(row["metrics"]["min_distance"])
        for row in rows
        if isinstance(row.get("metrics"), dict)
        and row["metrics"].get("min_distance") is not None
    ]
    avg_speed_vals = [
        float(row.get("metrics", {}).get("avg_speed", 0.0)) for row in rows
    ]

    success_rate = _mean(success_vals)
    mean_min_distance = _mean(min_dist_vals) if min_dist_vals else float("nan")
    mean_speed = _mean(avg_speed_vals)

    per_scenario: dict[str, dict] = {}
    for row in rows:
        sid = str(row.get("scenario_id", "unknown"))
        entry = per_scenario.setdefault(
            sid, {"episodes": 0, "successes": 0, "failed_seeds": []}
        )
        entry["episodes"] += 1
        if _episode_success(row):
            entry["successes"] += 1
        else:
            entry["failed_seeds"].append(int(row.get("seed", -1)))

    per_scenario_summary = [
        {
            "scenario_id": sid,
            "episodes": entry["episodes"],
            "successes": entry["successes"],
            "success_rate": entry["successes"] / max(entry["episodes"], 1),
            "failed_seeds": entry["failed_seeds"],
        }
        for sid, entry in sorted(per_scenario.items())
    ]

    failure_taxonomy = {
        "failed_episodes": sum(1 for v in success_vals if v < 0.5),
        "termination_reason_counts": dict(
            Counter(
                str(row.get("termination_reason", "unknown"))
                for row in rows
                if not _episode_success(row)
            )
        ),
    }

    report: dict = {
        "tag": args.tag,
        "checkpoint": str(args.checkpoint),
        "scenario_matrix": str(args.scenario_matrix),
        "device": args.device,
        "total_episodes": len(rows),
        "success_rate": success_rate,
        "mean_min_distance": mean_min_distance,
        "mean_avg_speed": mean_speed,
        "per_scenario": per_scenario_summary,
        "failure_taxonomy": failure_taxonomy,
        "run_summary": summary,
    }
    report = _nan_to_none(report)  # type: ignore[assignment]
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    safe_dist = 0.0 if math.isnan(mean_min_distance) else mean_min_distance
    logger.info("=== SAC Evaluation Results ===")
    logger.info("Episodes:         {}", len(rows))
    logger.info("Success rate:     {:.1%}", success_rate)
    logger.info("Mean min-dist:    {:.3f} m", safe_dist)
    logger.info("Mean avg-speed:   {:.3f} m/s", mean_speed)
    logger.info("Summary written:  {}", summary_path)

    gate_pass = success_rate >= args.min_success_rate
    if not gate_pass:
        logger.error(
            "Quality gate FAILED: success_rate={:.1%} < min={:.1%}",
            success_rate,
            args.min_success_rate,
        )
        return 1
    logger.success("Quality gate PASSED: success_rate={:.1%}", success_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
