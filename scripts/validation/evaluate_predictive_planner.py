#!/usr/bin/env python3
"""Evaluate predictive planner using map-runner scenarios and benchmark metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from loguru import logger

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.predictive_planner_config import build_predictive_planner_algo_config
from scripts.validation.predictive_eval_common import load_seed_manifest, make_subset_scenarios


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--scenario-matrix",
        type=Path,
        default=Path("configs/scenarios/sets/classic_crossing_subset.yaml"),
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
    )
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-success-rate", type=float, default=0.3)
    parser.add_argument("--min-distance", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tmp/predictive_planner/eval"),
    )
    parser.add_argument("--tag", type=str, default="predictive_eval")
    parser.add_argument(
        "--algo-config",
        type=Path,
        default=None,
        help="Optional YAML file to override predictive planner algorithm config.",
    )
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=None,
        help="Optional YAML map {scenario_id: [seed,...]} to evaluate hard-case subsets.",
    )
    parser.add_argument(
        "--comparison-jsonl",
        type=Path,
        default=None,
        help="Optional baseline JSONL to compute per-scenario deltas vs a reference run.",
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    """Return arithmetic mean with zero fallback."""
    return float(sum(values) / max(len(values), 1))


def _episode_success(row: dict) -> bool:
    """Resolve episode success with collision-aware fallback semantics."""
    metrics = row.get("metrics", {})
    if "success_rate" in metrics:
        return float(metrics.get("success_rate", 0.0)) >= 0.5
    success_val = metrics.get("success", False)
    if isinstance(success_val, bool):
        return success_val
    return float(success_val) >= 0.5


def _failure_taxonomy(rows: list[dict]) -> dict:
    """Build compact failure taxonomy from episode records."""
    failed = [row for row in rows if not _episode_success(row)]
    by_status = Counter(str(row.get("status", "unknown")) for row in failed)
    by_reason = Counter(str(row.get("termination_reason", "unknown")) for row in failed)
    by_scenario = Counter(str(row.get("scenario_id", "unknown")) for row in failed)
    return {
        "failed_episodes": len(failed),
        "status_counts": dict(by_status),
        "termination_reason_counts": dict(by_reason),
        "scenario_counts": dict(by_scenario),
    }


def _per_scenario_index(per_scenario_summary: list[dict]) -> dict[str, dict]:
    """Index per-scenario rows by scenario_id."""
    return {str(row.get("scenario_id", "unknown")): row for row in per_scenario_summary}


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """Run map-runner benchmark for predictive planner and enforce quality gates."""
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
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

    config_path = args.algo_config
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"Algorithm config not found: {config_path}")
    algo_cfg = build_predictive_planner_algo_config(
        checkpoint_path=args.checkpoint,
        device="cpu",
        config_path=config_path,
    )
    algo_cfg_path.write_text(yaml.safe_dump(algo_cfg, sort_keys=False), encoding="utf-8")

    scenarios_or_path: Path | list[dict]
    if args.seed_manifest is not None:
        if not args.seed_manifest.exists():
            raise FileNotFoundError(f"Seed manifest not found: {args.seed_manifest}")
        seed_manifest = load_seed_manifest(args.seed_manifest)
        subset = make_subset_scenarios(args.scenario_matrix, seed_manifest)
        if not subset:
            raise RuntimeError(
                f"No scenarios from {args.scenario_matrix} matched seed manifest {args.seed_manifest}"
            )
        scenarios_or_path = subset
    else:
        scenarios_or_path = args.scenario_matrix

    summary = run_map_batch(
        scenarios_or_path,
        jsonl_path,
        schema_path=args.schema_path,
        algo="prediction_planner",
        algo_config_path=str(algo_cfg_path),
        horizon=int(args.horizon),
        dt=float(args.dt),
        workers=int(args.workers),
        resume=False,
        benchmark_profile="experimental",
    )

    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("No episodes were written by map-runner evaluation.")

    success_vals = [1.0 if _episode_success(row) else 0.0 for row in rows]
    min_dist_vals = [
        float(row.get("metrics", {}).get("min_distance"))
        for row in rows
        if "min_distance" in row.get("metrics", {})
    ]
    avg_speed_vals = [float(row.get("metrics", {}).get("avg_speed", 0.0)) for row in rows]

    success_rate = _mean(success_vals)
    mean_min_distance = _mean(min_dist_vals) if min_dist_vals else float("nan")
    mean_speed = _mean(avg_speed_vals)
    per_scenario: dict[str, dict] = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id", "unknown"))
        scenario_entry = per_scenario.setdefault(
            scenario_id,
            {
                "episodes": 0,
                "successes": 0,
                "min_distance_values": [],
                "avg_speed_values": [],
                "failed_seeds": [],
            },
        )
        scenario_entry["episodes"] += 1
        row_success = _episode_success(row)
        if row_success:
            scenario_entry["successes"] += 1
        else:
            scenario_entry["failed_seeds"].append(int(row.get("seed", -1)))
        if "min_distance" in row.get("metrics", {}):
            scenario_entry["min_distance_values"].append(
                float(row.get("metrics", {}).get("min_distance"))
            )
        scenario_entry["avg_speed_values"].append(
            float(row.get("metrics", {}).get("avg_speed", 0.0))
        )

    per_scenario_summary = []
    for scenario_id, entry in sorted(per_scenario.items()):
        scenario_success = float(entry["successes"]) / max(int(entry["episodes"]), 1)
        scenario_min_distance = (
            _mean(entry["min_distance_values"]) if entry["min_distance_values"] else float("nan")
        )
        scenario_avg_speed = _mean(entry["avg_speed_values"])
        per_scenario_summary.append(
            {
                "scenario_id": scenario_id,
                "episodes": int(entry["episodes"]),
                "success_rate": scenario_success,
                "success_rate_delta_to_global": float(scenario_success - success_rate),
                "mean_min_distance": scenario_min_distance,
                "mean_min_distance_delta_to_global": float(
                    scenario_min_distance - mean_min_distance
                ),
                "mean_avg_speed": scenario_avg_speed,
                "failed_seeds": sorted(set(entry["failed_seeds"])),
            }
        )
    comparison_delta = None
    if args.comparison_jsonl is not None:
        if not args.comparison_jsonl.exists():
            raise FileNotFoundError(f"Comparison JSONL not found: {args.comparison_jsonl}")
        comparison_rows = [
            json.loads(line)
            for line in args.comparison_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cmp_per_scenario: dict[str, dict] = {}
        for row in comparison_rows:
            sid = str(row.get("scenario_id", "unknown"))
            bucket = cmp_per_scenario.setdefault(
                sid,
                {"episodes": 0, "successes": 0, "min_distance_values": []},
            )
            bucket["episodes"] += 1
            bucket["successes"] += 1 if _episode_success(row) else 0
            if "min_distance" in row.get("metrics", {}):
                bucket["min_distance_values"].append(
                    float(row.get("metrics", {}).get("min_distance"))
                )
        cmp_index = {
            sid: {
                "success_rate": float(v["successes"]) / max(int(v["episodes"]), 1),
                "mean_min_distance": (
                    _mean(v["min_distance_values"]) if v["min_distance_values"] else float("nan")
                ),
            }
            for sid, v in cmp_per_scenario.items()
        }
        cur_index = _per_scenario_index(per_scenario_summary)
        merged = []
        for sid in sorted(set(cur_index) | set(cmp_index)):
            cur = cur_index.get(sid)
            cmp = cmp_index.get(sid)
            if cur is None or cmp is None:
                continue
            merged.append(
                {
                    "scenario_id": sid,
                    "success_rate_delta_vs_comparison": float(
                        cur["success_rate"] - cmp["success_rate"]
                    ),
                    "mean_min_distance_delta_vs_comparison": float(
                        cur["mean_min_distance"] - cmp["mean_min_distance"]
                    ),
                }
            )
        comparison_delta = merged

    gates = {
        "min_success_rate": float(args.min_success_rate),
        "min_distance": float(args.min_distance),
        "pass_success_rate": bool(success_rate >= float(args.min_success_rate)),
        "pass_min_distance": bool(mean_min_distance >= float(args.min_distance)),
    }
    gates["pass_all"] = bool(gates["pass_success_rate"] and gates["pass_min_distance"])

    result = {
        "checkpoint": str(args.checkpoint),
        "scenario_matrix": str(args.scenario_matrix),
        "jsonl_path": str(jsonl_path),
        "algo_config_path": str(algo_cfg_path),
        "episodes": len(rows),
        "run_summary": summary,
        "metrics": {
            "success_rate": success_rate,
            "mean_min_distance": mean_min_distance,
            "mean_avg_speed": mean_speed,
        },
        "per_scenario_delta": per_scenario_summary,
        "failure_taxonomy": _failure_taxonomy(rows),
        "comparison_delta": comparison_delta,
        "quality_gates": gates,
    }

    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Saved evaluation summary to {}", summary_path)

    if not gates["pass_all"]:
        logger.error("Evaluation quality gates failed: {}", gates)
        return 2

    logger.success("Evaluation quality gates passed: {}", gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
