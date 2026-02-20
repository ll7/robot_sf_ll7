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
from robot_sf.training.scenario_loader import load_scenarios


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


def _load_seed_manifest(path: Path) -> dict[str, list[int]]:
    """Load scenario->seed manifest from YAML."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Seed manifest must be a mapping: {path}")
    out: dict[str, list[int]] = {}
    for key, value in data.items():
        if not isinstance(value, list):
            continue
        out[str(key)] = [int(v) for v in value]
    return out


def _make_subset_scenarios(
    scenario_matrix: Path, seed_manifest: dict[str, list[int]]
) -> list[dict]:
    """Load scenario matrix and inject explicit seed lists for selected scenarios."""
    scenarios = load_scenarios(scenario_matrix)
    selected: list[dict] = []
    base_dir = scenario_matrix.parent.resolve()
    for scenario in scenarios:
        scenario_id = str(
            scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
        )
        if scenario_id not in seed_manifest:
            continue
        scenario_copy = dict(scenario)
        map_file = scenario_copy.get("map_file")
        if isinstance(map_file, str):
            map_path = Path(map_file)
            if not map_path.is_absolute():
                scenario_copy["map_file"] = str((base_dir / map_path).resolve())
        scenario_copy["seeds"] = list(seed_manifest[scenario_id])
        selected.append(scenario_copy)
    return selected


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

    if args.algo_config is not None:
        if not args.algo_config.exists():
            raise FileNotFoundError(f"Algorithm config not found: {args.algo_config}")
        algo_cfg = yaml.safe_load(args.algo_config.read_text(encoding="utf-8")) or {}
        if not isinstance(algo_cfg, dict):
            raise TypeError(f"Algorithm config must be a mapping: {args.algo_config}")
    else:
        algo_cfg = {
            "predictive_checkpoint_path": str(args.checkpoint),
            "predictive_device": "cpu",
            "predictive_max_agents": 16,
            "predictive_horizon_steps": 8,
            "predictive_rollout_dt": 0.2,
            "max_linear_speed": 1.6,
            "max_angular_speed": 1.2,
            "goal_tolerance": 0.25,
            "predictive_goal_weight": 5.0,
            "predictive_collision_weight": 0.4,
            "predictive_near_miss_weight": 0.05,
            "predictive_velocity_weight": 0.01,
            "predictive_turn_weight": 0.01,
            "predictive_ttc_weight": 0.15,
            "predictive_ttc_distance": 0.8,
            "predictive_safe_distance": 0.35,
            "predictive_near_distance": 0.7,
            "predictive_progress_risk_weight": 1.2,
            "predictive_progress_risk_distance": 1.2,
            "predictive_hard_clearance_distance": 0.75,
            "predictive_hard_clearance_weight": 2.5,
            "predictive_adaptive_horizon_enabled": True,
            "predictive_horizon_boost_steps": 4,
            "predictive_near_field_distance": 2.4,
            "predictive_near_field_speed_cap": 0.75,
            "predictive_near_field_speed_samples": [0.1, 0.2, 0.35, 0.5],
            "predictive_near_field_heading_deltas": [
                -1.570796,
                -1.047198,
                -0.785398,
                -0.523599,
                0.0,
                0.523599,
                0.785398,
                1.047198,
                1.570796,
            ],
            "predictive_candidate_speeds": [0.0, 0.4, 0.7, 1.0],
            "predictive_candidate_heading_deltas": [-0.785398, -0.392699, 0.0, 0.392699, 0.785398],
            "occupancy_weight": 0.3,
        }
    # Always force requested checkpoint unless the caller explicitly asks otherwise in YAML.
    algo_cfg.setdefault("predictive_checkpoint_path", str(args.checkpoint))
    algo_cfg_path.write_text(yaml.safe_dump(algo_cfg, sort_keys=False), encoding="utf-8")

    scenarios_or_path: Path | list[dict]
    if args.seed_manifest is not None:
        if not args.seed_manifest.exists():
            raise FileNotFoundError(f"Seed manifest not found: {args.seed_manifest}")
        seed_manifest = _load_seed_manifest(args.seed_manifest)
        subset = _make_subset_scenarios(args.scenario_matrix, seed_manifest)
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
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not rows:
        raise RuntimeError("No episodes were written by map-runner evaluation.")

    success_vals = [1.0 if _episode_success(row) else 0.0 for row in rows]
    min_dist_vals = [float(row.get("metrics", {}).get("min_distance", 0.0)) for row in rows]
    avg_speed_vals = [float(row.get("metrics", {}).get("avg_speed", 0.0)) for row in rows]

    success_rate = _mean(success_vals)
    mean_min_distance = _mean(min_dist_vals)
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
        scenario_entry["min_distance_values"].append(
            float(row.get("metrics", {}).get("min_distance", 0.0))
        )
        scenario_entry["avg_speed_values"].append(
            float(row.get("metrics", {}).get("avg_speed", 0.0))
        )

    per_scenario_summary = []
    for scenario_id, entry in sorted(per_scenario.items()):
        scenario_success = float(entry["successes"]) / max(int(entry["episodes"]), 1)
        scenario_min_distance = _mean(entry["min_distance_values"])
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
            if line
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
            bucket["min_distance_values"].append(
                float(row.get("metrics", {}).get("min_distance", 0.0))
            )
        cmp_index = {
            sid: {
                "success_rate": float(v["successes"]) / max(int(v["episodes"]), 1),
                "mean_min_distance": _mean(v["min_distance_values"]),
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
