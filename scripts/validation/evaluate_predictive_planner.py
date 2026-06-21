#!/usr/bin/env python3
"""Evaluate predictive planner using map-runner scenarios and benchmark metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.observation_noise import load_observation_noise_spec
from robot_sf.benchmark.predictive_planner_config import build_predictive_planner_algo_config
from scripts.validation.predictive_eval_common import load_seed_manifest, make_subset_scenarios

_CONTRACT_VERSION = "benchmark-reset-v2"
_TRAINING_FAMILY = "prediction_planner"


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--scenario-matrix",
        type=Path,
        default=Path("configs/scenarios/sets/classic_cross_trap_subset.yaml"),
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
    parser.add_argument(
        "--observation-noise",
        type=Path,
        default=None,
        help="Optional observation-noise YAML profile passed through to map-runner evaluation.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    """Return arithmetic mean with zero fallback."""
    return float(sum(values) / max(len(values), 1))


def _episode_success(row: dict) -> bool:
    """Resolve episode success with collision-aware fallback semantics."""
    metrics = row.get("metrics", {})
    if "success_rate" in metrics:
        success_rate_value = metrics.get("success_rate")
        if success_rate_value is None or success_rate_value == "":
            return False
        return float(success_rate_value) >= 0.5
    success_val = metrics.get("success", False)
    if isinstance(success_val, bool):
        return success_val
    if success_val is None or success_val == "":
        return False
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


def _normal_z(confidence: float) -> float:
    """Return a normal critical value for common confidence levels."""
    if math.isclose(confidence, 0.90):
        return 1.6448536269514722
    if math.isclose(confidence, 0.99):
        return 2.5758293035489004
    return 1.959963984540054


def _wilson_interval(successes: int, total: int, confidence: float) -> list[float | None]:
    """Return Wilson score interval for a Bernoulli rate."""
    if total <= 0:
        return [None, None]
    z = _normal_z(confidence)
    p = float(successes) / float(total)
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total))) / denom
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def _bootstrap_mean_interval(
    values: list[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> list[float | None]:
    """Return deterministic bootstrap CI for a finite-value mean."""
    if not values or samples <= 0:
        return [None, None]
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(int(samples), len(arr)))
    means = np.mean(arr[indices], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return [
        float(np.quantile(means, max(0.0, alpha))),
        float(np.quantile(means, min(1.0, 1.0 - alpha))),
    ]


def _collision_metric_value(row: dict) -> float | None:
    """Return the explicit episode collision metric, or None when unavailable."""
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    for key in ("total_collision_count", "collisions"):
        value = metrics.get(key)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _collision_metric_status(rows: list[dict]) -> dict[str, object]:
    """Summarize whether every episode has an explicit collision metric."""
    available = sum(1 for row in rows if _collision_metric_value(row) is not None)
    total = len(rows)
    if total == 0:
        status = "not_available"
        reason = "no episode rows"
    elif available == total:
        status = "available"
        reason = None
    elif available == 0:
        status = "not_available"
        reason = "no episode rows had metrics.total_collision_count or metrics.collisions"
    else:
        status = "partial"
        reason = "some episode rows lacked metrics.total_collision_count or metrics.collisions"
    return {
        "status": status,
        "reason": reason,
        "denominator": total,
        "available": available,
        "required_fields": ["metrics.total_collision_count", "metrics.collisions"],
    }


def _integrity_summary(rows: list[dict]) -> dict[str, object]:
    """Detect contradiction patterns in evaluation episode records."""
    contradictions_by_episode: dict[str, list[str]] = {}
    for row in rows:
        metrics = row.get("metrics", {})
        success = _episode_success(row)
        total_collisions = float(
            metrics.get("total_collision_count", metrics.get("collisions", 0.0)) or 0.0
        )
        termination_reason = str(row.get("termination_reason", "unknown"))
        episode_id = str(row.get("episode_id", "unknown"))
        episode_reasons: list[str] = []
        if termination_reason == "collision" and success:
            episode_reasons.append("collision_with_success")
        if success and total_collisions > 0.0:
            episode_reasons.append("success_with_collision_metric")
        if episode_reasons:
            contradictions_by_episode.setdefault(episode_id, [])
            for reason in episode_reasons:
                if reason not in contradictions_by_episode[episode_id]:
                    contradictions_by_episode[episode_id].append(reason)
    contradictions: list[dict[str, object]] = []
    for episode_id, reasons in contradictions_by_episode.items():
        entry: dict[str, object] = {"episode_id": episode_id, "reasons": sorted(reasons)}
        contradictions.append(entry)
    contradictions.sort(key=lambda entry: str(entry.get("episode_id", "")))
    return {
        "pass": not contradictions,
        "contradiction_count": len(contradictions),
        "contradictions": contradictions,
    }


def _per_scenario_index(per_scenario_summary: list[dict]) -> dict[str, dict]:
    """Index per-scenario rows by scenario_id."""
    return {str(row.get("scenario_id", "unknown")): row for row in per_scenario_summary}


def _nan_to_none(value: object) -> object:
    """Recursively convert NaN floats to ``None`` for JSON compatibility."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """Run map-runner benchmark for predictive planner and enforce quality gates."""
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
        observation_noise=(
            load_observation_noise_spec(args.observation_noise)
            if args.observation_noise is not None
            else None
        ),
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
        if row.get("metrics", {}).get("min_distance") is not None
    ]
    collision_metric_values = [_collision_metric_value(row) for row in rows]
    collision_vals = [
        1.0 if float(value) > 0.0 else 0.0 for value in collision_metric_values if value is not None
    ]
    avg_speed_vals = [float(row.get("metrics", {}).get("avg_speed", 0.0)) for row in rows]

    success_rate = _mean(success_vals)
    collision_rate = _mean(collision_vals) if collision_vals else float("nan")
    mean_min_distance = _mean(min_dist_vals) if min_dist_vals else float("nan")
    mean_speed = _mean(avg_speed_vals)
    collision_status = _collision_metric_status(rows)
    success_count = int(sum(success_vals))
    collision_count = int(sum(collision_vals))
    uncertainty = {
        "confidence": float(args.confidence),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "success_rate_ci": _wilson_interval(
            success_count, len(success_vals), float(args.confidence)
        ),
        "collision_rate_ci": _wilson_interval(
            collision_count,
            len(collision_vals),
            float(args.confidence),
        ),
        "mean_min_distance_ci": _bootstrap_mean_interval(
            min_dist_vals,
            samples=int(args.bootstrap_samples),
            confidence=float(args.confidence),
            seed=int(args.bootstrap_seed),
        ),
        "methods": {
            "success_rate": "wilson_score",
            "collision_rate": "wilson_score",
            "mean_min_distance": "bootstrap_episode_mean",
        },
    }
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
        if row.get("metrics", {}).get("min_distance") is not None:
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
            if row.get("metrics", {}).get("min_distance") is not None:
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

    min_distance_available = bool(np.isfinite(mean_min_distance))
    gates = {
        "min_success_rate": float(args.min_success_rate),
        "min_distance": float(args.min_distance),
        "min_distance_available": min_distance_available,
        "collision_metric_status": collision_status,
        "collision_metric_available": collision_status["status"] == "available",
        "pass_success_rate": bool(success_rate >= float(args.min_success_rate)),
        "pass_min_distance": bool(
            mean_min_distance >= float(args.min_distance) if min_distance_available else True
        ),
    }
    gates["pass_all"] = bool(
        gates["pass_success_rate"]
        and gates["pass_min_distance"]
        and gates["collision_metric_available"]
    )
    integrity = _integrity_summary(rows)

    result = {
        "contract_version": _CONTRACT_VERSION,
        "training_family": _TRAINING_FAMILY,
        "artifact_role": "predictive_final_evaluation",
        "checkpoint": str(args.checkpoint),
        "scenario_matrix": str(args.scenario_matrix),
        "jsonl_path": str(jsonl_path),
        "algo_config_path": str(algo_cfg_path),
        "episodes": len(rows),
        "run_summary": summary,
        "metrics": {
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "mean_min_distance": mean_min_distance,
            "mean_avg_speed": mean_speed,
        },
        "uncertainty": uncertainty,
        "collision_metric_status": collision_status,
        "per_scenario_delta": per_scenario_summary,
        "failure_taxonomy": _failure_taxonomy(rows),
        "integrity": integrity,
        "comparison_delta": comparison_delta,
        "quality_gates": gates,
    }

    summary_path.write_text(json.dumps(_nan_to_none(result), indent=2), encoding="utf-8")
    logger.info("Saved evaluation summary to {}", summary_path)

    if not integrity["pass"]:
        logger.error("Evaluation integrity checks failed: {}", integrity)
        return 2
    if not gates["pass_all"]:
        logger.error("Evaluation quality gates failed: {}", gates)
        return 2

    logger.success("Evaluation quality gates passed: {}", gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
