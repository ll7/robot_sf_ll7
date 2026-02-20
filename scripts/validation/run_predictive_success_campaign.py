#!/usr/bin/env python3
"""Run predictive planner success-improvement campaign across checkpoints and planner configs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.training.scenario_loader import load_scenarios


@dataclass
class EvalResult:
    """Aggregate result for one checkpoint/config pair."""

    checkpoint: str
    variant: str
    suite: str
    episodes: int
    success_rate: float
    success_ci_low: float
    success_ci_high: float
    mean_min_distance: float
    mean_avg_speed: float
    jsonl_path: str


def parse_args() -> argparse.Namespace:
    """Parse campaign CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--scenario-matrix",
        type=Path,
        default=Path("configs/scenarios/classic_interactions.yaml"),
    )
    parser.add_argument(
        "--hard-seed-manifest",
        type=Path,
        default=Path("configs/benchmarks/predictive_hard_seeds_v1.yaml"),
    )
    parser.add_argument(
        "--planner-grid",
        type=Path,
        default=Path("configs/benchmarks/predictive_sweep_planner_grid_v1.yaml"),
    )
    parser.add_argument("--horizon", type=int, default=140)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tmp/predictive_planner/campaigns/latest_success_campaign"),
    )
    return parser.parse_args()


def _load_seed_manifest(path: Path) -> dict[str, list[int]]:
    """Load scenario->seed map from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Seed manifest must be a mapping: {path}")
    out: dict[str, list[int]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            out[str(key)] = [int(v) for v in value]
    return out


def _make_subset_scenarios(
    scenario_matrix: Path, seed_manifest: dict[str, list[int]]
) -> list[dict]:
    """Load scenarios and apply explicit seed sets for selected scenarios."""
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


def _load_planner_variants(path: Path) -> list[dict]:
    """Load planner sweep variants from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    variants = payload.get("variants", [])
    if not isinstance(variants, list):
        raise TypeError(f"planner grid variants must be a list: {path}")
    out = []
    for item in variants:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "unnamed"))
        params = item.get("params", {})
        if not isinstance(params, dict):
            params = {}
        out.append({"name": name, "params": params})
    if not out:
        raise RuntimeError(f"No planner variants found in {path}")
    return out


def _episode_success(row: dict) -> bool:
    """Resolve episode success with collision-aware fallback semantics."""
    metrics = row.get("metrics", {})
    if "success_rate" in metrics:
        return float(metrics.get("success_rate", 0.0)) >= 0.5
    value = metrics.get("success", False)
    if isinstance(value, bool):
        return value
    return float(value) >= 0.5


def _bootstrap_ci(values: np.ndarray, n_samples: int, seed: int) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean."""
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty((n_samples,), dtype=float)
    n = values.size
    for i in range(n_samples):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _base_algo_cfg(checkpoint: str) -> dict:
    """Base predictive planner algorithm config."""
    return {
        "predictive_checkpoint_path": checkpoint,
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


def _run_eval(
    *,
    scenarios_or_path: Path | list[dict],
    suite_name: str,
    checkpoint: str,
    variant_name: str,
    algo_cfg: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> EvalResult:
    """Run one evaluation and return aggregated metrics."""
    safe_ckpt = Path(checkpoint).stem.replace(".", "_")
    tag = f"{suite_name}__{variant_name}__{safe_ckpt}"
    algo_cfg_path = output_dir / f"{tag}_algo.yaml"
    jsonl_path = output_dir / f"{tag}.jsonl"
    algo_cfg_path.write_text(yaml.safe_dump(algo_cfg, sort_keys=False), encoding="utf-8")
    if jsonl_path.exists():
        jsonl_path.unlink()

    run_map_batch(
        scenarios_or_path,
        jsonl_path,
        schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
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
    success_vals = np.asarray([1.0 if _episode_success(row) else 0.0 for row in rows], dtype=float)
    min_dist_vals = np.asarray(
        [float(row.get("metrics", {}).get("min_distance", 0.0)) for row in rows],
        dtype=float,
    )
    speed_vals = np.asarray(
        [float(row.get("metrics", {}).get("avg_speed", 0.0)) for row in rows],
        dtype=float,
    )
    ci_low, ci_high = _bootstrap_ci(
        success_vals,
        n_samples=int(args.bootstrap_samples),
        seed=int(args.bootstrap_seed),
    )
    return EvalResult(
        checkpoint=checkpoint,
        variant=variant_name,
        suite=suite_name,
        episodes=len(rows),
        success_rate=float(np.mean(success_vals)) if success_vals.size > 0 else 0.0,
        success_ci_low=ci_low,
        success_ci_high=ci_high,
        mean_min_distance=float(np.mean(min_dist_vals)) if min_dist_vals.size > 0 else 0.0,
        mean_avg_speed=float(np.mean(speed_vals)) if speed_vals.size > 0 else 0.0,
        jsonl_path=str(jsonl_path),
    )


def _rank_key(hard: EvalResult, global_res: EvalResult) -> tuple[float, float, float]:
    """Ranking key: hard success first, then hard clearance, then global success."""
    return (hard.success_rate, hard.mean_min_distance, global_res.success_rate)


def _checkpoint_label(path_str: str) -> str:
    """Return compact checkpoint label for reports."""
    p = Path(path_str)
    if len(p.parts) >= 2:
        return "/".join(p.parts[-2:])
    return p.name


def main() -> int:
    """Execute full campaign and write machine + human-readable reports."""
    args = parse_args()
    variants = _load_planner_variants(args.planner_grid)
    hard_manifest = _load_seed_manifest(args.hard_seed_manifest)
    hard_scenarios = _make_subset_scenarios(args.scenario_matrix, hard_manifest)
    if not hard_scenarios:
        raise RuntimeError("Hard-case manifest did not match any scenarios.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for checkpoint in args.checkpoints:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        for variant in variants:
            name = str(variant["name"])
            cfg = _base_algo_cfg(checkpoint)
            cfg.update(dict(variant.get("params", {})))
            hard_res = _run_eval(
                scenarios_or_path=hard_scenarios,
                suite_name="hard",
                checkpoint=checkpoint,
                variant_name=name,
                algo_cfg=cfg,
                args=args,
                output_dir=args.output_dir,
            )
            global_res = _run_eval(
                scenarios_or_path=args.scenario_matrix,
                suite_name="global",
                checkpoint=checkpoint,
                variant_name=name,
                algo_cfg=cfg,
                args=args,
                output_dir=args.output_dir,
            )
            results.append(
                {
                    "checkpoint": checkpoint,
                    "variant": name,
                    "config": cfg,
                    "hard": hard_res.__dict__,
                    "global": global_res.__dict__,
                    "ranking_key": list(_rank_key(hard_res, global_res)),
                }
            )

    ranked = sorted(
        results,
        key=lambda r: tuple(r["ranking_key"]),
        reverse=True,
    )

    top = ranked[0]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario_matrix": str(args.scenario_matrix),
        "hard_seed_manifest": str(args.hard_seed_manifest),
        "planner_grid": str(args.planner_grid),
        "horizon": int(args.horizon),
        "dt": float(args.dt),
        "workers": int(args.workers),
        "bootstrap_samples": int(args.bootstrap_samples),
        "num_candidates": len(ranked),
        "best": top,
        "ranked": ranked,
    }

    json_path = args.output_dir / "campaign_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Predictive Success Campaign",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Scenario matrix: `{summary['scenario_matrix']}`",
        f"- Hard manifest: `{summary['hard_seed_manifest']}`",
        f"- Planner grid: `{summary['planner_grid']}`",
        f"- Candidates: `{summary['num_candidates']}`",
        "",
        "## Best Candidate",
        "",
        f"- Checkpoint: `{top['checkpoint']}`",
        f"- Variant: `{top['variant']}`",
        f"- Hard success: `{top['hard']['success_rate']:.4f}` "
        f"(95% CI `{top['hard']['success_ci_low']:.4f}`..`{top['hard']['success_ci_high']:.4f}`)",
        f"- Hard mean min-distance: `{top['hard']['mean_min_distance']:.4f}`",
        f"- Global success: `{top['global']['success_rate']:.4f}` "
        f"(95% CI `{top['global']['success_ci_low']:.4f}`..`{top['global']['success_ci_high']:.4f}`)",
        f"- Global mean min-distance: `{top['global']['mean_min_distance']:.4f}`",
        "",
        "## Ranking (top 10)",
        "",
        "| Rank | Variant | Checkpoint | Hard SR | Hard MinDist | Global SR | Global MinDist |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(ranked[:10], start=1):
        md_lines.append(
            "| "
            f"{i} | {row['variant']} | {_checkpoint_label(row['checkpoint'])} | "
            f"{row['hard']['success_rate']:.4f} | {row['hard']['mean_min_distance']:.4f} | "
            f"{row['global']['success_rate']:.4f} | {row['global']['mean_min_distance']:.4f} |"
        )

    md_path = args.output_dir / "campaign_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"summary": str(json_path), "report": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
