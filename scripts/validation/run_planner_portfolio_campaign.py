#!/usr/bin/env python3
"""Run multi-planner portfolio campaign across hard and global suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from scripts.validation.predictive_eval_common import load_seed_manifest, make_subset_scenarios


@dataclass
class EvalResult:
    """Aggregate result for one planner candidate on one suite."""

    candidate: str
    algo: str
    suite: str
    episodes: int
    success_rate: float
    success_ci_low: float
    success_ci_high: float
    mean_min_distance: float
    mean_avg_speed: float
    jsonl_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the portfolio campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--portfolio-grid",
        type=Path,
        default=Path("configs/benchmarks/portfolio_sweep_grid_v1.yaml"),
    )
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tmp/planner_portfolio/campaigns/latest"),
    )
    return parser.parse_args()


def _episode_success(row: dict[str, Any]) -> bool:
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
    value = metrics.get("success", 0.0)
    if isinstance(value, bool):
        return bool(value)
    if value is None or value == "":
        return False
    return float(value) >= 0.5


def _bootstrap_ci(values: np.ndarray, n_samples: int, seed: int) -> tuple[float, float]:
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


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _load_grid(path: Path) -> list[dict[str, Any]]:
    payload = _load_yaml(path)
    variants = payload.get("variants", [])
    if not isinstance(variants, list):
        raise TypeError("portfolio grid variants must be a list")
    out: list[dict[str, Any]] = []
    for item in variants:
        if not isinstance(item, dict):
            continue
        algo = str(item.get("algo", "")).strip().lower()
        name = str(item.get("name", "")).strip()
        if not algo or not name:
            continue
        out.append(item)
    if not out:
        raise RuntimeError(f"No valid variants in {path}")
    return out


def _load_algo_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return _load_yaml(path)


def _nan_to_none(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def _run_eval(
    *,
    scenarios_or_path: Path | list[dict],
    suite_name: str,
    candidate_name: str,
    algo: str,
    algo_cfg: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> EvalResult:
    cfg_hash = hashlib.sha1(
        json.dumps({"algo": algo, "cfg": algo_cfg}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    tag = f"{suite_name}__{algo}__{candidate_name}__{cfg_hash}"
    algo_cfg_path = output_dir / f"{tag}_algo.yaml"
    jsonl_path = output_dir / f"{tag}.jsonl"
    algo_cfg_path.write_text(yaml.safe_dump(algo_cfg, sort_keys=False), encoding="utf-8")
    if jsonl_path.exists():
        jsonl_path.unlink()

    run_map_batch(
        scenarios_or_path,
        jsonl_path,
        schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
        algo=algo,
        algo_config_path=str(algo_cfg_path),
        horizon=int(args.horizon),
        dt=float(args.dt),
        workers=int(args.workers),
        resume=False,
        benchmark_profile="experimental",
    )

    rows: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)

    success_vals = np.asarray([1.0 if _episode_success(row) else 0.0 for row in rows], dtype=float)
    min_dist_vals = np.asarray(
        [
            float(row.get("metrics", {}).get("min_distance"))
            for row in rows
            if row.get("metrics", {}).get("min_distance") is not None
        ],
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
        candidate=candidate_name,
        algo=algo,
        suite=suite_name,
        episodes=len(rows),
        success_rate=float(np.mean(success_vals)) if success_vals.size > 0 else 0.0,
        success_ci_low=ci_low,
        success_ci_high=ci_high,
        mean_min_distance=float(np.mean(min_dist_vals)) if min_dist_vals.size > 0 else float("nan"),
        mean_avg_speed=float(np.mean(speed_vals)) if speed_vals.size > 0 else 0.0,
        jsonl_path=str(jsonl_path),
    )


def _rank_key(hard: EvalResult, global_res: EvalResult) -> tuple[float, float, float, float]:
    hard_clear = hard.mean_min_distance if np.isfinite(hard.mean_min_distance) else float("-inf")
    global_clear = (
        global_res.mean_min_distance if np.isfinite(global_res.mean_min_distance) else float("-inf")
    )
    return (hard.success_rate, global_res.success_rate, hard_clear, global_clear)


def main() -> int:
    """Execute campaign and write JSON/Markdown reports."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hard_manifest = load_seed_manifest(args.hard_seed_manifest)
    hard_scenarios = make_subset_scenarios(args.scenario_matrix, hard_manifest)
    if not hard_scenarios:
        raise RuntimeError("Hard seed manifest did not match any scenarios.")

    variants = _load_grid(args.portfolio_grid)
    base_dir = args.portfolio_grid.parent.resolve()

    results: list[dict[str, Any]] = []
    for variant in variants:
        algo = str(variant["algo"]).strip().lower()
        candidate_name = str(variant["name"]).strip().replace(" ", "_")
        cfg_path_raw = variant.get("algo_config_path")
        cfg_path = None
        if isinstance(cfg_path_raw, str) and cfg_path_raw.strip():
            p = Path(cfg_path_raw)
            cfg_path = p if p.is_absolute() else (base_dir / p).resolve()
        cfg = _load_algo_config(cfg_path)

        params = variant.get("params", {})
        if isinstance(params, dict):
            cfg.update(params)

        hard_res = _run_eval(
            scenarios_or_path=hard_scenarios,
            suite_name="hard",
            candidate_name=candidate_name,
            algo=algo,
            algo_cfg=cfg,
            output_dir=args.output_dir,
            args=args,
        )
        global_res = _run_eval(
            scenarios_or_path=args.scenario_matrix,
            suite_name="global",
            candidate_name=candidate_name,
            algo=algo,
            algo_cfg=cfg,
            output_dir=args.output_dir,
            args=args,
        )

        results.append(
            {
                "candidate": candidate_name,
                "algo": algo,
                "config": cfg,
                "hard": hard_res.__dict__,
                "global": global_res.__dict__,
                "ranking_key": list(_rank_key(hard_res, global_res)),
            }
        )

    ranked = sorted(results, key=lambda r: tuple(r["ranking_key"]), reverse=True)
    best = ranked[0]

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario_matrix": str(args.scenario_matrix),
        "hard_seed_manifest": str(args.hard_seed_manifest),
        "portfolio_grid": str(args.portfolio_grid),
        "horizon": int(args.horizon),
        "dt": float(args.dt),
        "workers": int(args.workers),
        "bootstrap_samples": int(args.bootstrap_samples),
        "num_candidates": len(ranked),
        "best": best,
        "ranked": ranked,
    }

    json_path = args.output_dir / "campaign_summary.json"
    json_path.write_text(json.dumps(_nan_to_none(summary), indent=2), encoding="utf-8")

    md_lines = [
        "# Planner Portfolio Campaign",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Scenario matrix: `{summary['scenario_matrix']}`",
        f"- Hard manifest: `{summary['hard_seed_manifest']}`",
        f"- Grid: `{summary['portfolio_grid']}`",
        f"- Candidates: `{summary['num_candidates']}`",
        "",
        "## Best Candidate",
        "",
        f"- Candidate: `{best['candidate']}`",
        f"- Algorithm: `{best['algo']}`",
        f"- Hard success: `{best['hard']['success_rate']:.4f}`",
        f"- Global success: `{best['global']['success_rate']:.4f}`",
        f"- Hard mean min-distance: `{best['hard']['mean_min_distance']:.4f}`",
        f"- Global mean min-distance: `{best['global']['mean_min_distance']:.4f}`",
        "",
        "## Ranking",
        "",
        "| Rank | Candidate | Algo | Hard SR | Global SR | Hard MinDist | Global MinDist |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]

    for i, row in enumerate(ranked, start=1):
        md_lines.append(
            f"| {i} | {row['candidate']} | {row['algo']} | "
            f"{row['hard']['success_rate']:.4f} | {row['global']['success_rate']:.4f} | "
            f"{row['hard']['mean_min_distance']:.4f} | {row['global']['mean_min_distance']:.4f} |"
        )

    md_path = args.output_dir / "campaign_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"summary": str(json_path), "report": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
