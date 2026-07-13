#!/usr/bin/env python3
"""Compute-feasibility matrix runner (issue #5525).

Executes a planner/scenario/deadline matrix using the LatencyMeasurementHarness
from #5506/#5524 and produces cold/steady-state latency metrics, deadline-miss
rates, feasibility classifications, and full environment provenance per cell.

Output:
  output/compute_feasibility_matrix_v1/ (or --out)
    rows.jsonl          — one JSON record per cell
    manifest.json       — config+provenance header
    synthetic_summary.json  — per-planner latency summary table

Usage:
  uv run python scripts/benchmark/run_compute_feasibility_matrix.py \\
      --config configs/benchmarks/compute_feasibility_matrix_v1.yaml

Notes:
  - CPU-only, no SLURM, no training runs.
  - Fallback/degraded cells are classified explicitly (fail-closed).
  - Every accepted row passes the latency component-sum contract.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.latency_stress import (
    LatencyMeasurementHarness,
    classify_feasibility,
    collect_environment_provenance,
)
from robot_sf.benchmark.map_runner import _run_map_episode
from robot_sf.benchmark.utils import _config_hash, _git_hash_fallback

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "benchmarks" / "compute_feasibility_matrix_v1.yaml"


@dataclass(frozen=True)
class MatrixCell:
    """One position in the compute-feasibility matrix."""

    planner_key: str
    algo: str
    algo_config: dict[str, Any] | None
    map_label: str
    map_path: str
    pedestrian_count: int
    deadline_ms: float
    seed: int


def _git_head() -> str:
    """Return the current git HEAD commit hash, or ``unknown``."""
    return _git_hash_fallback() or "unknown"


def _resolve_scenario(
    *,
    map_path: str,
    map_label: str,
    pedestrian_count: int,
    horizon: int,
    repo_root: Path,
    scenario_path: Path,
) -> dict[str, Any]:
    """Build a scenario dict from map path and parameters.

    Returns:
        dict[str, Any]: Scenario definition for map_runner.
    """
    abs_map = Path(map_path)
    if not abs_map.is_absolute():
        abs_map = (repo_root / map_path).resolve()

    scenario: dict[str, Any] = {
        "name": f"feasibility_{map_label}_peds{pedestrian_count}",
        "map_file": str(abs_map),
        "simulation_config": {
            "max_episode_steps": horizon,
            "ped_density": 0.0,
        },
        "robot_config": {},
    }
    # Add pedestrians via single_pedestrians when count > 0:
    # we set ped_density so the simulator spawns random pedestrians,
    # but the exact count is controlled by ped_density * area.
    # For a reproducible fixed count, use single_pedestrians.
    if pedestrian_count > 0:
        # Load the map to find valid POIs for pedestrian goals.
        from robot_sf.training.scenario_loader import load_scenarios

        # Minimal scenario YAML to get map geometry
        temp_yaml = {"scenarios": [{"name": "temp", "map_file": str(abs_map)}]}
        scenarios = load_scenarios(
            yaml.dump(temp_yaml),
            scenario_config=Path(str(temp_yaml["scenarios"][0]["map_file"])),
        )
        if scenarios:
            # Use ped_density approach: density controls spawn rate
            # Scale density to approximate desired count for the map area

            # We'll let build_env_config handle the map loading
            # Use a higher ped_density for more pedestrians
            base_density = 0.02  # Base density
            scenario["simulation_config"]["ped_density"] = base_density * (
                1 + pedestrian_count * 0.1
            )
        else:
            # Fallback: use a higher density to approximate pedestrian count
            scenario["simulation_config"]["ped_density"] = 0.02 + pedestrian_count * 0.005

    scenario["scenario_path"] = str(scenario_path)
    return scenario


def _run_cell(
    cell: MatrixCell,
    *,
    horizon: int,
    dt: float,
    repo_root: Path,
    scenario_path: Path,
    observation_mode: str | None = None,
    observation_level: str | None = None,
    benchmark_track: str | None = None,
) -> dict[str, Any]:
    """Execute one matrix cell and return its latency + classification record.

    Returns:
        dict[str, Any]: Record with latency metrics, classification, provenance.
    """
    scenario = _resolve_scenario(
        map_path=cell.map_path,
        map_label=cell.map_label,
        pedestrian_count=cell.pedestrian_count,
        horizon=horizon,
        repo_root=repo_root,
        scenario_path=scenario_path,
    )

    cfg_hash = _config_hash(
        {
            "algo": cell.algo,
            "algo_config": cell.algo_config or {},
            "map_label": cell.map_label,
            "pedestrian_count": cell.pedestrian_count,
            "deadline_ms": cell.deadline_ms,
            "seed": cell.seed,
        }
    )

    ts = datetime.now(UTC).isoformat()
    wall_start = time.monotonic()

    harness = LatencyMeasurementHarness(
        deadline_ms=cell.deadline_ms,
        config_hash=cfg_hash,
    )

    status = "native"
    latency_metrics: dict[str, Any] = {}
    classification: str = "failed"

    try:
        with harness:
            _run_map_episode(
                scenario,
                seed=cell.seed,
                horizon=horizon,
                dt=dt,
                record_forces=False,
                snqi_weights=None,
                snqi_baseline=None,
                algo=cell.algo,
                algo_config=cell.algo_config,
                scenario_path=scenario_path,
                observation_mode=observation_mode,
                observation_level=observation_level,
                benchmark_track=benchmark_track,
            )

        # Require at least 2 cycles for steady-state classification
        metrics = harness.get_metrics()
        steady_totals = [c["total_ms"] for c in metrics.get("cycles", [])]
        if len(steady_totals) < 2:
            # Not enough cycles from episode — mark as not_available
            status = "not_available"
            latency_metrics = {
                "cold_start_latency_ms": metrics.get("cold_start_latency_ms"),
                "cycles_recorded": len(metrics.get("cycles", [])),
                "steady_state_cycles": 0,
            }
        else:
            classification = classify_feasibility(
                steady_state_latencies=[c["total_ms"] for c in steady_totals[1:]],
                deadline_ms=cell.deadline_ms,
            )
            latency_metrics = {
                "cold_start_latency_ms": metrics["cold_start_latency_ms"],
                "steady_state_latency_p50_ms": metrics["steady_state_latency_p50_ms"],
                "steady_state_latency_p95_ms": metrics["steady_state_latency_p95_ms"],
                "steady_state_latency_p99_ms": metrics["steady_state_latency_p99_ms"],
                "steady_state_latency_max_ms": metrics["steady_state_latency_max_ms"],
                "deadline_miss_rate": metrics["deadline_miss_rate"],
                "classification": classification,
                "steady_state_averages": metrics["steady_state_averages"],
                "steady_state_cycles": len(steady_totals) - 1,
                "total_cycles": len(steady_totals),
            }

    except Exception as e:  # noqa: BLE001
        # Fail-closed: any episode runner error produces a classified cell row.
        status = "failed"
        classification = "failed"
        latency_metrics = {"failure_reason": str(e)}

    wall_elapsed = (time.monotonic() - wall_start) * 1000.0

    prov = collect_environment_provenance(
        config_hash=cfg_hash,
    )

    return {
        "schema_version": "compute_feasibility_cell.v1",
        "ts": ts,
        "cell": {
            "planner_key": cell.planner_key,
            "algo": cell.algo,
            "algo_config": cell.algo_config,
            "map_label": cell.map_label,
            "map_path": cell.map_path,
            "pedestrian_count": cell.pedestrian_count,
            "deadline_ms": cell.deadline_ms,
            "seed": cell.seed,
            "config_hash": cfg_hash,
        },
        "status": status,
        "classification": classification,
        "latency": latency_metrics,
        "provenance": prov,
        "wall_ms": round(wall_elapsed, 2),
    }


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate matrix config YAML.

    Returns:
        dict[str, Any]: Parsed config.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != "compute_feasibility_matrix.v1":
        raise ValueError(
            f"Expected schema_version 'compute_feasibility_matrix.v1', "
            f"got {raw.get('schema_version')!r}"
        )
    return raw


def _expand_cells(config: dict[str, Any]) -> list[MatrixCell]:
    """Expand matrix config into list of cells.

    Returns:
        List of MatrixCell objects.
    """
    cells: list[MatrixCell] = []
    for planner in config["planners"]:
        for map_entry in config["maps"]:
            for ped_count in config["pedestrian_counts"]:
                for deadline in config["control_deadlines_ms"]:
                    for seed in config["seeds"]:
                        cells.append(
                            MatrixCell(
                                planner_key=planner["key"],
                                algo=planner["algo"],
                                algo_config=planner.get("algo_config"),
                                map_label=map_entry["label"],
                                map_path=map_entry["path"],
                                pedestrian_count=ped_count,
                                deadline_ms=float(deadline),
                                seed=seed,
                            )
                        )
    return cells


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build per-planner summary table from cell rows.

    Returns:
        dict[str, Any]: Summary grouped by planner and deadline.
    """
    planners: dict[str, dict[str, list[float]]] = {}
    failures: list[dict[str, Any]] = []

    for row in rows:
        key = row["cell"]["planner_key"]
        dl = row["cell"]["deadline_ms"]
        group = planners.setdefault(key, {}).setdefault(dl, [])
        if row["status"] != "native":
            failures.append(row)
            continue
        lat = row.get("latency", {})
        if "steady_state_latency_p95_ms" in lat:
            group.append(lat["steady_state_latency_p95_ms"])

    summary: dict[str, Any] = {"schema_version": "compute_feasibility_summary.v1"}
    for key, deadlines in sorted(planners.items()):
        entry: dict[str, Any] = {"planner_key": key}
        for dl, values in sorted(deadlines.items()):
            if values:
                entry[f"p95_ms_deadline_{int(dl)}"] = {
                    "values": [round(v, 3) for v in values],
                    "mean_ms": round(float(np.mean(values)), 3),
                    "median_ms": round(float(np.median(values)), 3),
                    "max_ms": round(float(np.max(values)), 3),
                    "n_cells": len(values),
                }
        summary[key] = entry

    summary["non_native_cells"] = len(failures)
    summary["total_cells"] = len(rows)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Execute compute-feasibility matrix with latency-stress harness."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Matrix config YAML path (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: output/compute_feasibility_matrix_v1/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matrix cells without executing.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    config = _load_config(config_path)
    cells = _expand_cells(config)

    out_dir = Path(args.out) if args.out else REPO_ROOT / "output" / "compute_feasibility_matrix_v1"

    def repo_rel(path: Path | str) -> str:
        try:
            return str(Path(path).relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    if args.dry_run:
        print(f"Dry run: {len(cells)} cells")
        for cell in cells:
            print(
                f"  {cell.planner_key} | {cell.map_label} | "
                f"peds={cell.pedestrian_count} | dl={cell.deadline_ms}ms | seed={cell.seed}"
            )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _config_hash(
        {"config_file": repo_rel(config_path), "schema": "compute_feasibility_matrix.v1"}
    )

    manifest = {
        "schema_version": "compute_feasibility_manifest.v1",
        "config_file": repo_rel(config_path),
        "config_hash": config_hash,
        "git_commit": _git_head(),
        "ts": datetime.now(UTC).isoformat(),
        "total_cells": len(cells),
        "planners": [p["key"] for p in config["planners"]],
        "maps": [m["label"] for m in config["maps"]],
        "pedestrian_counts": config["pedestrian_counts"],
        "deadlines_ms": config["control_deadlines_ms"],
        "seeds": config["seeds"],
        "horizon": config.get("horizon", 60),
        "dt": config.get("dt", 0.1),
    }

    scenario_path = config_path.parent if config_path.name.endswith(".yaml") else Path(".")

    print(f"Running {len(cells)} compute-feasibility matrix cells -> {repo_rel(out_dir)}")

    rows: list[dict[str, Any]] = []
    skipped = 0
    for idx, cell in enumerate(cells):
        existing = (
            out_dir
            / f"{cell.planner_key}_{cell.map_label}_{cell.pedestrian_count}_{cell.deadline_ms}_{cell.seed}.json"
        )
        if existing.is_file():
            row = json.loads(existing.read_text(encoding="utf-8"))
            rows.append(row)
            skipped += 1
            print(
                f"[{idx + 1}/{len(cells)}] SKIP (cached) {cell.planner_key} | {cell.map_label} | dl={cell.deadline_ms}ms"
            )
            continue

        print(
            f"[{idx + 1}/{len(cells)}] RUN {cell.planner_key} | {cell.map_label} | "
            f"peds={cell.pedestrian_count} | dl={cell.deadline_ms}ms | seed={cell.seed}"
        )
        try:
            row = _run_cell(
                cell,
                horizon=config.get("horizon", 60),
                dt=float(config.get("dt", 0.1)),
                repo_root=REPO_ROOT,
                scenario_path=scenario_path,
                observation_mode=config.get("observation_mode"),
                observation_level=config.get("observation_level"),
                benchmark_track=config.get("benchmark_track"),
            )
            rows.append(row)
            existing.write_text(
                json.dumps(row, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001
            # Fail-closed: every cell must produce a row.
            print(f"  ERROR: {e}")
            rows.append(
                {
                    "schema_version": "compute_feasibility_cell.v1",
                    "ts": datetime.now(UTC).isoformat(),
                    "cell": {
                        "planner_key": cell.planner_key,
                        "map_label": cell.map_label,
                        "pedestrian_count": cell.pedestrian_count,
                        "deadline_ms": cell.deadline_ms,
                        "seed": cell.seed,
                    },
                    "status": "failed",
                    "classification": "failed",
                    "latency": {"failure_reason": str(e)},
                    "provenance": {},
                }
            )

    # Write JSONL
    rows_path = out_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")

    # Write manifest
    manifest["completed_cells"] = len(rows)
    manifest["skipped_cached"] = skipped
    manifest["output_files"] = {
        "rows_jsonl": repo_rel(rows_path),
        "summary": repo_rel(out_dir / "synthetic_summary.json"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    # Write summary
    summary = _build_summary(rows)
    (out_dir / "synthetic_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    native = sum(1 for r in rows if r["status"] == "native")
    failed = sum(1 for r in rows if r["status"] == "failed")
    not_avail = sum(1 for r in rows if r["status"] == "not_available")
    print(
        f"\nDone: {native} native, {not_avail} not_available, {failed} failed / {len(rows)} total"
    )
    print(f"Output: {repo_rel(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
