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
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from robot_sf.benchmark.latency_stress import (
    LatencyMeasurementHarness,
    classify_feasibility,
    collect_environment_provenance,
)
from robot_sf.benchmark.map_runner import _run_map_episode
from robot_sf.benchmark.utils import _config_hash, _git_hash_fallback

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "benchmarks" / "compute_feasibility_matrix_v1.yaml"
_CELL_FAILURE_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    LookupError,
    OSError,
    AttributeError,
    ImportError,
    ArithmeticError,
    AssertionError,
)
_VALID_CELL_STATUSES = {"native", "not_available", "failed"}


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


@dataclass(frozen=True)
class RunnerSettings:
    """Normalized settings shared by every matrix cell."""

    horizon: int
    dt: float
    record_forces: bool
    observation_mode: str | None = None
    observation_level: str | None = None
    benchmark_track: str | None = None


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
) -> dict[str, Any]:
    """Build a scenario dict from map path and parameters.

    Returns:
        dict[str, Any]: Scenario definition for map_runner.
    """
    abs_map = Path(map_path)
    if not abs_map.is_absolute():
        abs_map = (repo_root / map_path).resolve()
    if not abs_map.is_file():
        raise FileNotFoundError(f"Map file not found: {abs_map}")

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
        # Density is a heuristic; map loading belongs to the episode setup and should not be
        # repeated here merely to decide whether a positive density is usable.
        scenario["simulation_config"]["ped_density"] = 0.02 + pedestrian_count * 0.005

    return scenario


def _normalize_horizon(value: Any) -> int:
    """Normalize an optional episode horizon and reject invalid values."""
    if value is None:
        return 60
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("horizon must be a positive integer or null")
    if value <= 0:
        raise ValueError("horizon must be > 0")
    return value


def _normalize_dt(value: Any) -> float:
    """Normalize an optional simulation step and reject invalid values."""
    if value is None:
        return 0.1
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("dt must be a positive number or null")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("dt must be finite and > 0")
    return normalized


def _validate_planners(planners: list[Any]) -> None:
    """Validate planner entries in a matrix config."""
    for index, planner in enumerate(planners):
        if not isinstance(planner, Mapping):
            raise TypeError(f"planners[{index}] must be a mapping")
        for field in ("key", "algo"):
            value = planner.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"planners[{index}].{field} must be a non-empty string")
        algo_config = planner.get("algo_config")
        if algo_config is not None and not isinstance(algo_config, Mapping):
            raise TypeError(f"planners[{index}].algo_config must be a mapping or null")


def _validate_maps(maps: list[Any]) -> None:
    """Validate map entries and fail closed for missing or directory paths."""
    for index, map_entry in enumerate(maps):
        if not isinstance(map_entry, Mapping):
            raise TypeError(f"maps[{index}] must be a mapping")
        map_path = map_entry.get("path")
        map_label = map_entry.get("label")
        if not isinstance(map_path, str) or not map_path.strip():
            raise ValueError(f"maps[{index}].path must be a non-empty string")
        if not isinstance(map_label, str) or not map_label.strip():
            raise ValueError(f"maps[{index}].label must be a non-empty string")
        resolved_map = Path(map_path)
        if not resolved_map.is_absolute():
            resolved_map = (REPO_ROOT / resolved_map).resolve()
        if not resolved_map.is_file():
            raise FileNotFoundError(f"Map file not found: {resolved_map}")


def _validate_matrix_axes(config: Mapping[str, Any]) -> None:
    """Validate pedestrian, deadline, and seed axes."""
    for index, count in enumerate(config["pedestrian_counts"]):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"pedestrian_counts[{index}] must be an integer >= 0")

    for index, deadline in enumerate(config["control_deadlines_ms"]):
        if isinstance(deadline, bool) or not isinstance(deadline, int | float):
            raise TypeError(f"control_deadlines_ms[{index}] must be a number")
        if not math.isfinite(float(deadline)) or deadline <= 0:
            raise ValueError(f"control_deadlines_ms[{index}] must be finite and > 0")

    for index, seed in enumerate(config["seeds"]):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seeds[{index}] must be an integer")


def _validate_runtime_fields(config: Mapping[str, Any]) -> None:
    """Validate fixed runtime settings in a matrix config."""
    _normalize_horizon(config.get("horizon"))
    _normalize_dt(config.get("dt"))
    record_forces = config.get("record_forces", False)
    if not isinstance(record_forces, bool):
        raise TypeError("record_forces must be a boolean")
    for field in ("observation_mode", "observation_level", "benchmark_track"):
        value = config.get(field)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{field} must be a string or null")


def _validate_config(config: Any) -> dict[str, Any]:
    """Validate the matrix shape and all fields needed by the runner."""
    if not isinstance(config, dict):
        raise ValueError("Matrix config must contain a mapping at its YAML root")
    if config.get("schema_version") != "compute_feasibility_matrix.v1":
        raise ValueError(
            "Expected schema_version 'compute_feasibility_matrix.v1', "
            f"got {config.get('schema_version')!r}"
        )
    required_lists = (
        "planners",
        "maps",
        "pedestrian_counts",
        "control_deadlines_ms",
        "seeds",
    )
    for field in required_lists:
        value = config.get(field)
        if not isinstance(value, list) or not value:
            raise ValueError(f"{field} must be a non-empty list")
    _validate_planners(config["planners"])
    _validate_maps(config["maps"])
    _validate_matrix_axes(config)
    _validate_runtime_fields(config)
    return config


def _cell_config_hash(
    cell: MatrixCell,
    *,
    matrix_config_hash: str | None,
    settings: RunnerSettings,
) -> str:
    """Return a cache/provenance hash for all cell-affecting runner inputs."""
    return _config_hash(
        {
            "matrix_config_hash": matrix_config_hash,
            "planner_key": cell.planner_key,
            "algo": cell.algo,
            "algo_config": cell.algo_config or {},
            "map_label": cell.map_label,
            "map_path": cell.map_path,
            "pedestrian_count": cell.pedestrian_count,
            "deadline_ms": cell.deadline_ms,
            "seed": cell.seed,
            "horizon": settings.horizon,
            "dt": settings.dt,
            "record_forces": settings.record_forces,
            "observation_mode": settings.observation_mode,
            "observation_level": settings.observation_level,
            "benchmark_track": settings.benchmark_track,
        }
    )


def _run_cell(
    cell: MatrixCell,
    *,
    settings: RunnerSettings,
    repo_root: Path,
    scenario_path: Path,
    matrix_config_hash: str | None = None,
) -> dict[str, Any]:
    """Execute one matrix cell and return its latency + classification record.

    Returns:
        dict[str, Any]: Record with latency metrics, classification, provenance.
    """
    scenario = _resolve_scenario(
        map_path=cell.map_path,
        map_label=cell.map_label,
        pedestrian_count=cell.pedestrian_count,
        horizon=settings.horizon,
        repo_root=repo_root,
    )

    cfg_hash = _cell_config_hash(
        cell,
        matrix_config_hash=matrix_config_hash,
        settings=settings,
    )

    ts = datetime.now(UTC).isoformat()
    wall_start = time.monotonic()

    harness = LatencyMeasurementHarness(
        deadline_ms=cell.deadline_ms,
        # The episode policy supplies its planner-config hash through metadata.  Keep the
        # full cell hash in the emitted row/cache contract without making the harness reject
        # a valid policy whose hash intentionally covers only planner configuration.
        config_hash=None,
    )

    status = "native"
    latency_metrics: dict[str, Any] = {}
    classification: str = "failed"

    try:
        with harness:
            _run_map_episode(
                scenario,
                seed=cell.seed,
                horizon=settings.horizon,
                dt=settings.dt,
                record_forces=settings.record_forces,
                snqi_weights=None,
                snqi_baseline=None,
                algo=cell.algo,
                algo_config=cell.algo_config,
                scenario_path=scenario_path,
                observation_mode=settings.observation_mode,
                observation_level=settings.observation_level,
                benchmark_track=settings.benchmark_track,
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
                steady_state_latencies=steady_totals[1:],
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

    except _CELL_FAILURE_EXCEPTIONS as e:
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
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _validate_config(raw)


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


def _summary_sample(row: Mapping[str, Any]) -> tuple[str, float, float] | None:
    """Extract one valid native summary sample, or return ``None``."""
    cell = row.get("cell")
    if not isinstance(cell, Mapping):
        return None
    key = cell.get("planner_key")
    deadline = cell.get("deadline_ms")
    if not isinstance(key, str) or not key.strip():
        return None
    if isinstance(deadline, bool) or not isinstance(deadline, int | float):
        return None
    dl = float(deadline)
    if not math.isfinite(dl) or dl <= 0.0 or row.get("status") != "native":
        return None
    lat = row.get("latency")
    if not isinstance(lat, Mapping):
        return None
    p95 = lat.get("steady_state_latency_p95_ms")
    if isinstance(p95, bool) or not isinstance(p95, int | float):
        return None
    p95_value = float(p95)
    if not math.isfinite(p95_value) or p95_value < 0.0:
        return None
    return key.strip(), dl, p95_value


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build per-planner summary table from cell rows.

    Returns:
        dict[str, Any]: Summary grouped by planner and deadline.
    """
    planners: dict[str, dict[str, list[float]]] = {}
    failures: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, Mapping):
            failures.append(row)
            continue
        sample = _summary_sample(row)
        if sample is None:
            failures.append(row)
            continue
        key, dl, p95_value = sample
        group = planners.setdefault(key, {}).setdefault(dl, [])
        group.append(p95_value)

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


def _load_cached_row(
    path: Path,
    *,
    cell: MatrixCell,
    config_hash: str,
) -> dict[str, Any] | None:
    """Load a cache row only when it is valid for the requested cell."""
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict):
            raise ValueError("cache root must be an object")
        if row.get("schema_version") != "compute_feasibility_cell.v1":
            raise ValueError("cache schema version is unsupported")
        if row.get("status") not in _VALID_CELL_STATUSES:
            raise ValueError("cache status is unsupported")
        cached_cell = row.get("cell")
        if not isinstance(cached_cell, Mapping):
            raise ValueError("cache cell must be an object")
        expected = {
            "planner_key": cell.planner_key,
            "algo": cell.algo,
            "algo_config": cell.algo_config,
            "map_label": cell.map_label,
            "map_path": cell.map_path,
            "pedestrian_count": cell.pedestrian_count,
            "deadline_ms": cell.deadline_ms,
            "seed": cell.seed,
            "config_hash": config_hash,
        }
        for key, value in expected.items():
            if cached_cell.get(key) != value:
                raise ValueError(f"cache cell field {key!r} does not match request")
        return row
    except _CELL_FAILURE_EXCEPTIONS as exc:
        logger.warning("Ignoring invalid cached row '{}': {}", path, exc)
        return None


def _failure_row(cell: MatrixCell, *, config_hash: str, reason: str) -> dict[str, Any]:
    """Build a complete failed-cell record for errors outside episode execution."""
    return {
        "schema_version": "compute_feasibility_cell.v1",
        "ts": datetime.now(UTC).isoformat(),
        "cell": {
            "planner_key": cell.planner_key,
            "algo": cell.algo,
            "algo_config": cell.algo_config,
            "map_label": cell.map_label,
            "map_path": cell.map_path,
            "pedestrian_count": cell.pedestrian_count,
            "deadline_ms": cell.deadline_ms,
            "seed": cell.seed,
            "config_hash": config_hash,
        },
        "status": "failed",
        "classification": "failed",
        "latency": {"failure_reason": reason},
        "provenance": {},
    }


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


def main() -> int:  # noqa: C901
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

    matrix_config_hash = _config_hash(
        {
            "config_file": repo_rel(config_path),
            "schema": "compute_feasibility_matrix.v1",
            "config": config,
        }
    )
    horizon = _normalize_horizon(config.get("horizon"))
    dt = _normalize_dt(config.get("dt"))
    record_forces = config.get("record_forces", False)
    settings = RunnerSettings(
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
        observation_mode=config.get("observation_mode"),
        observation_level=config.get("observation_level"),
        benchmark_track=config.get("benchmark_track"),
    )

    if args.dry_run:
        print(f"Dry run: {len(cells)} cells")
        for cell in cells:
            print(
                f"  {cell.planner_key} | {cell.map_label} | "
                f"peds={cell.pedestrian_count} | dl={cell.deadline_ms}ms | seed={cell.seed}"
            )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "compute_feasibility_manifest.v1",
        "config_file": repo_rel(config_path),
        "config_hash": matrix_config_hash,
        "git_commit": _git_head(),
        "ts": datetime.now(UTC).isoformat(),
        "total_cells": len(cells),
        "planners": [p["key"] for p in config["planners"]],
        "maps": [m["label"] for m in config["maps"]],
        "pedestrian_counts": config["pedestrian_counts"],
        "deadlines_ms": config["control_deadlines_ms"],
        "seeds": config["seeds"],
        "horizon": horizon,
        "dt": dt,
        "record_forces": record_forces,
    }

    scenario_path = config_path

    print(f"Running {len(cells)} compute-feasibility matrix cells -> {repo_rel(out_dir)}")

    rows: list[dict[str, Any]] = []
    skipped = 0
    for idx, cell in enumerate(cells):
        existing = (
            out_dir
            / f"{cell.planner_key}_{cell.map_label}_{cell.pedestrian_count}_{cell.deadline_ms}_{cell.seed}.json"
        )
        cell_config_hash = _cell_config_hash(
            cell,
            matrix_config_hash=matrix_config_hash,
            settings=settings,
        )
        if existing.is_file():
            row = _load_cached_row(existing, cell=cell, config_hash=cell_config_hash)
            if row is not None:
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
                settings=settings,
                repo_root=REPO_ROOT,
                scenario_path=scenario_path,
                matrix_config_hash=matrix_config_hash,
            )
            rows.append(row)
            existing.write_text(
                json.dumps(row, indent=2, default=str),
                encoding="utf-8",
            )
        except _CELL_FAILURE_EXCEPTIONS as e:
            # Fail-closed: every cell must produce a row.
            print(f"  ERROR: {e}")
            rows.append(_failure_row(cell, config_hash=cell_config_hash, reason=str(e)))

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
