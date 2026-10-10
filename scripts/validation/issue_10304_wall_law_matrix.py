#!/usr/bin/env python3
"""Run the #10304 successor bottleneck pedestrian wall-law matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point

from robot_sf.sim.simulator import init_simulators
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
)

SCENARIO_MATRIX = Path("configs/scenarios/issue_10304_wall_law_bottleneck_successor_v1.yaml")
SCENARIO_NAME = "classic_realworld_double_bottleneck_high"
BODY_RADIUS_M = 0.40
GOAL_RADIUS_M = 1.0
STALL_WINDOW_STEPS = 100
STALL_DISPLACEMENT_M = 0.20
DEFAULT_MAX_EPISODE_STEPS = 900
STATIONARY_ACTION = [(0.0, 0.0)]

VARIANTS: dict[str, dict[str, float | str]] = {
    "legacy": {"law": "legacy_shifted_gradient_v1"},
    "body_edge_exponential_v3": {"law": "body_edge_exponential_v3"},
    "range_only": {"law": "body_edge_exponential_v3_range_only"},
    "physical_margin": {"law": "body_edge_exponential_v3_physical_margin"},
    "contact_stiff": {"law": "body_edge_exponential_v3_contact_stiff"},
    "multi_segment": {"law": "body_edge_exponential_v3_multi_segment"},
    "physical_margin10x": {
        "law": "body_edge_exponential_v3_physical_margin",
        "factor_multiplier": 10.0,
    },
}


@dataclass(frozen=True)
class TrialResult:
    """Per-seed diagnostic counts for one wall-law variant."""

    variant: str
    seed: int
    pedestrians: int
    initial_overlaps: int
    new_overlaps: int
    any_overlaps: int
    stalled: int
    goal_completed: int
    min_wall_clearance_m: float
    steps: int


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parse_seed_spec(seed_spec: str) -> list[int]:
    seeds: list[int] = []
    for part in seed_spec.split(","):
        if not part:
            continue
        if "-" not in part:
            seeds.append(int(part))
            continue
        start, end = (int(value) for value in part.split("-", 1))
        if end < start:
            raise ValueError(f"seed range must ascend: {part}")
        seeds.extend(range(start, end + 1))
    return seeds


def _select_scenario() -> dict[str, Any]:
    return next(
        scenario
        for scenario in load_scenarios(SCENARIO_MATRIX)
        if scenario["name"] == SCENARIO_NAME
    )


def _obstacle_polygons(definition) -> list[Any]:
    return [polygon for obstacle in definition.obstacles for polygon in obstacle.iter_polygons()]


def _wall_clearance(point_xy: np.ndarray, obstacles: list[Any]) -> float:
    body = Point(float(point_xy[0]), float(point_xy[1])).buffer(BODY_RADIUS_M)
    return min(body.distance(obstacle) for obstacle in obstacles)


def _overlap_count(positions: np.ndarray, obstacles: list[Any]) -> tuple[int, float]:
    overlaps = 0
    min_clearance = math.inf
    for position in positions:
        body = Point(float(position[0]), float(position[1])).buffer(BODY_RADIUS_M)
        min_clearance = min(min_clearance, *(body.distance(obstacle) for obstacle in obstacles))
        if any(body.intersection(obstacle).area > 1e-9 for obstacle in obstacles):
            overlaps += 1
    return overlaps, min_clearance


def _final_targets(definition) -> np.ndarray:
    targets = []
    for ped in definition.single_pedestrians:
        if ped.trajectory:
            targets.append(ped.trajectory[-1])
        elif ped.goal is not None:
            targets.append(ped.goal)
        else:
            raise ValueError(f"single pedestrian {ped.id} has no final target")
    return np.asarray(targets, dtype=float)


def _apply_variant(config, variant: dict[str, float | str]) -> None:
    config.sim_config.obstacle_force_law = str(variant["law"])


def _run_one(payload: tuple[str, int]) -> TrialResult:
    variant_name, seed = payload
    from loguru import logger

    logger.remove()
    scenario = _select_scenario()
    scenario.setdefault("simulation_config", {})
    scenario["simulation_config"].update(
        {
            "route_spawn_seed": seed,
            "archetype_seed": seed,
            "response_law_seed": seed,
        }
    )
    config = build_robot_config_from_scenario(
        scenario,
        scenario_path=SCENARIO_MATRIX.resolve(),
    )
    config.sim_config.pedestrian_seed = seed
    config.sim_config.desired_speed_seed = seed
    _apply_variant(config, VARIANTS[variant_name])

    definition = next(iter(config.map_pool.map_defs.values()))
    simulator = init_simulators(
        config,
        definition,
        num_robots=1,
        random_start_pos=False,
    )[0]
    if multiplier := VARIANTS[variant_name].get("factor_multiplier"):
        simulator.pysf_sim.config.obstacle_force_config.factor *= float(multiplier)
    obstacles = _obstacle_polygons(definition)
    targets = _final_targets(definition)

    initial_positions = np.asarray(simulator.ped_pos, dtype=float)
    _, min_clearance = _overlap_count(initial_positions, obstacles)
    ever_overlapped = np.zeros(initial_positions.shape[0], dtype=bool)
    last_window: list[np.ndarray] = []
    final_positions = initial_positions
    steps = int(
        getattr(
            config.sim_config,
            "episode_step_limit",
            None,
        )
        or round(config.sim_config.sim_time_in_secs / config.sim_config.time_per_step_in_secs)
        or DEFAULT_MAX_EPISODE_STEPS
    )

    for step in range(steps):
        simulator.step_once(STATIONARY_ACTION)
        positions = np.asarray(simulator.ped_pos, dtype=float)
        final_positions = positions
        overlaps, clearance = _overlap_count(positions, obstacles)
        min_clearance = min(min_clearance, clearance)
        if overlaps:
            for row, position in enumerate(positions):
                if _wall_clearance(position, obstacles) <= 1e-9:
                    ever_overlapped[row] = True
        last_window.append(positions.copy())
        if len(last_window) > STALL_WINDOW_STEPS:
            last_window.pop(0)
        if np.all(np.linalg.norm(positions - targets, axis=1) <= GOAL_RADIUS_M):
            steps = step + 1
            break

    initial_overlap_mask = np.array(
        [
            any(
                Point(float(position[0]), float(position[1]))
                .buffer(BODY_RADIUS_M)
                .intersection(obstacle)
                .area
                > 1e-9
                for obstacle in obstacles
            )
            for position in initial_positions
        ],
        dtype=bool,
    )
    goal_distances = np.linalg.norm(final_positions - targets, axis=1)
    completed = goal_distances <= GOAL_RADIUS_M
    new_overlap_mask = ever_overlapped & ~initial_overlap_mask
    if len(last_window) >= 2:
        displacement = np.linalg.norm(last_window[-1] - last_window[0], axis=1)
    else:
        displacement = np.full(initial_positions.shape[0], math.inf)
    stalled = (~completed) & (displacement < STALL_DISPLACEMENT_M)

    return TrialResult(
        variant=variant_name,
        seed=seed,
        pedestrians=int(initial_positions.shape[0]),
        initial_overlaps=int(initial_overlap_mask.sum()),
        new_overlaps=int(new_overlap_mask.sum()),
        any_overlaps=int((ever_overlapped | initial_overlap_mask).sum()),
        stalled=int(stalled.sum()),
        goal_completed=int(completed.sum()),
        min_wall_clearance_m=float(min_clearance),
        steps=steps,
    )


def _summarize(results: list[TrialResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    for result in results:
        grouped[result.variant].append(result)

    summary = []
    for variant in VARIANTS:
        variant_results = grouped[variant]
        if not variant_results:
            continue
        pedestrians = sum(result.pedestrians for result in variant_results)
        summary.append(
            {
                "variant": variant,
                "seeds": len(variant_results),
                "pedestrians": pedestrians,
                "initial_overlaps": sum(result.initial_overlaps for result in variant_results),
                "new_overlaps": sum(result.new_overlaps for result in variant_results),
                "any_overlaps": sum(result.any_overlaps for result in variant_results),
                "stalled": sum(result.stalled for result in variant_results),
                "goal_completed": sum(result.goal_completed for result in variant_results),
                "goal_completion_rate": sum(result.goal_completed for result in variant_results)
                / pedestrians,
                "min_wall_clearance_m": min(
                    result.min_wall_clearance_m for result in variant_results
                ),
                "mean_steps": sum(result.steps for result in variant_results)
                / len(variant_results),
            }
        )
    return summary


def _markdown_table(summary: list[dict[str, Any]]) -> str:
    lines = [
        "| law | seeds | peds | initial overlaps | new overlaps | any overlaps | stalls | goals | completion | min wall clearance (m) | mean steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {variant} | {seeds} | {pedestrians} | {initial_overlaps} | "
            "{new_overlaps} | {any_overlaps} | {stalled} | "
            "{goal_completed}/{pedestrians} | {goal_completion_rate:.3f} | "
            "{min_wall_clearance_m:.6f} | {mean_steps:.1f} |".format(**row)
        )
    return "\n".join(lines)


def main() -> int:
    """Parse CLI arguments, run the matrix, and write JSON/Markdown artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1001-1200")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/issue_10304_wall_law_matrix")
    )
    args = parser.parse_args()

    seeds = _parse_seed_spec(args.seeds)
    workers = min(args.workers, 8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    tasks = [(variant, seed) for variant in VARIANTS for seed in seeds]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_run_one, tasks):
            results.append(result)
            print(
                f"[{len(results)}/{len(tasks)}] {result.variant} seed={result.seed} "
                f"new_overlaps={result.new_overlaps} stalled={result.stalled} "
                f"goals={result.goal_completed}/{result.pedestrians}",
                flush=True,
            )

    summary = _summarize(results)
    metadata = {
        "schema": "issue_10304_wall_law_matrix.v1",
        "scenario_matrix": str(SCENARIO_MATRIX),
        "scenario_name": SCENARIO_NAME,
        "seeds": seeds,
        "workers": workers,
        "host": platform.node(),
        "git_head": _git_head(),
        "elapsed_seconds": time.time() - started_at,
        "body_radius_m": BODY_RADIUS_M,
        "goal_radius_m": GOAL_RADIUS_M,
        "variants": VARIANTS,
    }
    payload = {
        "metadata": metadata,
        "summary": summary,
        "trials": [result.__dict__ for result in results],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    table = _markdown_table(summary)
    (args.output_dir / "matrix-table.md").write_text(table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
