#!/usr/bin/env python3
"""Run diagnostic-only topology-hypothesis traces for one policy-search case."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.map_runner import (
    _build_env_config,
    _build_policy,
    _policy_command_to_env_action,
    _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.termination_reason import route_complete_success
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.planner.grid_route import GridRoutePlannerAdapter, GridRoutePlannerConfig
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.policy_search_common import infer_scenario_family
from scripts.validation.run_policy_search_candidate import (
    _DEFAULT_FUNNEL,
    _DEFAULT_REGISTRY,
    _effective_candidate_runtime_for_scenario,
    _load_stage_scenarios,
    _load_yaml,
    _resolve_path,
    load_candidate_definition,
)
from scripts.validation.run_policy_search_step_diagnostics import (
    _json_ready,
    _obs_min_robot_ped_distance,
    _scenario_id,
    _seed_list,
    _select_scenario,
    _sim_min_robot_ped_distance,
)

_EPS = 1e-9
_INSUFFICIENT_HYPOTHESES_EXIT = 2


@dataclass(frozen=True)
class _RouteHypothesisPath:
    """One candidate route hypothesis through the local occupancy grid."""

    hypothesis_id: str
    path: list[tuple[int, int]]
    clearance_map: np.ndarray | None
    topology_signature: frozenset[tuple[int, int]]
    blocked_cell: tuple[int, int] | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="hybrid_rule_v3_waypoint2_route_lookahead8")
    parser.add_argument("--stage", default="full_matrix")
    parser.add_argument("--candidate-registry", type=Path, default=_DEFAULT_REGISTRY)
    parser.add_argument("--funnel-config", type=Path, default=_DEFAULT_FUNNEL)
    parser.add_argument(
        "--scenario-name",
        default="classic_realworld_double_bottleneck_high",
    )
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-hypotheses", type=int, default=3)
    parser.add_argument("--min-hypotheses", type=int, default=2)
    parser.add_argument("--block-radius-cells", type=int, default=3)
    parser.add_argument("--block-stride-cells", type=int, default=8)
    return parser.parse_args()


def _resolution(meta: dict[str, Any]) -> float:
    """Return occupancy-grid resolution in meters."""
    raw = np.asarray(meta.get("resolution", [0.2]), dtype=float).reshape(-1)
    value = float(raw[0]) if raw.size else 0.2
    return value if np.isfinite(value) and value > _EPS else 0.2


def _path_length(path: list[tuple[int, int]], *, resolution: float) -> float:
    """Return path length in meters."""
    if len(path) < 2:
        return 0.0
    return float(
        sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in pairwise(path)) * resolution
    )


def _path_overlap(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> float:
    """Return Jaccard overlap between two grid-cell paths."""
    left_cells = set(left)
    right_cells = set(right)
    union = left_cells | right_cells
    if not union:
        return 1.0
    return float(len(left_cells & right_cells) / len(union))


def _distinct_path(
    path: list[tuple[int, int]],
    topology_signature: frozenset[tuple[int, int]],
    accepted: list[_RouteHypothesisPath],
    *,
    max_overlap: float = 0.88,
    max_signature_overlap: float = 0.55,
) -> bool:
    """Return whether ``path`` is meaningfully distinct from accepted hypotheses."""
    for item in accepted:
        if _path_overlap(path, item.path) > max_overlap:
            return False
        if topology_signature and item.topology_signature:
            overlap = _path_overlap(list(topology_signature), list(item.topology_signature))
            if overlap > max_signature_overlap:
                return False
    return True


def _topology_signature(
    path: list[tuple[int, int]],
    blocked: np.ndarray,
    clearance_map: np.ndarray,
    *,
    clearance_threshold_cells: int,
) -> frozenset[tuple[int, int]]:
    """Return low-clearance path cells used as a compact corridor signature."""
    choke_cells: set[tuple[int, int]] = set()
    rows, cols = blocked.shape
    for row, col in path:
        up_blocked = row <= 0 or bool(blocked[row - 1, col])
        down_blocked = row >= rows - 1 or bool(blocked[row + 1, col])
        left_blocked = col <= 0 or bool(blocked[row, col - 1])
        right_blocked = col >= cols - 1 or bool(blocked[row, col + 1])
        if (up_blocked and down_blocked) or (left_blocked and right_blocked):
            choke_cells.add((row, col))
    if choke_cells:
        return frozenset(choke_cells)

    threshold = max(int(clearance_threshold_cells), 1)
    signature = {
        cell
        for cell in path
        if np.isfinite(float(clearance_map[cell])) and float(clearance_map[cell]) <= threshold
    }
    return frozenset(signature)


def _block_path_cell(
    blocked: np.ndarray,
    cell: tuple[int, int],
    *,
    radius: int,
    protected: set[tuple[int, int]],
) -> np.ndarray:
    """Return a blocked-grid copy with a local path cell masked out."""
    updated = blocked.copy()
    row, col = cell
    radius = max(int(radius), 0)
    for rr in range(max(0, row - radius), min(updated.shape[0], row + radius + 1)):
        for cc in range(max(0, col - radius), min(updated.shape[1], col + radius + 1)):
            if (rr, cc) not in protected:
                updated[rr, cc] = True
    return updated


def _find_alternative_paths(
    adapter: GridRoutePlannerAdapter,
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    max_hypotheses: int,
    block_radius_cells: int,
    block_stride_cells: int,
) -> list[_RouteHypothesisPath]:
    """Find diagnostic route alternatives by masking cells along the shortest path."""
    base_clearance = adapter._compute_clearance_map(blocked)
    base_path = adapter._astar(blocked, start, goal, clearance_map=base_clearance)
    if len(base_path) < 2:
        return []
    base_signature = _topology_signature(
        base_path,
        blocked,
        base_clearance,
        clearance_threshold_cells=max(block_radius_cells, 1),
    )

    hypotheses = [
        _RouteHypothesisPath(
            hypothesis_id="primary_route",
            path=base_path,
            clearance_map=base_clearance,
            topology_signature=base_signature,
        )
    ]
    protected = set(base_path[: max(2, len(base_path) // 12)])
    protected.update(base_path[-max(2, len(base_path) // 12) :])
    stride = max(int(block_stride_cells), 1)
    for idx in range(max(2, stride), max(len(base_path) - 2, 0), stride):
        if len(hypotheses) >= max(int(max_hypotheses), 1):
            break
        blocked_cell = base_path[idx]
        if blocked_cell in protected:
            continue
        perturbed = _block_path_cell(
            blocked,
            blocked_cell,
            radius=block_radius_cells,
            protected=protected,
        )
        clearance = adapter._compute_clearance_map(perturbed)
        path = adapter._astar(perturbed, start, goal, clearance_map=clearance)
        topology_signature = _topology_signature(
            path,
            blocked,
            base_clearance,
            clearance_threshold_cells=max(block_radius_cells, 1),
        )
        if len(path) < 2 or not _distinct_path(path, topology_signature, hypotheses):
            continue
        hypotheses.append(
            _RouteHypothesisPath(
                hypothesis_id=f"blocked_cell_{blocked_cell[0]}_{blocked_cell[1]}",
                path=path,
                clearance_map=clearance,
                topology_signature=topology_signature,
                blocked_cell=blocked_cell,
            )
        )
    return hypotheses


def _point_segment_distance(point: np.ndarray, start: np.ndarray, stop: np.ndarray) -> float:
    """Return 2-D distance from ``point`` to a segment."""
    segment = stop - start
    denom = float(np.dot(segment, segment))
    if denom <= _EPS:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    closest = start + t * segment
    return float(np.linalg.norm(point - closest))


def _path_dynamic_clearance(
    world_points: list[np.ndarray],
    ped_positions: np.ndarray,
    *,
    ped_radius: float,
) -> float | None:
    """Return nearest pedestrian clearance to a route polyline."""
    if len(world_points) < 2 or ped_positions.size == 0:
        return None
    distances: list[float] = []
    for ped_pos in ped_positions:
        distances.append(
            min(
                _point_segment_distance(np.asarray(ped_pos, dtype=float), start, stop)
                for start, stop in pairwise(world_points)
            )
        )
    return float(min(distances) - float(ped_radius)) if distances else None


def _extract_pedestrians(
    adapter: GridRoutePlannerAdapter,
    obs: dict[str, Any],
) -> tuple[np.ndarray, float]:
    """Return count-aware pedestrian positions and shared radius from an observation."""
    try:
        _robot_state, _goal_state, pedestrians = adapter._socnav_fields(obs)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        pedestrians = obs.get("pedestrians", {})
    if not isinstance(pedestrians, dict):
        pedestrians = {}
    positions = np.asarray(pedestrians.get("positions", []), dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 2) if positions.size % 2 == 0 else np.zeros((0, 2))
    if positions.ndim != 2:
        positions = np.zeros((0, 2), dtype=float)
    count_raw = np.asarray(pedestrians.get("count", [positions.shape[0]]), dtype=float).reshape(-1)
    count = int(count_raw[0]) if count_raw.size else int(positions.shape[0])
    count = max(0, min(count, int(positions.shape[0])))
    radius_raw = np.asarray(pedestrians.get("radius", [0.35]), dtype=float).reshape(-1)
    radius = float(radius_raw[0]) if radius_raw.size else 0.35
    if not np.isfinite(radius) or radius <= _EPS:
        radius = 0.35
    return positions[:count, :2], radius


def _side_label(path: list[np.ndarray], primary: list[np.ndarray]) -> str:
    """Label a hypothesis by its lateral side relative to the primary route chord."""
    if len(path) < 2 or len(primary) < 2:
        return "unknown"
    start = primary[0]
    stop = primary[-1]
    chord = stop - start
    if float(np.linalg.norm(chord)) <= _EPS:
        return "unknown"
    midpoint = path[len(path) // 2] - start
    cross = float(chord[0] * midpoint[1] - chord[1] * midpoint[0])
    if abs(cross) < 0.25:
        return "center"
    return "left" if cross > 0.0 else "right"


def _route_hypotheses_for_observation(  # noqa: C901
    adapter: GridRoutePlannerAdapter,
    obs: dict[str, Any],
    *,
    max_hypotheses: int,
    block_radius_cells: int,
    block_stride_cells: int,
) -> list[dict[str, Any]]:
    """Return JSON-ready topology hypotheses for the current observation."""
    try:
        robot_pos, heading, goal, radius = adapter._extract_state(obs)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return []
    payload = adapter._extract_grid_payload(obs)
    if payload is None:
        return []
    grid, meta = payload
    blocked = adapter._blocked_grid(grid, meta, radius)
    if blocked is None:
        return []
    start_rc = adapter._world_to_grid(robot_pos, meta, blocked.shape)
    goal_rc = adapter._world_to_grid(goal, meta, blocked.shape)
    if start_rc is None or goal_rc is None:
        return []
    start = adapter._nearest_free(blocked, start_rc, int(adapter.config.clearance_search_cells))
    stop = adapter._nearest_free(blocked, goal_rc, int(adapter.config.clearance_search_cells))
    if start is None or stop is None:
        return []

    route_paths = _find_alternative_paths(
        adapter,
        blocked,
        start,
        stop,
        max_hypotheses=max_hypotheses,
        block_radius_cells=block_radius_cells,
        block_stride_cells=block_stride_cells,
    )
    if not route_paths:
        return []

    resolution = _resolution(meta)
    ped_positions, ped_radius = _extract_pedestrians(adapter, obs)
    primary_world = [adapter._grid_to_world(cell, meta) for cell in route_paths[0].path]
    hypotheses: list[dict[str, Any]] = []
    for rank, route_path in enumerate(route_paths):
        world_points = [adapter._grid_to_world(cell, meta) for cell in route_path.path]
        clearance_map = route_path.clearance_map
        static_values: list[float] = []
        if clearance_map is not None:
            for cell in route_path.path:
                value = float(clearance_map[cell]) * resolution
                if np.isfinite(value):
                    static_values.append(value)
        waypoint_idx = min(
            max(int(adapter.config.waypoint_lookahead_cells), 1), len(world_points) - 1
        )
        tangent_heading = None
        if len(world_points) >= 2:
            tangent = (
                world_points[min(waypoint_idx + 1, len(world_points) - 1)]
                - world_points[waypoint_idx]
            )
            if float(np.linalg.norm(tangent)) > _EPS:
                tangent_heading = float(np.arctan2(tangent[1], tangent[0]))
        hypotheses.append(
            {
                "hypothesis_id": route_path.hypothesis_id,
                "rank": int(rank),
                "corridor_name": f"{_side_label(world_points, primary_world)}_corridor_{rank}",
                "blocked_cell": list(route_path.blocked_cell) if route_path.blocked_cell else None,
                "path_cell_count": len(route_path.path),
                "route_remaining_distance_m": _path_length(route_path.path, resolution=resolution),
                "route_progress_reference_m": -_path_length(route_path.path, resolution=resolution),
                "static_clearance_min_m": min(static_values) if static_values else None,
                "static_clearance_mean_m": float(np.mean(static_values)) if static_values else None,
                "static_clearance_waypoint_m": (
                    static_values[min(waypoint_idx, len(static_values) - 1)]
                    if static_values
                    else None
                ),
                "dynamic_clearance_min_m": _path_dynamic_clearance(
                    world_points,
                    ped_positions,
                    ped_radius=ped_radius,
                ),
                "waypoint_world": [
                    float(world_points[waypoint_idx][0]),
                    float(world_points[waypoint_idx][1]),
                ],
                "goal_world": [float(world_points[-1][0]), float(world_points[-1][1])],
                "route_tangent_heading": tangent_heading,
                "route_heading_error": None
                if tangent_heading is None
                else float((tangent_heading - float(heading) + np.pi) % (2.0 * np.pi) - np.pi),
            }
        )
    return hypotheses


def _summarize_hypotheses(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate hypothesis availability, route progress, and selected sources."""
    selected_source_counts = Counter(
        str(row.get("selected_local_command_source") or "unknown") for row in steps
    )
    availability_counts = Counter(str(row.get("topology_status", "unknown")) for row in steps)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        for hypothesis in row.get("topology_hypotheses", []):
            if isinstance(hypothesis, dict):
                grouped[int(hypothesis.get("rank", 0))].append(hypothesis)

    progress_by_rank: dict[str, dict[str, Any]] = {}
    for rank, rows in sorted(grouped.items()):
        first = rows[0]
        last = rows[-1]
        first_remaining = first.get("route_remaining_distance_m")
        last_remaining = last.get("route_remaining_distance_m")
        progress = (
            None
            if first_remaining is None or last_remaining is None
            else float(first_remaining) - float(last_remaining)
        )
        progress_by_rank[str(rank)] = {
            "samples": len(rows),
            "first_corridor_name": first.get("corridor_name"),
            "last_corridor_name": last.get("corridor_name"),
            "first_remaining_distance_m": first_remaining,
            "last_remaining_distance_m": last_remaining,
            "progress_delta_m": progress,
            "min_static_clearance_m": min(
                (
                    float(item["static_clearance_min_m"])
                    for item in rows
                    if item.get("static_clearance_min_m") is not None
                ),
                default=None,
            ),
            "min_dynamic_clearance_m": min(
                (
                    float(item["dynamic_clearance_min_m"])
                    for item in rows
                    if item.get("dynamic_clearance_min_m") is not None
                ),
                default=None,
            ),
        }

    return {
        "topology_status_counts": dict(sorted(availability_counts.items())),
        "selected_source_counts": dict(sorted(selected_source_counts.items())),
        "hypothesis_progress_by_rank": progress_by_rank,
    }


def _report_lines(payload: dict[str, Any], trace_path: Path) -> list[str]:
    """Build a Markdown diagnostic report."""
    summary = payload["summary"]
    lines = [
        f"# Topology-Hypothesis Diagnostics: {payload['candidate']}",
        "",
        "Diagnostic-only evidence; this is not benchmark success evidence.",
        "",
        f"- Scenario: `{payload['scenario_id']}`",
        f"- Seed: `{payload['seed']}`",
        f"- Horizon: `{payload['horizon']}`",
        f"- Trace JSON: `{trace_path}`",
        f"- Topology status counts: `{summary['topology_status_counts']}`",
        f"- Selected local command sources: `{summary['selected_source_counts']}`",
        "",
        "## Hypothesis Progress",
        "",
        "| Rank | First corridor | Last corridor | Samples | Start remaining | End remaining | Progress delta | Min static clearance | Min dynamic clearance |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in summary["hypothesis_progress_by_rank"].items():
        lines.append(
            "| "
            f"{rank} | {row['first_corridor_name']} | {row['last_corridor_name']} | "
            f"{row['samples']} | {row['first_remaining_distance_m']} | "
            f"{row['last_remaining_distance_m']} | {row['progress_delta_m']} | "
            f"{row['min_static_clearance_m']} | {row['min_dynamic_clearance_m']} |"
        )

    lines.extend(
        [
            "",
            "## Step Summary",
            "",
            "| Step | Status | Hypotheses | Selected source | Decision mode | Goal distance |",
            "|---:|---|---:|---|---|---:|",
        ]
    )
    for row in payload["steps"]:
        lines.append(
            "| "
            f"{row['step']} | {row['topology_status']} | {row['hypothesis_count']} | "
            f"{row['selected_local_command_source']} | {row['planner_mode']} | "
            f"{row['goal_distance']} |"
        )
    return lines


def main() -> int:  # noqa: C901, PLR0915
    """Run the topology-hypothesis diagnostic trace."""
    args = parse_args()
    funnel = _load_yaml(args.funnel_config)
    stages = funnel.get("stages")
    if not isinstance(stages, dict) or args.stage not in stages:
        raise KeyError(f"Unknown stage '{args.stage}' in {args.funnel_config}")
    stage_cfg = stages[args.stage]
    if not isinstance(stage_cfg, dict):
        raise TypeError(f"Stage config must be a mapping: {args.stage}")

    stage_matrix = _resolve_path(args.funnel_config.parent, stage_cfg.get("scenario_matrix"))
    if stage_matrix is None:
        raise ValueError(f"Stage '{args.stage}' is missing a resolvable scenario_matrix")
    seed_manifest = _resolve_path(args.funnel_config.parent, stage_cfg.get("seed_manifest"))

    entry, candidate_payload, algo_cfg, config_path = load_candidate_definition(
        args.candidate_registry,
        args.candidate,
    )
    algo = candidate_payload.get("algo") or entry.get("algo")
    if not isinstance(algo, str) or not algo.strip():
        raise ValueError(f"Candidate '{args.candidate}' is missing a valid algo field")

    loaded = _load_stage_scenarios(stage_matrix, seed_manifest)
    scenarios = load_scenarios(loaded) if isinstance(loaded, Path) else [dict(s) for s in loaded]
    scenario = _select_scenario(scenarios, args)
    family = infer_scenario_family(scenario)
    algo, effective_cfg = _effective_candidate_runtime_for_scenario(
        candidate_payload,
        algo_cfg,
        scenario,
        default_algo=algo.strip().lower(),
        config_anchor=config_path.parent,
    )
    scenario_seed_list = _seed_list(scenario)
    seed = (
        int(args.seed) if args.seed is not None else int(scenario_seed_list[int(args.seed_index)])
    )
    horizon = int(args.horizon or stage_cfg.get("horizon", 0) or 300)

    output_dir = args.output_dir or (
        Path("output")
        / "diagnostics"
        / "topology_hypotheses"
        / args.candidate
        / args.stage
        / f"{_scenario_id(scenario)}_seed{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = _scenario_with_episode_seed_defaults(scenario, seed=seed)
    env_config = _build_env_config(scenario, scenario_path=stage_matrix)
    policy_fn, algo_meta = _build_policy(
        algo,
        dict(effective_cfg),
        robot_kinematics="differential_drive",
    )
    planner_adapter = getattr(policy_fn, "_planner_adapter", None)
    planner_reset = getattr(policy_fn, "_planner_reset", None)
    planner_bind_env = getattr(policy_fn, "_planner_bind_env", None)
    planner_native_action = getattr(policy_fn, "_planner_native_env_action", False)

    route_adapter = GridRoutePlannerAdapter(
        config=GridRoutePlannerConfig(
            waypoint_lookahead_cells=int(
                effective_cfg.get("route_guide_waypoint_lookahead_cells", 8)
            ),
            obstacle_inflation_cells=int(
                effective_cfg.get("route_guide_obstacle_inflation_cells", 3)
            ),
            clearance_penalty_weight=float(
                effective_cfg.get("route_guide_clearance_penalty_weight", 0.5)
            ),
        )
    )

    env = make_robot_env(config=env_config, seed=seed, debug=False)
    steps: list[dict[str, Any]] = []
    done_info: dict[str, Any] = {}
    try:
        obs, _ = env.reset(seed=seed)
        if callable(planner_bind_env):
            planner_bind_env(env)
        if callable(planner_reset):
            planner_reset(seed=seed)

        for step_idx in range(horizon):
            robot_pos = np.array(env.simulator.robot_pos[0], dtype=float, copy=True)
            goal_pos = np.array(env.simulator.goal_pos[0], dtype=float, copy=True)
            goal_distance = float(np.linalg.norm(goal_pos - robot_pos))
            min_robot_ped_dist = _sim_min_robot_ped_distance(env)
            if min_robot_ped_dist is None:
                min_robot_ped_dist = _obs_min_robot_ped_distance(obs)
            topology_hypotheses = _route_hypotheses_for_observation(
                route_adapter,
                obs,
                max_hypotheses=int(args.max_hypotheses),
                block_radius_cells=int(args.block_radius_cells),
                block_stride_cells=int(args.block_stride_cells),
            )

            policy_command = policy_fn(obs)
            step_is_native = getattr(policy_fn, "_last_step_native", planner_native_action)
            env_action = (
                np.asarray(policy_command, dtype=np.float32)
                if step_is_native
                else _policy_command_to_env_action(
                    env=env,
                    config=env_config,
                    command=policy_command,
                )
            )
            planner_decision = None
            if planner_adapter is not None:
                last_decision = getattr(planner_adapter, "last_decision", None)
                if callable(last_decision):
                    planner_decision = last_decision()
            decision = planner_decision if isinstance(planner_decision, dict) else {}

            obs, reward, terminated, truncated, info = env.step(env_action)
            meta = info.get("meta", {}) if isinstance(info, dict) else {}
            is_success = route_complete_success(info if isinstance(info, dict) else {})
            topology_status = (
                "ok"
                if len(topology_hypotheses) >= int(args.min_hypotheses)
                else "insufficient_hypotheses"
            )
            steps.append(
                {
                    "step": int(step_idx),
                    "topology_status": topology_status,
                    "hypothesis_count": len(topology_hypotheses),
                    "topology_hypotheses": _json_ready(topology_hypotheses),
                    "selected_local_command_source": decision.get("selected_source"),
                    "planner_mode": decision.get("planner_mode"),
                    "planner_route_corridor": _json_ready(decision.get("route_corridor")),
                    "policy_command": _json_ready(policy_command),
                    "env_action": _json_ready(env_action),
                    "reward": float(reward),
                    "goal_distance": goal_distance,
                    "min_robot_ped_distance": min_robot_ped_dist,
                    "meta": _json_ready(meta),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "is_success": bool(is_success),
                    "is_pedestrian_collision": bool(meta.get("is_pedestrian_collision", False)),
                    "is_obstacle_collision": bool(meta.get("is_obstacle_collision", False)),
                    "is_robot_collision": bool(meta.get("is_robot_collision", False)),
                }
            )
            if terminated or truncated or is_success:
                done_info = {
                    "step": int(step_idx),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success": bool(is_success),
                    "meta": _json_ready(meta),
                    "family": family,
                }
                break
    finally:
        planner_summary = None
        if planner_adapter is not None:
            diagnostics = getattr(planner_adapter, "diagnostics", None)
            if callable(diagnostics):
                planner_summary = diagnostics()
        env.close()

    summary = _summarize_hypotheses(steps)
    has_sufficient_hypotheses = any(row.get("topology_status") == "ok" for row in steps)
    diagnostic_status = "diagnostic_complete" if has_sufficient_hypotheses else "not_available"
    payload = {
        "diagnostic_kind": "topology_hypothesis_trace",
        "diagnostic_status": diagnostic_status,
        "claim_boundary": "diagnostic_only_not_benchmark_success",
        "candidate": args.candidate,
        "stage": args.stage,
        "scenario_id": _scenario_id(scenario),
        "family": family,
        "seed": seed,
        "horizon": horizon,
        "algo": algo,
        "algo_config": _json_ready(effective_cfg),
        "algorithm_metadata": _json_ready(algo_meta),
        "planner_summary": _json_ready(planner_summary),
        "done_info": _json_ready(done_info),
        "summary": _json_ready(summary),
        "steps": steps,
    }

    trace_path = output_dir / "topology_hypotheses.json"
    trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = output_dir / "topology_hypotheses.md"
    report_path.write_text("\n".join(_report_lines(payload, trace_path)) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "trace": str(trace_path),
                "report": str(report_path),
                "scenario_id": payload["scenario_id"],
                "seed": seed,
                "diagnostic_status": diagnostic_status,
                "topology_status_counts": summary["topology_status_counts"],
                "selected_source_counts": summary["selected_source_counts"],
            },
            indent=2,
        )
    )
    return 0 if diagnostic_status == "diagnostic_complete" else _INSUFFICIENT_HYPOTHESES_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
