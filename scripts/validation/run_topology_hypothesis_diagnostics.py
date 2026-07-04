#!/usr/bin/env python3
"""Run diagnostic-only topology-hypothesis traces for one policy-search case."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from collections.abc import Collection
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def _first_float(value: Any, default: float) -> float:
    """Return the first finite scalar float from ``value`` or ``default``."""
    if value is None:
        return default
    raw = np.asarray(value, dtype=float).reshape(-1)
    if not raw.size or not np.isfinite(raw[0]):
        return default
    return float(raw[0])


def _resolution(meta: dict[str, Any]) -> float:
    """Return occupancy-grid resolution in meters."""
    value = _first_float(meta.get("resolution"), 0.2)
    return value if value > _EPS else 0.2


def _path_length(path: list[tuple[int, int]], *, resolution: float) -> float:
    """Return path length in meters."""
    if len(path) < 2:
        return 0.0
    return float(
        sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in pairwise(path)) * resolution
    )


def _path_overlap(left: Collection[tuple[int, int]], right: Collection[tuple[int, int]]) -> float:
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
            overlap = _path_overlap(topology_signature, item.topology_signature)
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
                hypothesis_id=f"masked_cell_{blocked_cell[0]}_{blocked_cell[1]}",
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
    distances = [
        min(_point_segment_distance(ped_pos, start, stop) for start, stop in pairwise(world_points))
        for ped_pos in ped_positions
    ]
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
    count = int(_first_float(pedestrians.get("count"), positions.shape[0]))
    count = max(0, min(count, int(positions.shape[0])))
    radius = _first_float(pedestrians.get("radius"), 0.35)
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


def _selection_score_example(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return compact per-hypothesis selection-score evidence for one step."""
    route_corridor = row.get("planner_route_corridor")
    if not isinstance(route_corridor, dict):
        return None
    scored_hypotheses = route_corridor.get("topology_hypotheses")
    if not isinstance(scored_hypotheses, list):
        return None

    example_rows: list[dict[str, Any]] = []
    for item in scored_hypotheses:
        if not isinstance(item, dict) or "score" not in item:
            continue
        example_rows.append(
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "score": item.get("score"),
                "score_rank": item.get("score_rank"),
                "score_margin_to_selected": item.get("score_margin_to_selected"),
                "score_components": item.get("score_components", {}),
                "selection_outcome": item.get("selection_outcome"),
                "rejection_reason": item.get("rejection_reason"),
            }
        )
    if not example_rows:
        return None
    return {
        "step": int(row.get("step", -1)),
        "hypotheses": example_rows,
    }


def _selected_topology_hypothesis_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the selected topology hypothesis diagnostic row for one step."""
    route_corridor = row.get("planner_route_corridor")
    if not isinstance(route_corridor, dict):
        return None

    selected = route_corridor.get("topology_hypothesis")
    if isinstance(selected, dict):
        return selected

    scored_hypotheses = route_corridor.get("topology_hypotheses")
    if not isinstance(scored_hypotheses, list):
        return None
    for item in scored_hypotheses:
        if isinstance(item, dict) and item.get("selection_outcome") == "selected":
            return item
    return None


def _selected_topology_hypothesis_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate selected-hypothesis and near-parity reason counts."""
    selected_counts: Counter[str] = Counter()
    near_parity_reason_counts: Counter[str] = Counter()

    for row in steps:
        selected = _selected_topology_hypothesis_row(row)
        if selected is None:
            continue
        hypothesis_id = selected.get("hypothesis_id")
        hypothesis_key = str(hypothesis_id) if hypothesis_id is not None else "unknown"
        selected_counts[hypothesis_key] += 1
        near_parity_reason = selected.get("near_parity_gate_reason")
        if near_parity_reason is not None:
            near_parity_reason_counts[str(near_parity_reason)] += 1

    return {
        "route_selector_selected_hypothesis_counts": dict(sorted(selected_counts.items())),
        "selected_row_near_parity_gate_reasons": dict(sorted(near_parity_reason_counts.items())),
    }


def _reuse_penalty_diagnostic(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the topology reuse-penalty diagnostic payload for one step."""
    route_corridor = row.get("planner_route_corridor")
    if not isinstance(route_corridor, dict):
        return None
    payload = route_corridor.get("topology_reuse_penalty")
    if not isinstance(payload, dict):
        return None
    return payload


def _reuse_penalty_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate topology reuse-penalty evidence from route-corridor diagnostics."""
    reason_counts: Counter[str] = Counter()
    applied_steps = 0
    eligible_steps = 0
    progress_gate_satisfied_steps = 0
    progress_suppressed_steps = 0
    max_recent_primary_selection_count = 0
    max_recent_progress_m = 0.0
    max_recent_progress_sample_count = 0
    for row in steps:
        reuse_penalty = _reuse_penalty_diagnostic(row)
        if reuse_penalty is None:
            continue
        if bool(reuse_penalty.get("reuse_penalty_applied", False)):
            applied_steps += 1
            reason = str(reuse_penalty.get("reuse_penalty_reason") or "unknown")
            reason_counts[reason] += 1
        if bool(reuse_penalty.get("eligible_near_parity_alternative_exists", False)):
            eligible_steps += 1
        if bool(reuse_penalty.get("primary_route_progress_gate_satisfied", False)):
            progress_gate_satisfied_steps += 1
        if bool(reuse_penalty.get("reuse_penalty_suppressed_by_progress", False)):
            progress_suppressed_steps += 1
        recent_count = reuse_penalty.get("recent_primary_selection_count", 0)
        if isinstance(recent_count, int | float):
            max_recent_primary_selection_count = max(
                max_recent_primary_selection_count,
                int(recent_count),
            )
        recent_progress = reuse_penalty.get("primary_route_recent_progress_m", 0.0)
        if isinstance(recent_progress, int | float):
            max_recent_progress_m = max(max_recent_progress_m, float(recent_progress))
        sample_count = reuse_penalty.get("primary_route_recent_progress_sample_count", 0)
        if isinstance(sample_count, int | float):
            max_recent_progress_sample_count = max(
                max_recent_progress_sample_count,
                int(sample_count),
            )
    return {
        "applied_steps": applied_steps,
        "eligible_near_parity_alternative_steps": eligible_steps,
        "max_recent_primary_selection_count": max_recent_primary_selection_count,
        "progress_gate_satisfied_steps": progress_gate_satisfied_steps,
        "progress_suppressed_steps": progress_suppressed_steps,
        "max_primary_route_recent_progress_m": max_recent_progress_m,
        "max_primary_route_recent_progress_sample_count": max_recent_progress_sample_count,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _terminal_outcome(done_info: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact terminal-outcome evidence for the diagnostic trace."""
    last_step = steps[-1] if steps else {}
    meta = done_info.get("meta") if isinstance(done_info.get("meta"), dict) else {}
    if not meta and isinstance(last_step.get("meta"), dict):
        meta = last_step["meta"]

    success = bool(done_info.get("success", last_step.get("is_success", False)))
    pedestrian_collision = bool(
        meta.get("is_pedestrian_collision", last_step.get("is_pedestrian_collision", False))
    )
    obstacle_collision = bool(
        meta.get("is_obstacle_collision", last_step.get("is_obstacle_collision", False))
    )
    robot_collision = bool(
        meta.get("is_robot_collision", last_step.get("is_robot_collision", False))
    )
    terminated = bool(done_info.get("terminated", last_step.get("terminated", False)))
    truncated = bool(done_info.get("truncated", last_step.get("truncated", False)))

    if success:
        outcome = "success"
    elif pedestrian_collision:
        outcome = "pedestrian_collision"
    elif obstacle_collision:
        outcome = "obstacle_collision"
    elif robot_collision:
        outcome = "robot_collision"
    elif truncated:
        outcome = "truncated"
    elif terminated:
        outcome = "terminated_without_success"
    else:
        outcome = "horizon_exhausted"

    return {
        "outcome": outcome,
        "step": done_info.get("step", last_step.get("step")),
        "terminated": terminated,
        "truncated": truncated,
        "success": success,
        "is_pedestrian_collision": pedestrian_collision,
        "is_obstacle_collision": obstacle_collision,
        "is_robot_collision": robot_collision,
    }


def _route_progress_state_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate planner route-progress terminal reasons from topology decisions."""
    reason_counts: Counter[str] = Counter()
    max_candidate_switch_count = 0
    max_stagnant_steps = 0
    max_progress_delta: float | None = None
    examples: list[dict[str, Any]] = []
    for row in steps:
        route_corridor = row.get("planner_route_corridor")
        if not isinstance(route_corridor, dict):
            continue
        progress = route_corridor.get("topology_route_progress")
        if not isinstance(progress, dict):
            continue
        reason = str(progress.get("terminal_reason", "unknown"))
        reason_counts[reason] += 1
        max_candidate_switch_count = max(
            max_candidate_switch_count,
            int(progress.get("candidate_switch_count", 0) or 0),
        )
        max_stagnant_steps = max(max_stagnant_steps, int(progress.get("stagnant_steps", 0) or 0))
        delta = progress.get("route_progress_delta_m")
        if isinstance(delta, int | float) and np.isfinite(delta):
            max_progress_delta = (
                float(delta)
                if max_progress_delta is None
                else max(float(delta), max_progress_delta)
            )
        if len(examples) < 5 and reason in {"near_parity_churn", "true_stall", "goal_progress"}:
            examples.append(
                {
                    "step": int(row.get("step", -1)),
                    "terminal_reason": reason,
                    "selected_hypothesis_id": progress.get("selected_hypothesis_id"),
                    "previous_selected_hypothesis_id": progress.get(
                        "previous_selected_hypothesis_id"
                    ),
                    "route_progress_delta_m": progress.get("route_progress_delta_m"),
                    "stagnant_steps": progress.get("stagnant_steps"),
                    "candidate_switch_count": progress.get("candidate_switch_count"),
                }
            )
    return {
        "schema_version": "topology_route_progress_summary.v1",
        "terminal_reason_counts": dict(sorted(reason_counts.items())),
        "near_parity_churn_steps": int(reason_counts.get("near_parity_churn", 0)),
        "true_stall_steps": int(reason_counts.get("true_stall", 0)),
        "goal_progress_steps": int(reason_counts.get("goal_progress", 0)),
        "max_candidate_switch_count": int(max_candidate_switch_count),
        "max_stagnant_steps": int(max_stagnant_steps),
        "max_route_progress_delta_m": max_progress_delta,
        "examples": examples,
    }


def _corrective_behavior_summary(
    steps: list[dict[str, Any]],
    *,
    selected_source_counts: Counter[str],
    influence_counts: Counter[str],
    progress_by_rank: dict[str, dict[str, Any]],
    hypothesis_switch_count: int,
    terminal_outcome: dict[str, Any],
    route_progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify whether topology near-parity evidence reached corrective behavior."""
    route_progress = route_progress or _route_progress_state_summary(steps)
    topology_command_steps = int(selected_source_counts.get("topology_hypothesis", 0))
    non_primary_influence_steps = int(
        sum(
            count
            for hypothesis_id, count in influence_counts.items()
            if hypothesis_id != "primary_route"
        )
    )
    progress_values = [
        float(row["progress_delta_m"])
        for row in progress_by_rank.values()
        if row.get("progress_delta_m") is not None
    ]
    max_progress_delta = max(progress_values, default=None)
    positive_route_progress = max_progress_delta is not None and max_progress_delta > _EPS
    terminal_success = bool(terminal_outcome.get("success", False))

    has_corrective_signal = (
        topology_command_steps > 0
        and bool(influence_counts)
        and positive_route_progress
        and terminal_success
    )
    if has_corrective_signal and non_primary_influence_steps > 0:
        decision = "continue"
        rationale = (
            "Topology hypothesis commands influenced the local command stream, at least one "
            "non-primary route was selected, route progress was positive, and the slice succeeded."
        )
    elif has_corrective_signal:
        decision = "revise"
        rationale = (
            "Topology commands influenced a successful progressing slice, but the influence stayed "
            "on the primary route; keep the lane diagnostic and revise toward non-primary corrective "
            "behavior."
        )
    elif topology_command_steps > 0 or bool(influence_counts):
        decision = "revise"
        rationale = (
            "Topology signals reached command arbitration, but route progress or terminal outcome "
            "does not yet support corrective behavior."
        )
    else:
        decision = "stop"
        rationale = (
            "This slice did not show topology command influence, so selection diversity alone is "
            "insufficient corrective-behavior evidence."
        )

    return {
        "selected_route_counts": dict(sorted(influence_counts.items())),
        "selected_source_counts": dict(sorted(selected_source_counts.items())),
        "topology_command_steps": topology_command_steps,
        "non_primary_topology_command_steps": non_primary_influence_steps,
        "hypothesis_switch_count": int(hypothesis_switch_count),
        "route_progress_terminal_reason_counts": route_progress["terminal_reason_counts"],
        "near_parity_churn_steps": int(route_progress["near_parity_churn_steps"]),
        "true_stall_steps": int(route_progress["true_stall_steps"]),
        "max_route_progress_delta_m": max_progress_delta,
        "positive_route_progress": positive_route_progress,
        "terminal_outcome": terminal_outcome,
        "decision": decision,
        "decision_rationale": rationale,
        "claim_boundary": "diagnostic_only_not_benchmark_success",
    }


def _summarize_hypotheses(
    steps: list[dict[str, Any]],
    done_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
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

    influence_examples: list[dict[str, Any]] = []
    influence_counts: Counter[str] = Counter()
    selection_score_examples: list[dict[str, Any]] = []
    previous_hypothesis_id: str | None = None
    hypothesis_switch_count = 0
    for row in steps:
        influence = row.get("topology_command_influence")
        if not isinstance(influence, dict):
            influence = {}
        else:
            hypothesis_id = str(influence.get("selected_hypothesis_id") or "unknown")
            influence_counts[hypothesis_id] += 1
            if previous_hypothesis_id is not None and hypothesis_id != previous_hypothesis_id:
                hypothesis_switch_count += 1
            previous_hypothesis_id = hypothesis_id
            if len(influence_examples) < 10:
                influence_examples.append(
                    {
                        "step": int(row.get("step", -1)),
                        "selected_hypothesis_id": hypothesis_id,
                        "reason": influence.get("reason"),
                        "selected_score": influence.get("selected_score"),
                        "selected_terms": influence.get("selected_terms", {}),
                    }
                )

        if len(selection_score_examples) < 10 and (example := _selection_score_example(row)):
            selection_score_examples.append(example)

    terminal = _terminal_outcome(done_info or {}, steps)
    selected_hypothesis_summary = _selected_topology_hypothesis_summary(steps)
    route_progress = _route_progress_state_summary(steps)
    return {
        "topology_status_counts": dict(sorted(availability_counts.items())),
        "selected_source_counts": dict(sorted(selected_source_counts.items())),
        **selected_hypothesis_summary,
        "hypothesis_progress_by_rank": progress_by_rank,
        "topology_route_progress": route_progress,
        "topology_command_influence_counts": dict(sorted(influence_counts.items())),
        "hypothesis_switch_count": hypothesis_switch_count,
        "topology_command_influence_examples": influence_examples,
        "topology_selection_score_examples": selection_score_examples,
        "topology_reuse_penalty": _reuse_penalty_summary(steps),
        "corrective_behavior": _corrective_behavior_summary(
            steps,
            selected_source_counts=selected_source_counts,
            influence_counts=influence_counts,
            progress_by_rank=progress_by_rank,
            hypothesis_switch_count=hypothesis_switch_count,
            terminal_outcome=terminal,
            route_progress=route_progress,
        ),
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
        "- Route-selector selected hypotheses: "
        f"`{summary['route_selector_selected_hypothesis_counts']}`",
        "- Selected-row near-parity gate reasons: "
        f"`{summary['selected_row_near_parity_gate_reasons']}`",
        f"- Topology command influence counts: `{summary['topology_command_influence_counts']}`",
        f"- Hypothesis switch count: `{summary['hypothesis_switch_count']}`",
        f"- Corrective-behavior decision: `{summary['corrective_behavior']['decision']}`",
        f"- Terminal outcome: `{summary['corrective_behavior']['terminal_outcome']['outcome']}`",
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

    corrective = summary["corrective_behavior"]
    lines.extend(
        [
            "",
            "## Corrective Behavior",
            "",
            f"- Decision: `{corrective['decision']}`",
            f"- Rationale: {corrective['decision_rationale']}",
            f"- Selected routes: `{corrective['selected_route_counts']}`",
            f"- Selected sources: `{corrective['selected_source_counts']}`",
            f"- Topology-command steps: `{corrective['topology_command_steps']}`",
            f"- Non-primary topology-command steps: `{corrective['non_primary_topology_command_steps']}`",
            f"- Max route-progress delta: `{corrective['max_route_progress_delta_m']}`",
            f"- Terminal outcome: `{corrective['terminal_outcome']}`",
            "",
        ]
    )

    if summary["topology_command_influence_examples"]:
        lines.extend(
            [
                "",
                "## Topology Command Influence Examples",
                "",
                "| Step | Hypothesis | Reason | Selected score | Route progress term | Tangent alignment term |",
                "|---:|---|---|---:|---:|---:|",
            ]
        )
        for row in summary["topology_command_influence_examples"]:
            terms = row.get("selected_terms", {})
            if not isinstance(terms, dict):
                terms = {}
            lines.append(
                "| "
                f"{row['step']} | {row['selected_hypothesis_id']} | {row['reason']} | "
                f"{row['selected_score']} | "
                f"{terms.get('corridor_subgoal_route_progress')} | "
                f"{terms.get('corridor_subgoal_tangent_alignment')} |"
            )

    if summary["topology_selection_score_examples"]:
        lines.extend(
            [
                "",
                "## Topology Selection Score Examples",
                "",
                "| Step | Hypothesis | Outcome | Rejection reason | Score | Margin to selected | Length penalty | Static clearance bonus |",
                "|---:|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for example in summary["topology_selection_score_examples"]:
            for item in example.get("hypotheses", []):
                terms = item.get("score_components", {})
                if not isinstance(terms, dict):
                    terms = {}
                lines.append(
                    "| "
                    f"{example['step']} | {item.get('hypothesis_id')} | "
                    f"{item.get('selection_outcome')} | {item.get('rejection_reason')} | "
                    f"{item.get('score')} | {item.get('score_margin_to_selected')} | "
                    f"{terms.get('length_penalty')} | {terms.get('static_clearance_bonus')} |"
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


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915
    """Run the topology-hypothesis diagnostic trace."""
    args = parse_args(argv)
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
                    "topology_command_influence": _json_ready(
                        decision.get("topology_command_influence")
                    ),
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
                try:
                    planner_summary = diagnostics()
                except Exception:
                    planner_summary = {
                        "status": "diagnostics_failed",
                    }
        env.close()

    summary = _summarize_hypotheses(steps, done_info)
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
