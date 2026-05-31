"""Diagnostic topology-hypothesis wrapper for the hybrid-rule local planner."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from robot_sf.planner.grid_route import GridRoutePlannerAdapter, build_grid_route_config
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
    build_hybrid_rule_local_planner_config,
)

_EPS = 1e-9
_TOPOLOGY_KEYS = {
    "diagnostic_only",
    "claim_boundary",
    "min_hypotheses",
    "max_hypotheses",
    "block_radius_cells",
    "block_stride_cells",
    "max_path_overlap",
    "length_weight",
    "static_clearance_weight",
    "fail_closed_on_missing_inputs",
    "fail_closed_on_insufficient_hypotheses",
    "route_hypothesis",
}


@dataclass(frozen=True)
class TopologyGuidedLocalPolicyConfig:
    """Configuration for the diagnostic topology-guided local policy."""

    hybrid_rule: HybridRuleLocalPlannerConfig
    route_hypothesis: Any
    diagnostic_only: bool = True
    claim_boundary: str = "diagnostic_only"
    min_hypotheses: int = 2
    max_hypotheses: int = 2
    block_radius_cells: int = 3
    block_stride_cells: int = 8
    max_path_overlap: float = 0.88
    length_weight: float = 1.0
    static_clearance_weight: float = 0.6
    fail_closed_on_missing_inputs: bool = True
    fail_closed_on_insufficient_hypotheses: bool = False


@dataclass(frozen=True)
class _RouteHypothesis:
    """One masked-route topology hypothesis."""

    hypothesis_id: str
    path: list[tuple[int, int]]
    clearance_map: np.ndarray | None
    blocked_cell: tuple[int, int] | None = None


def _path_overlap(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> float:
    """Return Jaccard overlap between two grid-cell paths."""
    left_cells = set(left)
    right_cells = set(right)
    union = left_cells | right_cells
    if not union:
        return 1.0
    return float(len(left_cells & right_cells) / len(union))


def _path_length(path: list[tuple[int, int]], *, resolution: float) -> float:
    """Return route length in metres."""
    if len(path) < 2:
        return 0.0
    return float(
        sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in pairwise(path)) * resolution
    )


def _first_float(value: Any, default: float) -> float:
    """Return the first finite float from a scalar-like payload."""
    try:
        raw = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return default
    if not raw.size or not np.isfinite(raw[0]):
        return default
    return float(raw[0])


def _block_path_cell(
    blocked: np.ndarray,
    cell: tuple[int, int],
    *,
    radius: int,
    protected: set[tuple[int, int]],
) -> np.ndarray:
    """Return a blocked-grid copy with one local route patch masked."""
    updated = blocked.copy()
    row, col = cell
    radius = max(int(radius), 0)
    for rr in range(max(0, row - radius), min(updated.shape[0], row + radius + 1)):
        for cc in range(max(0, col - radius), min(updated.shape[1], col + radius + 1)):
            if (rr, cc) not in protected:
                updated[rr, cc] = True
    return updated


def _static_clearance_summary(
    path: list[tuple[int, int]],
    clearance_map: np.ndarray | None,
    *,
    resolution: float,
) -> dict[str, float | None]:
    """Summarize static clearance along one route hypothesis.

    Returns:
        dict[str, float | None]: Minimum and mean static clearance in metres.
    """
    if clearance_map is None or not path:
        return {"static_clearance_min_m": None, "static_clearance_mean_m": None}
    values = [
        float(clearance_map[cell]) * resolution
        for cell in path
        if np.isfinite(float(clearance_map[cell]))
    ]
    if not values:
        return {"static_clearance_min_m": None, "static_clearance_mean_m": None}
    return {
        "static_clearance_min_m": float(min(values)),
        "static_clearance_mean_m": float(np.mean(values)),
    }


def build_topology_guided_local_policy_config(
    cfg: dict[str, Any] | None,
) -> TopologyGuidedLocalPolicyConfig:
    """Build the topology-guided local policy config from a YAML-style mapping.

    Returns:
        Parsed topology-guided local policy configuration.
    """
    raw = dict(cfg or {}) if isinstance(cfg, dict) else {}
    hybrid_payload = {key: value for key, value in raw.items() if key not in _TOPOLOGY_KEYS}
    route_payload = deepcopy(raw.get("route_hypothesis") or {})
    if not isinstance(route_payload, dict):
        route_payload = {}
    route_payload.setdefault(
        "waypoint_lookahead_cells",
        raw.get("route_guide_waypoint_lookahead_cells", 8),
    )
    route_payload.setdefault(
        "obstacle_inflation_cells",
        raw.get("route_guide_obstacle_inflation_cells", 3),
    )
    route_payload.setdefault(
        "clearance_penalty_weight",
        raw.get("route_guide_clearance_penalty_weight", 0.5),
    )
    return TopologyGuidedLocalPolicyConfig(
        hybrid_rule=build_hybrid_rule_local_planner_config(hybrid_payload),
        route_hypothesis=build_grid_route_config(route_payload),
        diagnostic_only=bool(raw.get("diagnostic_only", True)),
        claim_boundary=str(raw.get("claim_boundary", "diagnostic_only")),
        min_hypotheses=int(raw.get("min_hypotheses", 2)),
        max_hypotheses=int(raw.get("max_hypotheses", 2)),
        block_radius_cells=int(raw.get("block_radius_cells", 3)),
        block_stride_cells=int(raw.get("block_stride_cells", 8)),
        max_path_overlap=float(raw.get("max_path_overlap", 0.88)),
        length_weight=float(raw.get("length_weight", 1.0)),
        static_clearance_weight=float(raw.get("static_clearance_weight", 0.6)),
        fail_closed_on_missing_inputs=bool(raw.get("fail_closed_on_missing_inputs", True)),
        fail_closed_on_insufficient_hypotheses=bool(
            raw.get("fail_closed_on_insufficient_hypotheses", False)
        ),
    )


class TopologyGuidedHybridRulePlannerAdapter(HybridRuleLocalPlannerAdapter):
    """Hybrid-rule planner that selects among diagnostic topology hypotheses."""

    def __init__(self, config: TopologyGuidedLocalPolicyConfig | None = None) -> None:
        """Initialize the wrapped hybrid-rule scorer and route-hypothesis generator."""
        self.topology_config = config or build_topology_guided_local_policy_config({})
        super().__init__(self.topology_config.hybrid_rule)
        self._route_hypothesis = GridRoutePlannerAdapter(self.topology_config.route_hypothesis)
        self._topology_status_counts: Counter[str] = Counter()
        self._selected_hypothesis_counts: Counter[str] = Counter()
        self._last_topology_decision: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None) -> None:
        """Reset base planner state and topology-hypothesis diagnostics."""
        super().reset(seed=seed)
        self._topology_status_counts = Counter()
        self._selected_hypothesis_counts = Counter()
        self._last_topology_decision = None

    def _alternative_paths(
        self,
        blocked: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[_RouteHypothesis]:
        """Return the primary route plus distinct masked-route alternatives."""
        clearance_map = self._route_hypothesis._compute_clearance_map(blocked)
        base_path = self._route_hypothesis._astar(blocked, start, goal, clearance_map=clearance_map)
        if len(base_path) < 2:
            return []
        hypotheses = [
            _RouteHypothesis(
                hypothesis_id="primary_route",
                path=base_path,
                clearance_map=clearance_map,
            )
        ]
        protected = set(base_path[: max(2, len(base_path) // 12)])
        protected.update(base_path[-max(2, len(base_path) // 12) :])
        stride = max(int(self.topology_config.block_stride_cells), 1)
        for idx in range(max(2, stride), max(len(base_path) - 2, 0), stride):
            if len(hypotheses) >= max(int(self.topology_config.max_hypotheses), 1):
                break
            blocked_cell = base_path[idx]
            if blocked_cell in protected:
                continue
            perturbed = _block_path_cell(
                blocked,
                blocked_cell,
                radius=int(self.topology_config.block_radius_cells),
                protected=protected,
            )
            alt_clearance = self._route_hypothesis._compute_clearance_map(perturbed)
            path = self._route_hypothesis._astar(
                perturbed,
                start,
                goal,
                clearance_map=alt_clearance,
            )
            if len(path) < 2:
                continue
            if any(
                _path_overlap(path, item.path) > float(self.topology_config.max_path_overlap)
                for item in hypotheses
            ):
                continue
            hypotheses.append(
                _RouteHypothesis(
                    hypothesis_id=f"masked_cell_{blocked_cell[0]}_{blocked_cell[1]}",
                    path=path,
                    clearance_map=alt_clearance,
                    blocked_cell=blocked_cell,
                )
            )
        return hypotheses

    def _hypotheses_for_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return selectable route-corridor hypotheses for the current observation."""
        try:
            robot_pos, heading, goal, radius = self._route_hypothesis._extract_state(observation)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return {"status": "not_available", "reason": "missing_robot_or_goal_state"}
        payload = self._route_hypothesis._extract_grid_payload(observation)
        if payload is None:
            return {"status": "not_available", "reason": "missing_occupancy_grid"}
        grid, meta = payload
        blocked = self._route_hypothesis._blocked_grid(grid, meta, radius)
        if blocked is None:
            return {"status": "not_available", "reason": "missing_static_obstacle_channel"}
        start_rc = self._route_hypothesis._world_to_grid(robot_pos, meta, blocked.shape)
        goal_rc = self._route_hypothesis._world_to_grid(goal, meta, blocked.shape)
        if start_rc is None or goal_rc is None:
            return {"status": "not_available", "reason": "route_endpoint_outside_grid"}
        start = self._route_hypothesis._nearest_free(
            blocked,
            start_rc,
            int(self.topology_config.route_hypothesis.clearance_search_cells),
        )
        stop = self._route_hypothesis._nearest_free(
            blocked,
            goal_rc,
            int(self.topology_config.route_hypothesis.clearance_search_cells),
        )
        if start is None or stop is None:
            return {"status": "not_available", "reason": "no_free_route_endpoint"}

        route_paths = self._alternative_paths(blocked, start, stop)
        if len(route_paths) < int(self.topology_config.min_hypotheses):
            return {
                "status": "insufficient_hypotheses",
                "reason": "fewer_than_min_distinct_routes",
                "hypothesis_count": len(route_paths),
                "min_hypotheses": int(self.topology_config.min_hypotheses),
            }

        resolution = _first_float(meta.get("resolution"), 0.2)
        hypotheses: list[dict[str, Any]] = []
        for rank, route_path in enumerate(route_paths[: int(self.topology_config.max_hypotheses)]):
            geometry = self._route_hypothesis._route_geometry_from_path(
                path=route_path.path,
                clearance_map=route_path.clearance_map,
                meta=meta,
                robot_pos=robot_pos,
                heading=heading,
            )
            clearance = _static_clearance_summary(
                route_path.path,
                route_path.clearance_map,
                resolution=resolution,
            )
            route_remaining = _path_length(route_path.path, resolution=resolution)
            static_min = clearance["static_clearance_min_m"]
            score = -float(self.topology_config.length_weight) * route_remaining
            if static_min is not None:
                score += float(self.topology_config.static_clearance_weight) * float(static_min)
            hypotheses.append(
                {
                    "hypothesis_id": route_path.hypothesis_id,
                    "rank": int(rank),
                    "blocked_cell": list(route_path.blocked_cell)
                    if route_path.blocked_cell is not None
                    else None,
                    "path_cell_count": len(route_path.path),
                    "score": float(score),
                    "route_remaining_distance_m": float(route_remaining),
                    **clearance,
                    "route_corridor": geometry,
                }
            )
        selected = max(hypotheses, key=lambda item: float(item["score"]))
        return {
            "status": "ok",
            "reason": "selected_scored_route_hypothesis",
            "hypothesis_count": len(hypotheses),
            "selected_hypothesis_id": selected["hypothesis_id"],
            "selected_rank": int(selected["rank"]),
            "hypotheses": hypotheses,
        }

    def _route_corridor_diagnostics(
        self,
        observation: dict[str, Any],
        *,
        current_time: float,
    ) -> dict[str, Any] | None:
        """Select a topology hypothesis and expose it as route-corridor geometry.

        Returns:
            dict[str, Any] | None: Selected route-corridor payload, or ``None``
            when the diagnostic fail-closed gate is unavailable.
        """
        topology = self._hypotheses_for_observation(observation)
        status = str(topology.get("status", "unknown"))
        self._topology_status_counts[status] += 1
        self._last_topology_decision = topology
        if status != "ok":
            return None
        selected_id = str(topology["selected_hypothesis_id"])
        self._selected_hypothesis_counts[selected_id] += 1
        selected = next(
            item for item in topology["hypotheses"] if str(item["hypothesis_id"]) == selected_id
        )
        route_corridor = dict(selected["route_corridor"])
        route_remaining = route_corridor.get("route_remaining_distance")
        if isinstance(route_remaining, int | float | np.integer | np.floating) and np.isfinite(
            route_remaining
        ):
            remaining = float(route_remaining)
            self._route_distance_history.append((current_time, remaining))
            route_corridor["route_arc_progress_windows"] = self._distance_progress_windows(
                self._route_distance_history,
                current_time=current_time,
                current_distance=remaining,
            )
        route_corridor["topology_hypothesis"] = {
            key: value for key, value in selected.items() if key != "route_corridor"
        }
        route_corridor["topology_hypotheses"] = [
            {key: value for key, value in item.items() if key != "route_corridor"}
            for item in topology["hypotheses"]
        ]
        route_corridor["topology_status"] = "ok"
        return route_corridor

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return a command, stopping fail-closed when topology hypotheses are unavailable.

        Returns:
            tuple[float, float]: Selected linear and angular velocity.
        """
        command = super().plan(observation)
        topology = self._last_topology_decision or {}
        status = str(topology.get("status", "unknown"))
        fail_closed = (
            status == "not_available" and bool(self.topology_config.fail_closed_on_missing_inputs)
        ) or (
            status == "insufficient_hypotheses"
            and bool(self.topology_config.fail_closed_on_insufficient_hypotheses)
        )
        if fail_closed:
            self._last_command = (0.0, 0.0)
            if self._last_decision:
                self._last_decision["planner_mode"] = "TOPOLOGY_FAIL_CLOSED"
                self._last_decision["selected_command"] = [0.0, 0.0]
                self._last_decision["selected_source"] = "topology_fail_closed"
                self._last_decision["topology_guided"] = topology
            return (0.0, 0.0)
        if self._last_decision:
            self._last_decision["topology_guided"] = topology
        return command

    def diagnostics(self) -> dict[str, Any]:
        """Return aggregate base-planner and topology-selection diagnostics."""
        diagnostics = super().diagnostics()
        diagnostics["topology_guided"] = {
            "diagnostic_only": bool(self.topology_config.diagnostic_only),
            "claim_boundary": self.topology_config.claim_boundary,
            "min_hypotheses": int(self.topology_config.min_hypotheses),
            "max_hypotheses": int(self.topology_config.max_hypotheses),
            "status_counts": dict(sorted(self._topology_status_counts.items())),
            "selected_hypothesis_counts": dict(sorted(self._selected_hypothesis_counts.items())),
            "last_topology_decision": self._last_topology_decision,
        }
        return diagnostics


__all__ = [
    "TopologyGuidedHybridRulePlannerAdapter",
    "TopologyGuidedLocalPolicyConfig",
    "build_topology_guided_local_policy_config",
]
