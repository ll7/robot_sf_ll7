"""Testing-only topology-parallel NMPC prototype for issue #5310/#6152.

This module implements a bounded offline/research-only topology-parallel NMPC
planner adapter that evaluates 2-4 deterministic x-y-t maneuver hypotheses
concurrently under one wall-clock deadline.

Critical invariant: identical NMPC solver, objective, constraints, tolerances,
horizon, and iteration cap. Only deterministic, materially distinct x-y-t
initializations and the selection/hysteresis mechanism may differ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_sf.planner.nmpc_social import (
    NMPCSocialConfig,
    NMPCSocialPlannerAdapter,
    NMPCSolveResult,
    build_nmpc_social_config,
)


@dataclass
class HypothesisDiagnostics:
    """Per-hypothesis diagnostics from the topology-parallel NMPC solve."""

    label: str
    feasible: bool
    objective: float
    solver_status: str
    solver_iterations: int
    solver_runtime: float
    signed_side: int
    material_separation: float
    initialization_signature: dict[str, Any] = field(default_factory=dict)
    rollout_signature: dict[str, Any] = field(default_factory=dict)
    selection_rank: int = -1
    switch_reason: str = ""


@dataclass
class TopologyParallelNMPCConfig:
    """Configuration for the topology-parallel NMPC planner.

    max_hypotheses: Maximum number of hypotheses to evaluate in parallel (2-4).
    hypothesis_labels: Ordered tuple of hypothesis labels to evaluate.
    control_period_s: Nominal control period (wall-clock deadline for all solves).
    max_runtime_s: Hard runtime gate; fails closed on overrun.
    switch_hysteresis_ticks: Minimum ticks before switching selected hypothesis.
    nmpc_config: Base NMPC config shared identically across all hypotheses.
    """

    max_hypotheses: int = 3
    hypothesis_labels: tuple[str, ...] = ("pass_left", "yield_straight", "pass_right")
    control_period_s: float = 2.0
    max_runtime_s: float = 2.0
    switch_hysteresis_ticks: int = 2
    nmpc_config: NMPCSocialConfig | None = None


_HYPOTHESIS_SIDES: dict[str, int] = {
    "default": 0,
    "pass_left": 1,
    "yield_straight": 0,
    "pass_right": -1,
    "hard_left": 1,
    "hard_right": -1,
}


def _signed_side_for_label(label: str) -> int:
    return _HYPOTHESIS_SIDES.get(label, 0)


def _preferred_turn_for_label(label: str) -> float:
    side = _signed_side_for_label(label)
    return float(side) * 0.5


def _material_separation(
    left_states: np.ndarray,
    right_states: np.ndarray,
) -> float:
    """Compute minimum pairwise separation between two rollout state trajectories.

    Returns:
        float: Minimum Euclidean distance between any pair of rollout positions.
    """
    if left_states.size == 0 or right_states.size == 0:
        return float("inf")
    max_t = min(left_states.shape[0], right_states.shape[0])
    if max_t == 0:
        return float("inf")
    positions_left = left_states[:max_t, :2]
    positions_right = right_states[:max_t, :2]
    dists = np.linalg.norm(positions_left - positions_right, axis=1)
    return float(np.min(dists))


def _rollout_signature(rollout_states: np.ndarray) -> dict[str, Any]:
    """Compact geometric signature of a rollout trajectory.

    Returns:
        dict: Mean position, span, and state count.
    """
    if rollout_states.size == 0:
        return {"mean_x": 0.0, "mean_y": 0.0, "span_x": 0.0, "span_y": 0.0, "n_states": 0}
    positions = rollout_states[:, :2]
    return {
        "mean_x": float(np.mean(positions[:, 0])),
        "mean_y": float(np.mean(positions[:, 1])),
        "span_x": float(np.max(positions[:, 0]) - np.min(positions[:, 0])),
        "span_y": float(np.max(positions[:, 1]) - np.min(positions[:, 1])),
        "n_states": int(rollout_states.shape[0]),
    }


class TopologyParallelNMPCPlannerAdapter(NMPCSocialPlannerAdapter):
    """Offline/research-only topology-parallel NMPC planner adapter.

    Evaluates 2-4 deterministic maneuver hypotheses concurrently, selects
    the feasible-first/lowest-objective command, and applies two-tick switching
    hysteresis.
    """

    def __init__(self, config: TopologyParallelNMPCConfig | None = None) -> None:
        """Initialize the topology-parallel NMPC adapter."""
        if config is None:
            config = TopologyParallelNMPCConfig()
        super().__init__(config.nmpc_config)
        self.topo_config = config
        self._current_hypothesis_index: int = 0
        self._ticks_at_hypothesis: int = 0
        self._topo_stats: dict[str, Any] = {
            "calls": 0,
            "total_solves": 0,
            "hypothesis_switches": 0,
            "runtime_overruns": 0,
            "all_failed_count": 0,
        }
        self._last_hypothesis_diagnostics: list[HypothesisDiagnostics] = []

    def reset(self) -> None:
        """Clear per-episode state including hypothesis selection history."""
        super().reset()
        self._current_hypothesis_index = 0
        self._ticks_at_hypothesis = 0
        self._last_hypothesis_diagnostics = []

    def _evaluate_hypotheses(self, observation: dict[str, Any]) -> list[HypothesisDiagnostics]:
        """Solve all topology hypotheses and return their diagnostics.

        Each hypothesis uses the identical NMPC solver, objective, constraints,
        tolerances, horizon, and iteration cap. Only the preferred turn and
        initial guess differ.

        Returns:
            list[HypothesisDiagnostics]: Per-hypothesis solve diagnostics.
        """
        labels = self.topo_config.hypothesis_labels
        if not labels:
            return []

        results: list[HypothesisDiagnostics] = []
        deadline = time.perf_counter() + self.topo_config.max_runtime_s

        for label in labels:
            if time.perf_counter() > deadline:
                self._topo_stats["runtime_overruns"] = (
                    int(self._topo_stats.get("runtime_overruns", 0)) + 1
                )
                results.append(
                    HypothesisDiagnostics(
                        label=label,
                        feasible=False,
                        objective=float("inf"),
                        solver_status="deadline_exceeded",
                        solver_iterations=0,
                        solver_runtime=0.0,
                        signed_side=_signed_side_for_label(label),
                        material_separation=0.0,
                        initialization_signature={"deadline_exceeded": True},
                        rollout_signature={},
                        selection_rank=-1,
                        switch_reason="deadline_exceeded",
                    )
                )
                continue

            preferred_turn = _preferred_turn_for_label(label)
            t_hyp = time.perf_counter()
            result: NMPCSolveResult = self.solve_initialization(
                observation, preferred_turn=preferred_turn
            )
            t_hyp_elapsed = time.perf_counter() - t_hyp
            sig = _rollout_signature(result.rollout_states)
            diag = HypothesisDiagnostics(
                label=label,
                feasible=result.feasible,
                objective=result.objective,
                solver_status=result.solver_status,
                solver_iterations=result.solver_iterations,
                solver_runtime=t_hyp_elapsed,
                signed_side=_signed_side_for_label(label),
                material_separation=0.0,
                initialization_signature=dict(result.initialization_signature),
                rollout_signature=sig,
                selection_rank=-1,
                switch_reason="",
            )
            results.append(diag)

        # Compute pairwise material separation where applicable
        n = len(results)
        if n >= 2:
            for i in range(n):
                if not results[i].feasible:
                    continue
                for j in range(i + 1, n):
                    if not results[j].feasible:
                        continue
                    rs_i = self._last_result_states.get(results[i].label)
                    rs_j = self._last_result_states.get(results[j].label)
                    if rs_i is not None and rs_j is not None:
                        sep = _material_separation(rs_i, rs_j)
                        results[i].material_separation = max(results[i].material_separation, sep)
                        results[j].material_separation = max(results[j].material_separation, sep)

        return results

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return the command from the best topology hypothesis.

        Selection: feasible-first, then lowest-objective.
        Switching: two-tick hysteresis to prevent oscillation.
        """
        self._topo_stats["calls"] = int(self._topo_stats.get("calls", 0)) + 1
        self._last_result_states: dict[str, np.ndarray] = {}

        # Fast-path: at goal
        extracted_state = self._extract_state(observation)
        robot_pos = extracted_state[0]
        goal = extracted_state[3]
        goal_delta = goal - robot_pos
        goal_distance = float(np.linalg.norm(goal_delta))
        if goal_distance <= float(self.config.goal_tolerance):
            self._record_command(0.0, 0.0)
            return 0.0, 0.0

        # Evaluate all hypotheses
        self._topo_stats["total_solves"] = int(self._topo_stats.get("total_solves", 0)) + len(
            self.topo_config.hypothesis_labels
        )
        diagnostics = self._evaluate_hypotheses(observation)
        self._last_hypothesis_diagnostics = diagnostics

        # Select best hypothesis
        selected_idx = self._select_hypothesis(diagnostics)

        # Build context for command extraction
        preferred_turn = _preferred_turn_for_label(
            self.topo_config.hypothesis_labels[selected_idx]
            if selected_idx < len(self.topo_config.hypothesis_labels)
            else ""
        )
        context = self._solve_context(observation, preferred_turn=preferred_turn)
        result: NMPCSolveResult = self.solve_initialization(
            observation, preferred_turn=preferred_turn
        )
        command = self._command_from_solution(result, context)
        self._record_command(*command)
        return command

    def _select_hypothesis(self, diagnostics: list[HypothesisDiagnostics]) -> int:
        """Select the best hypothesis index.

        Feasible hypotheses are ranked by objective (lower is better).
        If no hypothesis is feasible, fall back to the current index.
        Two-tick hysteresis prevents switching on transient objective noise.

        Returns:
            int: Index of the selected hypothesis.
        """
        n = len(diagnostics)
        if n == 0:
            return 0

        # Score feasible hypotheses
        feasible_indices = [i for i, d in enumerate(diagnostics) if d.feasible]
        if not feasible_indices:
            self._topo_stats["all_failed_count"] = (
                int(self._topo_stats.get("all_failed_count", 0)) + 1
            )
            return min(self._current_hypothesis_index, n - 1)

        # Among feasible, pick lowest objective
        feasible_indices.sort(key=lambda i: diagnostics[i].objective)
        best_idx = feasible_indices[0]

        # Update ranks
        for rank, i in enumerate(feasible_indices):
            diagnostics[i].selection_rank = rank

        # Hysteresis: stay on current unless new best is significantly better
        # or we've accumulated enough ticks at the current hypothesis
        if best_idx == self._current_hypothesis_index:
            self._ticks_at_hypothesis += 1
            diagnostics[self._current_hypothesis_index].switch_reason = "already_selected"
            return self._current_hypothesis_index

        if self._ticks_at_hypothesis < self.topo_config.switch_hysteresis_ticks:
            diagnostics[best_idx].switch_reason = "suppressed_by_hysteresis"
            diagnostics[self._current_hypothesis_index].switch_reason = "hysteresis_hold"
            return self._current_hypothesis_index

        # Switch
        self._ticks_at_hypothesis = 0
        self._current_hypothesis_index = best_idx
        self._topo_stats["hypothesis_switches"] = (
            int(self._topo_stats.get("hypothesis_switches", 0)) + 1
        )
        diagnostics[best_idx].switch_reason = "new_best_selected"
        return best_idx


def build_topology_parallel_nmpc_config(
    cfg: dict[str, Any] | None,
) -> TopologyParallelNMPCConfig:
    """Build topology-parallel NMPC config from a mapping payload.

    Returns:
        TopologyParallelNMPCConfig: Parsed config with overrides applied.
    """
    if not isinstance(cfg, dict):
        return TopologyParallelNMPCConfig()
    nmpc_cfg_raw = cfg.get("nmpc_config")
    nmpc_cfg = None
    if isinstance(nmpc_cfg_raw, dict):
        nmpc_cfg = build_nmpc_social_config(nmpc_cfg_raw)

    raw_hypotheses = cfg.get("hypothesis_labels")
    hypotheses: tuple[str, ...] = TopologyParallelNMPCConfig.hypothesis_labels
    if isinstance(raw_hypotheses, (list, tuple)) and len(raw_hypotheses) >= 1:
        hypotheses = tuple(str(h).strip() for h in raw_hypotheses)

    return TopologyParallelNMPCConfig(
        max_hypotheses=int(cfg.get("max_hypotheses", TopologyParallelNMPCConfig.max_hypotheses)),
        hypothesis_labels=hypotheses,
        control_period_s=float(
            cfg.get("control_period_s", TopologyParallelNMPCConfig.control_period_s)
        ),
        max_runtime_s=float(cfg.get("max_runtime_s", TopologyParallelNMPCConfig.max_runtime_s)),
        switch_hysteresis_ticks=int(
            cfg.get("switch_hysteresis_ticks", TopologyParallelNMPCConfig.switch_hysteresis_ticks)
        ),
        nmpc_config=nmpc_cfg,
    )
