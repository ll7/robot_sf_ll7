"""Experimental topology-parallel NMPC planner for issue #5310."""

from __future__ import annotations

import hashlib
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from robot_sf.planner.nmpc_social import (
    NMPCSocialConfig,
    NMPCSocialPlannerAdapter,
    build_nmpc_social_config,
)

_HYPOTHESIS_ORDER = ("pass_left", "yield_straight", "pass_right")
_HYPOTHESIS_SIGN = {"pass_left": 1.0, "yield_straight": 0.0, "pass_right": -1.0}


@dataclass(frozen=True)
class TopologyParallelNMPCConfig:
    """Configuration for the testing-only topology-parallel NMPC arm."""

    nmpc: NMPCSocialConfig = field(default_factory=NMPCSocialConfig)
    hypothesis_labels: tuple[str, ...] = _HYPOTHESIS_ORDER
    max_hypotheses: int = 3
    initialization_turn_rate: float = 0.8
    yield_speed_scale: float = 0.55
    hysteresis_ticks: int = 2
    max_runtime_s: float = 5.0

    def __post_init__(self) -> None:
        """Reject ambiguous or unbounded experimental configuration."""
        labels = tuple(self.hypothesis_labels)
        if len(labels) < 2 or len(labels) > 4 or len(set(labels)) != len(labels):
            raise ValueError("hypothesis_labels must contain 2-4 unique labels")
        if any(label not in _HYPOTHESIS_ORDER for label in labels):
            raise ValueError(f"hypothesis_labels must come from {_HYPOTHESIS_ORDER!r}")
        if not 2 <= int(self.max_hypotheses) <= len(labels):
            raise ValueError("max_hypotheses must be between 2 and the number of labels")
        if float(self.initialization_turn_rate) < 0.0:
            raise ValueError("initialization_turn_rate must be >= 0")
        if not 0.0 < float(self.yield_speed_scale) <= 1.0:
            raise ValueError("yield_speed_scale must be in (0, 1]")
        if int(self.hysteresis_ticks) < 1:
            raise ValueError("hysteresis_ticks must be >= 1")
        if float(self.max_runtime_s) <= 0.0:
            raise ValueError("max_runtime_s must be > 0")


class TopologyParallelNMPCPlannerAdapter(NMPCSocialPlannerAdapter):
    """Solve the same NMPC problem from deterministic topology-shaped seeds."""

    def __init__(self, config: TopologyParallelNMPCConfig | None = None) -> None:
        """Initialize the experimental adapter with explicit topology settings."""
        self.topology_config = config or TopologyParallelNMPCConfig()
        super().__init__(self.topology_config.nmpc)

    def reset(self) -> None:
        """Reset solver statistics and topology commitment state."""
        super().reset()
        self._committed_hypothesis: str | None = None
        self._pending_hypothesis: str | None = None
        self._pending_ticks = 0
        self._last_decision: dict[str, Any] = {}

    @staticmethod
    def _signature(values: np.ndarray) -> str:
        """Return a stable compact signature for a numeric initialization or rollout."""
        normalized = np.asarray(values, dtype=float)
        return hashlib.sha256(np.round(normalized, 6).tobytes()).hexdigest()[:16]

    def _initializations(
        self,
        *,
        goal_heading_error: float,
        current_speed: float,
        goal_distance: float,
        preferred_turn: float,
        speed_cap: float,
    ) -> list[tuple[str, np.ndarray]]:
        """Create stable, geometrically different x-y-t control-sequence seeds.

        Returns:
            list[tuple[str, np.ndarray]]: Ordered hypothesis labels and flattened control seeds.
        """
        # Explicit topology seeds must not inherit the previous selected solution: otherwise
        # identical observations can silently change the hypothesis set between ticks.
        previous_solution = self._last_solution
        self._last_solution = None
        try:
            base = self._initial_guess(
                goal_heading_error=goal_heading_error,
                current_speed=current_speed,
                goal_distance=goal_distance,
                preferred_turn=preferred_turn,
                speed_cap=speed_cap,
            ).reshape(-1, 2)
        finally:
            self._last_solution = previous_solution

        horizon = max(int(self.config.horizon_steps), 1)
        initializations: list[tuple[str, np.ndarray]] = []
        labels = [
            label for label in _HYPOTHESIS_ORDER if label in self.topology_config.hypothesis_labels
        ]
        for label in labels[: int(self.topology_config.max_hypotheses)]:
            controls = base.copy()
            if label == "yield_straight":
                controls[:, 0] *= float(self.topology_config.yield_speed_scale)
                controls[:, 1] *= 0.25
            else:
                sign = _HYPOTHESIS_SIGN[label]
                profile = (
                    sign
                    * float(self.topology_config.initialization_turn_rate)
                    * (1.0 - np.arange(horizon, dtype=float) / max(horizon, 1))
                )
                controls[:, 1] = np.clip(
                    controls[:, 1] + profile,
                    -float(self.config.max_angular_speed),
                    float(self.config.max_angular_speed),
                )
            initializations.append((label, controls.reshape(-1)))
        return initializations

    def _select_hypothesis(self, rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
        """Select feasible-first/lowest-cost while bounding topology switching.

        Returns:
            tuple[dict[str, Any] | None, str]: Selected row and switch reason.
        """
        feasible = [row for row in rows if bool(row["feasible"])]
        if not feasible:
            self._committed_hypothesis = None
            self._pending_hypothesis = None
            self._pending_ticks = 0
            return None, "no_feasible_hypothesis"
        best = min(feasible, key=lambda row: float(row["objective"]))
        committed = next(
            (row for row in feasible if row["hypothesis"] == self._committed_hypothesis),
            None,
        )
        if committed is None:
            previous = self._committed_hypothesis
            self._committed_hypothesis = str(best["hypothesis"])
            self._pending_hypothesis = None
            self._pending_ticks = 0
            return best, "initial_selection" if previous is None else "committed_infeasible"
        if best["hypothesis"] == committed["hypothesis"]:
            self._pending_hypothesis = None
            self._pending_ticks = 0
            return committed, "committed_remains_best"

        candidate = str(best["hypothesis"])
        if candidate == self._pending_hypothesis:
            self._pending_ticks += 1
        else:
            self._pending_hypothesis = candidate
            self._pending_ticks = 1
        if self._pending_ticks >= int(self.topology_config.hysteresis_ticks):
            self._committed_hypothesis = candidate
            self._pending_hypothesis = None
            self._pending_ticks = 0
            return best, "hysteresis_confirmed"
        return committed, "hysteresis_hold"

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Solve all bounded topology seeds and emit the selected first command.

        Returns:
            tuple[float, float]: Selected linear and angular command.
        """
        self._stats["calls"] = int(self._stats.get("calls", 0)) + 1
        context, goal_heading_error, goal_distance = self._build_rollout_context(observation)
        if goal_distance <= float(self.config.goal_tolerance):
            self._last_decision = {
                "hypotheses": [],
                "selected_hypothesis": None,
                "switch_reason": "goal_reached",
                "hysteresis": self._hysteresis_diagnostics(),
            }
            self._record_command(0.0, 0.0)
            return 0.0, 0.0

        preferred_turn = self._preferred_avoidance_turn(
            robot_pos=context.robot_pos,
            heading=context.heading,
            ped_positions=context.ped_positions,
            ped_velocities=context.ped_velocities,
        )
        initializations = self._initializations(
            goal_heading_error=goal_heading_error,
            current_speed=context.current_speed,
            goal_distance=goal_distance,
            preferred_turn=preferred_turn,
            speed_cap=context.speed_cap,
        )
        started = perf_counter()
        rows: list[dict[str, Any]] = []
        for label, initial_guess in initializations:
            elapsed = perf_counter() - started
            if elapsed >= float(self.topology_config.max_runtime_s):
                rows.append(self._runtime_limited_row(label, initial_guess))
                continue
            solve = self._solve_context(context, initial_guess)
            self._stats["solver_successes"] += int(solve.feasible)
            self._stats["solver_failures"] += int(not solve.feasible)
            rows.append(
                {
                    "hypothesis": label,
                    "initialization_signature": self._signature(initial_guess),
                    "rollout_signature": self._signature(solve.rollout_states),
                    "feasible": bool(solve.feasible),
                    "objective": solve.objective,
                    "solver_status": solve.solver_status,
                    "solver_iterations": solve.solver_iterations,
                    "solver_runtime_s": solve.runtime_s,
                    "rollout_diagnostics": self._rollout_diagnostics(solve.rollout_states),
                    "_solve": solve,
                }
            )

        selected, switch_reason = self._select_hypothesis(rows)
        public_rows = [
            {key: value for key, value in row.items() if key != "_solve"} for row in rows
        ]
        self._last_decision = {
            "hypotheses": public_rows,
            "selected_hypothesis": None if selected is None else selected["hypothesis"],
            "switch_reason": switch_reason,
            "hysteresis": self._hysteresis_diagnostics(),
            "runtime_budget_s": float(self.topology_config.max_runtime_s),
        }
        if selected is None or selected["_solve"].solution is None:
            self._stats["fallback_stop_count"] += 1
            self._record_command(0.0, 0.0)
            return 0.0, 0.0

        solution = np.asarray(selected["_solve"].solution, dtype=float)
        self._last_solution = solution.copy()
        linear, angular = self._command_from_solution(solution, context=context)
        self._record_command(linear, angular)
        return linear, angular

    def _runtime_limited_row(self, label: str, initial_guess: np.ndarray) -> dict[str, Any]:
        """Represent an unexpanded hypothesis without claiming it was optimized.

        Returns:
            dict[str, Any]: Fail-closed diagnostic row.
        """
        return {
            "hypothesis": label,
            "initialization_signature": self._signature(initial_guess),
            "rollout_signature": self._signature(np.zeros((0, 3), dtype=float)),
            "feasible": False,
            "objective": None,
            "solver_status": "runtime_budget_exceeded",
            "solver_iterations": None,
            "solver_runtime_s": 0.0,
            "rollout_diagnostics": {"status": "not_optimized"},
            "_solve": None,
        }

    def _hysteresis_diagnostics(self) -> dict[str, Any]:
        """Return the bounded switch state for trace emission."""
        return {
            "committed_hypothesis": self._committed_hypothesis,
            "pending_hypothesis": self._pending_hypothesis,
            "pending_ticks": int(self._pending_ticks),
            "required_ticks": int(self.topology_config.hysteresis_ticks),
        }

    def _rollout_diagnostics(self, states: np.ndarray) -> dict[str, Any]:
        """Summarize the x-y-t rollout geometry without adding an objective term.

        Returns:
            dict[str, Any]: Compact rollout geometry summary.
        """
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[0] == 0:
            return {"status": "empty"}
        return {
            "status": "optimized",
            "steps": int(states.shape[0]),
            "terminal_position": [float(value) for value in states[-1, :2]],
            "lateral_span": float(np.ptp(states[:, 1])),
            "duration_s": float(states.shape[0]) * float(self.config.rollout_dt),
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return base NMPC diagnostics plus the latest per-hypothesis trace."""
        payload = super().diagnostics()
        payload["topology_parallel_nmpc"] = deepcopy(self._last_decision)
        return payload


def build_topology_parallel_nmpc_config(
    cfg: dict[str, Any] | None,
) -> TopologyParallelNMPCConfig:
    """Build the experimental topology-parallel config from a YAML mapping.

    Returns:
        TopologyParallelNMPCConfig: Parsed experimental planner configuration.
    """
    if not isinstance(cfg, dict):
        return TopologyParallelNMPCConfig()
    defaults = TopologyParallelNMPCConfig()
    raw_labels = cfg.get("hypothesis_labels", defaults.hypothesis_labels)
    if isinstance(raw_labels, str):
        raw_labels = tuple(item.strip() for item in raw_labels.split(",") if item.strip())
    try:
        labels = tuple(str(item) for item in raw_labels)
        ordered_labels = tuple(label for label in _HYPOTHESIS_ORDER if label in labels)
        max_hypotheses = int(cfg.get("max_hypotheses", defaults.max_hypotheses))
        topology = TopologyParallelNMPCConfig(
            nmpc=build_nmpc_social_config(cfg),
            hypothesis_labels=ordered_labels,
            max_hypotheses=max_hypotheses,
            initialization_turn_rate=float(
                cfg.get("initialization_turn_rate", defaults.initialization_turn_rate)
            ),
            yield_speed_scale=float(cfg.get("yield_speed_scale", defaults.yield_speed_scale)),
            hysteresis_ticks=int(cfg.get("hysteresis_ticks", defaults.hysteresis_ticks)),
            max_runtime_s=float(cfg.get("max_runtime_s", defaults.max_runtime_s)),
        )
    except (TypeError, ValueError) as exc:
        warnings.warn(
            f"Invalid topology-parallel NMPC config ({exc}); falling back to defaults.",
            RuntimeWarning,
            stacklevel=2,
        )
        return defaults
    return topology


__all__ = [
    "TopologyParallelNMPCConfig",
    "TopologyParallelNMPCPlannerAdapter",
    "build_topology_parallel_nmpc_config",
]
