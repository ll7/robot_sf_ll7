"""Experimental topology-parallel NMPC planner for issue #5310."""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from robot_sf.planner.nmpc_social import (
    NMPCSocialConfig,
    NMPCSocialPlannerAdapter,
    NMPCSolveResult,
    _parse_bool,
    build_nmpc_social_config,
)

_HYPOTHESIS_ORDER = ("pass_left", "yield_straight", "pass_right")
_DEFAULT_HYPOTHESIS = "default"
_VALID_HYPOTHESIS_LABELS = (_DEFAULT_HYPOTHESIS, *_HYPOTHESIS_ORDER)
_HYPOTHESIS_SIGN = {"pass_left": 1.0, "yield_straight": 0.0, "pass_right": -1.0}
_ARM_NAME = "topology_parallel_nmpc"


@dataclass(frozen=True)
class TopologyParallelNMPCConfig:
    """Configuration for the testing-only topology-parallel NMPC arm."""

    nmpc: NMPCSocialConfig = field(default_factory=NMPCSocialConfig)
    arm: str = _ARM_NAME
    enabled: bool = True
    hypothesis_labels: tuple[str, ...] = _HYPOTHESIS_ORDER
    max_hypotheses: int = 3
    initialization_turn_rate: float = 0.8
    yield_speed_scale: float = 0.55
    hysteresis_ticks: int = 2
    objective_tolerance: float = 1e-6
    control_period_s: float = 2.0
    max_runtime_s: float = 2.0

    def __post_init__(self) -> None:
        """Reject ambiguous, mismatched, or unbounded experimental configuration."""
        if self.arm != _ARM_NAME:
            raise ValueError(f"arm must be {_ARM_NAME!r}")
        if not isinstance(self.enabled, bool) or not self.enabled:
            raise ValueError("topology_parallel_nmpc requires enabled=true")

        _validate_hypotheses(tuple(self.hypothesis_labels), self.max_hypotheses)

        _require_finite("initialization_turn_rate", self.initialization_turn_rate, minimum=0.0)
        _require_finite(
            "yield_speed_scale", self.yield_speed_scale, minimum=0.0, strict_minimum=True
        )
        if float(self.yield_speed_scale) > 1.0:
            raise ValueError("yield_speed_scale must be in (0, 1]")
        hysteresis_ticks = _require_integer("hysteresis_ticks", self.hysteresis_ticks)
        if hysteresis_ticks < 1:
            raise ValueError("hysteresis_ticks must be >= 1")
        _require_finite("objective_tolerance", self.objective_tolerance, minimum=0.0)
        control_period_s = _require_finite("control_period_s", self.control_period_s, minimum=0.0)
        max_runtime_s = _require_finite("max_runtime_s", self.max_runtime_s, minimum=0.0)
        if not math.isclose(max_runtime_s, control_period_s, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("max_runtime_s must equal the predeclared control_period_s")


def _validate_hypotheses(labels: tuple[str, ...], max_hypotheses: Any) -> None:
    """Validate label cardinality and the explicit K=1 mode."""
    if not labels or any(not isinstance(label, str) for label in labels):
        raise ValueError("hypothesis_labels must contain string labels")
    if len(set(labels)) != len(labels):
        raise ValueError("hypothesis_labels must be unique")
    if any(label not in _VALID_HYPOTHESIS_LABELS for label in labels):
        raise ValueError(f"hypothesis_labels must come from {_VALID_HYPOTHESIS_LABELS!r}")
    count = _require_integer("max_hypotheses", max_hypotheses)
    if labels == (_DEFAULT_HYPOTHESIS,):
        if count != 1:
            raise ValueError("default K=1 mode requires max_hypotheses=1")
        return
    if _DEFAULT_HYPOTHESIS in labels or not 2 <= len(labels) <= 4:
        raise ValueError(
            "topology mode requires 2-4 labels; default initialization is a separate K=1 mode"
        )
    if not 2 <= count <= len(labels):
        raise ValueError("max_hypotheses must be between 2 and the number of labels")


def _require_integer(name: str, value: Any) -> int:
    """Return an exact integer config value, rejecting booleans and fractional values."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or numeric != float(integer):
        raise ValueError(f"{name} must be an integer")
    return integer


def _require_finite(
    name: str,
    value: Any,
    *,
    minimum: float,
    strict_minimum: bool = False,
) -> float:
    """Return a finite numeric config value with an explicit lower bound."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric) or (numeric <= minimum if strict_minimum else numeric < minimum):
        operator = ">" if strict_minimum else ">="
        raise ValueError(f"{name} must be finite and {operator} {minimum}")
    return numeric


class TopologyParallelNMPCPlannerAdapter(NMPCSocialPlannerAdapter):
    """Solve one unchanged NMPC problem per deterministic topology-shaped seed."""

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

    @property
    def _single_default_mode(self) -> bool:
        """Return whether this adapter is the explicit legacy-equivalence K=1 mode."""
        return tuple(self.topology_config.hypothesis_labels) == (_DEFAULT_HYPOTHESIS,)

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
        """Create stable x-y-t control-sequence seeds in canonical order.

        Returns:
            list[tuple[str, np.ndarray]]: Ordered hypothesis labels and flattened control seeds.
        """
        if self._single_default_mode:
            base = self._initial_guess(
                goal_heading_error=goal_heading_error,
                current_speed=current_speed,
                goal_distance=goal_distance,
                preferred_turn=preferred_turn,
                speed_cap=speed_cap,
            )
            return [(_DEFAULT_HYPOTHESIS, base)]

        # Topology seeds must not inherit the previous selected solution: otherwise identical
        # observations can silently change the hypothesis set between ticks.
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

    def _material_separation_threshold(self, context: Any) -> float:
        """Return a geometry threshold tied to the robot footprint and obstacle margin."""
        return max(
            float(context.robot_radius),
            0.5 * float(self.config.obstacle_margin),
            1e-3,
        )

    def _annotate_material_geometry(self, rows: list[dict[str, Any]], context: Any) -> None:
        """Mark rows that participate in a material, signed-side trajectory pair."""
        threshold = self._material_separation_threshold(context)
        self._initialize_geometry_diagnostics(rows, threshold)

        if self._single_default_mode:
            if rows:
                rows[0]["topology_valid"] = bool(rows[0]["feasible"])
            return

        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                self._annotate_pair(left, right, threshold)

    @staticmethod
    def _initialize_geometry_diagnostics(rows: list[dict[str, Any]], threshold: float) -> None:
        """Initialize public geometry-gate fields before pairwise comparison."""
        for row in rows:
            row["topology_valid"] = False
            row["rollout_diagnostics"]["minimum_topology_separation_m"] = threshold
            row["rollout_diagnostics"]["max_pairwise_separation_m"] = 0.0

    @staticmethod
    def _annotate_pair(left: dict[str, Any], right: dict[str, Any], threshold: float) -> None:
        """Mark one pair when it has opposite signed sides and material separation."""
        if not bool(left["feasible"]) or not bool(right["feasible"]):
            return
        left_sign = int(left["rollout_diagnostics"].get("side_sign", 0))
        right_sign = int(right["rollout_diagnostics"].get("side_sign", 0))
        if left_sign * right_sign != -1:
            return
        left_states = left["_solve"].rollout_states
        right_states = right["_solve"].rollout_states
        count = min(left_states.shape[0], right_states.shape[0])
        if count == 0:
            return
        separation = float(
            np.max(np.linalg.norm(left_states[:count, :2] - right_states[:count, :2], axis=1))
        )
        for row in (left, right):
            row["rollout_diagnostics"]["max_pairwise_separation_m"] = max(
                float(row["rollout_diagnostics"]["max_pairwise_separation_m"]), separation
            )
        if separation >= threshold:
            left["topology_valid"] = True
            right["topology_valid"] = True

    def _select_hypothesis(self, rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
        """Select feasible-first/lowest-cost with stable ties and bounded switching.

        Returns:
            tuple[dict[str, Any] | None, str]: Selected row and switch reason.
        """
        feasible = [
            row
            for row in rows
            if bool(row["feasible"])
            and bool(row.get("topology_valid", False))
            and row["objective"] is not None
            and np.isfinite(float(row["objective"]))
        ]
        committed_row = next(
            (row for row in rows if row["hypothesis"] == self._committed_hypothesis), None
        )
        if (
            not feasible
            and self._committed_hypothesis is not None
            and (committed_row is None or not bool(committed_row["feasible"]))
        ):
            # A previously validated commitment may become the only surviving feasible
            # alternative after a solver failure. Release the commitment immediately rather
            # than oscillating or reusing its unverified command.
            feasible = [
                row
                for row in rows
                if bool(row["feasible"])
                and row["objective"] is not None
                and np.isfinite(float(row["objective"]))
            ]
        if not feasible:
            self._committed_hypothesis = None
            self._pending_hypothesis = None
            self._pending_ticks = 0
            reason = (
                "no_feasible_hypothesis"
                if not any(bool(row["feasible"]) for row in rows)
                else "no_materially_distinct_hypotheses"
            )
            return None, reason

        minimum_objective = min(float(row["objective"]) for row in feasible)
        tied = [
            row
            for row in feasible
            if float(row["objective"])
            <= minimum_objective + float(self.topology_config.objective_tolerance)
        ]
        best = min(tied, key=self._hypothesis_rank)
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

    @staticmethod
    def _hypothesis_rank(row: dict[str, Any]) -> int:
        """Return the stable canonical rank used for objective ties."""
        label = str(row["hypothesis"])
        if label == _DEFAULT_HYPOTHESIS:
            return 0
        return _HYPOTHESIS_ORDER.index(label) + 1

    def _context_is_finite(self, context: Any) -> bool:
        """Reject non-finite simulator state before it reaches the optimizer.

        Returns:
            bool: ``True`` when all solver-relevant context values are finite.
        """
        scalar_values = (
            context.heading,
            context.current_speed,
            context.robot_radius,
            context.ped_radius,
            context.speed_cap,
        )
        arrays = (
            context.robot_pos,
            context.goal,
            context.ped_positions,
            context.ped_velocities,
        )
        if any(not np.isfinite(float(value)) for value in scalar_values):
            return False
        if any(not np.all(np.isfinite(np.asarray(value, dtype=float))) for value in arrays):
            return False
        for key in (
            "occupancy_grid",
            "occupancy_grid_meta_origin",
            "occupancy_grid_meta_resolution",
            "occupancy_grid_meta_size",
            "occupancy_grid_meta_use_ego_frame",
            "occupancy_grid_meta_channel_indices",
        ):
            value = context.observation.get(key)
            if value is not None and not np.all(np.isfinite(np.asarray(value, dtype=float))):
                return False
        return True

    @staticmethod
    def _observation_is_finite(observation: dict[str, Any]) -> bool:
        """Reject non-finite raw simulator fields before context construction.

        Returns:
            bool: ``True`` when all known numeric simulator fields are finite.
        """
        if not isinstance(observation, dict):
            return False
        containers = (
            observation.get("robot"),
            observation.get("goal"),
            observation.get("pedestrians"),
        )
        values: list[Any] = [
            observation.get(key)
            for key in (
                "robot_position",
                "robot_heading",
                "robot_speed",
                "robot_radius",
                "goal_current",
                "goal_next",
                "pedestrians_positions",
                "pedestrians_velocities",
                "pedestrians_count",
                "pedestrians_radius",
                "occupancy_grid",
                "occupancy_grid_meta_origin",
                "occupancy_grid_meta_resolution",
                "occupancy_grid_meta_size",
                "occupancy_grid_meta_use_ego_frame",
                "occupancy_grid_meta_channel_indices",
            )
            if key in observation
        ]
        for container in containers:
            if isinstance(container, dict):
                values.extend(value for value in container.values() if value is not None)
        try:
            return all(np.all(np.isfinite(np.asarray(value, dtype=float))) for value in values)
        except (TypeError, ValueError):
            return False

    def _decision_payload(
        self,
        rows: list[dict[str, Any]],
        *,
        selected: dict[str, Any] | None,
        switch_reason: str,
        cycle_runtime_s: float,
    ) -> dict[str, Any]:
        """Build the public trace without retaining solver objects.

        Returns:
            dict[str, Any]: JSON-compatible decision trace for the latest planner cycle.
        """
        public_rows = [
            {key: value for key, value in row.items() if key != "_solve"} for row in rows
        ]
        return {
            "hypotheses": public_rows,
            "selected_hypothesis": None if selected is None else selected["hypothesis"],
            "switch_reason": switch_reason,
            "hysteresis": self._hysteresis_diagnostics(),
            "cycle_runtime_s": float(cycle_runtime_s),
            "control_period_s": float(self.topology_config.control_period_s),
            "runtime_budget_s": float(self.topology_config.max_runtime_s),
            "config_contract": {
                "arm": self.topology_config.arm,
                "enabled": self.topology_config.enabled,
                "predeclared_threshold_equal": math.isclose(
                    float(self.topology_config.max_runtime_s),
                    float(self.topology_config.control_period_s),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
            },
        }

    def _stop(
        self,
        *,
        rows: list[dict[str, Any]],
        reason: str,
        cycle_runtime_s: float,
        reset_commitment: bool = True,
    ) -> tuple[float, float]:
        """Record a classified fail-closed stop.

        Returns:
            tuple[float, float]: The safe zero command.
        """
        if reset_commitment:
            self._committed_hypothesis = None
            self._pending_hypothesis = None
            self._pending_ticks = 0
        self._last_solution = None
        self._stats["fallback_stop_count"] += 1
        self._last_decision = self._decision_payload(
            rows,
            selected=None,
            switch_reason=reason,
            cycle_runtime_s=cycle_runtime_s,
        )
        self._record_command(0.0, 0.0)
        return 0.0, 0.0

    def _solve_one(
        self,
        context: Any,
        initial_guess: np.ndarray,
        *,
        deadline: float,
    ) -> NMPCSolveResult:
        """Run one solve and classify unexpected solver-boundary exceptions.

        Returns:
            NMPCSolveResult: A solver result or a classified fail-closed error result.
        """
        try:
            result = self._solve_context(context, initial_guess, deadline=deadline)
        except Exception:  # noqa: BLE001 - experimental solver boundary must fail closed.
            return NMPCSolveResult(
                feasible=False,
                solution=None,
                objective=None,
                solver_status="solver_exception",
                solver_iterations=None,
                runtime_s=0.0,
                rollout_states=np.zeros((0, 3), dtype=float),
            )
        if perf_counter() >= deadline:
            return NMPCSolveResult(
                feasible=False,
                solution=None,
                objective=None,
                solver_status="runtime_budget_exceeded",
                solver_iterations=result.solver_iterations,
                runtime_s=float(result.runtime_s),
                rollout_states=np.zeros((0, 3), dtype=float),
            )
        return result

    def _row_from_result(
        self, label: str, initial_guess: np.ndarray, solve: NMPCSolveResult, *, context: Any
    ) -> dict[str, Any]:
        """Convert one solver result into the trace row used by selection.

        Returns:
            dict[str, Any]: Internal trace row containing the solver result and public fields.
        """
        self._stats["solver_successes"] += int(solve.feasible)
        self._stats["solver_failures"] += int(not solve.feasible)
        return {
            "hypothesis": label,
            "initialization_signature": self._signature(initial_guess),
            "rollout_signature": self._signature(solve.rollout_states),
            "feasible": bool(solve.feasible),
            "objective": solve.objective,
            "solver_status": solve.solver_status,
            "solver_iterations": solve.solver_iterations,
            "solver_runtime_s": solve.runtime_s,
            "rollout_diagnostics": self._rollout_diagnostics(solve.rollout_states, context=context),
            "_solve": solve,
        }

    def _future_result(self, future: Any, *, deadline: float) -> NMPCSolveResult:
        """Return a future result or a classified deadline/future failure."""
        remaining = deadline - perf_counter()
        if remaining <= 0.0:
            return self._runtime_limited_result()
        try:
            return future.result(timeout=remaining)
        except FutureTimeoutError:
            return self._runtime_limited_result()
        except Exception:  # noqa: BLE001 - future boundary must fail closed.
            return self._solver_exception_result()

    def _collect_rows(
        self,
        context: Any,
        initializations: list[tuple[str, np.ndarray]],
        *,
        deadline: float,
    ) -> list[dict[str, Any]]:
        """Solve candidates concurrently and return rows in canonical input order.

        Returns:
            list[dict[str, Any]]: Solver rows ordered by the canonical hypothesis order.
        """
        executor = None
        futures: list[Any] = []
        if len(initializations) > 1:
            executor = ThreadPoolExecutor(
                max_workers=len(initializations), thread_name_prefix="nmpc"
            )
            futures = [
                executor.submit(self._solve_one, context, initial_guess, deadline=deadline)
                for _label, initial_guess in initializations
            ]

        rows: list[dict[str, Any]] = []
        try:
            for index, (label, initial_guess) in enumerate(initializations):
                solve = (
                    self._solve_one(context, initial_guess, deadline=deadline)
                    if executor is None
                    else self._future_result(futures[index], deadline=deadline)
                )
                rows.append(self._row_from_result(label, initial_guess, solve, context=context))
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
        return rows

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Solve all topology seeds under one hard wall-clock deadline.

        Returns:
            tuple[float, float]: The selected first command or a safe zero command.
        """
        started = perf_counter()
        self._stats["calls"] = int(self._stats.get("calls", 0)) + 1
        if not self._observation_is_finite(observation):
            return self._stop(
                rows=[],
                reason="invalid_simulator_state",
                cycle_runtime_s=perf_counter() - started,
            )
        context, goal_heading_error, goal_distance = self._build_rollout_context(observation)
        if not self._context_is_finite(context):
            return self._stop(
                rows=[], reason="invalid_simulator_state", cycle_runtime_s=perf_counter() - started
            )
        if goal_distance <= float(self.config.goal_tolerance):
            self._last_decision = self._decision_payload(
                [],
                selected=None,
                switch_reason="goal_reached",
                cycle_runtime_s=perf_counter() - started,
            )
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
        deadline = started + float(self.topology_config.max_runtime_s)
        rows = self._collect_rows(context, initializations, deadline=deadline)

        cycle_runtime_s = perf_counter() - started
        if cycle_runtime_s > float(self.topology_config.max_runtime_s):
            return self._stop(
                rows=rows,
                reason="runtime_budget_exceeded",
                cycle_runtime_s=cycle_runtime_s,
            )

        self._annotate_material_geometry(rows, context)
        selected, switch_reason = self._select_hypothesis(rows)
        if selected is None or selected["_solve"].solution is None:
            return self._stop(
                rows=rows,
                reason=switch_reason,
                cycle_runtime_s=cycle_runtime_s,
                reset_commitment=False,
            )

        solution = np.asarray(selected["_solve"].solution, dtype=float)
        if solution.size == 0 or not np.all(np.isfinite(solution)):
            return self._stop(
                rows=rows,
                reason="invalid_selected_solution",
                cycle_runtime_s=cycle_runtime_s,
            )
        self._last_solution = solution.copy()
        try:
            linear, angular = self._command_from_solution(solution, context=context)
        except Exception:  # noqa: BLE001 - command boundary must fail closed.
            return self._stop(
                rows=rows,
                reason="command_conversion_error",
                cycle_runtime_s=cycle_runtime_s,
            )
        self._last_decision = self._decision_payload(
            rows,
            selected=selected,
            switch_reason=switch_reason,
            cycle_runtime_s=cycle_runtime_s,
        )
        self._record_command(linear, angular)
        return linear, angular

    @staticmethod
    def _runtime_limited_result() -> NMPCSolveResult:
        """Return a classified result for a solve that missed the shared deadline."""
        return NMPCSolveResult(
            feasible=False,
            solution=None,
            objective=None,
            solver_status="runtime_budget_exceeded",
            solver_iterations=None,
            runtime_s=0.0,
            rollout_states=np.zeros((0, 3), dtype=float),
        )

    @staticmethod
    def _solver_exception_result() -> NMPCSolveResult:
        """Return a classified result for an unexpected future failure."""
        return NMPCSolveResult(
            feasible=False,
            solution=None,
            objective=None,
            solver_status="solver_exception",
            solver_iterations=None,
            runtime_s=0.0,
            rollout_states=np.zeros((0, 3), dtype=float),
        )

    def _hysteresis_diagnostics(self) -> dict[str, Any]:
        """Return the bounded switch state for trace emission."""
        return {
            "committed_hypothesis": self._committed_hypothesis,
            "pending_hypothesis": self._pending_hypothesis,
            "pending_ticks": int(self._pending_ticks),
            "required_ticks": int(self.topology_config.hysteresis_ticks),
        }

    def _rollout_diagnostics(self, states: np.ndarray, *, context: Any) -> dict[str, Any]:
        """Summarize x-y-t geometry without adding a topology objective term.

        Returns:
            dict[str, Any]: Compact geometry and signed-side diagnostics.
        """
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] < 3:
            return {"status": "empty", "side_sign": 0}
        goal_delta = np.asarray(context.goal, dtype=float) - np.asarray(
            context.robot_pos, dtype=float
        )
        goal_norm = float(np.linalg.norm(goal_delta))
        if not np.isfinite(goal_norm) or goal_norm <= 1e-9:
            side_offsets = np.zeros(states.shape[0], dtype=float)
        else:
            direction = goal_delta / goal_norm
            displacement = states[:, :2] - np.asarray(context.robot_pos, dtype=float)
            side_offsets = direction[0] * displacement[:, 1] - direction[1] * displacement[:, 0]
        peak_index = int(np.argmax(np.abs(side_offsets)))
        peak_offset = float(side_offsets[peak_index])
        threshold = self._material_separation_threshold(context)
        side_sign = 1 if peak_offset >= threshold else -1 if peak_offset <= -threshold else 0
        return {
            "status": "optimized",
            "steps": int(states.shape[0]),
            "terminal_position": [float(value) for value in states[-1, :2]],
            "lateral_span": float(np.ptp(states[:, 1])),
            "signed_lateral_offset_m": peak_offset,
            "side_sign": side_sign,
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
    """Build the experimental config and fail closed on semantic mismatches.

    Returns:
        TopologyParallelNMPCConfig: Parsed experimental planner configuration.
    """
    if cfg is None:
        return TopologyParallelNMPCConfig()
    if not isinstance(cfg, dict):
        raise ValueError("topology_parallel_nmpc config must be a mapping")

    defaults = TopologyParallelNMPCConfig()
    raw_labels = cfg.get("hypothesis_labels", defaults.hypothesis_labels)
    if isinstance(raw_labels, str):
        raw_labels = tuple(item.strip() for item in raw_labels.split(",") if item.strip())
    if not isinstance(raw_labels, (list, tuple)):
        raise ValueError("hypothesis_labels must be a list, tuple, or comma-separated string")
    labels = tuple(raw_labels)
    if any(not isinstance(label, str) for label in labels):
        raise ValueError("hypothesis_labels must contain strings")
    unknown = tuple(label for label in labels if label not in _VALID_HYPOTHESIS_LABELS)
    if unknown:
        raise ValueError(f"unknown hypothesis labels: {unknown!r}")
    if len(set(labels)) != len(labels):
        raise ValueError("hypothesis_labels must be unique")
    ordered_labels = (
        labels
        if labels == (_DEFAULT_HYPOTHESIS,)
        else tuple(label for label in _HYPOTHESIS_ORDER if label in labels)
    )

    try:
        topology = TopologyParallelNMPCConfig(
            nmpc=build_nmpc_social_config(cfg),
            arm=cfg.get("arm", defaults.arm),
            enabled=_parse_bool(cfg.get("enabled", defaults.enabled)),
            hypothesis_labels=ordered_labels,
            max_hypotheses=_require_integer(
                "max_hypotheses", cfg.get("max_hypotheses", defaults.max_hypotheses)
            ),
            initialization_turn_rate=float(
                cfg.get("initialization_turn_rate", defaults.initialization_turn_rate)
            ),
            yield_speed_scale=float(cfg.get("yield_speed_scale", defaults.yield_speed_scale)),
            hysteresis_ticks=_require_integer(
                "hysteresis_ticks", cfg.get("hysteresis_ticks", defaults.hysteresis_ticks)
            ),
            objective_tolerance=float(cfg.get("objective_tolerance", defaults.objective_tolerance)),
            control_period_s=float(cfg.get("control_period_s", defaults.control_period_s)),
            max_runtime_s=float(cfg.get("max_runtime_s", defaults.max_runtime_s)),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid topology-parallel NMPC config: {exc}") from exc
    return topology


__all__ = [
    "TopologyParallelNMPCConfig",
    "TopologyParallelNMPCPlannerAdapter",
    "build_topology_parallel_nmpc_config",
]
