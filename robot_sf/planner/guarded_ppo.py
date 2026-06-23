"""Safety guard for PPO benchmark actions.

This adapter keeps PPO as the primary action source and only intervenes when a
short-horizon rollout predicts unsafe pedestrian or obstacle clearance.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Protocol

import numpy as np

from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, _wrap_angle, build_risk_dwa_config
from robot_sf.planner.safety_shield import ShieldDecision
from robot_sf.planner.socnav import (
    OccupancyAwarePlannerMixin,
    ORCAPlannerAdapter,
    SocNavPlannerConfig,
)


class _CommandPlanner(Protocol):
    """Protocol for local planners that emit benchmark unicycle commands."""

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return a ``(linear, angular)`` command for one observation."""


@dataclass
class GuardedPPOConfig:
    """Configuration for the PPO safety guard."""

    rollout_dt: float = 0.2
    rollout_steps: int = 6
    goal_tolerance: float = 0.25
    near_field_distance: float = 2.0
    hard_ped_clearance: float = 0.58
    first_step_ped_clearance: float = 0.72
    hard_obstacle_clearance: float = 0.30
    min_ttc: float = 0.70
    obstacle_threshold: float = 0.5
    obstacle_search_cells: int = 12
    prior_blend_weight: float = 0.0
    prior_near_field_only: bool = True
    prior_progress_margin: float = 0.05
    prior_residual_mode: bool = False
    prior_residual_max_linear_delta: float = 0.25
    prior_residual_max_angular_delta: float = 0.35


class GuardedPPOAdapter(OccupancyAwarePlannerMixin):
    """Intervene on PPO actions only when they violate short-horizon safety checks."""

    def __init__(
        self,
        config: GuardedPPOConfig | None = None,
        *,
        fallback_adapter: _CommandPlanner | None = None,
        prior_adapter: _CommandPlanner | None = None,
    ) -> None:
        """Initialize guard and fallback local planner."""
        self.config = config or GuardedPPOConfig()
        self.fallback_adapter = fallback_adapter or RiskDWAPlannerAdapter()
        self.prior_adapter = prior_adapter
        self._last_action_adaptation: dict[str, Any] | None = None

    def _child_adapters(self) -> tuple[_CommandPlanner, ...]:
        """Return configured child planners that may own episode-local state."""
        if self.prior_adapter is None:
            return (self.fallback_adapter,)
        return (self.fallback_adapter, self.prior_adapter)

    def bind_env(self, env: Any) -> None:
        """Propagate environment binding to child planners that need map context."""
        for adapter in self._child_adapters():
            bind_env = getattr(adapter, "bind_env", None)
            if callable(bind_env):
                bind_env(env)

    def reset(self, *, seed: int | None = None) -> None:
        """Reset child planner state between benchmark episodes."""
        for adapter in self._child_adapters():
            reset = getattr(adapter, "reset", None)
            if not callable(reset):
                continue
            if seed is None:
                reset()
                continue
            try:
                reset(seed=seed)
            except TypeError:
                reset()

    def close(self) -> None:
        """Close child planners that own external resources."""
        for adapter in self._child_adapters():
            close = getattr(adapter, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _blend_commands(
        ppo_command: tuple[float, float],
        prior_command: tuple[float, float],
        weight: float,
    ) -> tuple[float, float]:
        """Blend PPO and prior commands as a lightweight residual correction.

        Returns:
            tuple[float, float]: Blended ``(linear, angular)`` command.
        """
        clipped_weight = float(np.clip(weight, 0.0, 1.0))
        return (
            (1.0 - clipped_weight) * float(ppo_command[0])
            + clipped_weight * float(prior_command[0]),
            (1.0 - clipped_weight) * float(ppo_command[1])
            + clipped_weight * float(prior_command[1]),
        )

    def _prior_command(self, observation: dict[str, Any]) -> tuple[float, float] | None:
        """Return the optional prior planner command, suppressing unavailable priors."""
        if self.prior_adapter is None:
            return None
        try:
            command = self.prior_adapter.plan(observation)
        except RuntimeError:
            return None
        return float(command[0]), float(command[1])

    def _residual_prior_command(
        self,
        ppo_command: tuple[float, float],
        prior_command: tuple[float, float],
    ) -> tuple[float, float]:
        """Return ORCA-prior command plus a bounded PPO residual.

        The residual is computed as ``ppo - prior`` so existing PPO-style actions remain
        interpretable while the prior stays the base command.
        """
        raw = (float(ppo_command[0]), float(ppo_command[1]))
        prior = (float(prior_command[0]), float(prior_command[1]))
        linear_bound = max(float(self.config.prior_residual_max_linear_delta), 0.0)
        angular_bound = max(float(self.config.prior_residual_max_angular_delta), 0.0)
        raw_residual = (raw[0] - prior[0], raw[1] - prior[1])
        clipped = (
            float(np.clip(raw_residual[0], -linear_bound, linear_bound)),
            float(np.clip(raw_residual[1], -angular_bound, angular_bound)),
        )
        adapted = (float(prior[0]) + clipped[0], float(prior[1]) + clipped[1])
        self._last_action_adaptation = {
            "mode": "prior_residual",
            "status": "adapted",
            "nominal_orca_action": [float(prior[0]), float(prior[1])],
            "raw_policy_action": [raw[0], raw[1]],
            "raw_residual_action": [raw_residual[0], raw_residual[1]],
            "bounded_residual_action": [clipped[0], clipped[1]],
            "adapted_action": [adapted[0], adapted[1]],
            "residual_bounds": {
                "linear": linear_bound,
                "angular": angular_bound,
            },
            "residual_clipped": bool(
                not np.isclose(raw_residual[0], clipped[0])
                or not np.isclose(raw_residual[1], clipped[1])
            ),
            "hard_guard_authoritative": True,
        }
        return adapted

    def _blend_is_preferred(
        self,
        ppo_eval: dict[str, float | bool],
        blend_eval: dict[str, float | bool],
    ) -> bool:
        """Return whether a safe prior blend improves safety without stalling progress."""
        if not bool(blend_eval["safe"]):
            return False
        progress_margin = float(self.config.prior_progress_margin)
        if float(blend_eval["progress"]) < float(ppo_eval["progress"]) - progress_margin:
            return False
        ppo_clear = min(float(ppo_eval["min_ped_clear"]), float(ppo_eval["first_ped_clear"]))
        blend_clear = min(
            float(blend_eval["min_ped_clear"]),
            float(blend_eval["first_ped_clear"]),
        )
        ppo_ttc = float(ppo_eval["min_ttc"])
        blend_ttc = float(blend_eval["min_ttc"])
        clear_improves = blend_clear > ppo_clear
        ttc_improves = (not np.isfinite(blend_ttc) and np.isfinite(ppo_ttc)) or (
            np.isfinite(blend_ttc) and np.isfinite(ppo_ttc) and blend_ttc > ppo_ttc
        )
        return bool(clear_improves or ttc_improves)

    def _extract_state(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """Extract robot, goal, and pedestrian state from structured observation.

        Returns:
            tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
                Robot position, heading, goal position, pedestrian positions, pedestrian velocities.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = self._as_1d_float(robot_state.get("position", [0.0, 0.0]), pad=2)[:2]
        heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        goal_next = self._as_1d_float(goal_state.get("next", [0.0, 0.0]), pad=2)[:2]
        goal_current = self._as_1d_float(goal_state.get("current", [0.0, 0.0]), pad=2)[:2]
        current_dist = float(np.linalg.norm(goal_current - robot_pos))
        next_dist = float(np.linalg.norm(goal_next - robot_pos))
        if current_dist > float(self.config.goal_tolerance):
            goal = goal_current
        elif next_dist > 1e-6:
            goal = goal_next
        else:
            goal = goal_current

        ped_positions_raw = ped_state.get("positions")
        ped_velocities_raw = ped_state.get("velocities")
        ped_pos = np.asarray([] if ped_positions_raw is None else ped_positions_raw, dtype=float)
        ped_vel = np.asarray([] if ped_velocities_raw is None else ped_velocities_raw, dtype=float)
        ped_count_arr = self._as_1d_float(ped_state.get("count", []), pad=0)
        ped_count = int(max(ped_count_arr[0], 0.0)) if ped_count_arr.size > 0 else None
        if (
            ped_pos.ndim == 1
            and ped_count is not None
            and ped_count > 0
            and ped_pos.size >= ped_count * 2
        ):
            ped_pos = ped_pos.reshape(-1, 2)[:ped_count]
        if (
            ped_vel.ndim == 1
            and ped_count is not None
            and ped_count > 0
            and ped_vel.size >= ped_count * 2
        ):
            ped_vel = ped_vel.reshape(-1, 2)[:ped_count]
        if ped_pos.ndim != 2 or ped_pos.shape[-1] != 2:
            ped_pos = np.zeros((0, 2), dtype=float)
        elif ped_count is not None:
            ped_pos = ped_pos[: min(ped_count, ped_pos.shape[0])]
        if ped_vel.ndim != 2 or ped_vel.shape[-1] != 2 or ped_vel.shape[0] != ped_pos.shape[0]:
            ped_vel = np.zeros_like(ped_pos)
        elif ped_count is not None:
            ped_vel = ped_vel[: min(ped_count, ped_vel.shape[0])]
        return robot_pos, heading, goal, ped_pos, ped_vel

    def _min_obstacle_clearance(self, point: np.ndarray, observation: dict[str, Any]) -> float:
        """Approximate obstacle clearance from occupancy grid payload.

        Returns:
            float: Clearance in meters, or ``inf`` when unavailable.
        """
        payload = self._extract_grid_payload(observation)
        if payload is None:
            return float("inf")
        grid, meta = payload
        channel = self._preferred_channel(meta)
        if channel < 0 or channel >= grid.shape[0]:
            return float("inf")

        rc = self._world_to_grid(point, meta, grid_shape=(grid.shape[1], grid.shape[2]))
        if rc is None:
            return 0.0
        row, col = rc
        channel_grid = np.asarray(grid[channel], dtype=float)
        threshold = float(self.config.obstacle_threshold)
        if channel_grid[row, col] >= threshold:
            return 0.0

        radius = max(int(self.config.obstacle_search_cells), 1)
        r0 = max(0, row - radius)
        r1 = min(channel_grid.shape[0], row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(channel_grid.shape[1], col + radius + 1)
        window = channel_grid[r0:r1, c0:c1]
        obs_idx = np.argwhere(window >= threshold)
        if obs_idx.size == 0:
            return float("inf")

        dr = obs_idx[:, 0] + r0 - row
        dc = obs_idx[:, 1] + c0 - col
        cell_dist = np.sqrt(dr.astype(float) ** 2 + dc.astype(float) ** 2)
        resolution = float(self._as_1d_float(meta.get("resolution", [0.2]), pad=1)[0])
        return float(np.min(cell_dist) * max(resolution, 1e-6))

    def _evaluate_command(
        self, observation: dict[str, Any], command: tuple[float, float]
    ) -> dict[str, float | bool]:
        """Evaluate a command over a short rollout horizon.

        Returns:
            dict[str, float | bool]: Safety summary including `safe` and clearance metrics.
        """
        robot_pos, heading, goal, ped_pos, ped_vel = self._extract_state(observation)
        dt = float(self.config.rollout_dt)
        steps = max(int(self.config.rollout_steps), 1)
        x = np.array(robot_pos, dtype=float)
        theta = float(heading)
        start_dist = float(np.linalg.norm(goal - robot_pos))
        min_ped_clear = float("inf")
        first_ped_clear = float("inf")
        min_obs_clear = float("inf")
        min_ttc = float("inf")

        for step in range(steps):
            t = (step + 1) * dt
            x = x + np.array(
                [
                    float(command[0]) * np.cos(theta) * dt,
                    float(command[0]) * np.sin(theta) * dt,
                ],
                dtype=float,
            )
            theta = _wrap_angle(theta + float(command[1]) * dt)

            if ped_pos.size > 0:
                ped_t = ped_pos + ped_vel * t
                ped_dist = np.linalg.norm(ped_t - x[None, :], axis=1)
                if ped_dist.size > 0:
                    min_ped_clear = min(min_ped_clear, float(np.min(ped_dist)))
                    if step == 0:
                        first_ped_clear = min(first_ped_clear, float(np.min(ped_dist)))

                    robot_vel = np.array(
                        [
                            float(command[0]) * np.cos(theta),
                            float(command[0]) * np.sin(theta),
                        ],
                        dtype=float,
                    )
                    rel_pos = ped_t - x[None, :]
                    rel_vel = ped_vel - robot_vel[None, :]
                    rel_speed_sq = np.sum(rel_vel * rel_vel, axis=1)
                    valid = rel_speed_sq > 1e-6
                    if np.any(valid):
                        ttc = -np.sum(rel_pos[valid] * rel_vel[valid], axis=1) / rel_speed_sq[valid]
                        ttc = ttc[ttc > 0.0]
                        if ttc.size > 0:
                            min_ttc = min(min_ttc, float(np.min(ttc)))
            min_obs_clear = min(min_obs_clear, self._min_obstacle_clearance(x, observation))

        end_dist = float(np.linalg.norm(goal - x))
        progress = start_dist - end_dist
        safe = (
            min_ped_clear >= float(self.config.hard_ped_clearance)
            and first_ped_clear >= float(self.config.first_step_ped_clearance)
            and min_obs_clear >= float(self.config.hard_obstacle_clearance)
            and (not np.isfinite(min_ttc) or min_ttc >= float(self.config.min_ttc))
        )
        return {
            "safe": bool(safe),
            "progress": float(progress),
            "min_ped_clear": float(min_ped_clear),
            "first_ped_clear": float(first_ped_clear),
            "min_obs_clear": float(min_obs_clear),
            "min_ttc": float(min_ttc),
        }

    def choose_command(
        self, observation: dict[str, Any], ppo_command: tuple[float, float]
    ) -> tuple[tuple[float, float], str]:
        """Choose between PPO, fallback planner, and stop.

        Returns:
            tuple[tuple[float, float], str]: Selected command and decision label.
        """
        return self.choose_command_decision(observation, ppo_command).as_command_result()

    def choose_command_decision(  # noqa: C901
        self, observation: dict[str, Any], ppo_command: tuple[float, float]
    ) -> ShieldDecision:
        """Choose a command and return structured safety-shield metadata.

        Returns:
            ShieldDecision: Proposed action, selected action, and shield decision metadata.
        """
        self._last_action_adaptation = {
            "mode": "direct_policy_command",
            "raw_policy_action": [float(ppo_command[0]), float(ppo_command[1])],
            "adapted_action": [float(ppo_command[0]), float(ppo_command[1])],
            # No bounded residual was applied on a direct pass-through, so the
            # residual was genuinely not clipped this decision. Emitting the key
            # keeps the per-decision residual-clip signal present even when no
            # learned checkpoint surfaces aggregate residual_clipping_stats.
            "residual_clipped": False,
            "hard_guard_authoritative": True,
        }
        robot_pos, _heading, goal, ped_pos, _ped_vel = self._extract_state(observation)
        if float(np.linalg.norm(goal - robot_pos)) <= float(self.config.goal_tolerance):
            return self._shield_decision(
                ppo_command=ppo_command,
                filtered_command=(0.0, 0.0),
                label="goal_reached",
                reason="goal_within_tolerance",
                ppo_eval={"safe": True},
                selected_eval={"safe": True},
                intervened=False,
            )

        if ped_pos.size > 0:
            current_min_dist = float(np.min(np.linalg.norm(ped_pos - robot_pos[None, :], axis=1)))
        else:
            current_min_dist = float("inf")

        prior_allowed_by_scene = not bool(
            self.config.prior_near_field_only
        ) or current_min_dist <= float(self.config.near_field_distance)
        ppo_eval = self._evaluate_command(observation, ppo_command)
        prior_command = self._prior_command(observation) if prior_allowed_by_scene else None
        prior_eval: dict[str, float | bool] | None = None
        if (
            bool(self.config.prior_residual_mode)
            and prior_command is not None
            and not bool(ppo_eval["safe"])
        ):
            prior_eval = self._evaluate_command(observation, prior_command)
            residual_command = self._residual_prior_command(ppo_command, prior_command)
            residual_eval = self._evaluate_command(observation, residual_command)
            residual_improves_ppo = self._blend_is_preferred(ppo_eval, residual_eval)
            residual_improves_prior = not bool(prior_eval["safe"]) or self._blend_is_preferred(
                prior_eval, residual_eval
            )
            if bool(residual_eval["safe"]) and residual_improves_ppo and residual_improves_prior:
                return self._shield_decision(
                    ppo_command=ppo_command,
                    filtered_command=residual_command,
                    label="prior_residual_safe",
                    reason="bounded_prior_residual_improved_short_horizon_safety",
                    ppo_eval=ppo_eval,
                    selected_eval=residual_eval,
                    fallback_policy="prior_residual",
                )

        prior_weight = float(self.config.prior_blend_weight)
        use_prior_in_scene = prior_command is not None and prior_weight > 0.0
        if use_prior_in_scene and prior_command is not None:
            blended_command = self._blend_commands(ppo_command, prior_command, prior_weight)
            blended_eval = self._evaluate_command(observation, blended_command)
            if self._blend_is_preferred(ppo_eval, blended_eval):
                return self._shield_decision(
                    ppo_command=ppo_command,
                    filtered_command=blended_command,
                    label="prior_blend_safe",
                    reason="prior_blend_improved_short_horizon_safety",
                    ppo_eval=ppo_eval,
                    selected_eval=blended_eval,
                    fallback_policy=type(self.prior_adapter).__name__,
                )

        if current_min_dist > float(self.config.near_field_distance) and bool(ppo_eval["safe"]):
            return self._shield_decision(
                ppo_command=ppo_command,
                filtered_command=(float(ppo_command[0]), float(ppo_command[1])),
                label="ppo_clear",
                reason="proposed_action_clear_of_near_field",
                ppo_eval=ppo_eval,
                selected_eval=ppo_eval,
                intervened=False,
            )
        if bool(ppo_eval["safe"]):
            return self._shield_decision(
                ppo_command=ppo_command,
                filtered_command=(float(ppo_command[0]), float(ppo_command[1])),
                label="ppo_safe",
                reason="proposed_action_satisfies_shield",
                ppo_eval=ppo_eval,
                selected_eval=ppo_eval,
                intervened=False,
            )

        if prior_command is not None:
            if prior_eval is None:
                prior_eval = self._evaluate_command(observation, prior_command)
            if bool(prior_eval["safe"]):
                return self._shield_decision(
                    ppo_command=ppo_command,
                    filtered_command=(float(prior_command[0]), float(prior_command[1])),
                    label="prior_safe",
                    reason="prior_command_satisfied_short_horizon_constraints",
                    ppo_eval=ppo_eval,
                    selected_eval=prior_eval,
                    fallback_policy=type(self.prior_adapter).__name__,
                )

        fallback_command = self.fallback_adapter.plan(observation)
        fallback_eval = self._evaluate_command(observation, fallback_command)
        if bool(fallback_eval["safe"]):
            return self._shield_decision(
                ppo_command=ppo_command,
                filtered_command=(float(fallback_command[0]), float(fallback_command[1])),
                label="fallback_safe",
                reason="fallback_command_satisfied_short_horizon_constraints",
                ppo_eval=ppo_eval,
                selected_eval=fallback_eval,
                fallback_policy=type(self.fallback_adapter).__name__,
            )

        stop_eval = self._evaluate_command(observation, (0.0, 0.0))
        if bool(stop_eval["safe"]):
            return self._shield_decision(
                ppo_command=ppo_command,
                filtered_command=(0.0, 0.0),
                label="stop_safe",
                reason="stop_command_satisfied_short_horizon_constraints",
                ppo_eval=ppo_eval,
                selected_eval=stop_eval,
                fallback_policy="stop",
            )

        if float(fallback_eval["min_ped_clear"]) > max(
            float(ppo_eval["min_ped_clear"]),
            float(stop_eval["min_ped_clear"]),
            float(prior_eval["min_ped_clear"]) if prior_eval is not None else float("-inf"),
        ):
            return self._shield_decision(
                ppo_command=ppo_command,
                filtered_command=(float(fallback_command[0]), float(fallback_command[1])),
                label="fallback_best_effort",
                reason="no_safe_command_available_fallback_has_largest_clearance",
                ppo_eval=ppo_eval,
                selected_eval=fallback_eval,
                fallback_policy=type(self.fallback_adapter).__name__,
                hard_constraint_violation=True,
            )
        return self._shield_decision(
            ppo_command=ppo_command,
            filtered_command=(0.0, 0.0),
            label="stop_best_effort",
            reason="no_safe_command_available_stop_has_largest_clearance",
            ppo_eval=ppo_eval,
            selected_eval=stop_eval,
            fallback_policy="stop",
            hard_constraint_violation=True,
        )

    def _shield_decision(  # noqa: PLR0913
        self,
        *,
        ppo_command: tuple[float, float],
        filtered_command: tuple[float, float],
        label: str,
        reason: str,
        ppo_eval: dict[str, float | bool],
        selected_eval: dict[str, float | bool],
        fallback_policy: str | None = None,
        intervened: bool | None = None,
        hard_constraint_violation: bool | None = None,
    ) -> ShieldDecision:
        """Build a structured shield decision for benchmark metadata.

        Returns:
            ShieldDecision: Serialized-ready action-filter decision.
        """
        explicit_intervened = (
            label not in {"ppo_clear", "ppo_safe", "goal_reached"}
            if intervened is None
            else bool(intervened)
        )
        explicit_violation = (
            bool(selected_eval.get("safe") is False)
            if hard_constraint_violation is None
            else bool(hard_constraint_violation)
        )
        action_adaptation = self._last_action_adaptation or {
            "mode": "unknown",
            "residual_clipped": False,
            "hard_guard_authoritative": True,
        }
        selected_matches_ppo = np.isclose(float(ppo_command[0]), float(filtered_command[0])) and (
            np.isclose(float(ppo_command[1]), float(filtered_command[1]))
        )
        adapted_action = action_adaptation.get("adapted_action")
        selected_matches_adaptation = False
        if isinstance(adapted_action, list | tuple) and len(adapted_action) >= 2:
            selected_matches_adaptation = bool(
                np.isclose(float(adapted_action[0]), float(filtered_command[0]))
                and np.isclose(float(adapted_action[1]), float(filtered_command[1]))
            )
        if not selected_matches_ppo and not selected_matches_adaptation:
            action_adaptation = {
                "mode": "guard_selected_command",
                "raw_policy_action": [float(ppo_command[0]), float(ppo_command[1])],
                "adapted_action": [float(filtered_command[0]), float(filtered_command[1])],
                # The guard selected a fallback/prior command rather than a
                # bounded residual, so no residual clipping occurred this step.
                "residual_clipped": False,
                "hard_guard_authoritative": True,
            }
        return ShieldDecision(
            proposed_action=(float(ppo_command[0]), float(ppo_command[1])),
            filtered_action=(float(filtered_command[0]), float(filtered_command[1])),
            decision_label=label,
            intervention_reason=reason,
            violated_constraints=self._violated_constraints(ppo_eval),
            prediction_source="short_horizon_rollout",
            prediction_horizon_steps=int(self.config.rollout_steps),
            prediction_dt=float(self.config.rollout_dt),
            uncertainty_metadata={"mode": "deterministic_rollout"},
            calibration_metadata={"status": "not_calibrated"},
            fallback_controller_state={
                "policy": fallback_policy or type(self.fallback_adapter).__name__,
                "prior_available": self.prior_adapter is not None,
                "action_adaptation": action_adaptation,
            },
            proposed_evaluation=dict(ppo_eval),
            selected_evaluation=dict(selected_eval),
            intervened=explicit_intervened,
            hard_constraint_violation=explicit_violation,
        )

    def _violated_constraints(self, evaluation: dict[str, float | bool]) -> tuple[str, ...]:
        """Return hard constraints violated by an evaluated proposed command."""
        if bool(evaluation.get("safe", True)):
            return ()
        violations: list[str] = []
        min_ped_clear = float(evaluation.get("min_ped_clear", float("inf")))
        first_ped_clear = float(evaluation.get("first_ped_clear", float("inf")))
        min_obs_clear = float(evaluation.get("min_obs_clear", float("inf")))
        min_ttc = float(evaluation.get("min_ttc", float("inf")))
        if min_ped_clear < float(self.config.hard_ped_clearance):
            violations.append("pedestrian_clearance")
        if first_ped_clear < float(self.config.first_step_ped_clearance):
            violations.append("first_step_pedestrian_clearance")
        if min_obs_clear < float(self.config.hard_obstacle_clearance):
            violations.append("obstacle_clearance")
        if np.isfinite(min_ttc) and min_ttc < float(self.config.min_ttc):
            violations.append("time_to_collision")
        return tuple(violations)


def build_guarded_ppo_config(cfg: dict[str, Any] | None) -> GuardedPPOConfig:
    """Build :class:`GuardedPPOConfig` from mapping payload.

    Returns:
        GuardedPPOConfig: Parsed guard configuration.
    """
    if not isinstance(cfg, dict):
        return GuardedPPOConfig()
    return GuardedPPOConfig(
        rollout_dt=float(cfg.get("guard_rollout_dt", 0.2)),
        rollout_steps=int(cfg.get("guard_rollout_steps", 6)),
        goal_tolerance=float(cfg.get("goal_tolerance", 0.25)),
        near_field_distance=float(cfg.get("guard_near_field_distance", 2.0)),
        hard_ped_clearance=float(cfg.get("guard_hard_ped_clearance", 0.58)),
        first_step_ped_clearance=float(cfg.get("guard_first_step_ped_clearance", 0.72)),
        hard_obstacle_clearance=float(cfg.get("guard_hard_obstacle_clearance", 0.30)),
        min_ttc=float(cfg.get("guard_min_ttc", 0.70)),
        obstacle_threshold=float(cfg.get("guard_obstacle_threshold", 0.5)),
        obstacle_search_cells=int(cfg.get("guard_obstacle_search_cells", 12)),
        prior_blend_weight=float(cfg.get("prior_blend_weight", 0.0)),
        prior_near_field_only=bool(cfg.get("prior_near_field_only", True)),
        prior_progress_margin=float(cfg.get("prior_progress_margin", 0.05)),
        prior_residual_mode=bool(
            cfg.get("prior_residual_mode", cfg.get("residual_enabled", False))
        ),
        prior_residual_max_linear_delta=float(
            cfg.get(
                "prior_residual_max_linear_delta",
                cfg.get("residual_linear_bound", 0.25),
            )
        ),
        prior_residual_max_angular_delta=float(
            cfg.get(
                "prior_residual_max_angular_delta",
                cfg.get("residual_angular_bound", 0.35),
            )
        ),
    )


def _build_socnav_orca_config(cfg: dict[str, Any] | None) -> SocNavPlannerConfig:
    """Build an ORCA planner config from guarded-PPO prior/fallback payloads.

    Returns:
        SocNavPlannerConfig: ORCA-compatible SocNav planner configuration.
    """
    if not isinstance(cfg, dict):
        return SocNavPlannerConfig()
    allowed = {field.name for field in fields(SocNavPlannerConfig)}
    return SocNavPlannerConfig(**{key: value for key, value in cfg.items() if key in allowed})


def build_guarded_ppo_prior(cfg: dict[str, Any] | None) -> _CommandPlanner | None:
    """Build an optional planner prior for residual guarded PPO.

    Returns:
        _CommandPlanner | None: ORCA prior when requested, otherwise ``None``.
    """
    root = cfg if isinstance(cfg, dict) else {}
    prior_policy = str(root.get("prior_policy", "none")).strip().lower()
    if prior_policy in {"", "none", "disabled"}:
        return None
    if prior_policy in {"orca", "socnav_orca"}:
        prior_cfg = root.get("prior_orca")
        if not isinstance(prior_cfg, dict):
            prior_cfg = {}
        return ORCAPlannerAdapter(
            config=_build_socnav_orca_config(prior_cfg),
            allow_fallback=bool(root.get("prior_allow_orca_fallback", False)),
        )
    raise ValueError(f"Unsupported guarded PPO prior_policy: {prior_policy}")


def build_guarded_ppo_fallback(cfg: dict[str, Any] | None) -> _CommandPlanner:
    """Build fallback local planner for guarded PPO.

    Returns:
        _CommandPlanner: Fallback adapter used when PPO action is unsafe.
    """
    root = cfg if isinstance(cfg, dict) else {}
    fallback_policy = str(root.get("fallback_policy", "risk_dwa")).strip().lower()
    if fallback_policy in {"orca", "socnav_orca"}:
        fallback_cfg = root.get("fallback_orca")
        if not isinstance(fallback_cfg, dict):
            fallback_cfg = {}
        return ORCAPlannerAdapter(
            config=_build_socnav_orca_config(fallback_cfg),
            allow_fallback=bool(root.get("fallback_allow_orca_fallback", False)),
        )
    if fallback_policy not in {"risk_dwa", "dwa", ""}:
        raise ValueError(f"Unsupported guarded PPO fallback_policy: {fallback_policy}")
    fallback_cfg = root.get("fallback_risk_dwa")
    if not isinstance(fallback_cfg, dict):
        fallback_cfg = {}
    return RiskDWAPlannerAdapter(config=build_risk_dwa_config(fallback_cfg))


__all__ = [
    "GuardedPPOAdapter",
    "GuardedPPOConfig",
    "build_guarded_ppo_config",
    "build_guarded_ppo_fallback",
    "build_guarded_ppo_prior",
]
