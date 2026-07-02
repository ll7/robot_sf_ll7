"""Gymnasium reward wrapper for PPO-Lagrangian constrained training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gymnasium import Wrapper

from robot_sf.training.safety_constraints import (
    LagrangeMultiplierState,
    SafetyConstraintSpec,
    step_safety_costs,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class ConstrainedRewardWrapper(Wrapper):
    """Apply Lagrangian safety-cost penalties while preserving raw task reward diagnostics."""

    def __init__(
        self,
        env: Any,
        constraints: Sequence[SafetyConstraintSpec],
        *,
        multiplier_state: LagrangeMultiplierState | None = None,
    ) -> None:
        """Initialize the wrapper around a Gymnasium-compatible environment."""
        super().__init__(env)
        self.constraints = tuple(constraints)
        if not self.constraints:
            raise ValueError("ConstrainedRewardWrapper requires at least one constraint")
        self.multiplier_state = multiplier_state or LagrangeMultiplierState.from_specs(
            self.constraints
        )

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset wrapped environment and clear partial episode costs.

        Returns:
            Wrapped environment reset observation and info.
        """
        self.multiplier_state.reset_episode_costs(self.constraints)
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply the constrained reward transform to one environment step.

        Returns:
            Gymnasium step tuple with constrained reward and added diagnostics.
        """
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        costs = step_safety_costs(info, self.constraints)
        multipliers = dict(self.multiplier_state.values)
        penalty = sum(multipliers.get(name, 0.0) * cost for name, cost in costs.items())
        constrained_reward = float(raw_reward) - penalty
        self.multiplier_state.observe_step(costs)

        info["constraint_costs"] = costs
        info["constraint_multipliers"] = multipliers
        info["raw_task_reward"] = float(raw_reward)
        info["constrained_reward"] = constrained_reward

        if terminated or truncated:
            info["constraint_episode"] = self.multiplier_state.episode_summary(self.constraints)
            self.multiplier_state.reset_episode_costs(self.constraints)

        return observation, constrained_reward, terminated, truncated, info

    def update_multipliers_from_episode(
        self,
        episode_costs: dict[str, float],
        *,
        episode_steps: int | None = None,
    ) -> dict[str, float]:
        """Update Lagrange multipliers from externally collected episode diagnostics.

        Returns:
            Updated multiplier values keyed by constraint name.
        """
        return self.multiplier_state.update_after_episode(
            self.constraints,
            episode_costs=episode_costs,
            episode_steps=episode_steps,
        )


__all__ = ["ConstrainedRewardWrapper"]
