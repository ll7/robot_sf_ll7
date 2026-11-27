"""Development sampler with early passing guidance and soft footprint clearance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from robot_sf.planner.socnav_sampling_v2 import _positive
from robot_sf.planner.velocity_sampler_r2 import _cross
from robot_sf.planner.velocity_sampler_r3 import (
    StallRecoveryVelocitySamplerAdapter,
    StallRecoveryWeights,
)


@dataclass(frozen=True)
class PassingWeights(StallRecoveryWeights):
    """Set soft passing and predicted pedestrian-clearance costs."""

    approach_range_m: float = 10.0
    approach_weight: float = 1.5
    passing_padding_m: float = 0.3
    clearance_margin_m: float = 0.1
    clearance_weight: float = 4.0

    def __post_init__(self) -> None:
        """Reject invalid passing settings."""
        super().__post_init__()
        values = np.array(
            [
                self.approach_range_m,
                self.approach_weight,
                self.passing_padding_m,
                self.clearance_margin_m,
                self.clearance_weight,
            ]
        )
        if not np.isfinite(values).all() or self.approach_range_m <= 0 or (values[1:] < 0).any():
            raise ValueError("invalid passing settings")


PassingWeightsT = TypeVar("PassingWeightsT", bound=PassingWeights)


class PassingVelocitySamplerAdapter(StallRecoveryVelocitySamplerAdapter[PassingWeightsT]):
    """Favor an early passing side without rejecting pedestrian-near commands."""

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the passing preference between episodes."""
        super().reset(seed=seed)
        self._approach_side = 0
        self._approach_steps = 0

    def plan(self, observation: dict) -> tuple[float, float]:
        """Set the passing offset from the observed robot and pedestrian radii.

        Returns:
            tuple[float, float]: Reachable linear and angular velocity targets.
        """
        robot, _, pedestrians = self._socnav_fields(observation)
        radius = _positive(robot.get("radius")) or self._velocity_drive.radius
        ped_radius = _positive(pedestrians.get("radius") if pedestrians else None) or 0.0
        self._passing_offset = radius + ped_radius + self.weights.passing_padding_m
        return super().plan(observation)

    def _select_candidate(  # noqa: PLR0913
        self, costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
    ) -> int:
        weights = self.weights
        relative = peds - points[0, 0]
        # Competing pedestrian preferences can block progress; retain the base
        # selector until at most one person is nearby.
        if (np.linalg.norm(relative, axis=1) < weights.approach_range_m).sum() > 1:
            self._approach_side = 0
            return super()._select_candidate(
                costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
            )
        adjusted = costs + weights.clearance_weight * np.maximum(
            0.0, weights.clearance_margin_m - ped_clear.min(axis=1)
        )
        ahead = relative @ direction
        lateral = _cross(np.broadcast_to(direction, relative.shape), relative)
        offset = getattr(self, "_passing_offset", 1.5)
        approaching = (
            (ahead > 0.0)
            & (ahead < weights.approach_range_m)
            & (np.abs(lateral) < offset + 0.5)
            & ((ped_vel @ direction) < -0.2)
        )
        if approaching.any():
            index = int(np.argmin(np.where(approaching, ahead, np.inf)))
            if not self._approach_side:
                self._approach_side = -1 if lateral[index] >= 0 else 1
            predicted = peds[index] + ped_vel[index] * times[-1]
            terminal_relative = points[:, -1] - predicted
            terminal_lateral = _cross(
                np.broadcast_to(direction, terminal_relative.shape), terminal_relative
            )
            adjusted += weights.approach_weight * np.maximum(
                0.0, offset - self._approach_side * terminal_lateral
            )
            self._approach_steps += 1
        else:
            self._approach_side = 0
        return super()._select_candidate(
            adjusted, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
        )

    def diagnostics(self) -> dict[str, Any]:
        """Report use of the early passing preference.

        Returns:
            dict[str, Any]: Planner identity and episode counters.
        """
        result = super().diagnostics()
        result.update(
            planner_type="PassingVelocitySamplerAdapter",
            implementation_mode="passing_velocity_sampling",
            approach_steps=self._approach_steps,
        )
        return result
