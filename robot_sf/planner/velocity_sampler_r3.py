"""Development sampler that relaxes pedestrian comfort after sustained stalling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TypeVar

import numpy as np

from robot_sf.planner.velocity_sampler_r2 import (
    MultiStageVelocitySamplerAdapter,
    MultiStageWeights,
)


@dataclass(frozen=True)
class StallRecoveryWeights(MultiStageWeights):
    """Set the displacement and time needed to relax pedestrian comfort."""

    pedestrian_weight: float = 3.0
    stall_seconds: float = 4.0
    stall_distance_m: float = 0.5
    recovery_pedestrian_weight: float = 0.75

    def __post_init__(self) -> None:
        """Reject invalid recovery settings before a run starts."""
        super().__post_init__()
        if (
            not np.isfinite(
                [self.stall_seconds, self.stall_distance_m, self.recovery_pedestrian_weight]
            ).all()
            or self.stall_seconds <= 0
            or self.stall_distance_m <= 0
            or not 0 <= self.recovery_pedestrian_weight <= self.pedestrian_weight
        ):
            raise ValueError("invalid stall recovery settings")


StallRecoveryWeightsT = TypeVar("StallRecoveryWeightsT", bound=StallRecoveryWeights)


class StallRecoveryVelocitySamplerAdapter(MultiStageVelocitySamplerAdapter[StallRecoveryWeightsT]):
    """Use the ordinary sampler until little displacement persists for several seconds."""

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the displacement anchor and recovery counters between episodes."""
        super().reset(seed=seed)
        self._stall_anchor = None
        self._stall_elapsed = 0.0
        self._recovery_steps = 0
        self._recovery_entries = 0
        self._recovering = False

    def plan(self, observation: dict) -> tuple[float, float]:
        """Temporarily relax the pedestrian cost; preserve the physical wall checks.

        Returns:
            tuple[float, float]: Reachable linear and angular velocity targets.
        """
        robot, _, _ = self._socnav_fields(observation)
        position = self._as_1d_float(robot["position"], pad=2)[:2]
        dt = self._simulation_timestep(observation)
        weights = self.weights
        if self._stall_anchor is None or (
            np.linalg.norm(position - self._stall_anchor) >= weights.stall_distance_m
        ):
            self._stall_anchor = position.copy()
            self._stall_elapsed = 0.0
        else:
            self._stall_elapsed += dt
        recovering = self._stall_elapsed >= weights.stall_seconds
        if recovering:
            self._recovery_steps += 1
            self._recovery_entries += int(not self._recovering)
            self.weights = replace(weights, pedestrian_weight=weights.recovery_pedestrian_weight)
        self._recovering = recovering
        try:
            return super().plan(observation)
        finally:
            self.weights = weights

    def diagnostics(self) -> dict[str, Any]:
        """Report how often the temporary comfort adjustment was active.

        Returns:
            dict[str, Any]: Planner identity and episode counters.
        """
        result = super().diagnostics()
        result.update(
            planner_type="StallRecoveryVelocitySamplerAdapter",
            implementation_mode="stall_recovery_velocity_sampling",
            recovery_steps=self._recovery_steps,
            recovery_entries=self._recovery_entries,
        )
        return result
