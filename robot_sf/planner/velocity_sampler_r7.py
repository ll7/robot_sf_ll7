"""Development sampler with a sustained turn for clear-space stall recovery."""

from __future__ import annotations

from math import atan2, pi
from typing import Any

import numpy as np

from robot_sf.common.math_utils import wrap_angle_pi_closed
from robot_sf.planner.velocity_sampler_r2 import MultiStageVelocitySamplerAdapter
from robot_sf.planner.velocity_sampler_r4 import PassingWeightsT
from robot_sf.planner.velocity_sampler_r6 import ClearanceVelocitySamplerAdapter


class RecoveryTurnVelocitySamplerAdapter(ClearanceVelocitySamplerAdapter[PassingWeightsT]):
    """Add a turn in place when the robot stalls facing away from the route."""

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the recovery turn counter between episodes."""
        super().reset(seed=seed)
        self._recovery_turn_index = None
        self._recovery_turn_steps = 0

    def _candidate_forecasts(  # noqa: PLR0913
        self, position, heading, speed0, turn0, speeds, turns, dt, drive, direction, peds, ped_vel
    ):
        self._recovery_turn_index = None
        if self._recovering and abs(speed0) <= 0.05:
            heading_error = wrap_angle_pi_closed(atan2(direction[1], direction[0]) - heading)
            nearest_pedestrian = np.linalg.norm(peds - position, axis=1).min(initial=np.inf)
            if abs(heading_error) > pi / 3.0 and nearest_pedestrian > self._passing_offset + 1.0:
                turn_limit = min(float(self.config.max_angular_speed), drive.max_angular_speed)
                self._recovery_turn_index = len(speeds)
                speeds = np.append(speeds, 0.0)
                turns = np.append(turns, np.copysign(turn_limit, heading_error))
        return super()._candidate_forecasts(
            position, heading, speed0, turn0, speeds, turns, dt, drive, direction, peds, ped_vel
        )

    def _select_candidate(  # noqa: PLR0913
        self, costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
    ) -> int:
        rotation = self._recovery_turn_index
        if (
            rotation is None
            or not safe[rotation]
            or ped_clear[rotation].min() <= self.weights.clearance_margin_m
            or ((points[safe, -1] - points[0, 0]) @ direction).max(initial=0.0) >= 0.1
        ):
            return super()._select_candidate(
                costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
            )

        # Let the turn compete when the wall-clear forecasts barely advance.
        # The previous profile and all other wall-clear profiles remain eligible.
        adjusted = costs.copy()
        adjusted[rotation] -= 1.0
        previous_side = self._locked_side
        previous_index = self._continuation_index
        self._locked_side = 0
        self._continuation_index = None
        try:
            chosen = MultiStageVelocitySamplerAdapter._select_candidate(
                self,
                adjusted,
                safe,
                speeds,
                brake_clear,
                ped_clear,
                points,
                times,
                peds,
                ped_vel,
                direction,
            )
        finally:
            self._locked_side = previous_side
            self._continuation_index = previous_index
        self._recovery_turn_steps += int(chosen == rotation)
        return chosen

    def diagnostics(self) -> dict[str, Any]:
        """Report chosen recovery turns.

        Returns:
            dict[str, Any]: Planner identity and episode counters.
        """
        result = super().diagnostics()
        result.update(
            planner_type="RecoveryTurnVelocitySamplerAdapter",
            implementation_mode="recovery_turn_velocity_sampling",
            recovery_turn_steps=self._recovery_turn_steps,
        )
        return result
