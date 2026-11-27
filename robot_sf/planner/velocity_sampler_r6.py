"""Development passing sampler with crowd clearance and narrow-pass waits."""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_sf.planner.velocity_sampler_r2 import swept_wall_clearance
from robot_sf.planner.velocity_sampler_r4 import PassingVelocitySamplerAdapter, PassingWeightsT


class ClearanceVelocitySamplerAdapter(PassingVelocitySamplerAdapter[PassingWeightsT]):
    """Prefer clear forecasts in crowds and yield when a narrow pass closes."""

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the crowd clearance and narrow-pass counters between episodes."""
        super().reset(seed=seed)
        self._crowd_clearance_steps = 0
        self._narrow_wait_steps = 0

    def _select_candidate(  # noqa: PLR0913
        self, costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
    ) -> int:
        weights = self.weights
        if (
            np.count_nonzero(np.linalg.norm(peds - points[0, 0], axis=1) < weights.approach_range_m)
            > 1
        ):
            clearance = ped_clear.min(axis=1)
            progress = (points[:, -1] - points[0, 0]) @ direction
            # Keep the crowd cost when a clear option offers comparable forward progress.
            forward = (progress > 0.0) & (progress >= 0.8 * progress[safe].max(initial=0.0))
            moving_clear = (
                safe & (speeds > 0.0) & forward & (clearance >= weights.clearance_margin_m)
            )
            if moving_clear.any():
                costs = costs + weights.clearance_weight * np.maximum(
                    0.0, weights.clearance_margin_m - clearance
                )
                self._crowd_clearance_steps += 1
        wait = self._narrow_pass_wait(safe, ped_clear, points, peds, ped_vel, direction)
        if wait is not None:
            safe = wait
            self._narrow_wait_steps += 1
        return super()._select_candidate(
            costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
        )

    def _narrow_pass_wait(self, safe, ped_clear, points, peds, ped_vel, direction):
        """Find wall-safe braking profiles when an oncoming walker blocks a narrow pass.

        Returns:
            The braking profile mask, or None when the normal selection can continue.
        """
        if len(peds) != 1 or self._recovering:
            return None
        relative = peds[0] - points[0, 0]
        normal = np.array([-direction[1], direction[0]])
        if (
            relative @ direction <= 0.0
            or abs(relative @ normal) >= self._passing_offset
            or ped_vel[0] @ direction >= -0.2
        ):
            return None
        target_speeds = np.array([p[0] if p[4] > 0 else p[2] for p in self._profiles])
        forward = safe & (target_speeds > 0.0)
        wait = safe & (target_speeds == 0.0)
        if (
            not forward.any()
            or not wait.any()
            or (forward & (ped_clear.min(axis=1) >= self.weights.clearance_scale)).any()
        ):
            return None
        bypass = peds[0] + np.array([-1.0, 1.0])[:, None] * self._passing_offset * normal
        bypass_paths = np.stack((np.broadcast_to(points[0, 0], bypass.shape), bypass), axis=1)
        lateral_wall_clearance = swept_wall_clearance(
            bypass_paths, self._walls, self._velocity_drive.radius
        )
        if (lateral_wall_clearance > 0.0).any():
            return None
        return wait

    def diagnostics(self) -> dict[str, Any]:
        """Report crowd clearance use and steps restricted to narrow-pass waits.

        Returns:
            dict[str, Any]: Planner identity and episode counters.
        """
        result = super().diagnostics()
        result.update(
            planner_type="ClearanceVelocitySamplerAdapter",
            implementation_mode="clearance_velocity_sampling",
            crowd_clearance_steps=self._crowd_clearance_steps,
            narrow_wait_steps=self._narrow_wait_steps,
        )
        return result
