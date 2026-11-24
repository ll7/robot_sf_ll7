"""Development velocity sampler with a guarded ORCA handoff during stalls."""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_sf.planner.clearance_geometry import swept_pedestrian_clearance
from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter
from robot_sf.planner.velocity_sampler_r2 import _braking_forecast, swept_wall_clearance
from robot_sf.planner.velocity_sampler_r4 import PassingWeightsT
from robot_sf.planner.velocity_sampler_r7 import RecoveryTurnVelocitySamplerAdapter


class ProgressVelocitySamplerAdapter(RecoveryTurnVelocitySamplerAdapter[PassingWeightsT]):
    """Consult ORCA in a pedestrian stall when no clear forecast advances the route."""

    _pedestrian_braking_wall_margin_m = 0.25

    def __init__(self, *, config: SocNavPlannerConfig, weights: PassingWeightsT) -> None:
        """Use the release ORCA defaults at the sampler's configured speed limits."""
        self._orca = ORCAPlannerAdapter(
            config=SocNavPlannerConfig(
                max_linear_speed=config.max_linear_speed, max_angular_speed=config.max_angular_speed
            ),
            allow_fallback=False,
        )
        super().__init__(config=config, weights=weights)

    def reset(self, *, seed: int | None = None) -> None:
        """Clear both planners and the handoff counters."""
        super().reset(seed=seed)
        self._orca.reset(seed=seed)
        self._orca_active = False
        self._orca_requested = False
        self._orca_recovery_steps = 0
        self._orca_recovery_entries = 0

    def bind_env(self, env: Any) -> None:
        """Bind both planners to the same map and drive."""
        super().bind_env(env)
        self._orca.bind_env(env)

    def _candidate_forecasts(  # noqa: PLR0913
        self, position, heading, speed0, turn0, speeds, turns, dt, drive, direction, peds, ped_vel
    ):
        self._drive_state = (position, heading, speed0, turn0, dt, drive, peds, ped_vel)
        return super()._candidate_forecasts(
            position, heading, speed0, turn0, speeds, turns, dt, drive, direction, peds, ped_vel
        )

    def _select_candidate(  # noqa: PLR0913
        self, costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
    ) -> int:
        chosen = super()._select_candidate(
            costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
        )
        nearby = np.linalg.norm(peds - points[0, 0], axis=1).min(initial=np.inf)
        started = self._recovering and abs(self._drive_state[2]) <= 0.1
        if (started or self._continuing_stall_handoff()) and nearby < self._passing_offset + 1.5:
            clear = safe & (ped_clear.min(axis=1) >= self.weights.clearance_margin_m)
            progress = (points[:, -1] - points[0, 0]) @ direction
            self._orca_requested = not (clear & (progress >= (2.0 / 3.0) * times[-1])).any()
        else:
            self._orca_requested = False
        return chosen

    def _continuing_stall_handoff(self) -> bool:
        """Allow a prior handoff to continue while it meets the progress guard.

        Returns:
            bool: Whether the previous step used ORCA.
        """
        return self._orca_active

    def plan(self, observation: dict) -> tuple[float, float]:
        """Use the sampler unless ORCA is requested and its braking path is clear.

        Returns:
            tuple[float, float]: Reachable linear and angular velocity targets.
        """
        sampled = super().plan(observation)
        if not self._orca_requested:
            self._orca_active = False
            return sampled

        position, heading, speed0, turn0, dt, drive, peds, ped_vel = self._drive_state
        speed, turn = self._orca.plan(observation)
        speed_cap = min(float(self.config.max_linear_speed), drive.max_linear_speed)
        turn_cap = min(float(self.config.max_angular_speed), drive.max_angular_speed)
        speed = float(np.clip(speed, 0.0, speed_cap))
        speed = speed0 + np.clip(
            speed - speed0, -drive.max_linear_decel * dt, drive.max_linear_accel * dt
        )
        turn = float(np.clip(turn, -turn_cap, turn_cap))
        turn = turn0 + np.clip(
            turn - turn0, -drive.max_angular_accel * dt, drive.max_angular_accel * dt
        )
        braking = _braking_forecast(
            position, heading, speed0, turn0, np.array([speed]), np.array([turn]), dt, drive
        )
        contact_radius = self._passing_offset - self.weights.passing_padding_m
        wall_clear = swept_wall_clearance(braking, self._walls, drive.radius).min()
        ped_clear = swept_pedestrian_clearance(
            braking, np.arange(braking.shape[1]) * dt, peds, ped_vel, contact_radius
        ).min()
        if wall_clear <= 1e-5 or ped_clear <= 0.0:
            self._orca_active = False
            return sampled
        # A sampled continuation was only evaluated on this step. Drop the
        # commitment before scoring the next observation.
        self._active_profile = None
        self._locked_side = 0
        self._last_side = 0
        self._approach_side = 0
        self._orca_recovery_steps += 1
        self._orca_recovery_entries += int(not self._orca_active)
        self._orca_active = True
        return float(speed), float(turn)

    def diagnostics(self) -> dict[str, Any]:
        """Record the number of decisions and entries using ORCA commands.

        Returns:
            dict[str, Any]: Planner identity and episode counters.
        """
        result = super().diagnostics()
        result.update(
            planner_type="ProgressVelocitySamplerAdapter",
            implementation_mode="progress_velocity_sampling",
            orca_recovery_steps=self._orca_recovery_steps,
            orca_recovery_entries=self._orca_recovery_entries,
        )
        return result
