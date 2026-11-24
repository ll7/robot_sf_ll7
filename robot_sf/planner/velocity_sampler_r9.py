"""Development velocity sampler with an ORCA handoff for imminent contact."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin
from typing import Any

import numpy as np

from robot_sf.planner.velocity_sampler_r4 import PassingWeights
from robot_sf.planner.velocity_sampler_r8 import ProgressVelocitySamplerAdapter


@dataclass(frozen=True)
class ContactWeights(PassingWeights):
    """Configure the contact-recovery sampler and its optional reciprocal cost."""

    orca_half_plane_weight: float = 0.0

    def __post_init__(self) -> None:
        """Keep the half-plane penalty finite and nonnegative."""
        super().__post_init__()
        if not isfinite(self.orca_half_plane_weight) or self.orca_half_plane_weight < 0.0:
            raise ValueError("orca_half_plane_weight must be finite and nonnegative")


class ContactVelocitySamplerAdapter(ProgressVelocitySamplerAdapter[ContactWeights]):
    """Consult ORCA when a pedestrian contact is predicted before the robot can stop."""

    def reset(self, *, seed: int | None = None) -> None:
        """Clear episode counters for the contact handoff."""
        super().reset(seed=seed)
        self._contact_requested = False
        self._orca_contact_requests = 0
        self._orca_contact_steps = 0

    def _continuing_stall_handoff(self) -> bool:
        # A contact handoff needs another contact prediction on the next decision,
        # unless the robot independently meets the original stall conditions.
        return super()._continuing_stall_handoff() and not self._contact_requested

    def _orca_half_plane_cost(
        self, points: np.ndarray, times: np.ndarray, peds: np.ndarray, ped_vel: np.ndarray
    ) -> np.ndarray:
        """Penalize first-segment velocities outside pedestrian ORCA half-planes.

        Returns:
            Largest positive signed distance to any half-plane, per candidate.
        """
        position, heading, speed, _, dt, drive, _, _ = self._drive_state
        cosine, sine = cos(heading), sin(heading)
        world_to_ego = np.array([[cosine, -sine], [sine, cosine]])
        lines = self._orca._build_orca_lines(
            robot_pos=position,
            robot_heading=heading,
            robot_velocity=np.array([speed, 0.0]),
            ped_positions=peds,
            ped_velocities=ped_vel @ world_to_ego,
            robot_radius=drive.radius,
            ped_radius=self._passing_offset - self.weights.passing_padding_m - drive.radius,
            time_step=dt,
        )
        if not lines:
            return np.zeros(len(points))
        velocities = (points[:, 1] - points[:, 0]) @ world_to_ego / times[1]
        offsets = np.stack([line.point for line in lines])[None, :, :] - velocities[:, None, :]
        tangents = np.stack([line.direction for line in lines])
        signed = tangents[:, 0] * offsets[..., 1] - tangents[:, 1] * offsets[..., 0]
        return np.maximum(0.0, signed.max(axis=1))

    def _select_candidate(  # noqa: PLR0913
        self, costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
    ) -> int:
        weight = self.weights.orca_half_plane_weight
        if weight and len(peds):
            costs = costs + weight * self._orca_half_plane_cost(points, times, peds, ped_vel)
        chosen = super()._select_candidate(
            costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
        )
        _, _, current_speed, _, dt, drive, _, _ = self._drive_state
        stopping_time = dt + max(current_speed, speeds[chosen]) / drive.max_linear_decel
        closest = ped_clear.min(axis=1)
        clear_sample = safe & (closest > 0.0) & (closest >= self.weights.clearance_margin_m)
        # Each clearance covers a swept interval. Include the one containing the stop.
        self._contact_requested = (
            not self._orca_requested
            and not clear_sample.any()
            and (ped_clear[chosen, times[:-1] < stopping_time] <= 0.0).any()
        )
        if self._contact_requested:
            self._orca_requested = True
            self._orca_contact_requests += 1
        return chosen

    def plan(self, observation: dict) -> tuple[float, float]:
        """Keep the inherited reachability and swept braking checks for both handoffs.

        Returns:
            tuple[float, float]: Reachable linear and angular targets.
        """
        command = super().plan(observation)
        self._orca_contact_steps += int(self._contact_requested and self._orca_active)
        return command

    def diagnostics(self) -> dict[str, Any]:
        """Report contact handoff use alongside the episode's total handoff use.

        Returns:
            dict[str, Any]: Planner identity and episode counters.
        """
        result = super().diagnostics()
        result.update(
            planner_type="ContactVelocitySamplerAdapter",
            implementation_mode="contact_velocity_sampling",
            orca_contact_requests=self._orca_contact_requests,
            orca_contact_steps=self._orca_contact_steps,
        )
        return result
