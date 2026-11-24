"""Select a velocity sampler when tracked walkers approach the active route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.planner.dra_mppi import DRAMPPIAdapter

if TYPE_CHECKING:
    from robot_sf.planner.dra_mppi import DRAMPPIConfig
    from robot_sf.planner.velocity_sampler_r9 import ContactVelocitySamplerAdapter


_MIN_LATERAL_SPEED_MPS = 0.2
_SIDE_ENTRANCE_DEPTH_RADII = 4.5
_DOWNSTREAM_CROWD_RADII = 3.0


@dataclass(frozen=True)
class EncounterConfig:
    """Time window, release delay and walker limit for a crossing handoff."""

    lookahead_s: float = 3.0
    clear_hold_s: float = 5.0
    max_crossing_walkers: int = 2

    def __post_init__(self) -> None:
        """Reject nonfinite times and invalid walker limits."""
        times = (self.lookahead_s, self.clear_hold_s)
        if not np.isfinite(times).all() or min(times) <= 0:
            raise ValueError("encounter times must be positive and finite")
        limit = self.max_crossing_walkers
        if type(limit) is not int or limit < 1:
            raise ValueError("the crossing walker limit must be a positive integer")


class EncounterPlannerAdapter(DRAMPPIAdapter):
    """Run DRA-MPPI until route geometry and pedestrian motion indicate an encounter."""

    def __init__(
        self,
        config: DRAMPPIConfig,
        sampler: ContactVelocitySamplerAdapter,
        encounter: EncounterConfig,
    ) -> None:
        """Keep both controllers and the encounter settings for one robot."""
        if not config.crossing_yield:
            raise ValueError("encounter sampling requires DRA-MPPI crossing geometry")
        self._sampler = sampler
        self._encounter = encounter
        super().__init__(config)

    def bind_env(self, env: Any) -> None:
        """Bind both controllers to the same live drive and static map."""
        super().bind_env(env)
        self._sampler.bind_env(env)

    def reset(self, *, seed: int | None = None) -> None:
        """Clear controller state, encounter memory, and per-episode counts."""
        super().reset()
        self._sampler.reset(seed=seed)
        self._hold_s = 0.0
        self._sampling = False
        self._sampler_steps = 0
        self._closed_handoff_steps = 0
        self._entries = 0
        self._crowding_target: tuple[float, float] | None = None
        self._crowded_openings = np.empty(0, dtype=bool)

    def _encounter_ahead(self, observation: dict[str, Any]) -> bool:
        """Find an entry into a sparse side passage or walkers staged at its entrances.

        Returns:
            Whether to use the velocity sampler on this control tick.
        """
        position, _, _, target, peds, velocities, ped_radius = self._planning_state(observation)
        if not len(peds) or not len(self._crossings):
            return False

        assert self._reference_start is not None
        tangent = self._crossing_direction
        normal = np.array([-tangent[1], tangent[0]])
        offset = (peds - target) @ normal
        along = (peds - self._reference_start) @ tangent
        progress = (position - self._reference_start) @ tangent
        radius = self.config.robot_radius_m
        contact_radius = radius + ped_radius
        side_depth = _SIDE_ENTRANCE_DEPTH_RADII * radius
        lo, hi = self._crossings.T
        lookahead = self._encounter.lookahead_s
        reach = self.config.reference_speed * lookahead
        upcoming = (hi > progress) & (lo - progress <= reach + radius)

        # Include walkers who have begun merging and still occupy the exit lane.
        in_side_passage = (np.abs(offset) <= side_depth)[None, :]
        near = (
            (along >= lo[:, None] - radius)
            & (along <= hi[:, None] + _DOWNSTREAM_CROWD_RADII * radius)
            & in_side_passage
        )
        if self._crowding_target != self._last_target_xy:
            self._crowding_target = self._last_target_xy
            self._crowded_openings = np.zeros(len(lo), dtype=bool)
        crowded = near.sum(axis=1) > self._encounter.max_crossing_walkers
        # Keep a busy opening with DRA-MPPI when the leading walker leaves it.
        self._crowded_openings |= upcoming & crowded
        sparse = ~(crowded | self._crowded_openings)
        if not sparse.any():
            return False
        lo, hi = lo[sparse], hi[sparse]

        # A pair at opposite entrances can start crossing after the robot commits.
        # React while the robot still has room to choose its approach.
        beside = (along >= lo[:, None] - radius) & (along <= hi[:, None] + radius) & in_side_passage
        left = (beside & (offset > contact_radius)).any(axis=1)
        right = (beside & (offset < -contact_radius)).any(axis=1)
        if (upcoming[sparse] & left & right).any():
            return True

        # Project the first entry into the route corridor using measured velocity.
        towards = -np.sign(offset) * (velocities @ normal)
        entry_time = np.maximum(np.abs(offset) - 0.75 * contact_radius, 0.0) / np.maximum(
            towards, _MIN_LATERAL_SPEED_MPS
        )
        entry_progress = along + entry_time * (velocities @ tangent)
        advance = entry_progress - progress
        arrival = (
            (towards >= _MIN_LATERAL_SPEED_MPS)
            & (entry_time <= lookahead)
            & (advance >= 0.0)
            & (advance <= min(reach, (target - position) @ tangent) + contact_radius)
        )
        in_opening = (
            (entry_progress >= lo[:, None] - radius) & (entry_progress <= hi[:, None] + radius)
        ).any(axis=0)
        return bool((arrival & in_opening).any())

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Refresh the sampler hold on a detected encounter and expire it after clear steps.

        Returns:
            Linear speed and angular rate targets from the selected controller.
        """
        if self._drive_settings is None:
            raise RuntimeError("encounter sampling requires a bound differential drive")
        dt = self._simulation_timestep(observation)
        if self._encounter_ahead(observation):
            self._hold_s = self._encounter.clear_hold_s
        else:
            self._hold_s = max(0.0, self._hold_s - dt)

        if self._hold_s > 1e-9:
            self._entries += int(not self._sampling)
            self._sampling = True
            command = self._sampler.plan(observation)
            self._sampler_steps += 1
            return command
        if self._sampling:
            self._closed_handoff_steps += self._sampler.diagnostics()["orca_recovery_steps"]
            self._sampler.reset()
            self._nominal.fill(0.0)
            self._beta = 1.0
            self._passing_side = 0
            self._sampling = False
        return super().plan(observation)

    def diagnostics(self) -> dict[str, Any]:
        """Count issued commands by controller, including the sampler's ORCA handoff.

        Returns:
            Episode diagnostics with mutually exclusive planner mode counts.
        """
        stats = super().diagnostics()
        handoffs = self._closed_handoff_steps + self._sampler.diagnostics()["orca_recovery_steps"]
        stats.update(
            encounter_entries=self._entries,
            mode_steps={
                "dra_mppi": self._planning_steps,
                "velocity_sampler": self._sampler_steps - handoffs,
                "orca_handoff": handoffs,
            },
        )
        return stats
