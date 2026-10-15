"""Opt-in empirical ORCA velocity filter; no nonholonomic AVO safety guarantee."""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from shapely import distance, linestrings
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union

from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.planner.socnav_orca import ORCAPlannerAdapter
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.robot.reverse_drive import bound_drive_settings


@dataclass
class ORCAAVOConfig(SocNavPlannerConfig):
    """Research-only v1 finite search; time constants use the control period."""

    avo_horizon: float = 6.0
    avo_sample_dt: float = 0.1
    avo_rings: int = 4
    avo_angles: int = 32

    def __post_init__(self, *args) -> None:
        """Reject invalid sampling contracts before starting an episode."""
        super().__post_init__(*args)
        for name in ("avo_horizon", "avo_sample_dt"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("avo_rings", "avo_angles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def build_orca_avo_config(payload: dict[str, Any]) -> ORCAAVOConfig:
    """Build the opt-in config without changing the base ORCA config schema.

    Returns:
        Validated experimental configuration.
    """
    keys = {f.name for f in fields(ORCAAVOConfig)}
    return ORCAAVOConfig(**{k: v for k, v in payload.items() if k in keys})


def reachable_targets(preferred, current, delta, drive, *, rings=4, angles=32):
    """Return deterministic candidates inside the speed and acceleration disks.

    The isotropic acceleration bound is the smaller of forward/braking authority.
    It is conservative for asymmetric drives. This disk is a holonomic surrogate,
    not the exact differential-drive endpoint sector.

    Returns:
        Finite candidate matrix, including the exact disk projection if feasible.
    """
    preferred, current = np.asarray(preferred, float), np.asarray(current, float)
    if (
        preferred.shape != (2,)
        or current.shape != (2,)
        or not np.isfinite([*preferred, *current, delta]).all()
        or delta <= 0
    ):
        raise ValueError("Velocities must be finite 2-vectors and delta positive")
    radius = delta * min(drive.max_linear_accel, drive.max_linear_decel)
    difference = preferred - current
    projection = current + difference * min(1.0, radius / max(np.linalg.norm(difference), 1e-15))
    braking = current * max(0.0, 1.0 - radius / max(np.linalg.norm(current), 1e-15))
    theta = np.arange(angles) * (2 * np.pi / angles)
    unit = np.column_stack((np.cos(theta), np.sin(theta)))
    lattice = current + (radius * np.arange(1, rings + 1)[:, None, None] / rings) * unit
    candidates = np.vstack((projection, current, braking, lattice.reshape(-1, 2)))
    speed_cap = drive.max_linear_speed
    candidates = candidates[np.linalg.norm(candidates, axis=1) <= speed_cap + 1e-12]
    if not len(candidates):
        raise ValueError("Current velocity is outside the live drive speed limit")
    return candidates


def acceleration_paths(targets, current, position, delta, times):
    """Integrate a(t)=(target-v(t))/delta exactly, including initial velocity.

    Returns:
        Array [candidate,time,xy] of exponential-lag positions.
    """
    targets = np.asarray(targets, float)
    times = np.asarray(times, float)
    return (
        np.asarray(position)[None, None, :]
        + targets[:, None, :] * times[None, :, None]
        + (np.asarray(current) - targets)[:, None, :]
        * delta
        * (-np.expm1(-times / delta))[None, :, None]
    )


def trajectories_clear(paths, times, pedestrians, velocities, radii, static_geometry, padding):
    """Exclude swept relative contacts with CV pedestrians and static geometry.

    Linear chords cover samples; padding bounds exponential-path interpolation
    error by a_max*sample_dt**2/8. Unknown pedestrian acceleration and subsequent
    differential-drive projection are outside this surrogate's guarantee.

    Returns:
        Boolean vector of admissible sampled candidates.
    """
    clear = np.ones(len(paths), dtype=bool)
    robot_radius, ped_radius = radii
    if not static_geometry.is_empty:
        clear &= distance(linestrings(paths), static_geometry) > robot_radius + padding
    if len(pedestrians):
        predicted = pedestrians[None, :, :] + times[:, None, None] * velocities[None, :, :]
        relative = paths[:, :, None, :] - predicted[None, :, :, :]
        start = relative[:, :-1]
        edge = relative[:, 1:] - start
        denom = np.sum(edge * edge, axis=-1)
        fraction = np.clip(-np.sum(start * edge, axis=-1) / np.maximum(denom, 1e-30), 0, 1)
        nearest = start + fraction[..., None] * edge
        clear &= np.all(
            np.sum(nearest * nearest, axis=-1) > (robot_radius + ped_radius + padding) ** 2,
            axis=(1, 2),
        )
    return clear


class ORCAAVOPlannerAdapter(ORCAPlannerAdapter):
    """Filter native ORCA proposals, then use its unchanged command conversion."""

    def __init__(self, config: ORCAAVOConfig | None = None, *, allow_fallback=False):
        """Require native ORCA and retain no hidden heuristic execution mode."""
        if allow_fallback:
            raise ValueError("orca_avo requires native ORCA; fallback is not supported")
        self._avo_drive = None
        self._avo_geometry = GeometryCollection()
        self._avo_last: dict[str, Any] = {}
        super().__init__(config or ORCAAVOConfig(), allow_fallback=False)

    def reset(self, *, seed: int | None = None) -> None:
        """Reset proposal diagnostics together with inherited episode state."""
        super().reset(seed=seed)
        self._avo_last = {}
        self._avo_no_safe_steps = 0
        self._avo_steps = 0

    def bind_env(self, env: Any) -> None:
        """Read actual drive settings and static geometry; reject unsupported plants."""
        drive = bound_drive_settings(env)
        if not isinstance(drive, DifferentialDriveSettings) or drive.min_linear_speed < 0:
            raise ValueError("orca_avo v1 requires a bound forward differential drive")
        if not np.isfinite(
            [drive.max_linear_accel, drive.max_linear_decel, drive.max_linear_speed, drive.radius]
        ).all():
            raise ValueError("orca_avo requires finite live drive limits")
        segments = list(env.simulator.iter_obstacle_segments())
        array = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
        if not np.isfinite(array).all():
            raise ValueError("orca_avo requires finite static geometry")
        self._avo_drive = drive
        self._avo_geometry = unary_union(linestrings(array)) if len(array) else GeometryCollection()
        super().bind_env(env)

    def plan_velocity_world(self, observation: dict) -> np.ndarray:
        """Return the closest safe sampled reachable target, or flagged braking.

        Returns:
            World velocity target passed to the unchanged ORCA projection.
        """
        if self._avo_drive is None:
            raise RuntimeError("Bind the actual drive/environment before using orca_avo")
        proposal = np.asarray(super().plan_velocity_world(observation), float)
        robot, _goal, pedestrians = self._socnav_fields(observation)
        heading = float(np.asarray(robot["heading"])[0])
        speed = float(np.asarray(robot["speed"])[0])
        current = speed * np.array([np.cos(heading), np.sin(heading)])
        delta = self._simulation_timestep(observation)
        drive = self._avo_drive
        targets = reachable_targets(
            proposal,
            current,
            delta,
            drive,
            rings=self.config.avo_rings,
            angles=self.config.avo_angles,
        )
        dt = min(delta, self.config.avo_sample_dt)
        times = np.linspace(
            0, self.config.avo_horizon, int(np.ceil(self.config.avo_horizon / dt)) + 1
        )
        paths = acceleration_paths(targets, current, robot["position"], delta, times)
        positions, ego_velocities, _count, radius = self._extract_pedestrians(pedestrians)
        velocities = ego_velocities @ np.array(
            [[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]]
        )
        padding = min(drive.max_linear_accel, drive.max_linear_decel) * dt**2 / 8
        clear = trajectories_clear(
            paths, times, positions, velocities, (drive.radius, radius), self._avo_geometry, padding
        )
        # Grid occupancy can include observed objects outside the bound static map.
        centers, obstacle_radii = self._extract_obstacles_from_grid(
            observation, np.asarray(robot["position"]), heading
        )
        for center, obstacle_radius in zip(centers, obstacle_radii, strict=True):
            clear &= trajectories_clear(
                paths,
                times,
                center[None, :],
                np.zeros((1, 2)),
                (drive.radius, obstacle_radius),
                GeometryCollection(),
                padding,
            )
        if np.any(clear):
            safe = targets[clear]
            target = safe[np.argmin(np.sum((safe - proposal) ** 2, axis=1))]
            status = "safe_sampled_target"
        else:
            # No assertion of safety when the safe sampled set is empty.
            radius = delta * min(drive.max_linear_accel, drive.max_linear_decel)
            target = current * max(0.0, 1.0 - radius / max(abs(speed), 1e-15))
            status = "no_safe_target_braking"
            self._avo_no_safe_steps += 1
        self._avo_steps += 1
        self._avo_last = {
            "status": status,
            "proposal_world_mps": proposal.tolist(),
            "current_world_mps": current.tolist(),
            "target_world_mps": target.tolist(),
            "ball_radius_mps": delta * min(drive.max_linear_accel, drive.max_linear_decel),
            "safe_candidates": int(np.sum(clear)),
            "candidates": len(targets),
        }
        return target

    def diagnostics(self) -> dict[str, Any]:
        """Expose filter identity and unresolved steps without claiming safety.

        Returns:
            Versioned empirical filter diagnostics and inherited ORCA trace stats.
        """
        return {
            **super().diagnostics(),
            "avo": {
                "schema_version": "orca_avo.v1",
                "readiness": "research_only",
                "guarantee": "empirical_only",
                "steps": self._avo_steps,
                "no_safe_target_steps": self._avo_no_safe_steps,
                "last": self._avo_last,
            },
        }
