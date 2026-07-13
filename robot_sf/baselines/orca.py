"""ORCA baseline planner for the Social Navigation Benchmark.

Restores the historical ``orca`` algorithm in the baseline registry (issue #5491). The
exact-repeat campaign for #5263 resolves cells with ``algo: orca`` and routes
episode execution through ``runner.run_episode(algo="orca")`` ->
``_load_baseline_planner``. That registry only knew the map-based ``ORCAPlannerAdapter``
(via ``map_runner``), so the executor crashed with
``Unknown algorithm 'orca'`` instead of running the cell.

This module plugs the adapter into the benchmark baseline contract (``step`` returns a
``{"vx", "vy"}`` world-velocity action, ``reset``/``get_metadata``/``close`` are
supported) so the historical ORCA cells execute on current ``main``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

import numpy as np

from robot_sf.baselines.interface import Observation
from robot_sf.planner.socnav import (
    ORCAPlannerAdapter,
    SocNavPlannerConfig,
)

# Fields accepted when building a SocNavPlannerConfig from a loose mapping.
_SOCNAV_CONFIG_FIELDS = tuple(SocNavPlannerConfig.__dataclass_fields__.keys())


def _build_socnav_config(config: dict[str, Any]) -> SocNavPlannerConfig:
    """Return a SocNavPlannerConfig filtered from a loose mapping.

    Unknown keys are dropped so an empty or partial planner config cannot crash the
    baseline constructor.

    Returns:
        SocNavPlannerConfig: Filtered planner configuration.
    """
    if not isinstance(config, dict):
        return SocNavPlannerConfig()
    filtered = {key: value for key, value in config.items() if key in _SOCNAV_CONFIG_FIELDS}
    return SocNavPlannerConfig(**filtered)


class OrcaPlanner:
    """Baseline ORCA planner wrapping ``ORCAPlannerAdapter``.

    Consumes the canonical benchmark ``Observation`` (robot + agents), adapts it into
    the SocNav structured observation expected by the adapter, and returns a
    world-frame ``{"vx", "vy"}`` action. Uses the benchmark-ready rvo2 solver
    when available.
    """

    def __init__(self, config: dict[str, Any] | SocNavPlannerConfig, *, seed: int | None = None):
        """Initialize the ORCA baseline planner.

        Args:
            config: Planner configuration object or dict payload.
            seed: Optional random seed (accepted for baseline API symmetry; ORCA is
                deterministic given rvo2).
        """
        self._seed = seed
        self._config = self._parse_config(config)
        if isinstance(config, dict):
            allow_fallback = bool(config.get("allow_fallback", False))
        else:
            allow_fallback = False
        self._adapter = ORCAPlannerAdapter(config=self._config, allow_fallback=allow_fallback)

    @staticmethod
    def _parse_config(config: dict[str, Any] | SocNavPlannerConfig) -> SocNavPlannerConfig:
        """Normalize config input into a SocNavPlannerConfig.

        Returns:
            SocNavPlannerConfig: Parsed configuration.
        """
        if isinstance(config, SocNavPlannerConfig):
            return config
        return _build_socnav_config(config if isinstance(config, dict) else {})

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal planner state and optionally reseed.

        Args:
            seed: Optional new seed.
        """
        if seed is not None:
            self._seed = seed
        self._adapter.reset()

    def _to_adapter_observation(self, obs: Observation | dict[str, Any]) -> dict[str, Any]:
        """Adapt a benchmark Observation into the SocNav structured observation.

        Returns:
            dict[str, Any]: Observation with ``robot``/``goal``/``pedestrians`` keys
            as expected by ``ORCAPlannerAdapter``.
        """
        if isinstance(obs, Observation):
            robot = dict(obs.robot)
            agents = list(obs.agents)
            dt = float(getattr(obs, "dt", 0.1) or 0.1)
        else:
            robot = dict(obs.get("robot", {}))
            agents = list(obs.get("agents", []))
            sim = obs.get("sim", {}) or {}
            dt_value = sim.get("timestep", [0.1])
            dt = float(np.asarray(dt_value, dtype=float).reshape(-1)[0]) if dt_value else 0.1

        # The SocNav-family ORCA adapter expects a richer robot state than the
        # canonical baseline Observation: heading, speed, and radius are required
        # by the rvo2 solver path. Default them from velocity/radius so a
        # minimal benchmark Observation still drives the planner.
        robot_position = np.asarray(robot.get("position", [0.0, 0.0]), dtype=float).reshape(-1)[:2]
        robot_velocity = np.asarray(robot.get("velocity", [0.0, 0.0]), dtype=float).reshape(-1)[:2]
        robot_radius = float(np.asarray(robot.get("radius", 0.3), dtype=float).reshape(-1)[0])
        heading_value = float(
            np.asarray(
                robot.get("heading", np.arctan2(robot_velocity[1], robot_velocity[0] + 1e-9)),
                dtype=float,
            ).reshape(-1)[0]
        )
        # The SocNav-family adapter indexes heading/radius as length-1 arrays, so
        # adapt them into that shape.
        heading = [heading_value]
        speed = float(np.linalg.norm(robot_velocity))
        adapted_robot = {
            "position": list(robot_position),
            "velocity": list(robot_velocity),
            "heading": heading,
            "speed": [speed],
            "radius": [robot_radius],
        }

        goal = robot.get("goal")
        goal_state = {
            "current": list(np.asarray(goal, dtype=float).reshape(-1)[:2])
            if goal is not None
            else [0.0, 0.0]
        }
        positions = [
            list(np.asarray(a.get("position", [0.0, 0.0]), dtype=float).reshape(-1)[:2])
            for a in agents
        ]
        velocities = [
            list(np.asarray(a.get("velocity", [0.0, 0.0]), dtype=float).reshape(-1)[:2])
            for a in agents
        ]
        count = len(agents)
        radius = (
            float(np.asarray(agents[0].get("radius", 0.3), dtype=float).reshape(-1)[0])
            if count > 0
            else 0.3
        )
        ped_state = {
            "positions": positions,
            "velocities": velocities,
            "count": [count],
            "radius": [radius],
        }
        return {
            "robot": adapted_robot,
            "goal": goal_state,
            "pedestrians": ped_state,
            "sim": {"timestep": [dt]},
        }

    def step(self, obs: Observation | dict[str, Any]) -> dict[str, float]:
        """Compute an ORCA world-velocity action for the given observation.

        Returns:
            Action dict ``{"vx", "vy"}`` in the world frame.
        """
        if isinstance(obs, dict):
            adapter_obs: dict[str, Any] = obs
        else:
            adapter_obs = self._to_adapter_observation(obs)
        world_velocity = np.asarray(
            self._adapter.plan_velocity_world(adapter_obs), dtype=float
        ).reshape(-1)
        if world_velocity.size < 2:
            return {"vx": 0.0, "vy": 0.0}
        return {"vx": float(world_velocity[0]), "vy": float(world_velocity[1])}

    def get_metadata(self) -> dict[str, Any]:
        """Return planner metadata for episode records.

        Returns:
            Metadata dict with algorithm, config hash, and status.
        """
        cfg = asdict(self._config)
        cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
        return {
            "algorithm": "orca",
            "config": cfg,
            "config_hash": cfg_hash,
            "status": "ok",
            "adapter": "ORCAPlannerAdapter",
        }

    def close(self) -> None:
        """Release planner resources (no-op for the local ORCA adapter)."""
        if hasattr(self._adapter, "close"):
            self._adapter.close()


__all__ = ["OrcaPlanner"]
