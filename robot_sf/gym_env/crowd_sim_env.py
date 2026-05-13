"""Crowd-only Gymnasium environment for Social Force pedestrian simulation."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger

from robot_sf.gym_env.env_config import SimulationSettings
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.render.sim_view import SimulationView
from robot_sf.sim.simulator import Simulator


@dataclass
class CrowdSimulationConfig:
    """Configuration for robot-free pedestrian simulation.

    This intentionally stays narrower than ``EnvSettings``: crowd simulation needs
    map selection, Social Force timing/density settings, and optional recording/rendering
    metadata, but it does not need robot, lidar, reward, or training configuration.
    """

    sim_config: SimulationSettings = field(default_factory=SimulationSettings)
    map_pool: MapDefinitionPool = field(default_factory=MapDefinitionPool)
    map_id: str | None = None
    peds_have_obstacle_forces: bool = True
    render_mode: str | None = None
    recording_enabled: bool = False
    recording_dir: str = "recordings"
    recording_path: str | None = None
    video_path: str | None = None
    video_fps: float | None = None


class CrowdSimEnv(gym.Env):
    """Gymnasium-style environment that advances only simulated pedestrians."""

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 60}

    def __init__(self, config: CrowdSimulationConfig | None = None):
        """Create a crowd simulation environment."""
        super().__init__()
        self.config = config or CrowdSimulationConfig()
        if self.config.render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode={self.config.render_mode!r}; "
                f"expected one of {self.metadata['render_modes']}"
            )

        self.render_mode = self.config.render_mode
        self.action_space = spaces.Box(low=0.0, high=0.0, shape=(0,), dtype=np.float32)
        self.observation_space = spaces.Dict({})
        self.map_def: MapDefinition
        self.map_id: str | None = None
        self.sim: Simulator
        configured_cap = self.config.sim_config.max_total_pedestrians
        self._pedestrian_capacity = 0 if configured_cap is None else int(configured_cap)
        if self._pedestrian_capacity < 0:
            raise ValueError("sim_config.max_total_pedestrians must be >= 0 when provided")
        self._step_count = 0
        self._warned_ignored_action = False
        self._sim_ui: SimulationView | None = None
        self._recording_file = None
        self._recording_path: Path | None = None
        # Honor factory-level seeding before the first constructor-time map selection.
        self._map_rng = np.random.default_rng(np.random.randint(0, np.iinfo(np.uint32).max))

        self._reset_simulator()
        self._sync_pedestrian_capacity()
        self.observation_space = self._make_observation_space()
        if self.config.recording_enabled:
            self._open_recording()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the crowd simulation and return the initial pedestrian state.

        Returns
        -------
        tuple[dict[str, np.ndarray], dict[str, Any]]
            Initial compact pedestrian observation and episode metadata.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._map_rng = self.np_random
        options = options or {}
        self._reset_simulator(map_id=options.get("map_id"))
        self._sync_pedestrian_capacity()
        observation = self._observation()
        info = self._info()
        self._record_event("reset", observation, info)
        return observation, info

    def step(
        self,
        action: Any | None = None,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance Social Force pedestrians by one fixed timestep.

        The optional action argument is accepted for Gymnasium compatibility. Non-``None``
        actions are ignored because this environment has no robot or trainable ego agent.

        Returns
        -------
        tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]
            Observation, zero reward, terminated flag, truncated flag, and metadata.
        """
        if action is not None and not self._warned_ignored_action:
            logger.warning("CrowdSimEnv ignores non-None actions; pedestrians step automatically.")
            self._warned_ignored_action = True

        self.sim.step_once([])
        self._step_count += 1
        observation = self._observation()
        truncated = self._step_count >= self.config.sim_config.max_sim_steps
        info = self._info()
        self._record_event("step", observation, info)
        if self.config.video_path:
            self.render()
        return observation, 0.0, False, truncated, info

    def render(self):
        """Render the current crowd state when rendering is enabled.

        Returns
        -------
        np.ndarray | None
            Latest RGB frame for ``rgb_array`` mode; otherwise ``None``.
        """
        if self.render_mode is None and not self.config.video_path:
            return None
        view = self._ensure_view()
        before = len(view.frames)
        view.render(self._visualizable_state(), target_fps=self._target_fps())
        if self.render_mode == "rgb_array":
            if len(view.frames) > before:
                return view.frames[-1]
            return np.asarray(view.screen)
        return None

    def close(self) -> None:
        """Close rendering and recording resources."""
        if self._sim_ui is not None:
            self._sim_ui.exit_simulation()
            self._sim_ui = None
        if self._recording_file is not None:
            self._recording_file.close()
            self._recording_file = None

    @property
    def recording_path(self) -> Path | None:
        """Return the JSONL recording path when recording is enabled.

        Returns
        -------
        Path | None
            Active recording path, or ``None`` when compact JSONL recording is disabled.
        """
        return self._recording_path

    def _reset_simulator(self, map_id: str | None = None) -> None:
        """Select a map and create a zero-robot simulator."""
        self.map_def, self.map_id = self._select_map(map_id or self.config.map_id)
        self.sim = Simulator(
            config=self.config.sim_config,
            map_def=self.map_def,
            robots=[],
            goal_proximity_threshold=self.config.sim_config.goal_radius,
            random_start_pos=False,
            peds_have_obstacle_forces=self.config.peds_have_obstacle_forces,
        )
        self._step_count = 0
        if self._sim_ui is not None:
            self._sim_ui.map_def = self.map_def
            self._sim_ui.obstacles = self.map_def.obstacles

    def _select_map(self, map_id: str | None) -> tuple[MapDefinition, str | None]:
        """Return the configured map and its best-effort identifier.

        Returns
        -------
        tuple[MapDefinition, str | None]
            Selected map definition and resolved map id when known.
        """
        if map_id is not None:
            return self.config.map_pool.get_map(map_id), map_id
        if self.config.map_pool.map_defs:
            map_ids = sorted(self.config.map_pool.map_defs.keys())
            chosen_index = int(self._map_rng.integers(len(map_ids)))
            chosen_id = map_ids[chosen_index]
            return self.config.map_pool.map_defs[chosen_id], chosen_id
        map_def = self.config.map_pool.choose_random_map()
        return map_def, None

    def _sync_pedestrian_capacity(self) -> None:
        """Freeze and validate the fixed pedestrian capacity for this env instance."""
        num_peds = int(self.sim.pysf_state.num_peds)
        if self._pedestrian_capacity == 0:
            self._pedestrian_capacity = num_peds
            return
        if num_peds > self._pedestrian_capacity:
            raise ValueError(
                "CrowdSimEnv requires a fixed pedestrian capacity across resets; "
                f"current simulator has {num_peds} pedestrians but capacity is "
                f"{self._pedestrian_capacity}. Set sim_config.max_total_pedestrians "
                "high enough for all selected maps."
            )

    def _make_observation_space(self) -> spaces.Dict:
        """Return the fixed observation space for this environment instance.

        Returns
        -------
        spaces.Dict
            Observation-space dictionary sized to the frozen pedestrian capacity.
        """
        vector_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._pedestrian_capacity, 2),
            dtype=np.float32,
        )
        return spaces.Dict(
            {
                "positions": vector_space,
                "velocities": vector_space,
                "goals": vector_space,
                "forces": vector_space,
                "step": spaces.Box(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    shape=(1,),
                    dtype=np.int64,
                ),
            }
        )

    def _pad_pedestrian_matrix(self, values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
        """Pad or reject pedestrian matrices to match the fixed observation-space capacity.

        Returns
        -------
        np.ndarray
            Matrix padded with trailing zero rows up to the frozen pedestrian capacity.
        """
        array = np.asarray(values, dtype=dtype)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"expected pedestrian matrix with shape (N, 2), got {array.shape}")
        if array.shape[0] > self._pedestrian_capacity:
            raise ValueError(
                "CrowdSimEnv observation exceeds fixed pedestrian capacity: "
                f"{array.shape[0]} > {self._pedestrian_capacity}"
            )
        padded = np.zeros((self._pedestrian_capacity, 2), dtype=dtype)
        if array.shape[0] > 0:
            padded[: array.shape[0]] = array
        return padded

    def _observation(self) -> dict[str, np.ndarray]:
        """Return compact dynamic pedestrian state.

        Returns
        -------
        dict[str, np.ndarray]
            Pedestrian positions, velocities, goals, forces, and current step index.
        """
        states = self.sim.pysf_state.pysf_states()
        return {
            "positions": self._pad_pedestrian_matrix(states[:, 0:2], dtype=np.float32),
            "velocities": self._pad_pedestrian_matrix(states[:, 2:4], dtype=np.float32),
            "goals": self._pad_pedestrian_matrix(states[:, 4:6], dtype=np.float32),
            "forces": self._pedestrian_forces(),
            "step": np.asarray([self._step_count], dtype=np.int64),
        }

    def _pedestrian_forces(self) -> np.ndarray:
        """Return the latest pedestrian force array, padded on reset when needed.

        Returns
        -------
        np.ndarray
            ``(num_pedestrians, 2)`` force array, zero-filled before the first simulator step.
        """
        num_peds = int(self.sim.pysf_state.num_peds)
        forces = np.asarray(self.sim.last_ped_forces, dtype=np.float32)
        if forces.shape == (num_peds, 2):
            return self._pad_pedestrian_matrix(forces, dtype=np.float32)
        return np.zeros((self._pedestrian_capacity, 2), dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        """Return static scene and episode metadata outside the step observation.

        Returns
        -------
        dict[str, Any]
            Map, pedestrian-count, timing, horizon, and recording metadata.
        """
        return {
            "map_id": self.map_id,
            "num_pedestrians": int(self.sim.pysf_state.num_peds),
            "time_per_step_in_secs": self.config.sim_config.time_per_step_in_secs,
            "sim_time_in_secs": self.config.sim_config.sim_time_in_secs,
            "max_sim_steps": self.config.sim_config.max_sim_steps,
            "recording_path": str(self._recording_path) if self._recording_path else None,
        }

    def _visualizable_state(self) -> SimpleNamespace:
        """Build a robot-free state object accepted by ``SimulationView``.

        Returns
        -------
        SimpleNamespace
            Minimal visualizable state containing only pedestrian and timing fields.
        """
        return SimpleNamespace(
            timestep=self._step_count,
            pedestrian_positions=np.asarray(self.sim.ped_pos, dtype=float),
            ped_actions=np.asarray(self.sim.last_ped_forces, dtype=float),
            time_per_step_in_secs=self.config.sim_config.time_per_step_in_secs,
        )

    def _ensure_view(self) -> SimulationView:
        """Create the renderer lazily only when rendering or video capture is used.

        Returns
        -------
        SimulationView
            Lazily constructed or cached simulation view.
        """
        if self._sim_ui is None:
            self._sim_ui = SimulationView(
                map_def=self.map_def,
                obstacles=self.map_def.obstacles,
                caption="RobotSF Crowd Simulation",
                focus_on_robot=False,
                focus_on_ego_ped=False,
                ped_radius=self.config.sim_config.ped_radius,
                goal_radius=self.config.sim_config.goal_radius,
                record_video=bool(self.config.video_path or self.render_mode == "rgb_array"),
                video_path=self.config.video_path,
                video_fps=self.config.video_fps,
            )
        return self._sim_ui

    def _target_fps(self) -> float:
        """Return the effective render cadence.

        Returns
        -------
        float
            Configured video FPS when set, otherwise the environment render FPS.
        """
        if self.config.video_fps is not None:
            return float(self.config.video_fps)
        return float(self.metadata["render_fps"])

    def _open_recording(self) -> None:
        """Open the compact JSONL recording stream."""
        if self.config.recording_path:
            path = Path(self.config.recording_path)
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            unique = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
            path = Path(self.config.recording_dir) / f"crowd_sim_{stamp}_{unique}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._recording_path = path
        self._recording_file = path.open("a", encoding="utf-8")

    def _record_event(
        self,
        event: str,
        observation: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> None:
        """Append one compact JSONL event when recording is enabled."""
        if self._recording_file is None:
            return
        payload = {
            "event": event,
            "step": int(self._step_count),
            "observation": {key: value.tolist() for key, value in observation.items()},
        }
        payload["info"] = info if event == "reset" else {}
        self._recording_file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._recording_file.flush()
