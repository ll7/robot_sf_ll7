"""
`robot_env.py` is a module that defines the simulation environment for a robot or multiple robots.
It includes classes and protocols for defining the robot's state, actions, and
observations within the environment.

`RobotEnv`: A class that represents the robot's environment. It inherits from `VectorEnv`
from the `gymnasium` library, which is a base class for environments that operate over
vectorized actions and observations. It includes methods for stepping through the environment,
resetting it, rendering it, and closing it.
It also defines the action and observation spaces for the robot.
"""

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from loguru import logger

from robot_sf.common.types import Line2D
from robot_sf.gym_env.base_env import BaseEnv
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.env_util import (
    init_collision_and_sensors,
    init_spaces,
    prepare_pedestrian_actions,
)
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.reward import simple_reward
from robot_sf.nav.obstacle import Obstacle
from robot_sf.nav.occupancy_grid import OccupancyGrid
from robot_sf.render.lidar_visual import render_lidar
from robot_sf.render.sim_view import VisualizableAction, VisualizableSimState
from robot_sf.robot.robot_state import RobotState
from robot_sf.sensor.range_sensor import lidar_ray_scan
from robot_sf.sensor.socnav_observation import (
    DEFAULT_MAX_PEDS,
    SocNavObservationFusion,
    socnav_observation_space,
)
from robot_sf.sim.simulator import (
    init_simulators,  # noqa: F401 (retained for backwards compatibility; may be removed later)
)


# Helper to compute a stable, short hash for env_config
# Placed near imports for reuse and clarity
def _stable_config_hash(cfg: EnvSettings) -> str:
    """TODO docstring. Document this function.

    Args:
        cfg: TODO docstring.

    Returns:
        16-character hexadecimal hash string representing the configuration.
    """
    try:
        payload = json.dumps(
            asdict(cfg) if is_dataclass(cfg) else cfg.__dict__,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        payload = repr(cfg)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def _build_step_info(meta: dict[str, Any]) -> dict[str, Any]:
    """Construct the info dict with collision/success flags for downstream consumers.

    Returns:
        Dictionary containing step, meta, collision, success, and is_success flags.
    """

    collision = bool(
        meta.get("is_pedestrian_collision")
        or meta.get("is_obstacle_collision")
        or meta.get("is_robot_collision")
    )
    success = bool(meta.get("is_route_complete") or meta.get("is_robot_at_goal"))
    return {
        "step": meta.get("step"),
        "meta": meta,
        "collision": collision,
        "success": success,
        "is_success": success,  # backward-compat key used by some scripts
    }


class RobotEnv(BaseEnv):
    """
    Representing a Gymnasium environment for training a self-driving robot
    with reinforcement learning.
    """

    def __init__(
        self,
        env_config: EnvSettings = EnvSettings(),
        reward_func: Callable[[dict], float] | None = None,
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str | None = None,
        video_fps: float | None = None,
        peds_have_obstacle_forces: bool = False,
        # New JSONL recording parameters
        use_jsonl_recording: bool = False,
        recording_dir: str = "recordings",
        suite_name: str = "robot_sim",
        scenario_name: str = "default",
        algorithm_name: str = "manual",
        recording_seed: int | None = None,
    ):
        """Initialize the robot environment.

        Args:
            env_config: Environment settings describing maps, sensors, and simulator behavior.
            reward_func: Optional callable used to compute rewards; defaults to ``simple_reward``.
            debug: Enables ``SimulationView`` visualization and rendering hooks.
            recording_enabled: When ``True``, record ``VisualizableSimState`` snapshots.
            record_video: Save simulator frames as a video via ``SimulationView``.
            video_path: Output path for the recorded video (when ``record_video`` is enabled).
            video_fps: Override frames-per-second for recorded videos.
            peds_have_obstacle_forces: Whether ped forces interact with obstacles.
            use_jsonl_recording: Enable structured JSONL recording instead of pickles.
            recording_dir: Directory for recordings.
            suite_name: Logical suite name stored in recording metadata.
            scenario_name: Scenario identifier stored in metadata.
            algorithm_name: Algorithm identifier stored in metadata.
            recording_seed: Optional seed stored alongside the recording metadata.
        """
        super().__init__(
            env_config=env_config,
            debug=debug,
            recording_enabled=recording_enabled,
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
            use_jsonl_recording=use_jsonl_recording,
            recording_dir=recording_dir,
            suite_name=suite_name,
            scenario_name=scenario_name,
            algorithm_name=algorithm_name,
            recording_seed=recording_seed,
        )

        # Initialize spaces based on the environment configuration and map
        self.action_space, self.observation_space, orig_obs_space = init_spaces(
            env_config,
            self.map_def,
        )

        # Assign the reward function; ensure a valid callable even if None passed via factory
        if reward_func is None:  # defensive: factory allows Optional
            logger.warning(
                "No reward_func provided to RobotEnv; falling back to simple_reward for safety.",
            )
        self.reward_func = reward_func or simple_reward

        # BaseEnv has already created self.simulator; avoid redundant initialization.
        # Initialize collision detectors and sensor data processors using existing simulator.
        occupancies, sensors = init_collision_and_sensors(
            self.simulator,
            env_config,
            orig_obs_space,
        )

        # Store configuration for factory pattern compatibility
        self.config = env_config

        # T043: Initialize occupancy grid if grid observation is enabled
        self.occupancy_grid = None
        if (
            hasattr(env_config, "include_grid_in_observation")
            and env_config.include_grid_in_observation
        ):
            self.occupancy_grid = OccupancyGrid(config=env_config.grid_config)
            logger.info(
                f"Occupancy grid initialized for observations: "
                f"shape={self.occupancy_grid.shape}, "
                f"resolution={env_config.grid_config.resolution}m"
            )

        if env_config.observation_mode == ObservationMode.SOCNAV_STRUCT:
            # Build SocNav-style observation space and fusion layer
            ped_count = getattr(self.simulator, "ped_pos", np.zeros((0, 2))).shape[0]
            max_peds = getattr(env_config.sim_config, "max_total_pedestrians", None)
            if max_peds is None:
                max_peds = max(DEFAULT_MAX_PEDS, ped_count)
            self.observation_space = socnav_observation_space(
                self.map_def,
                env_config,
                max_peds,
            )
            socnav_fusion = SocNavObservationFusion(
                simulator=self.simulator,
                env_config=env_config,
                max_pedestrians=max_peds,
                robot_index=0,
            )
            sensor_adapter = socnav_fusion
        else:
            sensor_adapter = sensors[0]

        # Setup initial state of the robot
        self.state = RobotState(
            self.simulator.robot_navs[0],
            occupancies[0],
            sensor_adapter,
            env_config.sim_config.time_per_step_in_secs,
            env_config.sim_config.sim_time_in_secs,
        )

        # Store last action executed by the robot
        self.last_action = None

    def step(self, action):
        """Execute one environment step.

        Args:
            action: Action sampled from ``action_space`` for the controlled robot.

        Returns:
            tuple: ``(obs, reward, terminated, truncated, info)`` per Gymnasium API.
        """
        # Process the action through the simulator
        action = self.simulator.robots[0].parse_action(action)
        # Perform simulation step
        self.simulator.step_once([action])
        # Get updated observation
        obs = self.state.step()

        # T044: Update occupancy grid if enabled
        if self.occupancy_grid is not None:
            # Extract obstacles from map
            obstacles = self._normalize_obstacles_for_grid(
                self.map_def.obstacles, self.map_def.bounds
            )
            # Extract updated pedestrian positions and radii
            ped_positions = self.simulator.ped_pos
            ped_radii = getattr(self.simulator, "ped_radii", None)
            if ped_radii is None:
                ped_radii = [0.35] * len(ped_positions)
            pedestrians = [
                (tuple(pos), radius) for pos, radius in zip(ped_positions, ped_radii, strict=True)
            ]
            # Get updated robot pose (already in RobotPose format: ((x, y), theta))
            robot_pose = self.simulator.robot_poses[0]
            # Regenerate grid
            self.occupancy_grid.generate(
                obstacles=obstacles,
                pedestrians=pedestrians,
                robot_pose=robot_pose,
                ego_frame=False,
            )
            # Update observation with new grid
            obs["occupancy_grid"] = self.occupancy_grid.to_observation()

        # Fetch metadata about the current state
        reward_dict = self.state.meta_dict()
        # add the action space to dict
        reward_dict["action_space"] = self.action_space
        # add action to dict
        reward_dict["action"] = action
        # Add last_action to reward_dict
        reward_dict["last_action"] = self.last_action
        # Determine if the episode has reached terminal state
        term = self.state.is_terminal
        # Compute the reward using the provided reward function
        reward = self.reward_func(reward_dict)
        # Update last_action for next step
        self.last_action = action

        # if recording is enabled, record the state
        if self.recording_enabled:
            self.record()

        # observation, reward, terminal, truncated,info
        info = _build_step_info(reward_dict)
        return (
            obs,
            reward,
            term,
            False,
            info,
        )

    def reset(self, seed=None, options=None):
        """Reset the environment and start a new episode.

        Args:
            seed: Optional random seed forwarded to ``BaseEnv`` reset.
            options: Optional Gymnasium reset options.

        Returns:
            tuple: ``(obs, info)`` with the initial observation and placeholder info dict.
        """
        super().reset(seed=seed, options=options)
        # Reset last_action
        self.last_action = None
        # Reset internal simulator state
        self.simulator.reset_state()
        # Reset the environment's state and return the initial observation
        obs = self.state.reset()

        # T043: Generate initial occupancy grid if enabled
        if self.occupancy_grid is not None:
            # Extract obstacles from map
            obstacles = self._normalize_obstacles_for_grid(
                self.map_def.obstacles, self.map_def.bounds
            )
            # Extract pedestrian positions and radii from simulator
            ped_positions = self.simulator.ped_pos
            ped_radii = getattr(self.simulator, "ped_radii", None)
            if ped_radii is None:
                # Default pedestrian radius if not available
                ped_radii = [0.35] * len(ped_positions)
            pedestrians = [
                (tuple(pos), radius) for pos, radius in zip(ped_positions, ped_radii, strict=True)
            ]
            # Get robot pose (already in RobotPose format: ((x, y), theta))
            robot_pose = self.simulator.robot_poses[0]
            # Generate grid
            self.occupancy_grid.generate(
                obstacles=obstacles,
                pedestrians=pedestrians,
                robot_pose=robot_pose,
                ego_frame=False,  # Use world frame by default
            )
            # Add grid to observation
            obs["occupancy_grid"] = self.occupancy_grid.to_observation()
            logger.debug(
                f"Initial occupancy grid generated: "
                f"obstacles={len(obstacles)}, pedestrians={len(pedestrians)}"
            )

        # Handle recording for both systems
        if self.recording_enabled:
            if self.use_jsonl_recording:
                # End previous episode if active, then start a new one
                try:
                    self.end_episode_recording()
                except (
                    RuntimeError,
                    ValueError,
                    AttributeError,
                ):  # pragma: no cover - safe if none active
                    pass
                config_hash = _stable_config_hash(self.env_config)
                self.start_episode_recording(config_hash=config_hash)
            else:
                # Legacy pickle recording
                self.save_recording()

        # info is necessary for the gym environment, but useless at the moment
        info = {"info": "test"}
        return obs, info

    @staticmethod
    def _normalize_obstacles_for_grid(
        obstacles: list[Obstacle] | list[Line2D], bounds: list[Line2D]
    ) -> list[Line2D]:
        """Convert obstacle objects/lines plus bounds into Line2D tuples for occupancy grids.

        Returns:
            list[Line2D]: Normalized line segments derived from map obstacles and bounds.
        """
        line_segments: list[Line2D] = []

        def _add_line(line) -> None:
            try:
                start, end = line
                line_segments.append((tuple(start), tuple(end)))
            except (TypeError, ValueError):
                pass  # Skip malformed lines

        for obstacle in obstacles:
            if isinstance(obstacle, Obstacle):
                for line in obstacle.lines:
                    if len(line) == 4:
                        # Obstacle.lines stores (x1, x2, y1, y2); convert to Line2D ((x1, y1), (x2, y2))
                        x1, x2, y1, y2 = line
                        _add_line(((x1, y1), (x2, y2)))
                    else:
                        _add_line(line)
            else:
                _add_line(obstacle)

        for bound in bounds:
            _add_line(bound)

        return line_segments

    def _prepare_visualizable_state(self):
        # Prepare action visualization, if any action was executed
        """TODO docstring. Document this function.

        Returns:
            VisualizableSimState containing the current simulation state for rendering.
        """
        action = (
            None
            if not self.last_action
            else VisualizableAction(
                self.simulator.robot_poses[0],
                self.last_action,
                self.simulator.goal_pos[0],
            )
        )

        # Robot position and LIDAR scanning visualization preparation
        robot_pos = self.simulator.robot_poses[0][0]
        distances, directions = lidar_ray_scan(
            self.simulator.robot_poses[0],
            self.state.occupancy,
            self.env_config.lidar_config,
        )

        # Construct ray vectors for visualization
        ray_vecs_np = render_lidar(robot_pos, distances, directions)

        # Prepare pedestrian action visualization
        ped_actions_np = prepare_pedestrian_actions(self.simulator)

        # Package the state for visualization
        state = VisualizableSimState(
            timestep=self.state.timestep,
            robot_action=action,
            robot_pose=self.simulator.robot_poses[0],
            pedestrian_positions=deepcopy(self.simulator.ped_pos),
            ray_vecs=ray_vecs_np,
            ped_actions=ped_actions_np,
            time_per_step_in_secs=self.env_config.sim_config.time_per_step_in_secs,
        )

        return state

    def render(self):
        """
        Render the environment visually if in debug mode.

        Raises RuntimeError if debug mode is not enabled.
        """
        if not self.sim_ui:
            raise RuntimeError(
                "Render unavailable: environment was created with debug=False (no sim_ui). "
                "Recreate via make_robot_env(..., debug=True) to enable visualization and frame capture.",
            )

        state = self._prepare_visualizable_state()

        # Execute rendering of the state through the simulation UI
        self.sim_ui.render(state)

    def record(self):
        """
        Records the current state as visualizable state and stores it in the list.
        """
        state = self._prepare_visualizable_state()

        # Use the new unified recording method
        self.record_simulation_step(state)

    def set_pedestrian_velocity_scale(self, scale: float = 1.0):
        """
        Set the pedestrian velocity visualization scaling factor.

        Args:
            scale (float): Scaling factor for pedestrian velocity arrows in visualization.
                          1.0 = actual size, 2.0 = double size for better visibility, etc.
        """
        if self.sim_ui:
            self.sim_ui.ped_velocity_scale = scale
        else:
            logger.warning("Cannot set velocity scale: debug mode not enabled")
