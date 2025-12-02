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

from loguru import logger

from robot_sf.gym_env.base_env import BaseEnv
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.env_util import (
    init_collision_and_sensors,
    init_spaces,
    prepare_pedestrian_actions,
)
from robot_sf.gym_env.reward import simple_reward
from robot_sf.render.lidar_visual import render_lidar
from robot_sf.render.sim_view import VisualizableAction, VisualizableSimState
from robot_sf.robot.robot_state import RobotState
from robot_sf.sensor.range_sensor import lidar_ray_scan
from robot_sf.sim.simulator import (
    init_simulators,  # noqa: F401 (retained for backwards compatibility; may be removed later)
)


# Helper to compute a stable, short hash for env_config
# Placed near imports for reuse and clarity
def _stable_config_hash(cfg: EnvSettings) -> str:
    """Stable config hash.

    Args:
        cfg: Auto-generated placeholder description.

    Returns:
        str: Auto-generated placeholder description.
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
    """Construct the info dict with collision/success flags for downstream consumers."""

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
        """
        Initialize the Robot Environment.

        Args:
            env_config: Environment configuration settings.
            reward_func: Optional reward function; falls back to :func:`simple_reward`.
            debug: Whether to enable debug visualizations.
            recording_enabled: Whether to record simulator states.
            record_video: Whether to capture videos via :class:`SimulationView`.
            video_path: Target path for recorded videos.
            video_fps: Optional FPS override for recordings.
            peds_have_obstacle_forces: Whether pedestrians exert obstacle forces.
            use_jsonl_recording: Whether to emit JSONL recording artifacts.
            recording_dir: Directory for JSONL or video artifacts.
            suite_name: Suite identifier stored with recordings.
            scenario_name: Scenario identifier stored with recordings.
            algorithm_name: Algorithm identifier stored with recordings.
            recording_seed: Optional deterministic seed saved with recordings.
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

        # Setup initial state of the robot
        self.state = RobotState(
            self.simulator.robot_navs[0],
            occupancies[0],
            sensors[0],
            env_config.sim_config.time_per_step_in_secs,
            env_config.sim_config.sim_time_in_secs,
        )

        # Store last action executed by the robot
        self.last_action = None

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action: Action sampled by the policy/agent.

        Returns:
            tuple: Observation, reward, termination flag, truncation flag, and info dict.
        """
        # Process the action through the simulator
        action = self.simulator.robots[0].parse_action(action)
        # Perform simulation step
        self.simulator.step_once([action])
        # Get updated observation
        obs = self.state.step()
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
        """
        Reset the environment to an initial state to begin a new episode.

        This method performs the following operations:
        1. Calls the superclass reset method with the provided seed and options.
        2. Clears the stored last action.
        3. Resets the internal state of the simulator.
        4. Resets the environment's state to obtain the initial observation.
        5. If recording is enabled, saves the current recording.

        Args:
            seed: Optional seed value for reproducible resets.
            options: Additional reset options.

        Returns:
            tuple: Initial observation and an info dict.
        """
        super().reset(seed=seed, options=options)
        # Reset last_action
        self.last_action = None
        # Reset internal simulator state
        self.simulator.reset_state()
        # Reset the environment's state and return the initial observation
        obs = self.state.reset()

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

    def _prepare_visualizable_state(self):
        """Prepare visualizable state.

        Returns:
            Any: Auto-generated placeholder description.
        """
        # Prepare action visualization, if any action was executed
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
