"""Refactored pedestrian environment implementation.

This module provides the canonical PedestrianEnv implementation backed by the
new abstract environment base classes and unified configuration system. The
legacy import path (robot_sf.gym_env.pedestrian_env) re-exports this class for
backward compatibility.
"""

import datetime
import pickle
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import loguru
import numpy as np
from gymnasium import spaces

from robot_sf.common.artifact_paths import get_artifact_category_path, resolve_artifact_path
from robot_sf.gym_env._stub_robot_model import StubRobotModel
from robot_sf.gym_env.abstract_envs import SingleAgentEnv
from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.gym_env.env_util import (
    init_ped_collision_and_sensors,
    init_ped_spaces,
    prepare_pedestrian_actions,
)
from robot_sf.gym_env.reward import simple_ped_reward
from robot_sf.gym_env.robot_env import _build_step_info
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinition
from robot_sf.ped_ego.pedestrian_state import PedestrianState
from robot_sf.render.lidar_visual import render_lidar
from robot_sf.render.sim_view import (
    SimulationView,
    VisualizableAction,
    VisualizableSimState,
)
from robot_sf.robot.robot_state import RobotState
from robot_sf.sensor.range_sensor import lidar_ray_scan
from robot_sf.sim.simulator import PedSimulator, init_ped_simulators

logger = loguru.logger


class RefactoredPedestrianEnv(SingleAgentEnv):
    """
    Refactored Pedestrian Environment using new architecture.

    This environment trains an adversarial pedestrian against a pre-trained robot.
    Demonstrates the new consistent interface and reduced code duplication.
    """

    # Type annotations for attributes to help type checker
    config: PedestrianSimulationConfig
    map_def: MapDefinition
    simulator: PedSimulator
    action_space: Any
    observation_space: Any
    orig_obs_space: Any
    robot_state: RobotState
    ped_state: PedestrianState
    state: PedestrianState
    last_obs_robot: Any
    last_action_robot: Any
    last_action_ped: Any
    robot_action_space: spaces.Space | None
    _robot_action_space_valid: bool

    def __init__(
        self,
        env_config: PedestrianSimulationConfig | PedEnvSettings | None = None,
        robot_model=None,
        reward_func: Callable[[dict], float] = simple_ped_reward,
        debug: bool = False,
        recording_enabled: bool = False,
        peds_have_obstacle_forces: bool | None = None,
        **kwargs,
    ):
        """
        Initialize the Pedestrian Environment.

        Args:
            env_config: Pedestrian simulation configuration (unified or legacy).
            robot_model: Pre-trained robot model for adversarial interaction.
            reward_func: Reward function for pedestrian training.
            debug: Enable debug mode with visualization.
            recording_enabled: Enable state recording.
            peds_have_obstacle_forces: Deprecated. Controls static obstacle forces for pedestrians.
            **kwargs: Additional keyword arguments forwarded to ``SingleAgentEnv``.
        """
        if env_config is None:
            env_config = PedestrianSimulationConfig()

        # Ensure pedestrian obstacle forces are configured consistently.
        if peds_have_obstacle_forces is not None:
            if hasattr(env_config, "peds_have_static_obstacle_forces"):
                env_config.peds_have_static_obstacle_forces = peds_have_obstacle_forces
            else:
                env_config.peds_have_obstacle_forces = peds_have_obstacle_forces

        # Store robot model
        if robot_model is None:
            robot_model = StubRobotModel()
        self.robot_model = robot_model

        # Store reward function
        self.reward_func = reward_func or simple_ped_reward

        self.robot_action_space = None
        self._robot_action_space_valid = True

        # Initialize base class
        super().__init__(
            config=env_config,
            debug=debug,
            recording_enabled=recording_enabled,
            **kwargs,
        )

        # Recording directory mirrors legacy behavior (canonical artifact category).
        self._recording_dir = get_artifact_category_path("recordings")

        # Track last actions/observations
        self.last_obs_robot = None
        self.last_action_robot = None
        self.last_action_ped = None

    def _setup_environment(self) -> None:
        """Initialize environment-specific components."""
        # Get map definition (respect explicit map_id when provided)
        map_id = getattr(self.config, "map_id", None)
        if map_id:
            try:
                self.map_def = self.config.map_pool.get_map(map_id)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
        else:
            self.map_def = self.config.map_pool.choose_random_map()

        # Initialize spaces
        self.action_space, self.observation_space, self.orig_obs_space = self._create_spaces()
        self._configure_robot_model_action_space()

        # Setup simulator and sensors
        self._setup_simulator()
        self._setup_sensors_and_collision()

        # Setup visualization if in debug mode
        if self.debug:
            self._setup_visualization()

    def _create_spaces(self):
        """Create action and observation spaces.

        Returns:
            Tuple of (action_space, observation_space, orig_obs_space) for the pedestrian.
        """
        # Use existing utility function
        combined_action_space, combined_observation_space, orig_obs_space = init_ped_spaces(
            self.config,
            self.map_def,
        )

        # Return pedestrian spaces (index 1)
        self.robot_action_space = combined_action_space[0]
        return combined_action_space[1], combined_observation_space[1], orig_obs_space

    def _configure_robot_model_action_space(self) -> None:
        """Configure and validate the robot model action space if available."""
        if hasattr(self.robot_model, "set_action_space") and self.robot_action_space is not None:
            try:
                self.robot_model.set_action_space(self.robot_action_space)
            except (AttributeError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.warning(f"Failed to set robot model action space: {exc}")

        self._robot_action_space_valid = self._validate_robot_model_action_space()

    def _validate_robot_model_action_space(self) -> bool:
        """Validate the robot model action space against the environment.

        Returns:
            bool: True when the model action space is compatible with the environment.
        """
        if self.robot_action_space is None:
            return True
        if not hasattr(self.robot_model, "action_space"):
            return True

        model_space = getattr(self.robot_model, "action_space", None)
        if model_space is None:
            return True

        if isinstance(self.robot_action_space, spaces.Box) and isinstance(model_space, spaces.Box):
            if model_space.shape != self.robot_action_space.shape:
                logger.warning(
                    "Robot model action space shape "
                    f"{model_space.shape} does not match env shape "
                    f"{self.robot_action_space.shape}. Falling back to null actions."
                )
                return False
            if not np.allclose(model_space.low, self.robot_action_space.low) or not np.allclose(
                model_space.high,
                self.robot_action_space.high,
            ):
                logger.warning(
                    "Robot model action space bounds do not match env bounds. "
                    "Falling back to null actions."
                )
                return False
            return True

        if hasattr(model_space, "shape") and hasattr(self.robot_action_space, "shape"):
            if model_space.shape != self.robot_action_space.shape:
                logger.warning(
                    "Robot model action space shape "
                    f"{model_space.shape} does not match env shape "
                    f"{self.robot_action_space.shape}. Falling back to null actions."
                )
                return False
        return True

    def _null_robot_action(self) -> np.ndarray:
        """Return a zero action matching the robot action space."""
        if isinstance(self.robot_action_space, spaces.Box):
            zeros = np.zeros(self.robot_action_space.shape, dtype=self.robot_action_space.dtype)
            return zeros
        shape = getattr(self.robot_action_space, "shape", (2,))
        return np.zeros(shape, dtype=float)

    def _format_robot_action(self, action: Any) -> np.ndarray | None:
        """Validate/clip robot action to the environment action space.

        Returns:
            np.ndarray | None: The formatted action or None when invalid.
        """
        if self.robot_action_space is None:
            return None
        try:
            arr = np.asarray(action, dtype=float)
        except (TypeError, ValueError):
            return None

        target_shape = getattr(self.robot_action_space, "shape", None)
        if target_shape is not None:
            if arr.shape != target_shape:
                if arr.size == int(np.prod(target_shape)):
                    arr = arr.reshape(target_shape)
                else:
                    return None

        if not np.all(np.isfinite(arr)):
            return None

        if isinstance(self.robot_action_space, spaces.Box):
            arr = np.clip(arr, self.robot_action_space.low, self.robot_action_space.high)
            arr = arr.astype(self.robot_action_space.dtype, copy=False)
        return arr

    def _setup_simulator(self) -> None:
        """Initialize the simulator."""
        self.simulator = init_ped_simulators(
            self.config,
            self.map_def,
            random_start_pos=True,
            peds_have_obstacle_forces=getattr(
                self.config,
                "peds_have_static_obstacle_forces",
                self.config.peds_have_obstacle_forces,
            ),
        )[0]

    def _setup_sensors_and_collision(self) -> None:
        """Initialize sensors and collision detection."""
        occupancies, sensors = init_ped_collision_and_sensors(
            self.simulator,
            self.config,
            self.orig_obs_space,
        )

        # Setup robot state
        self.robot_state = RobotState(
            nav=self.simulator.robot_navs[0],
            occupancy=occupancies[0],
            sensors=sensors[0],
            d_t=self.config.sim_config.time_per_step_in_secs,
            sim_time_limit=self.config.sim_config.sim_time_in_secs,
        )

        # Setup pedestrian state
        self.ped_state = PedestrianState(
            robot_occupancy=occupancies[0],
            ego_ped_occupancy=occupancies[1],
            sensors=sensors[1],
            d_t=self.config.sim_config.time_per_step_in_secs,
            sim_time_limit=self.config.sim_config.sim_time_in_secs,
        )

        # Store state references for base class
        self.state = self.ped_state

    def _setup_visualization(self) -> None:
        """Setup visualization for debug mode."""
        scaling_value = getattr(self.config, "render_scaling", None)
        scaling_value = 10 if scaling_value is None else int(scaling_value)
        self.sim_ui = SimulationView(
            scaling=scaling_value,
            map_def=self.map_def,
            obstacles=self.map_def.obstacles,
            robot_radius=self.config.robot_config.radius,
            ego_ped_radius=self.config.ego_ped_config.radius,
            ped_radius=self.config.sim_config.ped_radius,
            goal_radius=self.config.sim_config.goal_radius,
            record_video=self.record_video,
            video_path=self.video_path,
            video_fps=self.video_fps if self.video_fps is not None else 10.0,
        )

    def step(self, action):
        """Execute one environment step.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) following Gymnasium API.
        """
        # Parse pedestrian action
        action_ped = self.simulator.ego_ped.parse_action(action)
        self.last_action_ped = action_ped

        # Get robot action from model
        if not self._robot_action_space_valid:
            action_robot = self._null_robot_action()
        else:
            try:
                action_robot, _ = self.robot_model.predict(
                    self.last_obs_robot,
                    deterministic=True,
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning(f"Robot model predict failed ({exc}); using null action.")
                action_robot = self._null_robot_action()
            else:
                formatted = self._format_robot_action(action_robot)
                if formatted is None:
                    logger.warning("Robot model produced invalid action; using null action.")
                    action_robot = self._null_robot_action()
                else:
                    action_robot = formatted

        action_robot = self.simulator.robots[0].parse_action(action_robot)
        self.last_action_robot = action_robot

        # Execute simulation step
        self.simulator.step_once([action_robot], [action_ped])

        # Get updated observations
        self.last_obs_robot = self.robot_state.step()
        obs_ped = self.ped_state.step()

        # Get metadata and check terminal state
        meta = self.ped_state.meta_dict()
        terminated = self.ped_state.is_terminal

        # Calculate reward
        reward = self.reward_func(meta)

        # Record state if enabled
        if self.recording_enabled:
            self.record()

        info = _build_step_info(meta)
        return obs_ped, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Returns:
            Tuple of (observation, info) after environment reset.
        """
        super().reset(seed=seed, options=options)

        # Reset simulator
        self.simulator.reset_state()

        # Reset states
        self.last_obs_robot = self.robot_state.reset()
        obs_ped = self.ped_state.reset()

        # Reset action tracking
        self.last_action_robot = None
        self.last_action_ped = None

        if self.recording_enabled:
            self.save_recording()

        # Preserve legacy info payload shape.
        return obs_ped, {"info": "test"}

    def render(self, **kwargs):
        """Render the environment."""
        if not self.sim_ui:
            raise RuntimeError("Debug mode is not activated! Set debug=True!")

        state = self._prepare_visualizable_state()
        self.sim_ui.render(state)

    def _prepare_visualizable_state(self) -> VisualizableSimState:
        """Prepare state for visualization.

        Returns:
            VisualizableSimState containing current simulation state for rendering.
        """
        # Prepare robot action visualization
        robot_action = (
            None
            if not self.last_action_robot
            else VisualizableAction(
                self.simulator.robot_poses[0],
                self.last_action_robot,
                self.simulator.goal_pos[0],
            )
        )

        # Robot LIDAR visualization
        robot_pos = self.simulator.robot_poses[0][0]
        distances, directions = lidar_ray_scan(
            self.simulator.robot_poses[0],
            self.robot_state.occupancy,
            self.config.lidar_config,
        )
        robot_ray_vecs = render_lidar(robot_pos, distances, directions)

        # Prepare pedestrian action visualization
        ego_ped_action = (
            None
            if not self.last_action_ped
            else VisualizableAction(
                self.simulator.ego_ped_pose,
                self.last_action_ped,
                self.simulator.ego_ped_goal_pos,
            )
        )

        # Ego pedestrian LIDAR visualization
        ego_ped_pos = self.simulator.ego_ped_pos
        distances, directions = lidar_ray_scan(
            self.simulator.ego_ped_pose,
            self.ped_state.ego_ped_occupancy,
            self.config.lidar_config,
        )
        ego_ped_ray_vecs = render_lidar(ego_ped_pos, distances, directions)

        # Prepare NPC pedestrian actions
        ped_actions_np = prepare_pedestrian_actions(self.simulator)

        # Create visualizable state
        state = VisualizableSimState(
            self.robot_state.timestep,
            robot_action,
            self.simulator.robot_poses[0],
            deepcopy(self.simulator.ped_pos),
            robot_ray_vecs,
            ped_actions_np,
            self.simulator.ego_ped_pose,
            ego_ped_ray_vecs,
            ego_ped_action,
            time_per_step_in_secs=self.config.sim_config.time_per_step_in_secs,
        )

        return state

    def record(self):
        """Record current state for later playback."""
        state = self._prepare_visualizable_state()
        self.recorded_states.append(state)

    def save_recording(self, filename: str | None = None):
        """Save recorded states to a pickle file.

        Args:
            filename: Optional target filename ending with ``.pkl``. When omitted,
                a timestamped file is created under the recordings directory.
        """
        if filename is None:
            now = datetime.datetime.now()
            target_path = self._recording_dir / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        else:
            target_path = resolve_artifact_path(filename)

        if len(self.recorded_states) == 0:
            logger.warning("No states recorded, skipping save")
            return

        target_path.parent.mkdir(parents=True, exist_ok=True)

        with target_path.open("wb") as f:  # write binary
            pickle.dump((self.recorded_states, self.map_def), f)
            logger.info(f"Recording saved to {target_path}")
            logger.info("Reset state list")
            self.recorded_states = []


# Create alias for backward compatibility
PedestrianEnvRefactored = RefactoredPedestrianEnv
