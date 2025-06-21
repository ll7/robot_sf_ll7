"""
Refactored Pedestrian Environment

This demonstrates how to migrate PedestrianEnv to use the new abstract base classes
and unified configuration system.
"""

from copy import deepcopy
from typing import Callable

import loguru

from robot_sf.gym_env.abstract_envs import SingleAgentEnv
from robot_sf.gym_env.env_util import (
    init_ped_collision_and_sensors,
    init_ped_spaces,
    prepare_pedestrian_actions,
)
from robot_sf.gym_env.reward import simple_ped_reward
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.ped_ego.pedestrian_state import PedestrianState
from robot_sf.render.lidar_visual import render_lidar
from robot_sf.render.sim_view import (
    SimulationView,
    VisualizableAction,
    VisualizableSimState,
)
from robot_sf.robot.robot_state import RobotState
from robot_sf.sensor.range_sensor import lidar_ray_scan
from robot_sf.sim.simulator import init_ped_simulators

logger = loguru.logger


class RefactoredPedestrianEnv(SingleAgentEnv):
    """
    Refactored Pedestrian Environment using new architecture.

    This environment trains an adversarial pedestrian against a pre-trained robot.
    Demonstrates the new consistent interface and reduced code duplication.
    """

    def __init__(
        self,
        config: PedestrianSimulationConfig = None,
        robot_model=None,
        reward_func: Callable[[dict], float] = simple_ped_reward,
        debug: bool = False,
        recording_enabled: bool = False,
        peds_have_obstacle_forces: bool = False,
        **kwargs,
    ):
        """
        Initialize the Pedestrian Environment.

        Args:
            config: Pedestrian simulation configuration
            robot_model: Pre-trained robot model for adversarial interaction
            reward_func: Reward function for pedestrian training
            debug: Enable debug mode with visualization
            recording_enabled: Enable state recording
            peds_have_obstacle_forces: Whether pedestrians exert obstacle forces
        """
        if config is None:
            config = PedestrianSimulationConfig()

        # Store robot model
        if robot_model is None:
            raise ValueError("Robot model is required for pedestrian environment!")
        self.robot_model = robot_model

        # Store reward function
        self.reward_func = reward_func

        # Update config
        config.peds_have_obstacle_forces = peds_have_obstacle_forces

        # Initialize base class
        super().__init__(config=config, debug=debug, recording_enabled=recording_enabled, **kwargs)

    def _setup_environment(self) -> None:
        """Initialize environment-specific components."""
        # Get map definition
        self.map_def = self.config.map_pool.choose_random_map()

        # Initialize spaces
        self.action_space, self.observation_space, self.orig_obs_space = self._create_spaces()

        # Setup simulator and sensors
        self._setup_simulator()
        self._setup_sensors_and_collision()

        # Setup visualization if in debug mode
        if self.debug:
            self._setup_visualization()

    def _create_spaces(self):
        """Create action and observation spaces."""
        # Use existing utility function
        combined_action_space, combined_observation_space, orig_obs_space = init_ped_spaces(
            self.config, self.map_def
        )

        # Return pedestrian spaces (index 1)
        return combined_action_space[1], combined_observation_space[1], orig_obs_space

    def _setup_simulator(self) -> None:
        """Initialize the simulator."""
        self.simulator = init_ped_simulators(
            self.config,
            self.map_def,
            random_start_pos=True,
            peds_have_obstacle_forces=self.config.peds_have_obstacle_forces,
        )[0]

    def _setup_sensors_and_collision(self) -> None:
        """Initialize sensors and collision detection."""
        occupancies, sensors = init_ped_collision_and_sensors(
            self.simulator, self.config, self.orig_obs_space
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
        self.sim_ui = SimulationView(
            scaling=10,
            map_def=self.map_def,
            obstacles=self.map_def.obstacles,
            robot_radius=self.config.robot_config.radius,
            ego_ped_radius=self.config.ego_ped_config.radius,
            ped_radius=self.config.sim_config.ped_radius,
            goal_radius=self.config.sim_config.goal_radius,
        )

    def step(self, action):
        """Execute one environment step."""
        # Parse pedestrian action
        action_ped = self.simulator.ego_ped.parse_action(action)
        self.last_action_ped = action_ped

        # Get robot action from model
        action_robot, _ = self.robot_model.predict(self.last_obs_robot, deterministic=True)
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

        return obs_ped, reward, terminated, False, {"step": meta["step"], "meta": meta}

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset simulator
        self.simulator.reset_state()

        # Reset states
        self.last_obs_robot = self.robot_state.reset()
        obs_ped = self.ped_state.reset()

        # Reset action tracking
        self.last_action_robot = None
        self.last_action_ped = None

        return obs_ped, {}

    def render(self, **kwargs):
        """Render the environment."""
        if not self.sim_ui:
            raise RuntimeError("Debug mode is not activated! Set debug=True!")

        state = self._prepare_visualizable_state()
        self.sim_ui.render(state)

    def _prepare_visualizable_state(self) -> VisualizableSimState:
        """Prepare state for visualization."""
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


# Create alias for backward compatibility
PedestrianEnvRefactored = RefactoredPedestrianEnv
