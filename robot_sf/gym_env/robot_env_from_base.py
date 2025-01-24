"""
Robot environment implementation that inherits from BaseEnv.
Provides specific robot-related functionality while reusing common base features.
"""

from typing import Tuple, Dict, Any, Callable, Optional
import numpy as np
from copy import deepcopy

from robot_sf.gym_env.base_env import BaseEnv
from robot_sf.gym_env.env_config import RobotEnvSettings
from robot_sf.robot.robot_state import RobotState
from robot_sf.gym_env.env_util import init_collision_and_sensors, init_spaces
from robot_sf.gym_env.reward import simple_reward
from robot_sf.render.sim_view import VisualizableSimState, VisualizableAction
from robot_sf.sensor.range_sensor import lidar_ray_scan


class RobotEnvFromBase(BaseEnv):
    """Robot-specific environment implementation."""

    def __init__(
        self,
        env_config: RobotEnvSettings = RobotEnvSettings(),
        reward_func: Callable[[dict], float] = simple_reward,
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: Optional[str] = None,
        video_fps: Optional[float] = None,
        peds_have_obstacle_forces: bool = False,
    ):
        """Initialize robot environment."""
        super().__init__(
            env_config=env_config,
            debug=debug,
            recording_enabled=recording_enabled,
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
        )

        self.reward_func = reward_func
        self.peds_have_obstacle_forces = peds_have_obstacle_forces

        # Initialize robot-specific components
        self._setup_robot_environment()

    def _setup_robot_environment(self):
        """Setup robot-specific environment components."""
        # Initialize spaces
        self.action_space, self.observation_space, orig_obs_space = init_spaces(
            self.env_config, self.map_def
        )

        # Setup simulator
        self._setup_simulator()

        # Setup state management
        d_t = self.env_config.sim_config.time_per_step_in_secs
        max_ep_time = self.env_config.sim_config.sim_time_in_secs

        occupancies, sensors = init_collision_and_sensors(
            self.simulator, self.env_config, orig_obs_space
        )

        self.state = RobotState(
            self.simulator.robot_navs[0], occupancies[0], sensors[0], d_t, max_ep_time
        )

        # Setup visualization if needed
        self._setup_visualization()

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self.last_action = action

        # Execute action in simulator
        self.simulator.step_once([action])

        # Get new observation and metadata
        obs = self.state.step()
        reward_dict = self.state.meta_dict()

        # Check terminal state and compute reward
        term = self.state.is_terminal
        reward = self.reward_func(reward_dict)

        # Record if enabled
        if self.recording_enabled:
            self.record()

        return (
            obs,
            reward,
            term,
            False,
            {"step": reward_dict["step"], "meta": reward_dict},
        )

    def reset(self, seed=None, options=None):
        """Reset environment state."""
        super().reset(seed=seed)

        self.simulator.reset_state()
        obs = self.state.reset()

        if self.recording_enabled:
            self.save_recording()

        return obs, {}

    def _prepare_visualizable_state(self) -> VisualizableSimState:
        """Prepare state for visualization."""
        # Prepare action visualization
        action = (
            None
            if not self.last_action
            else VisualizableAction(
                self.simulator.robot_poses[0],
                self.last_action,
                self.simulator.goal_pos[0],
            )
        )

        # Get robot position and LIDAR data
        robot_pos = self.simulator.robot_poses[0][0]
        distances, directions = lidar_ray_scan(
            self.simulator.robot_poses[0],
            self.state.occupancy,
            self.env_config.lidar_config,
        )

        # Create ray vectors
        ray_vecs = zip(np.cos(directions) * distances, np.sin(directions) * distances)
        ray_vecs_np = np.array(
            [
                [[robot_pos[0], robot_pos[1]], [robot_pos[0] + x, robot_pos[1] + y]]
                for x, y in ray_vecs
            ]
        )

        # Get pedestrian actions
        ped_actions = zip(
            self.simulator.pysf_sim.peds.pos(),
            self.simulator.pysf_sim.peds.pos() + self.simulator.pysf_sim.peds.vel() * 2,
        )
        ped_actions_np = np.array([[pos, vel] for pos, vel in ped_actions])

        return VisualizableSimState(
            self.state.timestep,
            action,
            self.simulator.robot_poses[0],
            deepcopy(self.simulator.ped_pos),
            ray_vecs_np,
            ped_actions_np,
        )
