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

from copy import deepcopy
from typing import Callable

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
from robot_sf.render.sim_view import (
    VisualizableAction,
    VisualizableSimState,
)
from robot_sf.robot.robot_state import RobotState
from robot_sf.sensor.range_sensor import lidar_ray_scan
from robot_sf.sim.simulator import init_simulators


class RobotEnv(BaseEnv):
    """
    Representing a Gymnasium environment for training a self-driving robot
    with reinforcement learning.
    """

    def __init__(
        self,
        env_config: EnvSettings = EnvSettings(),
        reward_func: Callable[[dict], float] = simple_reward,
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str = None,
        video_fps: float = None,
        peds_have_obstacle_forces: bool = False,
    ):
        """
        Initialize the Robot Environment.

        Parameters:
        - env_config (EnvSettings): Configuration for environment settings.
        - reward_func (Callable[[dict], float]): Reward function that takes
            a dictionary as input and returns a float as reward.
        - debug (bool): If True, enables debugging information such as
            visualizations.
        - recording_enabled (bool): If True, enables recording of the simulation
        - record_video: If True, saves simulation as video file
        - video_path: Path where to save the video file
        """
        super().__init__(
            env_config=env_config,
            debug=debug,
            recording_enabled=recording_enabled,
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
        )

        # Initialize spaces based on the environment configuration and map
        self.action_space, self.observation_space, orig_obs_space = init_spaces(
            env_config, self.map_def
        )

        # Assign the reward function; ensure a valid callable even if None passed via factory
        if reward_func is None:  # defensive: factory allows Optional
            logger.warning(
                "No reward_func provided to RobotEnv; falling back to simple_reward for safety."
            )
            self.reward_func = simple_reward
        else:
            self.reward_func = reward_func

        # Initialize simulator with a random start position
        self.simulator = init_simulators(
            env_config,
            self.map_def,
            random_start_pos=True,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
        )[0]

        # Initialize collision detectors and sensor data processors
        occupancies, sensors = init_collision_and_sensors(
            self.simulator, env_config, orig_obs_space
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

        Parameters:
        - action: Action to be executed.

        Returns:
        - obs: Observation after taking the action.
        - reward: Calculated reward for the taken action.
        - term: Boolean indicating if the episode has terminated.
        - truncated: Boolean indicating if the episode was truncated.
        - info: Additional information as dictionary.
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
        return (
            obs,
            reward,
            term,
            False,
            {"step": reward_dict["step"], "meta": reward_dict},
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

        Parameters:
            seed (Optional[int]): The seed value for environment reset.
            options (Optional[dict]): Additional options for the reset process.

        Returns:
            tuple: A tuple containing:
                - obs: The initial observation after the environment reset.
                - info (dict): A dictionary with auxiliary information.
        """
        super().reset(seed=seed, options=options)
        # Reset last_action
        self.last_action = None
        # Reset internal simulator state
        self.simulator.reset_state()
        # Reset the environment's state and return the initial observation
        obs = self.state.reset()
        # if recording is enabled, save the recording and reset the state list
        if self.recording_enabled:
            self.save_recording()

        # info is necessary for the gym environment, but useless at the moment
        info = {"info": "test"}
        return obs, info

    def _prepare_visualizable_state(self):
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
                "Recreate via make_robot_env(..., debug=True) to enable visualization and frame capture."
            )

        state = self._prepare_visualizable_state()

        # Execute rendering of the state through the simulation UI
        self.sim_ui.render(state)

    def record(self):
        """
        Records the current state as visualizable state and stores it in the list.
        """
        state = self._prepare_visualizable_state()
        self.recorded_states.append(state)

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
