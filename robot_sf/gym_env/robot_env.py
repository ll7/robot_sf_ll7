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

import os
import datetime
from typing import Tuple, Callable, List
from copy import deepcopy
import pickle

import loguru
import numpy as np

from gymnasium import Env
from gymnasium.utils import seeding

from robot_sf.robot.robot_state import RobotState
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.sensor.range_sensor import lidar_ray_scan

from robot_sf.render.sim_view import (
    SimulationView,
    VisualizableAction,
    VisualizableSimState,
)
from robot_sf.sim.simulator import init_simulators
from robot_sf.gym_env.reward import simple_reward
from robot_sf.gym_env.env_util import init_collision_and_sensors, init_spaces

logger = loguru.logger

Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]


class RobotEnv(Env):
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
        video_fps: int = 10,
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

        # Environment configuration details
        self.env_config = env_config

        # Extract first map definition; currently only supports using the first map
        self.map_def = env_config.map_pool.choose_random_map()

        # Initialize spaces based on the environment configuration and map
        self.action_space, self.observation_space, orig_obs_space = init_spaces(
            env_config, self.map_def
        )

        # Assign the reward function and debug flag
        self.reward_func = reward_func
        self.debug = debug

        # Initialize the list to store recorded states
        self.recorded_states: List[VisualizableSimState] = []
        self.recording_enabled = recording_enabled

        # Initialize simulator with a random start position
        self.simulator = init_simulators(
            env_config, self.map_def, random_start_pos=True
        )[0]

        # Delta time per simulation step and maximum episode time
        d_t = env_config.sim_config.time_per_step_in_secs
        max_ep_time = env_config.sim_config.sim_time_in_secs

        # Initialize collision detectors and sensor data processors
        occupancies, sensors = init_collision_and_sensors(
            self.simulator, env_config, orig_obs_space
        )

        # Setup initial state of the robot
        self.state = RobotState(
            self.simulator.robot_navs[0], occupancies[0], sensors[0], d_t, max_ep_time
        )

        # Store last action executed by the robot
        self.last_action = None

        # If in debug mode or video recording is enabled, create simulation view
        if debug or record_video:
            self.sim_ui = SimulationView(
                scaling=10,
                map_def=self.map_def,
                obstacles=self.map_def.obstacles,
                robot_radius=env_config.robot_config.radius,
                ped_radius=env_config.sim_config.ped_radius,
                goal_radius=env_config.sim_config.goal_radius,
                record_video=record_video,
                video_path=video_path,
                video_fps=video_fps,
            )

            # Only show window if in debug mode
            if debug:
                self.sim_ui.show()

    def step(self, action):
        """
        Execute one time step within the environment.

        Parameters:
        - action: Action to be executed.

        Returns:
        - obs: Observation after taking the action.
        - reward: Calculated reward for the taken action.
        - term: Boolean indicating if the episode has terminated.
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
        Reset the environment state to start a new episode.

        Returns:
        - obs: The initial observation after resetting the environment.
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
        ray_vecs = zip(np.cos(directions) * distances, np.sin(directions) * distances)
        ray_vecs_np = np.array(
            [
                [[robot_pos[0], robot_pos[1]], [robot_pos[0] + x, robot_pos[1] + y]]
                for x, y in ray_vecs
            ]
        )

        # Prepare pedestrian action visualization
        ped_actions = zip(
            self.simulator.pysf_sim.peds.pos(),
            self.simulator.pysf_sim.peds.pos() + self.simulator.pysf_sim.peds.vel() * 2,
        )
        ped_actions_np = np.array([[pos, vel] for pos, vel in ped_actions])

        # Package the state for visualization
        state = VisualizableSimState(
            self.state.timestep,
            action,
            self.simulator.robot_poses[0],
            deepcopy(self.simulator.ped_pos),
            ray_vecs_np,
            ped_actions_np,
        )

        return state

    def render(self):
        """
        Render the environment visually if in debug mode.

        Raises RuntimeError if debug mode is not enabled.
        """
        if not self.sim_ui:
            raise RuntimeError(
                "Debug mode is not activated! Consider setting " "debug=True!"
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

    def save_recording(self, filename: str = None):
        """
        save the recorded states to a file
        filname: str, must end with *.pkl
        resets the recorded states list at the end
        """
        if filename is None:
            now = datetime.datetime.now()
            # get current working directory
            cwd = os.getcwd()
            filename = f'{cwd}/recordings/{now.strftime("%Y-%m-%d_%H-%M-%S")}.pkl'

        # only save if there are recorded states
        if len(self.recorded_states) == 0:
            logger.warning("No states recorded, skipping save")
            # TODO: First env.reset will always have no recorded states
            return

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "wb") as f:  # write binary
            pickle.dump((self.recorded_states, self.map_def), f)
            logger.info(f"Recording saved to {filename}")
            logger.info("Reset state list")
            self.recorded_states = []

    def exit(self):
        """
        Clean up and exit the simulation UI, if it exists.
        """
        if self.sim_ui:
            self.sim_ui.exit_simulation()
