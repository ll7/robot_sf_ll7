"""
`pedestrian_env.py` is a module that defines the simulation environment for a pedestrian.
It includes classes and protocols for defining the pedetsrians's state, actions, and
observations within the environment.

`PedestrianEnv`: A class that represents the pedestrian's environment. It inherits from `Env`
from the `gymnasium` library, which is a base class for environments that operate over
actions and observations. It includes methods for stepping through the environment,
resetting it, rendering it, and closing it.
It also defines the action and observation spaces for the pedestrian.
"""

import datetime
import os
import pickle
from copy import deepcopy
from typing import Callable, List

import loguru
import numpy as np
from gymnasium import Env

from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.gym_env.env_util import init_ped_collision_and_sensors, init_ped_spaces
from robot_sf.gym_env.reward import simple_ped_reward
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


class PedestrianEnv(Env):
    """
    Representing a Gymnasium environment for training a adversial pedestrian
    with reinforcement learning.
    """

    def __init__(
        self,
        env_config: PedEnvSettings = None,
        reward_func: Callable[[dict], float] = simple_ped_reward,
        robot_model=None,
        debug: bool = False,
        recording_enabled: bool = False,
    ):
        """
        Initialize the pedestrian Environment.

        Parameters:
        - env_config (PedEnvSettings): Configuration for environment settings.
        - reward_func (Callable[[dict], float]): Reward function that takes
            a dictionary as input and returns a float as reward.
        - debug (bool): If True, enables debugging information such as
            visualizations.
        - recording_enabled (bool): If True, enables recording of the simulation
        """
        if env_config is None:
            env_config = PedEnvSettings()

        # Environment configuration details
        self.env_config = env_config

        # Extract first map definition; currently only supports using the first map
        self.map_def = env_config.map_pool.choose_random_map()

        # Initialize spaces based on the environment configuration and map
        combined_action_space, combined_observation_space, orig_obs_space = init_ped_spaces(
            env_config, self.map_def
        )

        # Assign the action and observation spaces
        self.action_space = combined_action_space[1]
        self.observation_space = combined_observation_space[1]

        # Assign the reward function and debug flag
        self.reward_func = reward_func
        self.debug = debug

        # Initialize the list to store recorded states
        self.recorded_states: List[VisualizableSimState] = []
        self.recording_enabled = recording_enabled

        # Initialize simulator with a random start position
        self.simulator = init_ped_simulators(env_config, self.map_def, random_start_pos=True)[0]

        # Initialize collision detectors and sensor data processors
        occupancies, sensors = init_ped_collision_and_sensors(
            self.simulator, env_config, orig_obs_space
        )

        # Setup initial state of the robot
        self.robot_state = RobotState(
            nav=self.simulator.robot_navs[0],
            occupancy=occupancies[0],
            sensors=sensors[0],
            d_t=env_config.sim_config.time_per_step_in_secs,
            sim_time_limit=env_config.sim_config.sim_time_in_secs,
        )

        # Setup initial state of the pedestrian
        self.ped_state = PedestrianState(
            robot_occupancy=occupancies[0],  # robot occupancy
            ego_ped_occupancy=occupancies[1],  # ego_ped occupancy
            sensors=sensors[1],
            d_t=env_config.sim_config.time_per_step_in_secs,
            sim_time_limit=env_config.sim_config.sim_time_in_secs,
        )

        # Assign the robot model
        if robot_model is None:
            raise ValueError("Please provide a valid robot_model during initialization.")
        self.robot_model = robot_model

        # Store last state executed by the robot
        self.last_action_robot = None
        self.last_obs_robot = None

        # Store last action executed by the pedestrian
        self.last_action_ped = None

        # If in debug mode, create a simulation view to visualize the state
        if debug:
            self.sim_ui = SimulationView(
                scaling=10,
                map_def=self.map_def,
                obstacles=self.map_def.obstacles,
                robot_radius=env_config.robot_config.radius,
                ego_ped_radius=env_config.ego_ped_config.radius,
                ped_radius=env_config.sim_config.ped_radius,
                goal_radius=env_config.sim_config.goal_radius,
            )

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
        action_ped = action

        # Process the action through the simulator
        action_robot, _ = self.robot_model.predict(self.last_obs_robot, deterministic=True)
        action_robot = self.simulator.robots[0].parse_action(action_robot)
        # action_robot = (0.0, 0.0) #TODO: remove noop after testing
        self.last_action_robot = action_robot

        action_ped = self.simulator.ego_ped.parse_action(action_ped)
        self.last_action_ped = action_ped

        # Perform simulation step
        self.simulator.step_once([action_robot], [action_ped])

        # Get updated observation
        self.last_obs_robot = self.robot_state.step()
        obs_ped = self.ped_state.step()

        # Fetch metadata about the current state
        meta = self.ped_state.meta_dict()

        # Determine if the episode has reached terminal state
        term = self.ped_state.is_terminal

        # Compute the reward using the provided reward function
        reward = self.reward_func(meta)

        # if recording is enabled, record the state
        if self.recording_enabled:
            self.record()
        truncated = False
        return obs_ped, reward, term, truncated, {"step": meta["step"], "meta": meta}

    def reset(self, seed=None, options=None):
        """
        Reset the environment state to start a new episode.

        Returns:
        - obs: The initial observation after resetting the environment.
        - info: Empty dictionary.
        """
        super().reset(seed=seed, options=options)
        # Reset last actions
        self.last_action_robot = None
        self.last_action_ped = None
        # Reset internal simulator state
        self.simulator.reset_state()
        # Reset the environment's state and return the initial observation
        self.last_obs_robot = self.robot_state.reset()
        obs_ped = self.ped_state.reset()

        # if recording is enabled, save the recording and reset the state list
        if self.recording_enabled:
            self.save_recording()
        # TODO: find fix for vec_env needing tuple
        return obs_ped, {"info": "test"}

    def _prepare_visualizable_state(self):
        # Prepare action visualization, if any action was executed
        robot_action = (
            None
            if not self.last_action_robot
            else VisualizableAction(
                self.simulator.robot_poses[0],
                self.last_action_robot,
                self.simulator.goal_pos[0],
            )
        )

        # Robot position and LIDAR scanning visualization preparation
        robot_pos = self.simulator.robot_poses[0][0]
        distances, directions = lidar_ray_scan(
            self.simulator.robot_poses[0],
            self.robot_state.occupancy,
            self.env_config.lidar_config,
        )

        # Construct ray vectors for visualization
        robot_ray_vecs = render_lidar(robot_pos, distances, directions)

        # Prepare npc_pedestrian action visualization
        ped_actions = zip(
            self.simulator.pysf_sim.peds.pos(),
            self.simulator.pysf_sim.peds.pos() + self.simulator.pysf_sim.peds.vel() * 2,
        )
        ped_actions_np = np.array([[pos, vel] for pos, vel in ped_actions])
        # Prepare action and LIDAR visualization for the ego pedestrian
        ego_ped_action = (
            None
            if not self.last_action_ped
            else VisualizableAction(
                self.simulator.ego_ped_pose,
                self.last_action_ped,
                self.simulator.ego_ped_goal_pos,
            )
        )

        ego_ped_pos = self.simulator.ego_ped_pos
        distances, directions = lidar_ray_scan(
            self.simulator.ego_ped_pose,
            self.ped_state.ego_ped_occupancy,
            self.env_config.lidar_config,
        )
        ego_ped_ray_vecs = render_lidar(ego_ped_pos, distances, directions)

        # Package the state for visualization
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
            time_per_step_in_secs=self.env_config.sim_config.time_per_step_in_secs,
        )

        return state

    def render(self):
        """
        Render the environment visually if in debug mode.

        Raises RuntimeError if debug mode is not enabled.
        """
        if not self.sim_ui:
            raise RuntimeError("Debug mode is not activated! Consider setting debug=True!")

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
            filename = f"recordings/{now.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"

        # only save if there are recorded states
        if len(self.recorded_states) == 0:
            logger.warning("No states recorded, skipping save")
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
