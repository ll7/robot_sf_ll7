import os
from math import ceil
from dataclasses import dataclass
from typing import Tuple, Union
from copy import deepcopy

import numpy as np
from gym import Env, spaces

from robot_sf.simulation_config import load_config
from robot_sf.occupancy import ContinuousOccupancy
from robot_sf.range_sensor import ContinuousLidarScanner, LidarScannerSettings
from robot_sf.sim_view import SimulationView, VisualizableAction, VisualizableSimState
from robot_sf.robot import DifferentialDriveRobot, RobotSettings, rel_pos
from robot_sf.simulator import Simulator


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]


@dataclass
class SimulationSettings:
    sim_length: int=200
    d_t: float=0.4
    peds_speed_mult: float=1.3
    lidar_n_rays: int=272
    norm_obs: bool=True


class RobotEnv(Env):
    """Representing an OpenAI Gym environment wrapper for
    training a robot with reinforcement leanring"""

    def __init__(self, debug: bool=False, difficulty: int=0,
                 sim_config: SimulationSettings = SimulationSettings(),
                 lidar_config: LidarScannerSettings = LidarScannerSettings(),
                 robot_config: RobotSettings = RobotSettings()):

        self.sim_config = sim_config
        self.lidar_config = lidar_config
        self.robot_config = robot_config

        self.max_sim_steps = ceil(sim_config.sim_length / sim_config.d_t)
        self.env_type = 'RobotEnv'

        path_to_config = os.path.join(os.path.dirname(__file__), "config", "map_config.toml")
        box_size, force_config, obstacles = load_config(path_to_config)

        self.sim_env: Simulator = None
        self.occupancy = ContinuousOccupancy(
            box_size,
            lambda: self.sim_env.robot_pose[0],
            lambda: self.sim_env.goal_pos,
            lambda: self.sim_env.pysf_sim.env.obstacles_raw,
            lambda: self.sim_env.current_positions,
            robot_config.radius)

        self.lidar_sensor = ContinuousLidarScanner(lidar_config, self.occupancy)
        robot_factory = lambda s, g: DifferentialDriveRobot(s, g, self.robot_config)

        self.sim_env = Simulator(
            box_size, force_config, obstacles, robot_factory,
            difficulty, sim_config.peds_speed_mult, sim_config.d_t)
        self.target_distance_max = np.sqrt(2) * (self.sim_env.box_size * 2)
        # info: max distance is length of box diagonal

        action_low  = np.array([-robot_config.max_linear_speed, -robot_config.max_angular_speed])
        action_high = np.array([ robot_config.max_linear_speed,  robot_config.max_angular_speed])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        state_max = np.concatenate((
                lidar_config.max_scan_dist * np.ones((lidar_config.scan_length,)),
                np.array([robot_config.max_linear_speed, robot_config.max_angular_speed,
                          self.target_distance_max, np.pi])), axis=0)
        state_min = np.concatenate((
                np.zeros((lidar_config.scan_length,)),
                np.array([0, -robot_config.max_angular_speed, 0, -np.pi])
            ), axis=0)
        self.observation_space = spaces.Box(low=state_min, high=state_max, dtype=np.float64)

        self.episode = 0
        self.timestep = 0
        self.last_action: Union[PolarVec2D, None] = None
        if debug:
            self.sim_ui = SimulationView(self.sim_env.box_size * 2, self.sim_env.box_size * 2)
        # TODO: provide a callback that shuts the simulator down on cancellation by user via UI

    def render(self, mode='human'):
        action = None if not self.last_action else \
            VisualizableAction(self.sim_env.robot.pose, self.last_action, self.sim_env.goal_pos)

        state = VisualizableSimState(
            self.timestep,
            action,
            self.sim_env.robot.pose,
            deepcopy(self.occupancy.pedestrian_coords),
            deepcopy(self.occupancy.obstacle_coords))

        self.sim_ui.render(state)

    def step(self, action: np.ndarray):
        action_parsed = (action[0], action[1])
        self.sim_env.step_once(action_parsed)
        self.last_action = action_parsed

        norm_ranges, rob_state = self._get_obs()
        obs = np.concatenate((norm_ranges, rob_state), axis=0)

        reward, done = self._reward()
        self.timestep += 1
        return obs, reward, done, { 'step': self.episode }

    def reset(self):
        self.episode += 1
        self.timestep = 0
        self.last_action = None

        self.sim_env.reset_state()

        self.distance_init = self.sim_env.dist_to_goal
        norm_ranges, rob_state = self._get_obs()
        obs = np.concatenate((norm_ranges, rob_state), axis=0)
        return obs

    def _reward(self) -> Tuple[float, bool]:
        step_discount = 1.0 / self.max_sim_steps
        reward, is_terminal = -step_discount, False
        if self.occupancy.is_robot_collision:
            reward -= 1
            is_terminal = True
        if self.occupancy.is_robot_at_goal:
            reward += 2
            is_terminal = True
        if self.timestep >= self.max_sim_steps:
            is_terminal = True
        return reward, is_terminal

    def _get_obs(self):
        ranges_np = self.lidar_sensor.get_scan(self.sim_env.robot_pose)
        speed_x, speed_rot = self.sim_env.robot.current_speed
        target_distance, target_angle = rel_pos(self.sim_env.robot.pose, self.sim_env.goal_pos)

        if self.sim_config.norm_obs:
            ranges_np /= self.lidar_config.max_scan_dist
            speed_x /= self.robot_config.max_linear_speed
            speed_rot = speed_rot / self.robot_config.max_angular_speed
            target_distance /= self.target_distance_max
            target_angle = target_angle / np.pi

        return ranges_np, np.array([speed_x, speed_rot, target_distance, target_angle])
