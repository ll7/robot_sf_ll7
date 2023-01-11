from math import ceil
from dataclasses import dataclass
from typing import Tuple, Union
from copy import deepcopy

import numpy as np
from gym import Env, spaces

from robot_sf.sim_config import MapDefinitionPool
from robot_sf.occupancy import ContinuousOccupancy
from robot_sf.range_sensor import ContinuousLidarScanner, LidarScannerSettings
from robot_sf.sim_view import SimulationView, VisualizableAction, VisualizableSimState
from robot_sf.robot import DifferentialDriveRobot, RobotSettings, rel_pos
from robot_sf.ped_robot_force import PedRobotForceConfig
from robot_sf.simulator import Simulator


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]


@dataclass
class SimulationSettings:
    sim_length: int=200
    norm_obs: bool=True
    difficulty: int=2
    d_t: float=0.4
    peds_speed_mult: float=1.3
    prf_config: PedRobotForceConfig = PedRobotForceConfig()


@dataclass
class EnvSettings:
    sim_config: SimulationSettings = SimulationSettings()
    lidar_config: LidarScannerSettings = LidarScannerSettings()
    robot_config: RobotSettings = RobotSettings()
    map_pool: MapDefinitionPool = MapDefinitionPool()


class RobotEnv(Env):
    """Representing an OpenAI Gym environment wrapper for
    training a robot with reinforcement leanring"""

    def __init__(self, env_config: EnvSettings = EnvSettings(), debug: bool=False):
        self.sim_config = env_config.sim_config
        self.lidar_config = env_config.lidar_config
        self.robot_config = env_config.robot_config

        map_def = env_config.map_pool.choose_random_map()
        box_size = map_def.box_size

        self.env_type = 'RobotEnv'
        self.max_sim_steps = ceil(self.sim_config.sim_length / self.sim_config.d_t)
        self.max_target_dist = np.sqrt(2) * (box_size * 2) # the box diagonal
        self.action_space, self.observation_space = \
            RobotEnv._build_gym_spaces(self.max_target_dist, self.robot_config, self.lidar_config)

        self.sim_env: Simulator = None
        self.occupancy = ContinuousOccupancy(
            box_size,
            lambda: self.sim_env.robot_pose[0],
            lambda: self.sim_env.goal_pos,
            lambda: self.sim_env.pysf_sim.env.obstacles_raw,
            lambda: self.sim_env.current_positions,
            self.robot_config.radius)

        self.lidar_sensor = ContinuousLidarScanner(self.lidar_config, self.occupancy)
        robot_factory = lambda s, g: DifferentialDriveRobot(s, g, self.robot_config)
        self.sim_env = Simulator(box_size, self.sim_config, map_def, robot_factory)

        self.episode = 0
        self.timestep = 0
        self.last_action: Union[PolarVec2D, None] = None
        if debug:
            self.sim_ui = SimulationView(box_size * 2, box_size * 2)
        # TODO: provide a callback that shuts the simulator down on cancellation by user via UI

    def render(self, mode='human'):
        if not self.sim_ui:
            raise RuntimeError('Debug mode is not activated! Consider setting debug=True!')

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

        norm_ranges, rob_state = self._get_obs()
        obs = np.concatenate((norm_ranges, rob_state), axis=0)
        return obs

    def _reward(self) -> Tuple[float, bool]:
        step_discount = 0.1 / self.max_sim_steps
        reward, is_terminal = -step_discount, False
        if self.occupancy.is_robot_collision:
            reward -= 1
            is_terminal = True
        if self.occupancy.is_robot_at_goal:
            reward += 1
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
            target_distance /= self.max_target_dist
            target_angle = target_angle / np.pi

        return ranges_np, np.array([speed_x, speed_rot, target_distance, target_angle])

    @staticmethod
    def _build_gym_spaces(
            max_target_dist: float, robot_config: RobotSettings, \
            lidar_config: LidarScannerSettings) -> Tuple[spaces.Box, spaces.Box]:
        action_low  = np.array([-robot_config.max_linear_speed, -robot_config.max_angular_speed])
        action_high = np.array([ robot_config.max_linear_speed,  robot_config.max_angular_speed])
        action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        state_max = np.concatenate((
                lidar_config.max_scan_dist * np.ones((lidar_config.scan_length,)),
                np.array([robot_config.max_linear_speed, robot_config.max_angular_speed,
                            max_target_dist, np.pi])), axis=0)
        state_min = np.concatenate((
                np.zeros((lidar_config.scan_length,)),
                np.array([0, -robot_config.max_angular_speed, 0, -np.pi])
            ), axis=0)
        observation_space = spaces.Box(low=state_min, high=state_max, dtype=np.float64)
        return action_space, observation_space
