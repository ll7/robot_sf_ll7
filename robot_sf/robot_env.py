from dataclasses import dataclass, field
from typing import Tuple, Union, Callable
from copy import deepcopy

import numpy as np
from gym import Env, spaces

from robot_sf.sim_config import EnvSettings
from robot_sf.occupancy import ContinuousOccupancy
from robot_sf.range_sensor import ContinuousLidarScanner
from robot_sf.sim_view import SimulationView, VisualizableAction, VisualizableSimState
from robot_sf.robot import DifferentialDriveRobot, rel_pos, angle
from robot_sf.simulator import Simulator


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]


@dataclass
class EnvState:
    is_pedestrian_collision: bool
    is_obstacle_collision: bool
    is_robot_at_goal: bool
    is_timesteps_exceeded: bool

    @property
    def is_terminal(self) -> bool:
        return self.is_timesteps_exceeded or self.is_pedestrian_collision or \
            self.is_obstacle_collision or self.is_robot_at_goal


@dataclass
class SimpleReward:
    max_sim_steps: float
    step_discount: float = field(init=False)

    def __post_init__(self):
        # TODO: think of removing the step discount, gamma already discounts rewards
        self.step_discount = 0.1 / self.max_sim_steps

    def __call__(self, state: EnvState) -> float:
        reward = -self.step_discount
        if state.is_pedestrian_collision or state.is_obstacle_collision:
            reward -= 2
        if state.is_robot_at_goal:
            reward += 1
        return reward


def build_action_space(max_linear_speed: float, max_angular_speed) -> spaces.Box:
    drive_actuator = np.array([max_linear_speed, max_angular_speed], dtype=np.float64)
    return spaces.Box(low=drive_actuator * -1.0, high=drive_actuator, dtype=np.float64)


def build_norm_observation_space(num_rays: int) -> spaces.Box:
    range_sensor = np.ones((num_rays), dtype=np.float64)
    drive_state = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    state_max = np.concatenate((range_sensor, drive_state), axis=0)
    state_min = np.concatenate((range_sensor * 0.0, drive_state * -1.0), axis=0)
    return spaces.Box(low=state_min, high=state_max, dtype=np.float64)


class RobotEnv(Env):
    """Representing an OpenAI Gym environment wrapper for
    training a robot with reinforcement leanring"""

    def __init__(
            self, env_config: EnvSettings = EnvSettings(),
            reward_func: Union[Callable[[EnvState], float], None] = None,
            debug: bool=False):
        self.sim_config = env_config.sim_config
        self.lidar_config = env_config.lidar_config
        self.robot_config = env_config.robot_config

        map_def = env_config.map_pool.choose_random_map()
        width, height = map_def.width, map_def.height

        self.env_type = 'RobotEnv'
        self.max_sim_steps = self.sim_config.max_sim_steps
        self.max_target_dist = np.sqrt(2) * (max(width, height) * 2) # the box diagonal
        self.action_space = build_action_space(
            self.robot_config.max_linear_speed, self.robot_config.max_angular_speed)
        self.observation_space = build_norm_observation_space(self.lidar_config.num_rays)
        self.reward_func = reward_func if reward_func else SimpleReward(self.max_sim_steps)

        self.sim_env: Simulator
        self.occupancy = ContinuousOccupancy(
            width, height,
            lambda: self.sim_env.robot_pose[0],
            lambda: self.sim_env.goal_pos,
            lambda: self.sim_env.pysf_sim.env.obstacles_raw,
            lambda: self.sim_env.ped_positions,
            self.robot_config.radius,
            self.sim_config.ped_radius,
            self.sim_config.goal_radius)

        self.lidar_sensor = ContinuousLidarScanner(self.lidar_config, self.occupancy)
        robot_factory = lambda s: DifferentialDriveRobot(s, self.robot_config)
        goal_detection = lambda: self.occupancy.is_robot_at_goal
        self.sim_env = Simulator(self.sim_config, map_def, robot_factory, goal_detection)

        self.episode = 0
        self.timestep = 0
        self.last_action: Union[PolarVec2D, None] = None
        if debug:
            self.sim_ui = SimulationView(
                robot_radius=self.robot_config.radius,
                ped_radius=self.sim_config.ped_radius,
                goal_radius=self.sim_config.goal_radius)
            self.sim_ui.show()

    def step(self, action: np.ndarray):
        action_parsed = (action[0], action[1])
        self.sim_env.step_once(action_parsed)
        self.last_action = action_parsed
        obs = self._get_obs()
        state = EnvState(
            self.occupancy.is_pedestrian_collision,
            self.occupancy.is_obstacle_collision,
            self.occupancy.is_robot_at_goal,
            self.timestep > self.max_sim_steps)
        reward, done = self.reward_func(state), state.is_terminal
        self.timestep += 1
        return obs, reward, done, { 'step': self.episode, 'meta': state }

    def reset(self):
        self.episode += 1
        self.timestep = 0
        self.last_action = None
        self.sim_env.reset_state()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        ranges_np = self.lidar_sensor.get_scan(self.sim_env.robot_pose)
        speed_x, speed_rot = self.sim_env.robot.current_speed
        target_distance, target_angle = rel_pos(self.sim_env.robot.pose, self.sim_env.goal_pos)
        next_target_angle = 0.0 if not self.sim_env.next_goal_pos else \
            angle(self.sim_env.robot.pos, self.sim_env.goal_pos, self.sim_env.next_goal_pos)

        # normalize observations within [0, 1] or [-1, 1]
        ranges_np /= self.lidar_config.max_scan_dist
        speed_x /= self.robot_config.max_linear_speed
        speed_rot /= self.robot_config.max_angular_speed
        target_distance /= self.max_target_dist
        target_angle /= np.pi
        next_target_angle /= np.pi

        state = np.array([speed_x, speed_rot, target_distance, target_angle, next_target_angle])
        return np.concatenate((ranges_np, state), axis=0)

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

    def exit(self):
        if self.sim_ui:
            self.sim_ui.exit()
