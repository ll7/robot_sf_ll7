from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Union, Callable
from copy import deepcopy

import numpy as np
from gym import Env, spaces

from robot_sf.sim_config import EnvSettings
from robot_sf.occupancy import ContinuousOccupancy
from robot_sf.range_sensor import lidar_ray_scan
from robot_sf.sim_view import SimulationView, VisualizableAction, VisualizableSimState
from robot_sf.robot import DifferentialDriveRobot, rel_pos, angle
from robot_sf.simulator import Simulator


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]


def simple_reward(meta: dict) -> float:
    step_discount = 0.1 / meta["max_sim_steps"]
    reward = -step_discount
    if meta["is_pedestrian_collision"] or meta["is_obstacle_collision"]:
        reward -= 2
    if meta["is_robot_at_goal"]:
        reward += 1
    return reward


def is_terminal(meta: dict) -> bool:
    return meta["is_timesteps_exceeded"] or meta["is_pedestrian_collision"] or \
        meta["is_obstacle_collision"] or meta["is_robot_at_goal"]


def build_action_space(max_linear_speed: float, max_angular_speed) -> spaces.Box:
    high = np.array([max_linear_speed, max_angular_speed], dtype=np.float32)
    low = np.array([0.0, -max_angular_speed], dtype=np.float32)
    return spaces.Box(low=low, high=high)


def build_norm_observation_space(
        num_rays: int, max_scan_dist: float, max_linear_speed: float,
        max_angular_speed: float, max_target_dist: float) -> Tuple[spaces.Box, np.ndarray]:
    max_drive_state = np.array([max_linear_speed, max_angular_speed, max_target_dist, np.pi, np.pi])
    min_drive_state = np.array([0.0, -max_angular_speed, 0.0, -np.pi, -np.pi])
    max_lidar_state = np.full((num_rays), max_scan_dist)
    min_lidar_state = np.zeros((num_rays))
    max_obs = np.concatenate((max_lidar_state, max_drive_state), axis=0, dtype=np.float32)
    min_obs = np.concatenate((min_lidar_state, min_drive_state), axis=0, dtype=np.float32)
    low, high = min_obs / max_obs, max_obs / max_obs
    norm_obs_space = spaces.Box(low=low, high=high)
    return norm_obs_space, max_obs


def target_sensor_obs(
        robot_pose: RobotPose,
        goal_pos: Vec2D,
        next_goal_pos: Union[Vec2D, None]) -> Tuple[float, float, float]:
    robot_pos, _ = robot_pose
    target_distance, target_angle = rel_pos(robot_pose, goal_pos)
    next_target_angle = 0.0 if not next_goal_pos else angle(robot_pos, goal_pos, next_goal_pos)
    return target_distance, target_angle, next_target_angle


@dataclass
class SensorFusion:
    lidar_sensor: Callable[[], np.ndarray]
    robot_speed_sensor: Callable[[], PolarVec2D]
    target_sensor: Callable[[], Tuple[float, float, float]]
    max_values: np.ndarray = field(default_factory=lambda: np.ones(()))

    def next_obs(self) -> np.ndarray:
        lidar_state = self.lidar_sensor()
        speed_x, speed_rot = self.robot_speed_sensor()
        target_distance, target_angle, next_target_angle = self.target_sensor()
        drive_state = np.array([speed_x, speed_rot, target_distance, target_angle, next_target_angle])
        return np.concatenate((lidar_state, drive_state), axis=0) / self.max_values


def collect_metadata(env) -> dict:
    # TODO: add RobotEnv type hint
    return {
        "step": env.episode * env.max_sim_steps,
        "episode": env.episode,
        "step_of_episode": env.timestep,
        "is_pedestrian_collision": env.occupancy.is_pedestrian_collision,
        "is_obstacle_collision": env.occupancy.is_obstacle_collision,
        "is_robot_at_goal": env.sim_env.navigator.reached_waypoint,
        "is_timesteps_exceeded": env.timestep > env.max_sim_steps,
        "max_sim_steps": env.max_sim_steps
    }


class RobotEnv(Env):
    """Representing an OpenAI Gym environment for training
    a self-driving robot with reinforcement learning"""

    def __init__(
            self, env_config: EnvSettings = EnvSettings(),
            metadata_collector: Callable[[RobotEnv], dict] = collect_metadata,
            reward_func: Callable[[dict], float] = simple_reward,
            term_func: Callable[[dict], bool] = is_terminal,
            debug: bool = False):
        self.reward_func = reward_func
        self.term_func = term_func
        self.metadata_collector = metadata_collector
        sim_config = env_config.sim_config
        lidar_config = env_config.lidar_config
        robot_config = env_config.robot_config
        map_def = env_config.map_pool.choose_random_map()

        self.env_type = 'RobotEnv'
        self.max_sim_steps = sim_config.max_sim_steps
        self.action_space = build_action_space(
            robot_config.max_linear_speed, robot_config.max_angular_speed)
        self.observation_space, obs_norm = build_norm_observation_space(
            lidar_config.num_rays, lidar_config.max_scan_dist, robot_config.max_linear_speed,
            robot_config.max_angular_speed, map_def.max_target_dist)

        robot = DifferentialDriveRobot(robot_config)
        goal_proximity = robot_config.radius + sim_config.goal_radius
        self.sim_env = Simulator(sim_config, map_def, robot, goal_proximity)

        self.occupancy = ContinuousOccupancy(
            map_def.width, map_def.height, lambda: robot.pos, lambda: self.sim_env.goal_pos,
            lambda: self.sim_env.pysf_sim.env.obstacles_raw, lambda: self.sim_env.ped_positions,
            robot_config.radius, sim_config.ped_radius, sim_config.goal_radius)

        ray_sensor = lambda: lidar_ray_scan(robot.pose, self.occupancy, lidar_config)
        target_sensor = lambda: target_sensor_obs(
            robot.pose, self.sim_env.goal_pos, self.sim_env.next_goal_pos)
        self.sensor_fusion = SensorFusion(
            ray_sensor, lambda: robot.current_speed, target_sensor, obs_norm)

        self.episode = 0
        self.timestep = 0
        self.last_action: Union[PolarVec2D, None] = None
        if debug:
            self.sim_ui = SimulationView(
                robot_radius=robot_config.radius,
                ped_radius=sim_config.ped_radius,
                goal_radius=sim_config.goal_radius)
            self.sim_ui.show()

    def step(self, action: np.ndarray):
        action_parsed = (action[0], action[1])
        self.sim_env.step_once(action_parsed)
        self.last_action = action_parsed
        obs = self.sensor_fusion.next_obs()

        meta = self.metadata_collector(self)
        self.timestep += 1
        return obs, self.reward_func(meta), self.term_func(meta), { "step": meta["step"], "meta": meta }

    def reset(self):
        self.episode += 1
        self.timestep = 0
        self.last_action = None
        self.sim_env.reset_state()
        return self.sensor_fusion.next_obs()

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
