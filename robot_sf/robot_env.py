from __future__ import annotations
from typing import Tuple, Union, Callable
from copy import deepcopy

import numpy as np
from gym import Env

from robot_sf.sim_config import EnvSettings
from robot_sf.nav.occupancy import ContinuousOccupancy
from robot_sf.sensor.range_sensor import lidar_ray_scan, lidar_sensor_space
from robot_sf.sensor.goal_sensor import target_sensor_obs, target_sensor_space
from robot_sf.sensor.sensor_fusion import fused_sensor_space, SensorFusion, OBS_RAYS, OBS_DRIVE_STATE
from robot_sf.sim.sim_view import SimulationView, VisualizableAction, VisualizableSimState
from robot_sf.sim.simulator import Simulator
from robot_sf.ped_npc.ped_robot_force import PedRobotForce
from robot_sf.robot.differential_drive import DifferentialDriveAction
from robot_sf.robot.bicycle_drive import BicycleAction


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]


def simple_reward(meta: dict) -> float:
    step_discount = 0.01 / meta["max_sim_steps"]
    reward = -step_discount
    if meta["is_pedestrian_collision"] or meta["is_obstacle_collision"]:
        reward -= 2
    if meta["is_robot_at_goal"]:
        reward += 1
    return reward


def disturbance_reward(meta: dict) -> float:
    step_discount = 0.1 / meta["max_sim_steps"]
    reward = -step_discount

    forces = meta["ped_robot_forces"]
    intensities = (np.clip(np.sqrt(np.sum(forces**2, axis=1)), 0.1, 1.1) - 0.1) / 1.0
    total_disturbance = np.clip(np.sum(intensities), 0.0, 1.0) / meta["max_sim_steps"]
    reward -= total_disturbance * 10

    if meta["is_pedestrian_collision"] or meta["is_obstacle_collision"]:
        reward -= 3
    if meta["is_robot_at_goal"]:
        reward += 1
    return reward


def is_terminal(meta: dict) -> bool:
    return meta["is_timesteps_exceeded"] or meta["is_pedestrian_collision"] or \
        meta["is_obstacle_collision"] or meta["is_robot_at_goal"]


def collect_metadata(env) -> dict:
    # TODO: add RobotEnv type hint
    return {
        "step": env.episode * env.max_sim_steps,
        "episode": env.episode,
        "step_of_episode": env.timestep,
        "is_pedestrian_collision": env.occupancy.is_pedestrian_collision,
        "is_obstacle_collision": env.occupancy.is_obstacle_collision,
        "is_robot_at_goal": env.sim_env.robot_nav.reached_waypoint,
        "is_route_complete": env.sim_env.robot_nav.reached_destination,
        "is_timesteps_exceeded": env.timestep > env.max_sim_steps,
        "max_sim_steps": env.max_sim_steps,
        "ped_robot_forces": [f for f in env.sim_env.pysf_sim.forces
                             if isinstance(f, PedRobotForce)][0].last_forces
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
        robot = env_config.robot_factory()
        goal_proximity = robot_config.radius + sim_config.goal_radius
        self.sim_env = Simulator(sim_config, map_def, robot, goal_proximity)

        self.occupancy = ContinuousOccupancy(
            map_def.width, map_def.height, lambda: robot.pos, lambda: self.sim_env.goal_pos,
            lambda: self.sim_env.pysf_sim.env.obstacles_raw[:, :4], lambda: self.sim_env.ped_positions,
            robot_config.radius, sim_config.ped_radius, sim_config.goal_radius)

        self.action_space = robot.action_space
        self.observation_space, orig_obs_space = fused_sensor_space(
            sim_config.stack_steps, robot.observation_space,
            target_sensor_space(map_def.max_target_dist),
            lidar_sensor_space(lidar_config.num_rays, lidar_config.max_scan_dist))

        ray_sensor = lambda: lidar_ray_scan(robot.pose, self.occupancy, lidar_config)
        target_sensor = lambda: target_sensor_obs(
            robot.pose, self.sim_env.goal_pos, self.sim_env.next_goal_pos)
        self.sensor_fusion = SensorFusion(
            ray_sensor, lambda: robot.current_speed, target_sensor, orig_obs_space)

        self.episode = 0
        self.timestep = 0
        self.last_action: Union[DifferentialDriveAction, BicycleAction, None] = None
        if debug:
            self.sim_ui = SimulationView(
                scaling=10,
                obstacles=map_def.obstacles,
                robot_radius=robot_config.radius,
                ped_radius=sim_config.ped_radius,
                goal_radius=sim_config.goal_radius)
            self.sim_ui.show()

    def step(self, action: np.ndarray):
        action_parsed = self.sim_env.robot.parse_action(action)
        self.sim_env.step_once(action_parsed)
        self.last_action = action_parsed
        obs = self.sensor_fusion.next_obs()

        meta = self.metadata_collector(self)
        masked_meta = { "step": meta["step"], "meta": meta } # info: SB3 crashes otherwise
        self.timestep += 1
        return obs, self.reward_func(meta), self.term_func(meta), masked_meta

    def reset(self):
        self.episode += 1
        self.timestep = 0
        self.last_action = None
        self.sim_env.reset_state()
        self.sensor_fusion.reset_cache()
        return self.sensor_fusion.next_obs()

    def render(self, mode='human'):
        if not self.sim_ui:
            raise RuntimeError('Debug mode is not activated! Consider setting debug=True!')

        action = None if not self.last_action else VisualizableAction(
            self.sim_env.robot.pose, self.last_action, self.sim_env.goal_pos)

        state = VisualizableSimState(
            self.timestep, action, self.sim_env.robot.pose,
            deepcopy(self.sim_env.ped_positions))

        self.sim_ui.render(state)

    def exit(self):
        if self.sim_ui:
            self.sim_ui.exit()
