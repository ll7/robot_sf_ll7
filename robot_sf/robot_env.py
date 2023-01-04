import os
from typing import Tuple, List, Union
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


class RobotEnv(Env):
    """Representing an OpenAI Gym environment wrapper for
    training a robot with reinforcement leanring"""

    # TODO: transform this into cohesive data structures
    def __init__(self, lidar_n_rays: int=272, collision_distance: float=0.7,
                 visual_angle_portion: float=1.0, lidar_range: float=10.0,
                 v_linear_max: float=1, v_angular_max: float=1,
                 rewards: Union[List[float], None]=None,
                 max_v_x_delta: float=.5, max_v_rot_delta: float=.5, d_t: Union[float, None]=None,
                 normalize_obs_state: bool=True, sim_length: int=200,
                 scan_noise: Union[List[float], None]=None, difficulty: int=0,
                 peds_speed_mult: float=1.3, debug: bool=False):

        scan_noise = scan_noise if scan_noise else [0.005, 0.002]

        self.lidar_range = lidar_range
        self.sim_length = sim_length  # maximum simulation length (in seconds)
        self.env_type = 'RobotEnv'
        self.rewards = rewards if rewards else [1, 100, 40]
        self.normalize_obs_state = normalize_obs_state

        self.linear_max =  v_linear_max
        self.angular_max = v_angular_max

        path_to_config = os.path.join(os.path.dirname(__file__), "config", "map_config.toml")
        box_size, config, obstacles = load_config(path_to_config)
        robot_radius = config.robot_radius

        self.sim_env: Simulator = None
        self.occupancy = ContinuousOccupancy(
            box_size,
            lambda: self.sim_env.robot_pose[0],
            lambda: self.sim_env.goal_pos,
            lambda: self.sim_env.pysf_sim.env.obstacles_raw,
            lambda: self.sim_env.current_positions,
            robot_radius)
        lidar_settings = LidarScannerSettings(
            lidar_range, visual_angle_portion, lidar_n_rays, scan_noise)
        self.lidar_sensor = ContinuousLidarScanner(lidar_settings, self.occupancy)

        robot_settings = RobotSettings(robot_radius, self.linear_max, self.angular_max, collision_distance)
        robot_factory = lambda spawn_pose, goal: \
            DifferentialDriveRobot(spawn_pose, goal, robot_settings)

        self.sim_env = Simulator(box_size, config, obstacles, robot_factory, peds_speed_mult, d_t)
        self.target_distance_max = np.sqrt(2) * (self.sim_env.box_size * 2) # length of box diagonal
        self.d_t = self.sim_env.d_t

        action_low  = np.array([-max_v_x_delta, -max_v_rot_delta])
        action_high = np.array([ max_v_x_delta,  max_v_rot_delta])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        state_max = np.concatenate((
                self.lidar_range * np.ones((lidar_settings.scan_length,)),
                np.array([self.linear_max, self.angular_max, self.target_distance_max, np.pi])
            ), axis=0)
        state_min = np.concatenate((
                np.zeros((lidar_settings.scan_length,)),
                np.array([0, -self.angular_max, 0, -np.pi])
            ), axis=0)
        self.observation_space = spaces.Box(low=state_min, high=state_max, dtype=np.float64)

        self.duration = 0.0
        self.rotation_counter = 0
        self.distance_init = float('inf')

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
        (dot_x, dot_orient), dist_before, dist_after, is_overdrive = \
            self.sim_env.step_once(action_parsed)

        norm_ranges, rob_state = self._get_obs()
        obs = np.concatenate((norm_ranges, rob_state), axis=0)
        self.rotation_counter += np.abs(dot_orient * self.d_t)

        # determine the reward and whether the episode is done
        reward, done = self._reward()
        # reward, done = self._reward(dist_before, dist_after, dot_x, norm_ranges, is_overdrive)
        self.duration += self.d_t
        self.timestep += 1
        self.last_action = action_parsed
        return obs, reward, done, { 'step': self.episode }

    def reset(self):
        self.episode += 1
        self.duration = 0.0
        self.rotation_counter = 0
        self.timestep = 0
        self.last_action = None

        self.sim_env.reset_state()

        self.distance_init = self.sim_env.dist_to_goal
        norm_ranges, rob_state = self._get_obs()
        obs = np.concatenate((norm_ranges, rob_state), axis=0)
        return obs

    def _is_end_of_episode_with_failure(self) -> bool:
        return self.occupancy.is_robot_collision or self.duration > self.sim_length

    def _is_end_of_episode_with_success(self) -> bool:
        return self.occupancy.is_robot_at_goal

    def _reward(self) -> Tuple[float, bool]:
        reward, is_terminal = -0.01, False
        if self.occupancy.is_robot_collision:
            reward -= 1
            is_terminal = True
        if self.occupancy.is_robot_at_goal:
            reward += 1.01
            is_terminal = True
        if self.duration > self.sim_length:
            is_terminal = True
        return reward, is_terminal

    def _get_obs(self):
        ranges_np = self.lidar_sensor.get_scan(self.sim_env.robot_pose)
        speed_x, speed_rot = self.sim_env.robot.current_speed
        target_distance, target_angle = rel_pos(self.sim_env.robot.pose, self.sim_env.goal_pos)

        if self.normalize_obs_state:
            ranges_np /= self.lidar_range
            speed_x /= self.linear_max
            speed_rot = speed_rot / self.angular_max
            target_distance /= self.target_distance_max
            target_angle = target_angle / np.pi

        return ranges_np, np.array([speed_x, speed_rot, target_distance, target_angle])
