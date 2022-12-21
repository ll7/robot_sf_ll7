from math import dist
from typing import Tuple, List, Union
from copy import deepcopy

import numpy as np
from gym import Env, spaces

from robot_sf.map_continuous import ContinuousOccupancy
from robot_sf.range_sensor_continuous import ContinuousLidarScanner, LidarScannerSettings
from robot_sf.sim_view import SimulationView, VisualizableAction, VisualizableSimState
from robot_sf.vector import RobotPose, PolarVec2D
from robot_sf.robot import DifferentialDriveRobot, RobotSettings
from robot_sf.extenders_py_sf.extender_sim import ExtdSimulator


Vec2D = Tuple[float, float]


class RobotEnv(Env):
    """Representing an OpenAI Gym environment wrapper for
    training a robot with reinforcement leanring"""

    # TODO: transform this into cohesive data structures
    def __init__(self, lidar_n_rays: int=272, collision_distance: float=0.7,
                 visual_angle_portion: float=1.0, lidar_range: float=10.0,
                 v_linear_max: float=1, v_angular_max: float=1,
                 rewards: Union[List[float], None]=None,
                 max_v_x_delta: float=.5, max_v_rot_delta: float=.5, d_t: Union[float, None]=None,
                 normalize_obs_state: bool=True, sim_length: int=200, difficulty: int=0,
                 scan_noise: Union[List[float], None]=None,
                 peds_speed_mult: float=1.3, debug: bool=False):

        scan_noise = scan_noise if scan_noise else [0.005, 0.002]

        # info: this gets initialized by env.reset()
        self.robot: DifferentialDriveRobot = None
        self.target_coords: Vec2D = None

        self.lidar_range = lidar_range
        self.closest_obstacle = self.lidar_range

        self.sim_length = sim_length  # maximum simulation length (in seconds)
        self.env_type = 'RobotEnv'
        self.rewards = rewards if rewards else [1, 100, 40]
        self.normalize_obs_state = normalize_obs_state

        self.linear_max =  v_linear_max
        self.angular_max = v_angular_max

        sparsity_levels = [500, 200, 100, 50, 20]
        self.sim_env = ExtdSimulator(difficulty, sparsity_levels[difficulty], d_t, peds_speed_mult)
        self.target_distance_max = np.sqrt(2) * (self.sim_env.box_size * 2)
        self.d_t = self.sim_env.d_t

        self.robot_map = ContinuousOccupancy(
            self.sim_env.box_size,
            lambda: self.sim_env.pysf_sim.env.obstacles_raw,
            lambda: self.sim_env.current_positions)
        lidar_settings = LidarScannerSettings(
            lidar_range, visual_angle_portion, lidar_n_rays, scan_noise)
        lidar_sensor = ContinuousLidarScanner(lidar_settings, self.robot_map)

        robot_settings = RobotSettings(self.linear_max, self.angular_max, collision_distance)
        self.robot_factory = lambda robot_map, robot_pose: \
            DifferentialDriveRobot(robot_pose, lidar_sensor, robot_map, robot_settings)

        action_low  = np.array([-max_v_x_delta, -max_v_rot_delta])
        action_high = np.array([ max_v_x_delta,  max_v_rot_delta])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        state_max = np.concatenate((
                self.lidar_range * np.ones((lidar_sensor.settings.scan_length,)),
                np.array([self.linear_max, self.angular_max, self.target_distance_max, np.pi])
            ), axis=0)
        state_min = np.concatenate((
                np.zeros((lidar_sensor.settings.scan_length,)),
                np.array([0, -self.angular_max, 0, -np.pi])
            ), axis=0)
        self.observation_space = spaces.Box(low=state_min, high=state_max, dtype=np.float64)

        self.duration = 0
        self.rotation_counter = 0
        self.distance_init = float('inf')

        self.episode = 0
        self.timestep = 0
        self.last_action: PolarVec2D = None
        if debug:
            self.sim_ui = SimulationView(self.robot_map.box_size * 2, self.robot_map.box_size * 2)
        # TODO: provide a callback that shuts the simulator down on cancellation by user via UI

    def render(self, mode='human'):
        action = None if not self.last_action else \
            VisualizableAction(
                deepcopy(self.robot.pose),
                deepcopy(self.last_action),
                deepcopy(self.target_coords))

        state = VisualizableSimState(
            self.timestep,
            action,
            deepcopy(self.robot.pose),
            deepcopy(self.robot_map.pedestrian_coords),
            deepcopy(self.robot_map.obstacle_coords))

        self.sim_ui.render(state)

    def step(self, action: np.ndarray):
        coords_with_direction = self.robot.pose.coords_with_orient
        self.sim_env.move_robot(coords_with_direction)
        self.sim_env.step_once()
        self.robot_map.update_moving_objects()

        dist_before = dist(self.robot.pos, self.target_coords)
        action_parsed = PolarVec2D(action[0], action[1])
        movement, saturate_input = self.robot.apply_action(action_parsed, self.d_t)
        dot_x, dot_orient = movement.dist, movement.orient
        dist_after = dist(self.robot.pos, self.target_coords)

        # scan for collisions with LiDAR sensor, generate new observation
        ranges = self.robot.get_scan()
        norm_ranges, rob_state = self._get_obs(ranges)
        self.rotation_counter += np.abs(dot_orient * self.d_t)

        # determine the reward and whether the episode is done
        reward, done = self._reward(dist_before, dist_after, dot_x, norm_ranges, saturate_input)
        self.timestep += 1
        self.last_action = action_parsed
        return np.concatenate((norm_ranges, rob_state), axis=0), \
            reward, done, { 'step': self.episode }

    def _reward(self, dist_0, dist_1, dot_x, ranges, saturate_input) -> Tuple[float, bool]:
        # if pedestrian / obstacle is hit or time expired
        if self.robot.is_pedestrians_collision(.8) or \
                self.robot.is_obstacle_collision(self.robot.config.rob_collision_radius) or \
                self.robot.is_out_of_bounds or self.duration > self.sim_length:
            bonus_unclipped = (self.distance_init - dist_1) / self.target_distance_max
            final_distance_bonus = np.clip(bonus_unclipped, -1, 1)
            reward = -self.rewards[1] * (1 - final_distance_bonus)
            done = True

        # if target is reached
        elif self.robot.is_target_reached(self.target_coords, tolerance=1):
            cum_rotations = (self.rotation_counter / (2 * np.pi))
            rotations_penalty = self.rewards[2] * cum_rotations / (1e-5 + self.duration)

            # reward is proportional to distance covered / speed in getting to target
            reward = np.maximum(self.rewards[1] / 2, self.rewards[1] - rotations_penalty)
            done = True

        else:
            self.duration += self.d_t
            reward = self.rewards[0] * ((dist_0 - dist_1) / (self.linear_max * self.d_t) \
                - int(saturate_input) + (1 - min(ranges)) \
                    * (dot_x / self.linear_max) * int(dist_0 > dist_1))
            done = False

        return reward, done

    def reset(self):
        self.episode += 1
        self.duration = 0
        self.rotation_counter = 0
        self.timestep = 0
        self.last_action = None

        self.sim_env.reset_state()
        self.target_coords, robot_pose = \
            self._pick_robot_spawn_and_target_pos(self.robot_map)
        self.robot = self.robot_factory(self.robot_map, robot_pose)

        # initialize Scan to get dimension of state (depends on ray cast)
        dist_to_goal, _ = robot_pose.rel_pos(self.target_coords)
        self.distance_init = dist_to_goal
        ranges = self.robot.get_scan()
        norm_ranges, rob_state = self._get_obs(ranges)
        return np.concatenate((norm_ranges, rob_state), axis=0)

    def _pick_robot_spawn_and_target_pos(
            self, robot_map: ContinuousOccupancy) -> Tuple[Vec2D, RobotPose]:
        # TODO: don't spawn inside polygons -> move this logic into the map
        low_bound, high_bound = robot_map.position_bounds()
        count = 0
        min_distance = (high_bound[0] - low_bound[0]) / 20 # TODO: why divide by 20?????
        while True:
            t_x, t_y = np.random.uniform(
                low=np.array(low_bound)[:2], high=np.array(high_bound)[:2], size=2)
            target_coords = (t_x, t_y)
            # ensure that the target is not occupied by obstacles
            dists = np.linalg.norm(robot_map.obstacle_coords[:, :2] - target_coords)
            if np.amin(dists) > min_distance:
                break
            count +=1
            # TODO: rather exhaustively check if the map is ok on environment creation
            if count >= 100:
                raise ValueError('suitable initial coordinates not found')

        check_out_of_bounds = lambda coords: not robot_map.is_in_bounds(coords[0], coords[1])
        robot_coords = np.random.uniform(low=low_bound, high=high_bound, size=3)
        robot_pose = RobotPose((robot_coords[0], robot_coords[1]), robot_coords[2])

        # if initial coords are too close (1.5m) to an obstacle,
        # pedestrian or the target, generate new coords
        while robot_map.is_collision(robot_pose.pos, 1.5) or \
                check_out_of_bounds(robot_pose.coords) or \
                robot_pose.rel_pos(target_coords)[0] < (high_bound[0] - low_bound[0]) / 2:
            robot_coords = np.random.uniform(low=low_bound, high=high_bound, size=3)
            robot_pose = RobotPose((robot_coords[0], robot_coords[1]), robot_coords[2])

        return target_coords, robot_pose

    def _get_obs(self, ranges_np: np.ndarray):
        speed_x = self.robot.state.current_speed.dist
        speed_rot = self.robot.state.current_speed.orient

        target_distance, target_angle = self.robot.state.current_pose.rel_pos(self.target_coords)
        self.closest_obstacle = np.amin(ranges_np)

        if self.normalize_obs_state:
            ranges_np /= self.lidar_range
            speed_x /= self.linear_max
            speed_rot = speed_rot / self.angular_max
            target_distance /= self.target_distance_max
            target_angle = target_angle / np.pi

        return ranges_np, np.array([speed_x, speed_rot, target_distance, target_angle])
