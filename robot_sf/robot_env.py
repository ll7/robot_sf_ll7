from typing import Tuple

import numpy as np
from gym import Env, spaces

from robot_sf.map import BinaryOccupancyGrid
from robot_sf.robot import DifferentialDriveRobot, LidarScannerSettings, MovementVec2D, RobotPose, RobotSettings, Vec2D
from robot_sf.extenders_py_sf.extender_sim import ExtdSimulator


def initialize_robot(
        robot_map: BinaryOccupancyGrid,
        visualization_angle_portion: float,
        lidar_range: int,
        lidar_n_rays: int,
        spawn_pos: RobotPose,
        robot_collision_radius,
        wheel_max_linear_speed: float,
        wheel_max_angular_speed: float):

    # initialize robot with map
    robot_settings = RobotSettings(
        wheel_max_linear_speed, wheel_max_angular_speed, robot_collision_radius)
    lidar_settings = LidarScannerSettings(lidar_range, visualization_angle_portion, lidar_n_rays)
    robot = DifferentialDriveRobot(spawn_pos, robot_settings, lidar_settings, robot_map)
    return robot


def initialize_simulator(peds_sparsity, difficulty, dt, peds_speed_mult) -> ExtdSimulator:
    sim_env = ExtdSimulator(difficulty=difficulty)
    sim_env.set_ped_sparsity(peds_sparsity)
    sim_env.peds.step_width = dt
    sim_env.peds.max_speed_multiplier = peds_speed_mult
    return sim_env


def initialize_map(sim_env: ExtdSimulator) -> BinaryOccupancyGrid:
    # initialize map
    map_height = 2 * sim_env.box_size
    map_length = 2 * sim_env.box_size
    robot_map = BinaryOccupancyGrid(map_height, map_length, peds_sim_env=sim_env)
    robot_map.center_map_frame()
    robot_map.update_from_peds_sim(fixed_objects_map = True)
    robot_map.update_from_peds_sim()
    return robot_map


class RobotEnv(Env):
    """Representing a OpenAI Gym environment wrapper for
    training a robot with reinforcement leanring"""

    # TODO: transform this into cohesive data structures
    def __init__(self, lidar_n_rays: int=135,
                 collision_distance = 0.7, visualization_angle_portion = 0.5, lidar_range = 10,
                 v_linear_max = 1, v_angular_max = 1, rewards = None, max_v_x_delta = .5, 
                 initial_margin = .3, max_v_rot_delta = .5, dt = None, normalize_obs_state = True,
                 sim_length = 200, difficulty = 0, peds_speed_mult = 1.3):

        # TODO: get rid of most of these instance variables
        #       -> encapsulate statefulness inside a "state" object
        # self.scan_noise = scan_noise if scan_noise else [0.005, 0.002]

        self.robot = [] # TODO: init this properly
        self.target_coords = [] # TODO: init this properly
        self.lidar_range = lidar_range
        self.closest_obstacle = self.lidar_range

        self.sim_length = sim_length  # maximum simulation length (in seconds)
        self.env_type = 'RobotEnv'
        self.rewards = rewards if rewards else [1, 100, 40]
        self.normalize_obs_state = normalize_obs_state

        self.linear_max =  v_linear_max
        self.angular_max = v_angular_max

        # TODO: don't initialize the entire simulator just for retrieving some settings
        sim_env_test = ExtdSimulator()
        self.target_distance_max = np.sqrt(2) * (sim_env_test.box_size * 2)

        action_low  = np.array([-max_v_x_delta, -max_v_rot_delta])
        action_high = np.array([ max_v_x_delta,  max_v_rot_delta])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        state_max = np.concatenate((
                self.lidar_range * np.ones((lidar_n_rays,)),
                np.array([self.linear_max, self.angular_max, self.target_distance_max, np.pi])
            ), axis=0)
        state_min = np.concatenate((
                np.zeros((lidar_n_rays,)),
                np.array([0, -self.angular_max, 0, -np.pi])
            ), axis=0)
        self.observation_space = spaces.Box(low=state_min, high=state_max, dtype=np.float64)

        # aligning peds_sim_env and robot step_width
        self.dt = sim_env_test.peds.step_width if dt is None else dt
        self.map_boundaries_factory = lambda map: map.position_bounds(initial_margin)

        sparsity_levels = [500, 200, 100, 50, 20]
        self.simulator_factory = lambda: initialize_simulator(
            sparsity_levels[difficulty], difficulty, self.dt, peds_speed_mult)

        self.robot_factory = lambda robot_map, robot_pose: initialize_robot(
            robot_map,
            visualization_angle_portion,
            self.lidar_range,
            lidar_n_rays,
            robot_pose,
            collision_distance,
            self.linear_max,
            self.angular_max)

    def render(self, mode='human'):
        # TODO: visualize the game state with something like e.g. pygame
        pass

    def step(self, action_np: np.ndarray):
        action = MovementVec2D(action_np[0], action_np[1])

        # TODO: perform the robot's movement inside the robot class
        saturate_input = False
        coords_with_direction = self.robot.state.current_pose.coords_with_orient
        self.robot.map.peds_sim_env.move_robot(coords_with_direction)

        # initial distance
        dist_to_goal_before_action = self.robot.state.current_pose.target_rel_position(self.target_coords)[0]
        dot_x = self.robot.state.current_speed.dist + action.dist
        dot_orient = self.robot.state.current_speed.orient + action.orient

        # clip action into a valid range
        # TODO: this shouldn't be necessary, all actions from the action space are valid by definition!!!
        dot_x = np.clip(dot_x, 0, self.linear_max)
        saturate_input = dot_x < 0 or dot_x > self.linear_max
        dot_orient = np.clip(dot_orient, -self.angular_max, self.angular_max)
        saturate_input = saturate_input or abs(dot_orient) > self.angular_max

        self.robot.map.peds_sim_env.step(1)
        # TODO: the map shouldn't know about the simulator at all
        #       -> just pass in the pedestrians' new positions, that's it
        self.robot.map.update_from_peds_sim()
        self.robot.update_robot_speed(dot_x, dot_orient)
        self.robot.compute_odometry(self.dt)
        ranges, rob_state = self._get_obs()

        # new distance
        dist_to_goal_after_action = self.robot.state.current_pose.target_rel_position(self.target_coords)[0]
        self.rotation_counter += np.abs(dot_orient * self.dt)

        reward, done = self._reward(dist_to_goal_before_action, dist_to_goal_after_action, dot_x, ranges, saturate_input)
        return (ranges, rob_state), reward, done, None

    def _reward(self, dist_0, dist_1, dot_x, ranges, saturate_input) -> Tuple[float, bool]:
        # if pedestrian / obstacle is hit or time expired
        if self.robot.check_pedestrians_collision(.8) or \
                self.robot.check_obstacle_collision(self.robot.config.rob_collision_radius) or \
                self.robot.check_out_of_bounds(margin = 0.01) or self.duration > self.sim_length:
            final_distance_bonus = np.clip((self.distance_init - dist_1) / self.target_distance_max , -1, 1)
            reward = -self.rewards[1] * (1 - final_distance_bonus)
            done = True

        # if target is reached
        elif self.robot.check_target_reached(self.target_coords, tolerance=1):
            cum_rotations = (self.rotation_counter/(2*np.pi))
            rotations_penalty = self.rewards[2] * cum_rotations / (1e-5+self.duration)

            # reward is proportional to distance covered / speed in getting to target
            reward = np.maximum(self.rewards[1] / 2, self.rewards[1] - rotations_penalty)
            done = True

        else:
            self.duration += self.dt
            reward = self.rewards[0] * ((dist_0 - dist_1) / (self.linear_max * self.dt) \
                - int(saturate_input) + (1 - min(ranges)) * (dot_x / self.linear_max) * int(dist_0 > dist_1))
            done = False

        return reward, done

    def reset(self):
        self.duration = 0
        self.rotation_counter = 0

        # TODO: don't initialize the entire simulator for each episode
        #       -> initialize the simulator only once in constructor scope, then reset it
        sim_env = self.simulator_factory()

        # TODO: generate a couple of maps on environment startup and pick randomly from them
        robot_map = initialize_map(sim_env)

        self.target_coords, robot_pose = \
            self._pick_robot_spawn_and_target_pos(robot_map)
        self.robot = self.robot_factory(robot_map, robot_pose)

        # initialize Scan to get dimension of state (depends on ray cast)
        dist_to_goal, _ = robot_pose.target_rel_position(self.target_coords)
        self.distance_init = dist_to_goal
        return self._get_obs()

    def _pick_robot_spawn_and_target_pos(self, map: BinaryOccupancyGrid) -> Tuple[np.ndarray, RobotPose]:
        low_bound, high_bound = self.map_boundaries_factory(map)
        count = 0
        min_distance = (high_bound[0] - low_bound[0]) / 20 # TODO: why divide by 20?????
        while True:
            target_coords = np.random.uniform(
                low=np.array(low_bound)[:2], high=np.array(high_bound)[:2], size=2)
            # ensure that the target is not occupied by obstacles
            dists = np.sqrt(np.sum((map.obstacles_coordinates - target_coords)**2, axis=1))
            if np.amin(dists) > min_distance:
                break
            count +=1
            # TODO: rather exhaustively check if the map is ok on environment creation
            if count >= 100:
                raise ValueError('suitable initial coordinates not found')

        check_out_of_bounds = lambda coords: not map.check_if_valid_world_coordinates(coords, margin=0.2).any()
        robot_coords = np.random.uniform(low=low_bound, high=high_bound, size=3)
        robot_pose = RobotPose(Vec2D(robot_coords[0], robot_coords[1]), robot_coords[2])

        # if initial condition is too close (1.5m) to obstacle,
        # pedestrians or target, generate new initial condition
        while map.check_collision(robot_pose.pos, 1.5) or check_out_of_bounds(robot_pose.coords) or \
                robot_pose.target_rel_position(target_coords)[0] < (high_bound[0] - low_bound[0]) / 2:
            robot_coords = np.random.uniform(low=low_bound, high=high_bound, size=3)
            robot_pose = RobotPose(Vec2D(robot_coords[0], robot_coords[1]), robot_coords[2])

        return target_coords, robot_pose

    def _get_obs(self):
        ranges = self.robot.scanner.scan_structure['data']['ranges']
        ranges_np = np.nan_to_num(np.array(ranges), nan=self.lidar_range)
        speed_x = self.robot.state.current_speed.dist
        speed_rot = self.robot.state.current_speed.orient

        target_distance, target_angle = self.robot.state.current_pose.target_rel_position(self.target_coords)
        self.closest_obstacle = np.amin(ranges_np)

        if self.normalize_obs_state:
            ranges_np /= self.lidar_range
            speed_x /= self.linear_max
            speed_rot = speed_rot / self.angular_max
            target_distance /= self.target_distance_max
            target_angle = target_angle / np.pi

        return ranges_np, np.array([speed_x, speed_rot, target_distance, target_angle])
