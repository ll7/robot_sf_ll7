import numpy as np
from gym import Env, spaces

from robot_sf.map import BinaryOccupancyGrid
from robot_sf.robot import DifferentialDriveRobot, LidarScannerSettings, RobotPose, RobotSettings, Vec2D
from robot_sf.extenders_py_sf.extender_sim import ExtdSimulator


def initialize_robot(
        visualization_angle_portion: float,
        lidar_range: int,
        lidar_n_rays,
        init_state,
        robot_collision_radius,
        peds_sparsity,
        diff,
        wheel_max_linear_speed: float,
        wheel_max_angular_speed: float):

    # TODO: don't initialize the environment here ...
    #       -> create separate functions to init env / map
    sim_env = ExtdSimulator(difficulty=diff)
    sim_env.set_ped_sparsity(peds_sparsity)

    # initialize map
    map_height = 2 * sim_env.box_size
    map_length = 2 * sim_env.box_size
    robot_map = BinaryOccupancyGrid(map_height, map_length, peds_sim_env=sim_env)
    robot_map.center_map_frame()
    robot_map.update_from_peds_sim(fixedObjectsMap = True)
    robot_map.update_from_peds_sim()

    # initialize robot with map
    robot_settings = RobotSettings(
        wheel_max_linear_speed, wheel_max_angular_speed, robot_collision_radius)
    lidar_settings = LidarScannerSettings(lidar_range, visualization_angle_portion, lidar_n_rays)
    spawn_pos = RobotPose(Vec2D(init_state[0], init_state[1]), init_state[2])
    robot = DifferentialDriveRobot(spawn_pos, robot_settings, lidar_settings, robot_map)
    return robot


class RobotEnv(Env):

    # TODO: transform this into cohesive data structures
    def __init__(self, lidar_n_rays: int=135,
                 collision_distance = 0.7, visualization_angle_portion = 0.5, lidar_range = 10,
                 v_linear_max = 1, v_angular_max = 1, rewards = None, max_v_x_delta = .5, 
                 initial_margin = .3, max_v_rot_delta = .5, dt = None, normalize_obs_state = True,
                 sim_length = 200, difficulty = 0, scan_noise = None, n_chunk_sections = 18, peds_speed_mult = 1.3):

        self.n_chunk_sections = n_chunk_sections
        sparsity_levels = [500, 200, 100, 50, 20]

        self.peds_speed_mult = peds_speed_mult
        self.peds_sparsity = sparsity_levels[difficulty]
        self._difficulty = difficulty

        self.scan_noise = scan_noise if scan_noise else [0.005, 0.002]
        self.data_render = []

        self.robot = []
        self.lidar_n_rays = lidar_n_rays
        self.collision_distance = collision_distance
        self.target_coordinates = []
        self.visualization_angle_portion = visualization_angle_portion
        self.lidar_range = lidar_range

        self.closest_obstacle = self.lidar_range
        self.sim_length = sim_length  # maximum simulation length (in seconds)
        self.env_type = 'RobotEnv'
        self.rewards = rewards if rewards else [1, 100, 40]
        self.normalize_obs_state = normalize_obs_state

        self.max_v_x_delta = max_v_x_delta
        self.max_v_rot_delta = max_v_rot_delta

        self.linear_max =  v_linear_max
        self.angular_max = v_angular_max

        self.target_distance_max = []
        self.action_space = []
        self.observation_space = []

        self.dt = dt
        self.initial_margin = initial_margin

        sim_env_test = ExtdSimulator()
        self.target_distance_max = np.sqrt(2) * (sim_env_test.box_size * 2)

        self.action_low  = np.array([-self.max_v_x_delta, -self.max_v_rot_delta])
        self.action_high = np.array([ self.max_v_x_delta,  self.max_v_rot_delta])
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float64)

        state_max = np.concatenate((
                self.lidar_range * np.ones((self.lidar_n_rays,)),
                np.array([self.linear_max, self.angular_max, self.target_distance_max, np.pi])
            ), axis=0)
        state_min = np.concatenate((
                np.zeros((self.lidar_n_rays,)),
                np.array([0, -self.angular_max, 0, -np.pi])
            ), axis=0)
        self.observation_space = spaces.Box(low=state_min, high=state_max, dtype=np.float64)
        self.n_observations = self.observation_space.shape[0]

        # aligning peds_sim_env and robot step_width
        self.dt = sim_env_test.peds.step_width if self.dt is None else self.dt

    def get_max_iterations(self):
        return int(round(self.sim_length / self.dt))

    # def get_actions_structure(self):
    #     pass

    def get_observations_structure(self):
        return [self.lidar_n_rays, (self.n_observations - self.lidar_n_rays)]

    # def get_controller_input(self):
    #     pass

    def step(self, action):
        self._step(action)

    def render(self, mode = 'human', iter_params = (0, 0, 0)):
        pass # nothing to do here ...

    def reset(self):
        self._reset()

    def _step(self, action):
        info = {}
        saturate_input = False
        self.robot.map.peds_sim_env.addRobot(self.robot.get_current_pose())

        # initial distance
        dist_0 = self.robot.target_rel_position(self.target_coordinates[:2])[0]
        dot_x = self.robot.current_speed.dist + action[0]
        dot_orient = self.robot.current_speed.orient + action[1]

        if dot_x < 0 or dot_x > self.linear_max:
            dot_x = np.clip(dot_x, 0, self.linear_max) 
            saturate_input = True

        if abs(dot_orient)> self.angular_max:
            saturate_input = True
            dot_orient = np.clip( dot_orient ,-self.angular_max , self.angular_max)

        if self.robot_state_history is None:
            self.robot_state_history = np.array([[dot_x, dot_orient]])
        else:
            self.robot_state_history = np.append(self.robot_state_history, np.array([[dot_x, dot_orient]]), axis = 0)
        
        self.robot.map.peds_sim_env.step(1)
        self.robot.map.update_from_peds_sim() 
        self.robot.update_robot_speed(dot_x, dot_orient)
        self.robot.compute_odometry(self.dt)
        ranges, rob_state = self._get_obs()

        # new distance
        dist_1 = self.robot.target_rel_position(self.target_coordinates[:2])[0]
        self.robot.get_scan(scan_noise = self.scan_noise)
        self.rotation_counter += np.abs(dot_orient * self.dt)

        # if pedestrian / obstacle is hit or time expired
        if self.robot.check_pedestrians_collision(.8) or self.robot.check_obstacle_collision() or \
            self.robot.check_out_of_bounds(margin = 0.01) or self.duration > self.sim_length :
            final_distance_bonus = np.clip((self.distance_init - dist_1) / self.target_distance_max , -1, 1)
            reward = -self.rewards[1] * (1 - final_distance_bonus)
            done = True

        # if target is reached
        elif self.robot.check_target_reach(self.target_coordinates[:2], tolerance=1):
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

        info['robot_map'] = self.robot.chunk(self.n_chunk_sections)
        return (ranges, rob_state), reward, done, info

    def _reset(self):
        self.duration = 0
        self.rotation_counter = 0
        self.robot_state_history = None

        self.robot = initialize_robot(
            self.visualization_angle_portion,
            self.lidar_range,
            self.lidar_n_rays,
            [0, 0, 0],
            self.collision_distance,
            self.peds_sparsity,
            self._difficulty,
            self.linear_max, self.angular_max)

        low_bound, high_bound = self._get_position_bounds()

        count = 0
        min_distance = (high_bound[0] - low_bound[0]) / 20
        while True:
            self.target_coordinates = np.random.uniform(low = np.array(low_bound), high = np.array(high_bound),size=3)
            if np.amin(np.sqrt(np.sum((self.robot.map.obstaclesCoordinates - self.target_coordinates[:2])**2, axis = 1))) > min_distance:
                break
            count +=1
            if count >= 100:
                raise ValueError('suitable initial coordinates not found')

        # if initial condition is too close (1.5m) to obstacle, pedestrians or target, generate new initial condition
        while self.robot.check_collision(1.5) or self.robot.check_out_of_bounds(margin = 0.2) or \
                self.robot.target_rel_position(self.target_coordinates[:2])[0] < (high_bound[0]-low_bound[0]) / 2 :

            init_coordinates = np.random.uniform(low = low_bound, high = high_bound,size=3)
            self.robot.set_robot_pose(init_coordinates[0], init_coordinates[1], init_coordinates[2])

        # initialize Scan to get dimension of state (depends on ray cast) 
        self.distance_init = self.robot.target_rel_position(self.target_coordinates[:2])[0]

        # if step time externally defined, align peds sim env
        if self.dt != self.robot.map.peds_sim_env.peds.step_width:
            self.robot.map.peds_sim_env.peds.step_width = self.dt

        if self.peds_speed_mult != self.robot.map.peds_sim_env.peds.max_speed_multiplier:
            self.robot.map.peds_sim_env.peds.max_speed_multiplier = self.peds_speed_mult

        return self._get_obs()

    def _get_position_bounds(self):
        # define bounds for initial and target position of robot (margin = margin from map limit)
        margin = self.initial_margin
        x_idx_min = round(margin * self.robot.map.grid_size['x'])
        x_idx_max = round((1 - margin) * self.robot.map.grid_size['x'])

        y_idx_min = round(margin * self.robot.map.grid_size['y'])
        y_idx_max = round((1 - margin) * self.robot.map.grid_size['y'])

        low_bound  = [self.robot.map.X[0, x_idx_min], self.robot.map.Y[y_idx_min, 0], -np.pi]
        high_bound = [self.robot.map.X[0, x_idx_max], self.robot.map.Y[y_idx_max, 0],  np.pi]
        return low_bound, high_bound

    def _get_obs(self):
        ranges = self.robot.scanner.scan_structure['data']['ranges']
        ranges_np = np.nan_to_num(np.array(ranges), nan=self.lidar_range)
        speed_x = self.robot.current_speed.dist
        speed_rot = self.robot.current_speed.orient

        target_coords = self.target_coordinates[:2]
        target_distance, target_angle = self.robot.target_rel_position(target_coords)
        self.closest_obstacle = np.amin(ranges_np)

        if self.normalize_obs_state:
            ranges_np /= self.lidar_range
            speed_x /= self.linear_max
            speed_rot = speed_rot/self.angular_max
            target_distance /= self.target_distance_max
            target_angle = target_angle / np.pi

        return ranges_np, np.array([speed_x, speed_rot, target_distance, target_angle])
