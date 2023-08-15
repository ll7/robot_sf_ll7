from math import ceil
from typing import Tuple, Callable, List, Protocol, Any
from dataclasses import dataclass, field
from copy import deepcopy
from multiprocessing.pool import ThreadPool

import numpy as np
from gym.vector import VectorEnv
from gym import Env, spaces
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.navigation import RouteNavigator

from robot_sf.sim_config import EnvSettings
from robot_sf.nav.occupancy import ContinuousOccupancy
from robot_sf.sensor.range_sensor import lidar_ray_scan, lidar_sensor_space
from robot_sf.sensor.goal_sensor import target_sensor_obs, target_sensor_space
from robot_sf.sensor.sensor_fusion import fused_sensor_space, SensorFusion, OBS_RAYS, OBS_DRIVE_STATE
from robot_sf.sim.sim_view import SimulationView, VisualizableAction, VisualizableSimState
from robot_sf.sim.simulator import Simulator


Vec2D = Tuple[float, float]
PolarVec2D = Tuple[float, float]
RobotPose = Tuple[Vec2D, float]


class Robot(Protocol):
    @property
    def observation_space(self) -> spaces.Box:
        raise NotImplementedError()

    @property
    def action_space(self) -> spaces.Box:
        raise NotImplementedError()

    @property
    def pos(self) -> Vec2D:
        raise NotImplementedError()

    @property
    def pose(self) -> RobotPose:
        raise NotImplementedError()

    @property
    def current_speed(self) -> PolarVec2D:
        raise NotImplementedError()

    def apply_action(self, action: Any, d_t: float):
        raise NotImplementedError()

    def reset_state(self, new_pose: RobotPose):
        raise NotImplementedError()

    def parse_action(self, action: Any) -> Any:
        raise NotImplementedError()


@dataclass
class RobotState:
    nav: RouteNavigator
    occupancy: ContinuousOccupancy
    sensors: SensorFusion
    d_t: float
    sim_time_limit: float
    episode: int = field(init=False, default=0)
    is_at_goal: bool = field(init=False, default=False)
    is_collision_with_ped: bool = field(init=False, default=False)
    is_collision_with_obst: bool = field(init=False, default=False)
    is_collision_with_robot: bool = field(init=False, default=False)
    is_timeout: bool = field(init=False, default=False)
    sim_time_elapsed: float = field(init=False, default=0.0)
    timestep: int = field(init=False, default=0)

    @property
    def max_sim_steps(self) -> int:
        return int(ceil(self.sim_time_limit / self.d_t))

    @property
    def is_terminal(self) -> bool:
        return self.is_at_goal or self.is_timeout or self.is_collision_with_robot \
            or self.is_collision_with_ped or self.is_collision_with_obst

    @property
    def is_waypoint_complete(self) -> bool:
        return self.nav.reached_waypoint

    @property
    def is_route_complete(self) -> bool:
        return self.nav.reached_destination

    def reset(self):
        self.episode += 1
        self.timestep = 0
        self.sim_time_elapsed = 0.0
        self.is_collision_with_ped = False
        self.is_collision_with_obst = False
        self.is_collision_with_robot = False
        self.is_at_goal = False
        self.is_timeout = False
        self.sensors.reset_cache()
        return self.sensors.next_obs()

    def step(self):
        # TODO: add check for robot-robot collisions as well
        self.timestep += 1
        self.sim_time_elapsed += self.d_t
        self.is_collision_with_ped = self.occupancy.is_pedestrian_collision
        self.is_collision_with_obst = self.occupancy.is_obstacle_collision
        self.is_collision_with_robot = self.occupancy.is_robot_robot_collision
        self.is_at_goal = self.occupancy.is_robot_at_goal
        self.is_timeout = self.sim_time_elapsed > self.sim_time_limit
        return self.sensors.next_obs()

    def meta_dict(self) -> dict:
        return {
            "step": self.episode * self.max_sim_steps,
            "episode": self.episode,
            "step_of_episode": self.timestep,
            "is_pedestrian_collision": self.is_collision_with_ped,
            "is_robot_collision": self.is_collision_with_robot,
            "is_obstacle_collision": self.is_collision_with_obst,
            "is_robot_at_goal": self.is_waypoint_complete,
            "is_route_complete": self.is_route_complete,
            "is_timesteps_exceeded": self.is_timeout,
            "max_sim_steps": self.max_sim_steps
        }


def simple_reward(
        meta: dict,
        max_episode_step_discount: float=-0.1,
        ped_coll_penalty: float=-5,
        obst_coll_penalty: float=-2,
        reach_waypoint_reward: float=1) -> float:
    reward = max_episode_step_discount / meta["max_sim_steps"]
    if meta["is_pedestrian_collision"] or meta["is_robot_collision"]:
        reward += ped_coll_penalty
    if meta["is_obstacle_collision"]:
        reward += obst_coll_penalty
    if meta["is_robot_at_goal"]:
        reward += reach_waypoint_reward
    return reward


def init_simulators(env_config: EnvSettings, map_def: MapDefinition, num_robots: int = 1):
    num_sims = ceil(num_robots / map_def.num_start_pos)
    goal_proximity = env_config.robot_config.radius + env_config.sim_config.goal_radius
    sims: List[Simulator] = []

    for i in range(num_sims):
        n = map_def.num_start_pos if i < num_sims - 1 else num_robots % map_def.num_start_pos
        sim_robots = [env_config.robot_factory() for _ in range(n)]
        sims.append(Simulator(env_config.sim_config, map_def, sim_robots, goal_proximity))

    return sims


def init_collision_and_sensors(
        sim: Simulator, env_config: EnvSettings, orig_obs_space: spaces.Dict):
    num_robots = len(sim.robots)
    sim_config = env_config.sim_config
    robot_config = env_config.robot_config
    lidar_config = env_config.lidar_config

    occupancies = [ContinuousOccupancy(
            sim.map_def.width, sim.map_def.height,
            lambda: sim.robot_pos[i], lambda: sim.goal_pos[i],
            lambda: sim.pysf_sim.env.obstacles_raw[:, :4], lambda: sim.ped_pos,
            robot_config.radius, sim_config.ped_radius, sim_config.goal_radius)
        for i in range(num_robots)]

    sensor_fusions: List[SensorFusion] = []
    for r_id in range(num_robots):
        ray_sensor = lambda r_id=r_id: lidar_ray_scan(
            sim.robots[r_id].pose, occupancies[r_id], lidar_config)
        target_sensor = lambda r_id=r_id: target_sensor_obs(
            sim.robots[r_id].pose, sim.goal_pos[r_id], sim.next_goal_pos[r_id])
        speed_sensor = lambda r_id=r_id: sim.robots[r_id].current_speed
        sensor_fusions.append(SensorFusion(
            ray_sensor, speed_sensor, target_sensor,
            orig_obs_space, sim_config.use_next_goal))

    return occupancies, sensor_fusions


def init_spaces(env_config: EnvSettings, map_def: MapDefinition):
    robot = env_config.robot_factory()
    action_space = robot.action_space
    observation_space, orig_obs_space = fused_sensor_space(
        env_config.sim_config.stack_steps, robot.observation_space,
        target_sensor_space(map_def.max_target_dist),
        lidar_sensor_space(env_config.lidar_config.num_rays,
                           env_config.lidar_config.max_scan_dist))
    return action_space, observation_space, orig_obs_space


class RobotEnv(Env):
    """Representing an OpenAI Gym environment for training
    a self-driving robot with reinforcement learning"""

    def __init__(
            self, env_config: EnvSettings = EnvSettings(),
            reward_func: Callable[[dict], float] = simple_reward,
            debug: bool = False):

        map_def = env_config.map_pool.map_defs[0] # info: only use first map
        self.action_space, self.observation_space, orig_obs_space = init_spaces(env_config, map_def)

        self.reward_func, self.debug = reward_func, debug
        self.simulator = init_simulators(env_config, map_def)[0]
        d_t = env_config.sim_config.time_per_step_in_secs
        max_ep_time = env_config.sim_config.sim_time_in_secs

        occupancies, sensors = init_collision_and_sensors(self.simulator, env_config, orig_obs_space)
        self.state = RobotState(self.simulator.robot_navs[0], occupancies[0], sensors[0], d_t, max_ep_time)

        self.last_action = None
        if debug:
            self.sim_ui = SimulationView(
                scaling=6,
                obstacles=map_def.obstacles,
                robot_radius=env_config.robot_config.radius,
                ped_radius=env_config.sim_config.ped_radius,
                goal_radius=env_config.sim_config.goal_radius)
            self.sim_ui.show()

    def step(self, action):
        action = self.simulator.robots[0].parse_action(action)
        self.last_action = action
        self.simulator.step_once([action])
        obs = self.state.step()
        meta = self.state.meta_dict()
        term = self.state.is_terminal
        reward = self.reward_func(meta)
        return obs, reward, term, { "step": meta["step"], "meta": meta }

    def reset(self):
        self.simulator.reset_state()
        obs = self.state.reset()
        return obs

    def render(self):
        if not self.sim_ui:
            raise RuntimeError('Debug mode is not activated! Consider setting debug=True!')

        action = None if not self.last_action else VisualizableAction(
            self.simulator.robot_poses[0], self.last_action, self.simulator.goal_pos[0])

        state = VisualizableSimState(
            self.state.timestep, action, self.simulator.robot_poses[0],
            deepcopy(self.simulator.ped_pos))

        self.sim_ui.render(state)

    def exit(self):
        if self.sim_ui:
            self.sim_ui.exit()


class MultiRobotEnv(VectorEnv):
    """Representing an OpenAI Gym environment for training
    multiple self-driving robots with reinforcement learning"""

    def __init__(
            self, env_config: EnvSettings = EnvSettings(),
            reward_func: Callable[[dict], float] = simple_reward,
            debug: bool = False, num_robots: int = 1):

        map_def = env_config.map_pool.map_defs[0] # info: only use first map
        action_space, observation_space, orig_obs_space = init_spaces(env_config, map_def)
        super(MultiRobotEnv, self).__init__(num_robots, observation_space, action_space)
        self.action_space = spaces.Box(
            low=np.array([self.single_action_space.low for _ in range(num_robots)]),
            high=np.array([self.single_action_space.high for _ in range(num_robots)]),
            dtype=self.single_action_space.low.dtype)

        self.reward_func, self.debug = reward_func, debug
        self.simulators = init_simulators(env_config, map_def, num_robots)
        self.states: List[RobotState] = []
        d_t = env_config.sim_config.time_per_step_in_secs
        max_ep_time = env_config.sim_config.sim_time_in_secs

        for sim in self.simulators:
            occupancies, sensors = init_collision_and_sensors(sim, env_config, orig_obs_space)
            states = [RobotState(nav, occ, sen, d_t, max_ep_time)
                      for nav, occ, sen in zip(sim.robot_navs, occupancies, sensors)]
            self.states.extend(states)

        self.sim_worker_pool = ThreadPool(len(self.simulators))
        self.obs_worker_pool = ThreadPool(num_robots)

    def step(self, actions):
        actions = [self.simulators[0].robots[0].parse_action(a) for a in actions]
        i = 0
        actions_per_simulator = []
        for sim in self.simulators:
            num_robots = len(sim.robots)
            actions_per_simulator.append(actions[i:i+num_robots])
            i += num_robots

        self.sim_worker_pool.map(
            lambda s_a: s_a[0].step_once(s_a[1]),
            zip(self.simulators, actions_per_simulator))

        obs = self.obs_worker_pool.map(lambda s: s.step(), self.states)

        metas = [state.meta_dict() for state in self.states]
        masked_metas = [{ "step": meta["step"], "meta": meta } for meta in metas]
        masked_metas = (*masked_metas,)
        terms = [state.is_terminal for state in self.states]
        rewards = [self.reward_func(meta) for meta in metas]

        for i, (sim, state, term) in enumerate(zip(self.simulators, self.states, terms)):
            if term:
                sim.reset_state()
                obs[i] = state.reset()

        obs = { OBS_DRIVE_STATE: np.array([o[OBS_DRIVE_STATE] for o in obs]),
                OBS_RAYS: np.array([o[OBS_RAYS] for o in obs])}

        return obs, rewards, terms, masked_metas

    def reset(self):
        self.sim_worker_pool.map(lambda sim: sim.reset_state(), self.simulators)
        obs = self.obs_worker_pool.map(lambda s: s.reset(), self.states)

        obs = { OBS_DRIVE_STATE: np.array([o[OBS_DRIVE_STATE] for o in obs]),
                OBS_RAYS: np.array([o[OBS_RAYS] for o in obs]) }
        return obs

    def render(self, robot_id: int=0):
        # TODO: add support for PyGame rendering
        pass

    def close_extras(self, **kwargs):
        self.sim_worker_pool.close()
        self.obs_worker_pool.close()
