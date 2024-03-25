"""
`MultiRobotEnv`: A class that extends `RobotEnv` to handle multiple robots in the environment.
It overrides the `step_async` method to apply actions to all robots in the environment.
"""

from typing import Callable, List

from multiprocessing.pool import ThreadPool

import numpy as np

from gymnasium.vector import VectorEnv
from gymnasium import spaces

from robot_sf.robot.robot_state import RobotState
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.sensor.sensor_fusion import OBS_RAYS, OBS_DRIVE_STATE
from robot_sf.sim.simulator import init_simulators
from robot_sf.gym_env.reward import simple_reward

from robot_sf.gym_env.env_util import init_collision_and_sensors, init_spaces


class MultiRobotEnv(VectorEnv):
    """Representing a Gymnasium environment for training
    multiple self-driving robots with reinforcement learning"""

    def __init__(
            self, env_config: EnvSettings = EnvSettings(),
            reward_func: Callable[[dict], float] = simple_reward,
            debug: bool = False, num_robots: int = 1):

        map_def = env_config.map_pool.map_defs["uni_campus_big"] # info: only use first map
        action_space, observation_space, orig_obs_space = init_spaces(
            env_config,
            map_def
            )
        super(MultiRobotEnv, self).__init__(
            num_robots,
            observation_space,
            action_space
            )
        self.action_space = spaces.Box(
            low=np.array(
                [self.single_action_space.low for _ in range(num_robots)]
                ),
            high=np.array(
                [self.single_action_space.high for _ in range(num_robots)]
                ),
            dtype=self.single_action_space.low.dtype)

        self.reward_func, self.debug = reward_func, debug
        self.simulators = init_simulators(
            env_config,
            map_def,
            num_robots,
            random_start_pos=False
            )
        self.states: List[RobotState] = []
        d_t = env_config.sim_config.time_per_step_in_secs
        max_ep_time = env_config.sim_config.sim_time_in_secs

        for sim in self.simulators:
            occupancies, sensors = init_collision_and_sensors(
                sim, env_config,
                orig_obs_space)
            states = [
                RobotState(nav, occ, sen, d_t, max_ep_time)
                for nav, occ, sen in zip(sim.robot_navs, occupancies, sensors)
                ]
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

        for i, (sim, state, term) in enumerate(
            zip(self.simulators, self.states, terms)
            ):
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