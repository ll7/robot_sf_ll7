"""
`MultiRobotEnv`: A class that extends `RobotEnv` to handle multiple robots in the environment.
It overrides the `step_async` method to apply actions to all robots in the environment.
"""

from collections.abc import Callable
from multiprocessing.pool import ThreadPool
from typing import Any

import numpy as np
from gymnasium import spaces
from loguru import logger

from robot_sf.gym_env.abstract_envs import MultiAgentEnv
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.env_util import init_collision_and_sensors, init_spaces
from robot_sf.gym_env.reward import simple_reward
from robot_sf.gym_env.unified_config import MultiRobotConfig
from robot_sf.robot.robot_state import RobotState
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS
from robot_sf.sim.simulator import init_simulators


class MultiRobotEnv(MultiAgentEnv):
    """Representing a Gymnasium environment for training
    multiple self-driving robots with reinforcement learning"""

    # Type annotations
    config: EnvSettings | MultiRobotConfig
    sim_worker_pool: ThreadPool
    obs_worker_pool: ThreadPool

    def __init__(
        self,
        env_config: EnvSettings | MultiRobotConfig = EnvSettings(),
        reward_func: Callable[[dict], float] | None = simple_reward,
        debug: bool = False,
        num_robots: int | None = None,
    ):
        """Init.

        Args:
            env_config: environment config.
            reward_func: reward func.
            debug: Debug mode flag.
            num_robots: number of robots.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        if isinstance(env_config, MultiRobotConfig):
            resolved_num_robots = env_config.num_robots
            if num_robots is not None and num_robots != resolved_num_robots:
                raise ValueError(
                    "num_robots argument ("
                    f"{num_robots}) must match env_config.num_robots ({resolved_num_robots}).",
                )
        else:
            resolved_num_robots = num_robots or 1

        # Initialize pools as None early so destructor/cleanup paths don't fail
        # if they are referenced before full initialization.
        self.sim_worker_pool = None  # type: ignore[assignment]
        self.obs_worker_pool = None  # type: ignore[assignment]

        map_def = env_config.map_pool.map_defs["uni_campus_big"]  # info: only use first map
        action_space, observation_space, orig_obs_space = init_spaces(env_config, map_def)

        # Keep a reference to the per-agent spaces for later composition
        self.single_action_space = action_space
        self.single_observation_space = observation_space

        # Initialize abstract multi-agent base with config and agent count.
        # `MultiAgentEnv` will forward the config to BaseSimulationEnv.
        super().__init__(env_config, resolved_num_robots, debug=debug)  # type: ignore[arg-type]

        # Combined action space for all robots (vectorized)
        self.action_space = spaces.Box(
            low=np.array([self.single_action_space.low for _ in range(resolved_num_robots)]),
            high=np.array([self.single_action_space.high for _ in range(resolved_num_robots)]),
            dtype=self.single_action_space.low.dtype,
        )

        # Ensure a usable reward function even if None explicitly provided
        if reward_func is None:
            self.reward_func = simple_reward
        else:
            self.reward_func = reward_func
        self.debug = debug
        self.simulators = init_simulators(
            env_config,
            map_def,
            resolved_num_robots,
            random_start_pos=False,
        )
        self.states: list[RobotState] = []
        d_t = env_config.sim_config.time_per_step_in_secs
        max_ep_time = env_config.sim_config.sim_time_in_secs

        for sim in self.simulators:
            occupancies, sensors = init_collision_and_sensors(sim, env_config, orig_obs_space)
            states = [
                RobotState(nav, occ, sen, d_t, max_ep_time)
                for nav, occ, sen in zip(sim.robot_navs, occupancies, sensors, strict=False)
            ]
            self.states.extend(states)
        self.sim_worker_pool = ThreadPool(len(self.simulators))
        self.obs_worker_pool = ThreadPool(resolved_num_robots)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one environment step with multi-agent actions."""
        actions = action  # Rename for clarity in multi-agent context
        actions = [self.simulators[0].robots[0].parse_action(a) for a in actions]
        i = 0
        actions_per_simulator = []
        for sim in self.simulators:
            num_robots = len(sim.robots)
            actions_per_simulator.append(actions[i : i + num_robots])
            i += num_robots

        self.sim_worker_pool.map(
            lambda s_a: s_a[0].step_once(s_a[1]),
            zip(self.simulators, actions_per_simulator, strict=False),
        )

        obs = self.obs_worker_pool.map(lambda s: s.step(), self.states)

        metas = [state.meta_dict() for state in self.states]
        masked_metas = [{"step": meta["step"], "meta": meta} for meta in metas]
        masked_metas_tuple = (*masked_metas,)
        terms = [state.is_terminal for state in self.states]
        rewards = [self.reward_func(meta) for meta in metas]

        for i, (sim, state, term) in enumerate(
            zip(self.simulators, self.states, terms, strict=False),
        ):
            if term:
                sim.reset_state()
                obs[i] = state.reset()

        obs_dict = {
            OBS_DRIVE_STATE: np.array([o[OBS_DRIVE_STATE] for o in obs]),
            OBS_RAYS: np.array([o[OBS_RAYS] for o in obs]),
        }

        # Return in Gymnasium format (obs, reward, terminated, truncated, info)
        # For multi-agent, we aggregate rewards
        total_reward = sum(rewards)
        any_terminated = any(terms)
        return obs_dict, total_reward, any_terminated, False, {"agents": masked_metas_tuple}

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Reset the environment."""
        self.sim_worker_pool.map(lambda sim: sim.reset_state(), self.simulators)
        obs = self.obs_worker_pool.map(lambda s: s.reset(), self.states)

        obs_dict = {
            OBS_DRIVE_STATE: np.array([o[OBS_DRIVE_STATE] for o in obs]),
            OBS_RAYS: np.array([o[OBS_RAYS] for o in obs]),
        }
        return obs_dict, {}

    def render(self, **kwargs) -> None:
        """Render the environment."""
        # TODO: add support for PyGame rendering
        pass

    def close_extras(self, **kwargs):
        """Close extras.

        Args:
            kwargs: Additional keyword arguments forwarded downstream.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        if getattr(self, "sim_worker_pool", None) is not None:
            try:
                self.sim_worker_pool.close()
            except (AttributeError, OSError, RuntimeError) as e:
                # Best-effort close; log common failure modes but avoid
                # silencing unrelated exceptions.
                logger.warning("Failed to close sim_worker_pool: {}", e)
        if getattr(self, "obs_worker_pool", None) is not None:
            try:
                self.obs_worker_pool.close()
            except (AttributeError, OSError, RuntimeError) as e:
                logger.warning("Failed to close obs_worker_pool: {}", e)

    # --- Abstract base compatibility -------------------------------------------------
    def _setup_environment(self) -> None:
        """Minimal environment setup hook called by BaseSimulationEnv.__init__.

        We intentionally keep this lightweight because the full simulator
        and per-agent states are initialized later in the concrete
        constructor. This satisfies the abstract contract.
        """
        # no-op: actual simulator initialization happens in __init__ below
        return None

    def _create_spaces(self) -> tuple[spaces.Space, spaces.Space]:
        """Return per-agent action and observation spaces.

        The factory path computes these before calling the base ctor; if
        they are not yet available, compute them on-demand from the
        environment config.
        """
        try:
            return self.single_action_space, self.single_observation_space
        except AttributeError:
            # Fallback: compute from config
            map_def = self.config.map_pool.map_defs["uni_campus_big"]
            action_space, observation_space, _ = init_spaces(self.config, map_def)
            return action_space, observation_space

    def _setup_agents(self) -> None:
        """Hook to initialise agents; already done in __init__, so keep no-op.

        Implemented to satisfy abstract base requirements.
        """
        return None

    def _step_agents(self, actions: list) -> tuple[list, list[float], list[bool], list[dict]]:
        """Execute a multi-agent step and return the standard tuple.

        This delegates to the environment's `step` implementation to avoid
        duplicating logic.
        """
        obs, reward, terminated, _truncated, info = self.step(actions)
        # Convert single values back to lists for multi-agent interface
        # Note: This may need adjustment based on actual multi-agent requirements
        return [obs], [reward], [terminated], [info]
