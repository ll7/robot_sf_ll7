"""
`MultiRobotEnv`: A class that extends `RobotEnv` to handle multiple robots in the environment.
It overrides the `step_async` method to apply actions to all robots in the environment.
"""

from collections.abc import Callable
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from loguru import logger

from robot_sf.gym_env.abstract_envs import MultiAgentEnv
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.env_util import (
    init_collision_and_sensors,
    init_spaces,
    prepare_pedestrian_actions,
)
from robot_sf.gym_env.reward import simple_reward
from robot_sf.gym_env.unified_config import MultiRobotConfig
from robot_sf.render.lidar_visual import render_lidar
from robot_sf.render.sim_view import SimulationView, VisualizableSimState
from robot_sf.robot.robot_state import RobotState
from robot_sf.sensor.range_sensor import lidar_ray_scan
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS
from robot_sf.sim.simulator import init_simulators


class MultiRobotEnv(MultiAgentEnv):
    """Representing a Gymnasium environment for training
    multiple self-driving robots with reinforcement learning"""

    # Type annotations
    config: EnvSettings | MultiRobotConfig
    sim_worker_pool: ThreadPool
    obs_worker_pool: ThreadPool
    _state_bindings: list[tuple[Any, int, RobotState]]
    _sim_views: list[SimulationView]

    def __init__(
        self,
        env_config: EnvSettings | MultiRobotConfig = EnvSettings(),
        reward_func: Callable[[dict], float] | None = simple_reward,
        debug: bool = False,
        num_robots: int | None = None,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str | None = None,
        video_fps: float | None = None,
    ):
        """TODO docstring. Document this function.

        Args:
            env_config: TODO docstring.
            reward_func: TODO docstring.
            debug: TODO docstring.
            num_robots: TODO docstring.
            recording_enabled: TODO docstring.
            record_video: TODO docstring.
            video_path: TODO docstring.
            video_fps: TODO docstring.
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
        super().__init__(
            env_config,
            resolved_num_robots,
            debug=debug,
            recording_enabled=recording_enabled,
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
        )  # type: ignore[arg-type]
        self.map_def = map_def

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
        self._state_bindings = []
        self._sim_views = []
        d_t = env_config.sim_config.time_per_step_in_secs
        max_ep_time = env_config.sim_config.sim_time_in_secs

        for sim in self.simulators:
            occupancies, sensors = init_collision_and_sensors(sim, env_config, orig_obs_space)
            states = [
                RobotState(nav, occ, sen, d_t, max_ep_time)
                for nav, occ, sen in zip(sim.robot_navs, occupancies, sensors, strict=False)
            ]
            self.states.extend(states)
            for robot_idx, state in enumerate(states):
                self._state_bindings.append((sim, robot_idx, state))
        self.sim_worker_pool = ThreadPool(len(self.simulators))
        self.obs_worker_pool = ThreadPool(resolved_num_robots)
        self._setup_render_views()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one environment step with multi-agent actions.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) following Gymnasium API.
        """
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

        for i, ((sim, _robot_idx, state), term) in enumerate(
            zip(self._state_bindings, terms, strict=False),
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment.

        Returns:
            Tuple of (observation, info) after environment reset.
        """
        super().reset(seed=seed, options=options)
        self.sim_worker_pool.map(lambda sim: sim.reset_state(), self.simulators)
        obs = self.obs_worker_pool.map(lambda s: s.reset(), self.states)

        obs_dict = {
            OBS_DRIVE_STATE: np.array([o[OBS_DRIVE_STATE] for o in obs]),
            OBS_RAYS: np.array([o[OBS_RAYS] for o in obs]),
        }
        return obs_dict, {}

    def render(self, **kwargs) -> None:
        """Render the environment for each robot view."""
        if not self._sim_views:
            raise RuntimeError("Render unavailable: set debug=True or record_video=True.")
        for binding, sim_view in zip(self._state_bindings, self._sim_views, strict=False):
            state = self._build_visualizable_state(binding)
            sim_view.render(state)

    def close_extras(self, **kwargs):
        """TODO docstring. Document this function.

        Args:
            kwargs: TODO docstring.
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
        for sim_view in self._sim_views:
            try:
                sim_view.exit_simulation()
            except (AttributeError, OSError, RuntimeError) as e:
                logger.warning("Failed to close simulation view: {}", e)

    def close(self):
        """Close pools and simulation views."""
        self.close_extras()
        super().close()

    def _setup_render_views(self) -> None:
        """Initialize one SimulationView per robot when rendering is enabled."""
        if not (self.debug or self.record_video):
            return
        scaling_value = getattr(self.config, "render_scaling", None)
        scaling_value = 20 if scaling_value is None else int(scaling_value)
        if self.video_fps is None:
            fps = 1.0 / float(self.config.sim_config.time_per_step_in_secs)
        else:
            fps = float(self.video_fps)

        total_views = len(self._state_bindings)
        for global_idx in range(total_views):
            resolved_video_path = self._resolve_video_path_for_robot(
                self.video_path,
                robot_idx=global_idx,
                total_robots=total_views,
            )
            sim_view = SimulationView(
                scaling=scaling_value,
                map_def=self.map_def,
                obstacles=self.map_def.obstacles,
                robot_radius=self.config.robot_config.radius,
                ped_radius=self.config.sim_config.ped_radius,
                goal_radius=self.config.sim_config.goal_radius,
                record_video=bool(self.record_video),
                video_path=resolved_video_path,
                video_fps=fps,
                caption=f"RobotSF MultiRobotEnv - robot {global_idx}",
            )
            self._sim_views.append(sim_view)
        if self._sim_views:
            # Keep legacy compatibility for code that expects a single `sim_ui`.
            self.sim_ui = self._sim_views[0]

    @staticmethod
    def _resolve_video_path_for_robot(
        base_video_path: str | None,
        *,
        robot_idx: int,
        total_robots: int,
    ) -> str | None:
        """Resolve a deterministic per-robot video path.

        Returns:
            str | None: Per-robot path or ``None`` when recording has no output file.
        """
        if not base_video_path:
            return None
        base_path = Path(base_video_path)
        suffix = base_path.suffix or ".mp4"
        stem = base_path.stem or "episode"
        if total_robots <= 1:
            return str(base_path.with_suffix(suffix))
        resolved = base_path.with_name(f"{stem}_robot{robot_idx}{suffix}")
        return str(resolved)

    def _build_visualizable_state(
        self,
        binding: tuple[Any, int, RobotState],
    ) -> VisualizableSimState:
        """Build render state for one robot.

        Returns:
            VisualizableSimState: State payload compatible with ``SimulationView.render``.
        """
        sim, robot_idx, state = binding
        robot_pose = sim.robot_poses[robot_idx]
        robot_pos = robot_pose[0]
        distances, directions = lidar_ray_scan(
            robot_pose,
            state.occupancy,
            self.config.lidar_config,
        )
        ray_vecs_np = render_lidar(robot_pos, distances, directions)
        ped_actions_np = prepare_pedestrian_actions(sim)
        return VisualizableSimState(
            timestep=state.timestep,
            robot_action=None,
            robot_pose=robot_pose,
            pedestrian_positions=np.asarray(sim.ped_pos, dtype=float).copy(),
            ray_vecs=ray_vecs_np,
            ped_actions=ped_actions_np,
            time_per_step_in_secs=self.config.sim_config.time_per_step_in_secs,
        )

    # --- Abstract base compatibility -------------------------------------------------
    def _setup_environment(self) -> None:
        """Minimal environment setup hook called by BaseSimulationEnv.__init__.

        We intentionally keep this lightweight because the full simulator
        and per-agent states are initialized later in the concrete
        constructor. This satisfies the abstract contract.

        Returns:
            None (setup is deferred to __init__).
        """
        # no-op: actual simulator initialization happens in __init__ below
        return None

    def _create_spaces(self) -> tuple[spaces.Space, spaces.Space]:
        """Return per-agent action and observation spaces.

        The factory path computes these before calling the base ctor; if
        they are not yet available, compute them on-demand from the
        environment config.

        Returns:
            Tuple of (action_space, observation_space) for a single agent.
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

        Returns:
            None (agents already initialized in __init__).
        """
        return None

    def _step_agents(self, actions: list) -> tuple[list, list[float], list[bool], list[dict]]:
        """Execute a multi-agent step and return the standard tuple.

        This delegates to the environment's `step` implementation to avoid
        duplicating logic.

        Returns:
            Tuple of ([observations], [rewards], [terminated], [info_dicts]) for all agents.
        """
        obs, reward, terminated, _truncated, info = self.step(actions)
        # Convert single values back to lists for multi-agent interface
        # Note: This may need adjustment based on actual multi-agent requirements
        return [obs], [reward], [terminated], [info]
