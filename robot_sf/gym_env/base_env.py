"""
Base environment for the simulation environment.
Provides common functionality for all environments.
"""

import datetime
import pickle

from gymnasium import Env
from loguru import logger

from robot_sf.common.artifact_paths import resolve_artifact_path
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.planner import GlobalPlanner, PlannerConfig
from robot_sf.render.jsonl_recording import JSONLRecorder
from robot_sf.render.sim_view import SimulationView, VisualizableSimState
from robot_sf.sim.registry import get_backend
from robot_sf.sim.simulator import init_simulators


class BaseEnv(Env):
    """Base environment class that handles common functionality."""

    def __init__(
        self,
        env_config: EnvSettings = EnvSettings(),
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str | None = None,
        video_fps: float | None = None,
        peds_have_obstacle_forces: bool = False,
        # New JSONL recording parameters
        use_jsonl_recording: bool = False,
        recording_dir: str = "recordings",
        suite_name: str = "robot_sim",
        scenario_name: str = "default",
        algorithm_name: str = "manual",
        recording_seed: int | None = None,
    ):
        """TODO docstring. Document this function.

        Args:
            env_config: TODO docstring.
            debug: TODO docstring.
            recording_enabled: TODO docstring.
            record_video: TODO docstring.
            video_path: TODO docstring.
            video_fps: TODO docstring.
            peds_have_obstacle_forces: TODO docstring.
            use_jsonl_recording: TODO docstring.
            recording_dir: TODO docstring.
            suite_name: TODO docstring.
            scenario_name: TODO docstring.
            algorithm_name: TODO docstring.
            recording_seed: TODO docstring.
        """
        super().__init__()

        # Environment configuration details
        self.env_config = env_config

        # Set video FPS if not provided
        if video_fps is None:
            video_fps = 1 / self.env_config.sim_config.time_per_step_in_secs
            logger.info(f"Video FPS not provided, setting to {video_fps}")

        # Extract first map definition; currently only supports using the first map
        self.map_def = env_config.map_pool.choose_random_map()
        attach_planner_to_map(self.map_def, self.env_config)

        self.debug = debug

        # Initialize the list to store recorded states
        self.recorded_states: list[VisualizableSimState] = []
        self.recording_enabled = recording_enabled

        # Route recordings through the canonical artifact tree (or override) by default.
        self._recording_dir = resolve_artifact_path(recording_dir)

        # New JSONL recording system
        self.use_jsonl_recording = use_jsonl_recording
        self.jsonl_recorder: JSONLRecorder | None = None

        if use_jsonl_recording and recording_enabled:
            # Use provided seed or generate from environment config
            seed = recording_seed if recording_seed is not None else getattr(env_config, "seed", 0)

            self.jsonl_recorder = JSONLRecorder(
                output_dir=self._recording_dir,
                suite=suite_name,
                scenario=scenario_name,
                algorithm=algorithm_name,
                seed=seed,
            )

        # Initialize simulator via backend registry (falls back to legacy init on error)
        backend_key = getattr(env_config, "backend", "fast-pysf")
        try:
            factory = get_backend(backend_key)
            self.simulator = factory(
                env_config,
                self.map_def,
                peds_have_obstacle_forces,
            )
            logger.debug("Initialized simulator with backend={}", backend_key)
        except KeyError:
            logger.warning(
                "Unknown backend '{}' — falling back to legacy init_simulators().",
                backend_key,
            )
            self.simulator = init_simulators(
                env_config,
                self.map_def,
                random_start_pos=True,
                peds_have_obstacle_forces=peds_have_obstacle_forces,
            )[0]

        # Store last action executed by the robot
        self.last_action = None

        # Initialize sim_ui attribute (required for exit() method)
        self.sim_ui = None

        # If in debug mode or video recording is enabled, create simulation view
        if debug or record_video:
            # Prefer config-driven render scaling when provided; else default to 10.
            scaling_value = getattr(env_config, "render_scaling", None)
            scaling_value = 10 if scaling_value is None else int(scaling_value)
            self.sim_ui = SimulationView(
                scaling=scaling_value,
                map_def=self.map_def,
                obstacles=self.map_def.obstacles,
                robot_radius=env_config.robot_config.radius,
                ped_radius=env_config.sim_config.ped_radius,
                goal_radius=env_config.sim_config.goal_radius,
                record_video=record_video,
                video_path=video_path,
                video_fps=video_fps,
            )

    def render(self):
        """TODO docstring. Document this function."""
        raise NotImplementedError

    def step(self, action):
        """TODO docstring. Document this function.

        Args:
            action: TODO docstring.
        """
        raise NotImplementedError

    def exit(self):
        """
        Clean up and exit the simulation UI, if it exists.
        """
        if self.sim_ui:
            self.sim_ui.exit_simulation()

    def save_recording(self, filename: str | None = None):
        """
        Save the recorded states to a file.

        Args:
            filename: Optional target filename ending with `.pkl`. When omitted, a timestamped
                filename is generated under the configured recording directory.
        """
        if filename is None:
            now = datetime.datetime.now()
            target_path = self._recording_dir / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        else:
            target_path = resolve_artifact_path(filename)

        if len(self.recorded_states) == 0:
            logger.warning("No states recorded, skipping save")
            # TODO: First env.reset will always have no recorded states
            return

        target_path.parent.mkdir(parents=True, exist_ok=True)

        with target_path.open("wb") as f:  # write binary
            pickle.dump((self.recorded_states, self.map_def), f)
            logger.info(f"Recording saved to {target_path}")
            logger.info("Reset state list")
            self.recorded_states = []

    def start_episode_recording(self, config_hash: str = "unknown") -> None:
        """Start recording a new episode with JSONL recorder."""
        if self.jsonl_recorder is not None:
            self.jsonl_recorder.start_episode(config_hash=config_hash)

    def record_simulation_step(self, state: VisualizableSimState) -> None:
        """Record a single simulation step to both recording systems."""
        # Legacy pickle recording
        if self.recording_enabled and not self.use_jsonl_recording:
            self.recorded_states.append(state)

        # New JSONL recording
        if self.jsonl_recorder is not None:
            self.jsonl_recorder.record_step(state)

    def record_entity_reset(self, entity_ids: list[int], state: VisualizableSimState) -> None:
        """Record an entity reset event."""
        if self.jsonl_recorder is not None:
            self.jsonl_recorder.record_entity_reset(entity_ids, state)

    def end_episode_recording(self) -> None:
        """End the current episode recording."""
        if self.jsonl_recorder is not None:
            self.jsonl_recorder.end_episode()

    def close_recorder(self) -> None:
        """Close the recorder and clean up resources."""
        if self.jsonl_recorder is not None:
            self.jsonl_recorder.close()
            self.jsonl_recorder = None


def attach_planner_to_map(map_def, env_config) -> None:
    """Attach a GlobalPlanner instance to the map when enabled in config."""
    if not getattr(env_config, "use_planner", False):
        return
    if getattr(map_def, "_use_planner", False) and getattr(map_def, "_global_planner", None):
        return

    clearance = getattr(env_config, "planner_clearance_margin", 0.3)
    robot_cfg = getattr(env_config, "robot_config", None)
    robot_radius = getattr(robot_cfg, "radius", 0.4) if robot_cfg else 0.4

    planner_cfg = PlannerConfig(robot_radius=robot_radius, min_safe_clearance=clearance)
    map_def._global_planner = GlobalPlanner(map_def, planner_cfg)
    map_def._use_planner = True
