"""
Base environment for the simulation environment.
Provides common functionality for all robot-based environments.
"""

from typing import List, Optional
import datetime
import pickle

from gymnasium import Env

from robot_sf.gym_env.env_config import BaseEnvSettings
from robot_sf.render.sim_view import (
    SimulationView,
    VisualizableSimState,
)
from robot_sf.sim.simulator import init_simulators


class BaseEnv(Env):
    """Base environment class that handles common functionality."""

    def __init__(
        self,
        env_config: BaseEnvSettings = BaseEnvSettings(),
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: Optional[str] = None,
        video_fps: Optional[float] = None,
    ):
        """Initialize base environment with common settings."""
        super().__init__()

        self.env_config = env_config
        self.debug = debug
        self.recording_enabled = recording_enabled
        self.record_video = record_video
        self.video_path = video_path
        self.video_fps = video_fps

        # Common attributes
        self.recorded_states: List[VisualizableSimState] = []
        self.map_def = env_config.map_pool.choose_random_map()
        self.last_action = None
        self.simulator = None
        self.sim_ui = None

    def _setup_simulator(self):
        """Initialize simulator with config settings."""
        self.simulator = init_simulators(
            self.env_config, self.map_def, random_start_pos=True
        )[0]

    def _setup_visualization(self):
        """Setup visualization if debug or video recording is enabled."""
        if self.debug or self.record_video:
            self.sim_ui = SimulationView(
                scaling=10,
                map_def=self.map_def,
                obstacles=self.map_def.obstacles,
                robot_radius=self.env_config.robot_config.radius,
                ped_radius=self.env_config.sim_config.ped_radius,
                goal_radius=self.env_config.sim_config.goal_radius,
                record_video=self.record_video,
                video_path=self.video_path,
                video_fps=self.video_fps,
            )

    def render(self):
        """Render the environment if visualization is enabled."""
        if not self.sim_ui:
            raise RuntimeError(
                "Debug mode is not activated! Consider setting debug=True!"
            )

        state = self._prepare_visualizable_state()
        self.sim_ui.render(state)

    def record(self):
        """Record current state if recording is enabled."""
        if self.recording_enabled:
            self.recorded_states.append(self._prepare_visualizable_state())

    def save_recording(self, filename: Optional[str] = None):
        """Save recorded states to file."""
        if not self.recorded_states:
            return

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.pkl"

        with open(filename, "wb") as f:
            pickle.dump(self.recorded_states, f)

        self.recorded_states.clear()

    def exit(self):
        """Clean up environment resources."""
        if self.sim_ui:
            self.sim_ui.exit_simulation()

    def _prepare_visualizable_state(self) -> VisualizableSimState:
        """Prepare state for visualization. Must be implemented by child classes."""
        raise NotImplementedError
