"""
Abstract base classes for environment hierarchy.

This module defines the abstract base classes that provide consistent
interfaces and shared functionality for all simulation environments.
"""

from abc import ABC, abstractmethod
from typing import Any

from gymnasium import Env, spaces

from robot_sf.gym_env.unified_config import BaseSimulationConfig
from robot_sf.render.sim_view import VisualizableSimState


class BaseSimulationEnv(Env, ABC):
    """
    Abstract base class for all simulation environments.

    Provides common functionality for:
    - Environment setup and teardown
    - Recording and visualization
    - Configuration management
    - Shared utility methods
    """

    def __init__(
        self,
        config: BaseSimulationConfig,
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str | None = None,
        video_fps: float | None = None,
        **kwargs,
    ):
        """Initialize shared simulation plumbing used by all environments.

        Args:
            config: Unified simulation config describing backends, map sources, and
                runtime toggles.
            debug: Whether to enable verbose logging or visualization helpers.
            recording_enabled: Enables capture of ``VisualizableSimState`` snapshots per
                step so runs can be replayed later.
            record_video: If ``True``, exports a video when ``save_recording`` is called.
            video_path: Optional override for the recording destination path.
            video_fps: Frames-per-second to use for rendered videos; defaults to the view
                FPS when ``None``.
            kwargs: Additional keyword arguments consumed by subclasses (e.g., custom
                simulator hooks).
        """
        super().__init__()
        self.config = config
        self.debug = debug
        self.recording_enabled = recording_enabled
        self.record_video = record_video
        self.video_path = video_path
        self.video_fps = video_fps

        # Common state
        self.recorded_states: list[VisualizableSimState] = []
        self.sim_ui = None
        self.map_def = None
        self.applied_seed: int | None = None

        # Initialize common components
        self._setup_environment()

    @abstractmethod
    def _setup_environment(self) -> None:
        """Initialize environment-specific components."""
        pass

    @abstractmethod
    def _create_spaces(self) -> tuple[spaces.Space, spaces.Space]:
        """Create action and observation spaces."""
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one environment step."""
        pass

    @abstractmethod
    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Reset the environment."""
        pass

    @abstractmethod
    def render(self, **kwargs) -> None:
        """Render the environment."""
        pass

    def exit(self) -> None:
        """Clean up and exit the simulation."""
        if self.sim_ui:
            self.sim_ui.exit_simulation()

    def save_recording(self, filename: str | None = None) -> None:
        """Save recorded states to file."""
        from robot_sf.gym_env.base_env import BaseEnv

        # Reuse existing implementation
        base_env = BaseEnv.__new__(BaseEnv)
        base_env.recorded_states = self.recorded_states
        base_env.map_def = self.map_def
        base_env.save_recording(filename)


class SingleAgentEnv(BaseSimulationEnv, ABC):
    """
    Abstract base class for single-agent environments.

    Handles common patterns for single robot/agent simulation including:
    - State management
    - Sensor fusion
    - Collision detection
    """

    def __init__(self, config: BaseSimulationConfig, **kwargs):
        """Configure base state for a single-agent environment.

        Args:
            config: Simulation config that defines the map, backend, and robot-specific
                parameters.
            kwargs: Extra keyword arguments forwarded to :class:`BaseSimulationEnv`.
        """
        self.state = None
        self.simulator = None
        self.last_action = None
        # Call super().__init__() after initializing instance variables
        # because parent's _setup_environment() may depend on these attributes
        super().__init__(config, **kwargs)

    @abstractmethod
    def _setup_simulator(self) -> None:
        """Initialize the simulator."""
        pass

    @abstractmethod
    def _setup_sensors_and_collision(self) -> None:
        """Initialize sensors and collision detection."""
        pass

    @abstractmethod
    def _prepare_visualizable_state(self) -> VisualizableSimState:
        """Prepare state for visualization."""
        pass


class MultiAgentEnv(BaseSimulationEnv, ABC):
    """
    Abstract base class for multi-agent environments.

    Handles common patterns for multi-agent simulation including:
    - Vectorized operations
    - Agent coordination
    - Parallel simulation
    """

    def __init__(self, config: BaseSimulationConfig, num_agents: int, **kwargs):
        """Configure shared resources for a multi-agent simulation.

        Args:
            config: Simulation config applied to every agent instance.
            num_agents: Number of agents to spawn and coordinate.
            kwargs: Additional keyword arguments passed to :class:`BaseSimulationEnv`.
        """
        super().__init__(config, **kwargs)
        self.num_agents = num_agents
        self.agents = []
        self.simulators = []

    @abstractmethod
    def _setup_agents(self) -> None:
        """Initialize multiple agents."""
        pass

    @abstractmethod
    def _step_agents(
        self,
        actions: list[Any],
    ) -> tuple[list[Any], list[float], list[bool], list[dict]]:
        """Execute step for all agents."""
        pass
