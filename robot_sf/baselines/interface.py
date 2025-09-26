"""Planner interface protocol for the Social Navigation Benchmark.

Defines the standard interface that all baseline planners must implement.
Ensures a consistent API across different planning algorithms including
SocialForce, PPO, Random, and future baselines.
"""

from __future__ import annotations

from typing import Any, Protocol


class PlannerProtocol(Protocol):
    """Protocol defining the standard interface for navigation planners.

    Responsibilities:
    - Initialization with configuration and seed
    - Action generation from observations
    - State reset with optional seed
    - Configuration updates
    - Resource cleanup
    """

    def __init__(self, config: Any, *, seed: int | None = None) -> None:
        """Initialize the planner.

        Args:
            config: Planner-specific configuration object or dict.
            seed: Optional random seed for deterministic behavior.
        """

    def step(self, obs: dict[str, Any] | Any) -> dict[str, float]:
        """Generate an action from an observation.

        Args:
            obs: Observation from the environment.

        Returns:
            Action dictionary (e.g., {"vx": 1.0, "vy": 0.5} or
            {"v": 1.0, "omega": 0.2}).
        """

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal planner state.

        Args:
            seed: Optional random seed.
        """

    def configure(self, config: Any) -> None:
        """Update the planner's configuration.

        Args:
            config: New configuration object or dict.
        """

    def close(self) -> None:
        """Release resources (models, files, etc.)."""


__all__ = ["PlannerProtocol"]
