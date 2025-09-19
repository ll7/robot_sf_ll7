"""Planner interface protocol for the Social Navigation Benchmark.

This module defines the standard interface that all baseline planners
must implement to be compatible with the benchmark system.

The protocol ensures consistent API across different planning algorithms
including SocialForce, PPO, Random, and future baselines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Union


class PlannerProtocol(Protocol):
    """Protocol defining the standard interface for navigation planners.

    All baseline planners must implement this interface to be compatible
    with the Social Navigation Benchmark system.

    The protocol defines a minimal interface that supports:
    - Initialization with configuration and seed
    - Action generation from observations
    - State reset with optional seed
    - Configuration updates
    - Resource cleanup
    """

    def __init__(self, config: Any, *, seed: Optional[int] = None) -> None:
        """Initialize the planner with configuration and optional seed.

        Args:
            config: Planner-specific configuration object or dict
            seed: Optional random seed for deterministic behavior
        """
        ...

    def step(self, obs: Union[Dict[str, Any], Any]) -> Dict[str, float]:
        """Generate action from observation.

        Args:
            obs: Observation from the environment, either as dict or
                structured observation object

        Returns:
            Action dictionary with control values (e.g., {"vx": 1.0, "vy": 0.5}
            or {"v": 1.0, "omega": 0.2} for unicycle mode)
        """
        ...

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Reset the planner's internal state.

        Args:
            seed: Optional random seed for deterministic behavior
        """
        ...

    def configure(self, config: Any) -> None:
        """Update the planner's configuration.

        Args:
            config: New configuration object or dict
        """
        ...

    def close(self) -> None:
        """Clean up resources (files, models, etc.)."""
        ...


__all__ = ["PlannerProtocol"]
