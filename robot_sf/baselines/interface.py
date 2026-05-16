"""Planner interface protocol for the Social Navigation Benchmark.

Defines the standard interface that all baseline planners must implement.
Ensures a consistent API across different planning algorithms including
SocialForce, PPO, Random, and future baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ObservationContract:
    """Planner-facing observation assumptions declared as lightweight metadata."""

    mode: str
    supported_modes: tuple[str, ...]
    required_inputs: tuple[str, ...]
    active_mode: str | None = None
    frame: str = "world"
    normalization: str = "raw"
    pedestrian_ordering: str = "distance_ascending"
    missing_value: str | None = None
    notes: str = ""

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable observation contract payload."""
        return {
            "mode": self.mode,
            "active_mode": self.active_mode or self.mode,
            "supported_modes": list(self.supported_modes),
            "required_inputs": list(self.required_inputs),
            "frame": self.frame,
            "normalization": self.normalization,
            "pedestrian_ordering": self.pedestrian_ordering,
            "missing_value": self.missing_value,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ActionContract:
    """Planner action assumptions before any benchmark/environment conversion."""

    command_space: str
    output_keys: tuple[str, ...]
    frame: str = "robot"
    normalization: str = "raw"
    units: str = "mps_radps"
    scaling: str = "none"
    compatible_robot_kinematics: tuple[str, ...] = (
        "differential_drive",
        "bicycle_drive",
        "holonomic",
        "mixed",
        "unknown",
    )
    notes: str = ""
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable action contract payload."""
        return {
            "command_space": self.command_space,
            "output_keys": list(self.output_keys),
            "frame": self.frame,
            "normalization": self.normalization,
            "units": self.units,
            "scaling": self.scaling,
            "compatible_robot_kinematics": list(self.compatible_robot_kinematics),
            "bounds": {key: list(value) for key, value in self.bounds.items()},
            "notes": self.notes,
        }


@dataclass(frozen=True)
class PlannerMetadata:
    """First-class planner compatibility metadata for benchmark-facing workflows."""

    planner_id: str
    observation_contract: ObservationContract
    action_contract: ActionContract
    reset_contract: str = "seeded_reset"
    scenario_requirements: tuple[str, ...] = ()
    compatibility_scope: str = "metadata_only"
    notes: str = ""

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable planner metadata payload."""
        return {
            "planner_id": self.planner_id,
            "observation_contract": self.observation_contract.to_metadata(),
            "action_contract": self.action_contract.to_metadata(),
            "reset_contract": self.reset_contract,
            "scenario_requirements": list(self.scenario_requirements),
            "compatibility_scope": self.compatibility_scope,
            "notes": self.notes,
        }


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


__all__ = [
    "ActionContract",
    "ObservationContract",
    "PlannerMetadata",
    "PlannerProtocol",
]
