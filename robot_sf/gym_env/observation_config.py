"""Observation-specific configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_OBSERVATION_STACK_STEPS = 3


def _validate_stack_steps(stack_steps: int) -> int:
    """Return a positive stack depth or raise a clear configuration error."""
    stack_steps = int(stack_steps)
    if stack_steps <= 0:
        raise ValueError("stack_steps must be > 0")
    return stack_steps


@dataclass
class ObservationStackSettings:
    """Configuration for temporal history stacked into observations."""

    stack_steps: int = DEFAULT_OBSERVATION_STACK_STEPS

    def __post_init__(self) -> None:
        """Validate the configured observation history depth."""
        self.stack_steps = _validate_stack_steps(self.stack_steps)


def sync_observation_stack_settings(config: Any) -> ObservationStackSettings:
    """Synchronize observation-owned stack settings with the legacy simulation alias.

    ``SimulationSettings.stack_steps`` is retained for existing scripts and saved configs. New
    code should read and write ``observation_stack.stack_steps`` through the helpers here.

    Returns:
        The normalized observation stack settings attached to ``config``.
    """
    observation_stack = getattr(config, "observation_stack", None)
    sim_config = getattr(config, "sim_config", None)
    legacy_stack_steps = getattr(sim_config, "stack_steps", None)

    if observation_stack is None:
        stack_steps = (
            DEFAULT_OBSERVATION_STACK_STEPS
            if legacy_stack_steps is None
            else _validate_stack_steps(legacy_stack_steps)
        )
        observation_stack = ObservationStackSettings(stack_steps=stack_steps)
        config.observation_stack = observation_stack
    elif isinstance(observation_stack, dict):
        observation_stack = ObservationStackSettings(**observation_stack)
        config.observation_stack = observation_stack
    elif not isinstance(observation_stack, ObservationStackSettings):
        raise TypeError(
            "observation_stack must be an ObservationStackSettings instance, dict, or None"
        )

    if legacy_stack_steps is not None:
        validated_legacy_stack_steps = _validate_stack_steps(legacy_stack_steps)
        if (
            validated_legacy_stack_steps != DEFAULT_OBSERVATION_STACK_STEPS
            and observation_stack.stack_steps == DEFAULT_OBSERVATION_STACK_STEPS
        ):
            observation_stack.stack_steps = validated_legacy_stack_steps

    if sim_config is not None and hasattr(sim_config, "stack_steps"):
        sim_config.stack_steps = observation_stack.stack_steps
    return observation_stack


def get_observation_stack_steps(config: Any) -> int:
    """Return the effective observation history depth for an environment config."""
    return sync_observation_stack_settings(config).stack_steps


def set_observation_stack_steps(config: Any, stack_steps: int) -> None:
    """Set observation history depth while keeping the legacy simulation alias in sync."""
    stack_steps = _validate_stack_steps(stack_steps)
    observation_stack = getattr(config, "observation_stack", None)
    if isinstance(observation_stack, dict):
        observation_stack = ObservationStackSettings(**observation_stack)
    elif observation_stack is None:
        observation_stack = ObservationStackSettings()
    elif not isinstance(observation_stack, ObservationStackSettings):
        raise TypeError(
            "observation_stack must be an ObservationStackSettings instance, dict, or None"
        )

    observation_stack.stack_steps = stack_steps
    config.observation_stack = observation_stack

    sim_config = getattr(config, "sim_config", None)
    if sim_config is not None and hasattr(sim_config, "stack_steps"):
        sim_config.stack_steps = stack_steps
