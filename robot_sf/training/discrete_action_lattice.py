"""Discrete unicycle command lattices for distributional value learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def _require_finite_bounded_values(
    *,
    values: tuple[float, ...],
    field_name: str,
    max_abs_value: float,
    max_field_name: str,
) -> None:
    """Validate one lattice axis against its configured absolute bound."""

    if not values:
        raise ValueError(f"{field_name} must contain at least one command value")
    for value in values:
        if not isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        if abs(value) > max_abs_value:
            raise ValueError(f"{field_name} exceed {max_field_name}")


@dataclass(frozen=True)
class UnicycleCommand:
    """Absolute unicycle velocity target selected by a discrete policy."""

    linear_velocity: float
    angular_velocity: float

    def as_tuple(self) -> tuple[float, float]:
        """Return ``(linear_velocity, angular_velocity)`` for adapter-facing code.

        Returns:
            Tuple ordered as linear velocity, then angular velocity.
        """

        return (self.linear_velocity, self.angular_velocity)


@dataclass(frozen=True)
class DiscreteUnicycleActionLattice:
    """Cartesian lattice of absolute unicycle velocity commands.

    The lattice is intentionally small and deterministic so QR-DQN/IQN-style
    value heads can bridge Robot SF's continuous native action space through a
    benchmark-compatible unicycle command surface.
    """

    linear_values: tuple[float, ...]
    angular_values: tuple[float, ...]
    max_linear_speed: float
    max_angular_speed: float

    def __post_init__(self) -> None:
        """Validate lattice values fail closed before training uses them."""

        linear_values = tuple(float(value) for value in self.linear_values)
        angular_values = tuple(float(value) for value in self.angular_values)
        object.__setattr__(self, "linear_values", linear_values)
        object.__setattr__(self, "angular_values", angular_values)
        object.__setattr__(self, "max_linear_speed", float(self.max_linear_speed))
        object.__setattr__(self, "max_angular_speed", float(self.max_angular_speed))

        if self.max_linear_speed <= 0.0 or not isfinite(self.max_linear_speed):
            raise ValueError("max_linear_speed must be a finite positive value")
        if self.max_angular_speed <= 0.0 or not isfinite(self.max_angular_speed):
            raise ValueError("max_angular_speed must be a finite positive value")
        _require_finite_bounded_values(
            values=linear_values,
            field_name="linear_values",
            max_abs_value=self.max_linear_speed,
            max_field_name="max_linear_speed",
        )
        _require_finite_bounded_values(
            values=angular_values,
            field_name="angular_values",
            max_abs_value=self.max_angular_speed,
            max_field_name="max_angular_speed",
        )

    @property
    def action_count(self) -> int:
        """Return the number of discrete command candidates.

        Returns:
            Cartesian product size for the configured lattice axes.
        """

        return len(self.linear_values) * len(self.angular_values)

    def command_at(self, index: int) -> UnicycleCommand:
        """Return the deterministic command for ``index``.

        The ordering is row-major over linear velocity, then angular velocity.

        Returns:
            Absolute unicycle velocity command at the requested action index.
        """

        if index < 0 or index >= self.action_count:
            raise IndexError(f"action index {index} outside [0, {self.action_count})")
        linear_index, angular_index = divmod(index, len(self.angular_values))
        return UnicycleCommand(
            linear_velocity=self.linear_values[linear_index],
            angular_velocity=self.angular_values[angular_index],
        )

    def commands(self) -> tuple[UnicycleCommand, ...]:
        """Return all commands in stable index order.

        Returns:
            Tuple of all configured commands in action-index order.
        """

        return tuple(self.command_at(index) for index in range(self.action_count))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable lattice contract.

        Returns:
            Dictionary containing lattice axes, bounds, and command-space tag.
        """

        return {
            "linear_values": list(self.linear_values),
            "angular_values": list(self.angular_values),
            "max_linear_speed": self.max_linear_speed,
            "max_angular_speed": self.max_angular_speed,
            "command_space": "unicycle_vw",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DiscreteUnicycleActionLattice:
        """Build a lattice from a JSON-compatible payload.

        Returns:
            Validated action lattice.
        """

        unknown = set(payload) - {
            "linear_values",
            "angular_values",
            "max_linear_speed",
            "max_angular_speed",
            "command_space",
        }
        if unknown:
            raise ValueError(f"unknown lattice keys: {sorted(unknown)}")
        command_space = payload.get("command_space", "unicycle_vw")
        if command_space != "unicycle_vw":
            raise ValueError("distributional RL lattices must use command_space='unicycle_vw'")
        return cls(
            linear_values=tuple(payload["linear_values"]),
            angular_values=tuple(payload["angular_values"]),
            max_linear_speed=payload["max_linear_speed"],
            max_angular_speed=payload["max_angular_speed"],
        )

    def to_json_file(self, path: Path) -> None:
        """Write the lattice contract to ``path``."""

        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_json_file(cls, path: Path) -> DiscreteUnicycleActionLattice:
        """Read a lattice contract from ``path``.

        Returns:
            Validated action lattice loaded from JSON.
        """

        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
