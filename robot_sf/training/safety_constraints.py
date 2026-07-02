"""Safety-cost extraction and Lagrange multiplier state for constrained training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

SUPPORTED_SAFETY_COST_SOURCES = frozenset(
    {
        "collision_any",
        "pedestrian_collision",
        "robot_or_obstacle",
        "near_miss",
        "comfort_exposure",
        "ttc_risk",
    }
)


@dataclass(frozen=True)
class SafetyConstraintSpec:
    """Configuration for one episodic safety-cost budget."""

    name: str
    budget_per_episode: float
    source_key: str
    multiplier_init: float = 1.0
    multiplier_lr: float = 0.05
    multiplier_max: float = 50.0
    normalize_by_episode_steps: bool = False

    def __post_init__(self) -> None:
        """Validate the constraint spec fail-closed before training starts."""
        if not self.name:
            raise ValueError("SafetyConstraintSpec.name must be non-empty")
        if self.source_key not in SUPPORTED_SAFETY_COST_SOURCES:
            raise ValueError(f"Unsupported safety-cost source: {self.source_key}")
        _require_non_negative_finite(self.budget_per_episode, "budget_per_episode")
        _require_non_negative_finite(self.multiplier_init, "multiplier_init")
        _require_non_negative_finite(self.multiplier_lr, "multiplier_lr")
        _require_non_negative_finite(self.multiplier_max, "multiplier_max")
        if self.multiplier_init > self.multiplier_max:
            raise ValueError("multiplier_init must be <= multiplier_max")


@dataclass
class LagrangeMultiplierState:
    """Mutable Lagrange multiplier and episode-cost accumulator state."""

    values: dict[str, float]
    episode_costs: dict[str, float] = field(default_factory=dict)
    completed_episodes: int = 0
    episode_steps: int = 0

    @classmethod
    def from_specs(cls, specs: Sequence[SafetyConstraintSpec]) -> LagrangeMultiplierState:
        """Build multiplier state initialized from constraint specs.

        Returns:
            New state with multiplier values and zeroed episode cost accumulators.
        """
        return cls(
            values={spec.name: spec.multiplier_init for spec in specs},
            episode_costs={spec.name: 0.0 for spec in specs},
        )

    def observe_step(self, costs: Mapping[str, float]) -> None:
        """Accumulate one step of extracted safety costs."""
        for name, cost in costs.items():
            self.episode_costs[name] = self.episode_costs.get(name, 0.0) + _finite_non_negative(
                cost
            )
        self.episode_steps += 1

    def episode_summary(
        self,
        specs: Sequence[SafetyConstraintSpec],
    ) -> dict[str, dict[str, float] | int]:
        """Return current episode costs, budgets, violations, and multipliers.

        Returns:
            Serializable episode diagnostics for callbacks and manifests.
        """
        costs = dict(self.episode_costs)
        budgets = {spec.name: spec.budget_per_episode for spec in specs}
        violations: dict[str, float] = {}
        for spec in specs:
            observed_cost = costs.get(spec.name, 0.0)
            if spec.normalize_by_episode_steps and self.episode_steps > 0:
                observed_cost /= self.episode_steps
            violations[spec.name] = observed_cost - spec.budget_per_episode
        return {
            "costs": costs,
            "budgets": budgets,
            "violations": violations,
            "multipliers_before_update": dict(self.values),
            "episode_steps": self.episode_steps,
        }

    def update_after_episode(
        self,
        specs: Sequence[SafetyConstraintSpec],
        *,
        episode_costs: Mapping[str, float] | None = None,
        episode_steps: int | None = None,
    ) -> dict[str, float]:
        """Update multipliers from completed episode costs and clip to configured bounds.

        Returns:
            Updated multiplier values keyed by constraint name.
        """
        costs = self.episode_costs if episode_costs is None else episode_costs
        steps = self.episode_steps if episode_steps is None else episode_steps
        for spec in specs:
            observed_cost = _finite_non_negative(costs.get(spec.name, 0.0))
            if spec.normalize_by_episode_steps and steps > 0:
                observed_cost /= steps
            next_value = self.values.get(spec.name, spec.multiplier_init) + spec.multiplier_lr * (
                observed_cost - spec.budget_per_episode
            )
            self.values[spec.name] = min(spec.multiplier_max, max(0.0, next_value))
        self.completed_episodes += 1
        return dict(self.values)

    def reset_episode_costs(self, specs: Sequence[SafetyConstraintSpec]) -> None:
        """Reset per-episode cost accumulators after terminal diagnostics are emitted."""
        self.episode_costs = {spec.name: 0.0 for spec in specs}
        self.episode_steps = 0


def step_safety_costs(
    info: Mapping[str, Any],
    specs: Sequence[SafetyConstraintSpec],
) -> dict[str, float]:
    """Extract configured safety costs from one environment ``info`` payload.

    Returns:
        Safety cost scalar for each configured constraint name.
    """
    return {spec.name: _source_cost(info, spec.source_key) for spec in specs}


def _source_cost(info: Mapping[str, Any], source_key: str) -> float:
    """Return one supported safety-cost scalar from RobotEnv step metadata.

    Returns:
        Non-negative cost for the requested source.
    """
    meta = info.get("meta")
    meta_mapping = meta if isinstance(meta, Mapping) else {}
    if source_key == "collision_any":
        return float(
            bool(info.get("collision"))
            or bool(meta_mapping.get("is_pedestrian_collision"))
            or bool(meta_mapping.get("is_obstacle_collision"))
            or bool(meta_mapping.get("is_robot_collision"))
        )
    if source_key == "pedestrian_collision":
        return float(bool(meta_mapping.get("is_pedestrian_collision")))
    if source_key == "robot_or_obstacle":
        return float(
            bool(meta_mapping.get("is_robot_collision"))
            or bool(meta_mapping.get("is_obstacle_collision"))
        )
    if source_key == "near_miss":
        return _finite_non_negative(meta_mapping.get("near_misses"))
    if source_key == "comfort_exposure":
        return _finite_non_negative(meta_mapping.get("comfort_exposure"))
    if source_key == "ttc_risk":
        time_to_collision = _finite_positive(meta_mapping.get("time_to_collision"))
        return 0.0 if time_to_collision is None else 1.0 / time_to_collision
    raise ValueError(f"Unsupported safety-cost source: {source_key}")


def _finite_non_negative(value: Any) -> float:
    """Coerce finite non-negative floats and clamp invalid values to zero.

    Returns:
        Finite non-negative float, or zero for invalid input.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(result):
        return 0.0
    return max(0.0, result)


def _finite_positive(value: Any) -> float | None:
    """Coerce finite positive floats, returning ``None`` for invalid values.

    Returns:
        Positive finite float, or ``None`` when no positive value is available.
    """
    result = _finite_non_negative(value)
    if result <= 0.0:
        return None
    return result


def _require_non_negative_finite(value: Any, field_name: str) -> None:
    """Raise if ``value`` is not finite and non-negative."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and non-negative") from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")


__all__ = [
    "SUPPORTED_SAFETY_COST_SOURCES",
    "LagrangeMultiplierState",
    "SafetyConstraintSpec",
    "step_safety_costs",
]
