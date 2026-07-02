"""Tests for constrained-RL safety-cost extraction."""

from __future__ import annotations

import math

import pytest

from robot_sf.training.safety_constraints import (
    LagrangeMultiplierState,
    SafetyConstraintSpec,
    step_safety_costs,
)


def _spec(name: str, source_key: str, **overrides: float) -> SafetyConstraintSpec:
    return SafetyConstraintSpec(
        name=name,
        source_key=source_key,
        budget_per_episode=overrides.pop("budget_per_episode", 0.0),
        **overrides,
    )


def test_collision_any_uses_top_level_collision_flag() -> None:
    """Top-level RobotEnv collision flag counts as one collision cost."""
    costs = step_safety_costs(
        {"collision": True, "meta": {}}, [_spec("collision", "collision_any")]
    )

    assert costs == {"collision": 1.0}


def test_collision_any_uses_pedestrian_collision_metadata() -> None:
    """Pedestrian collision metadata counts even if top-level flag is missing."""
    costs = step_safety_costs(
        {"meta": {"is_pedestrian_collision": True}},
        [_spec("collision", "collision_any")],
    )

    assert costs == {"collision": 1.0}


@pytest.mark.parametrize("value", [None, float("nan"), -1.0, "not-a-number"])
def test_near_miss_clamps_invalid_or_missing_values_to_zero(value: object) -> None:
    """Invalid near-miss metadata is fail-closed to zero cost, not NaN propagation."""
    costs = step_safety_costs(
        {"meta": {"near_misses": value}},
        [_spec("near_miss", "near_miss")],
    )

    assert costs == {"near_miss": 0.0}


@pytest.mark.parametrize("meta", [{}, {"comfort_exposure": None}, {"comfort_exposure": math.nan}])
def test_comfort_exposure_handles_absent_and_non_finite_values(meta: dict[str, object]) -> None:
    """Comfort exposure extraction tolerates absent or non-finite metadata."""
    costs = step_safety_costs({"meta": meta}, [_spec("comfort", "comfort_exposure")])

    assert costs == {"comfort": 0.0}


def test_unknown_cost_source_fails_closed() -> None:
    """Unknown cost sources are rejected before training starts."""
    with pytest.raises(ValueError, match="Unsupported safety-cost source"):
        SafetyConstraintSpec(name="unknown", source_key="unknown", budget_per_episode=0.0)


def test_multiplier_update_clips_to_bounds() -> None:
    """Lagrange multiplier updates are clipped to the configured range."""
    specs = [
        _spec(
            "collision",
            "collision_any",
            multiplier_init=0.1,
            multiplier_lr=10.0,
            multiplier_max=1.0,
        )
    ]
    state = LagrangeMultiplierState.from_specs(specs)

    updated = state.update_after_episode(specs, episode_costs={"collision": 10.0})

    assert updated == {"collision": 1.0}
    assert state.completed_episodes == 1


def test_normalized_external_episode_costs_require_episode_steps() -> None:
    """Externally supplied costs must include steps for step-normalized constraints."""
    specs = [
        _spec(
            "near_miss",
            "near_miss",
            budget_per_episode=0.5,
            normalize_by_episode_steps=True,
        )
    ]
    state = LagrangeMultiplierState.from_specs(specs)

    with pytest.raises(ValueError, match="episode_steps must be provided"):
        state.update_after_episode(specs, episode_costs={"near_miss": 1.0})


def test_ttc_risk_cost_is_bounded_for_tiny_positive_values() -> None:
    """TTC risk source should not create unbounded rewards for tiny positive TTC."""
    costs = step_safety_costs(
        {"meta": {"time_to_collision": 1e-12}},
        [_spec("ttc", "ttc_risk")],
    )

    assert costs == {"ttc": 10000.0}
