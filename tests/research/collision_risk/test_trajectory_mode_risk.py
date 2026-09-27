"""Focused contract tests for discrete supplied-marginal mode risk."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_sf.research.collision_risk import (
    CandidateAction,
    CollisionRiskInputError,
    RiskEstimatorConfig,
    RiskSchemaError,
    TrajectoryModeRiskInput,
    estimate_trajectory_mode_risk,
)


def _config(**overrides: object) -> RiskEstimatorConfig:
    """Return a small fixed grid for mode-risk fixtures."""
    values: dict[str, object] = {
        "horizon_steps": 3,
        "dt_s": 0.1,
        "n_samples": 256,
        "robot_radius_m": 0.2,
        "pedestrian_radius_m": 0.2,
        "seed": 17,
    }
    values.update(overrides)
    return RiskEstimatorConfig(**values)


def _action(config: RiskEstimatorConfig) -> CandidateAction:
    """Return a stationary candidate on the configured waypoint grid."""
    return CandidateAction(
        action_id="stationary",
        waypoints=np.zeros((config.horizon_steps + 1, 2), dtype=float),
    )


def _mode(
    config: RiskEstimatorConfig,
    means: np.ndarray,
    covariance: np.ndarray | None = None,
    *,
    times: np.ndarray | None = None,
    actor_radius_m: float = 0.2,
) -> TrajectoryModeRiskInput:
    """Build one validated mode input without navigation-layer types."""
    steps = config.horizon_steps + 1
    covariances = (
        np.zeros((steps, 2, 2), dtype=float)
        if covariance is None
        else np.asarray(covariance, dtype=float)
    )
    return TrajectoryModeRiskInput(
        actor_id=4,
        mode_id="crossing",
        actor_radius_m=actor_radius_m,
        time_offsets_s=(np.arange(steps, dtype=float) * config.dt_s if times is None else times),
        mean_positions=means,
        covariances=covariances,
    )


def test_zero_covariance_is_deterministic_and_reports_stable_peak() -> None:
    """Zero covariance yields exact grid-time contact events and zero MC error."""
    config = _config()
    means = np.array([[1.0, 0.0], [0.3, 0.0], [1.0, 0.0], [1.0, 0.0]])
    result = estimate_trajectory_mode_risk(_action(config), _mode(config, means), config)

    assert result.per_time_probability == (0.0, 1.0, 0.0, 0.0)
    assert result.per_time_mc_standard_error == (0.0, 0.0, 0.0, 0.0)
    assert result.peak_risk_time_index == 1
    assert result.estimated_time_union_bound == 1.0
    assert result.claim_boundary.startswith("finite-sample point estimate")
    with pytest.raises(RiskSchemaError, match="first maximum risk"):
        replace(result, peak_risk_time_index=0).validate()


def test_nonzero_covariance_is_seed_reproducible_and_has_mc_error() -> None:
    """The same seed gives the same marginal samples and standard errors."""
    config = _config(n_samples=4096, seed=23)
    means = np.zeros((4, 2), dtype=float)
    covariance = np.tile(np.diag([0.04, 0.04]), (4, 1, 1))
    mode = _mode(config, means, covariance)

    first = estimate_trajectory_mode_risk(_action(config), mode, config)
    second = estimate_trajectory_mode_risk(_action(config), mode, config)

    assert first == second
    assert any(error > 0.0 for error in first.per_time_mc_standard_error)
    # For a centered isotropic Gaussian and circular contact region, the exact
    # marginal is Rayleigh: P(r <= R) = 1 - exp(-R^2 / (2 sigma^2)).
    expected = 1.0 - np.exp(-(0.4**2) / (2.0 * 0.04))
    assert all(value == pytest.approx(expected, abs=0.02) for value in first.per_time_probability)


def test_time_grid_must_match_candidate_configuration() -> None:
    """A strictly increasing but misaligned grid fails before sampling."""
    config = _config()
    means = np.ones((4, 2), dtype=float)
    mode = _mode(config, means, times=np.array([0.0, 0.1, 0.2, 0.31]))

    with pytest.raises(CollisionRiskInputError, match="configured action grid"):
        estimate_trajectory_mode_risk(_action(config), mode, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mean_positions", np.ones((3, 2)), "mean_positions must have shape"),
        ("mean_positions", np.full((4, 2), np.nan), "mean_positions must contain only finite"),
        (
            "time_offsets_s",
            np.array([0.0, 0.1, np.nan, 0.3]),
            "time_offsets_s must contain only finite",
        ),
        ("covariances", np.zeros((4, 2, 3)), "covariances must have shape"),
        ("covariances", np.full((4, 2, 2), np.nan), "covariances must contain only finite"),
        (
            "covariances",
            np.tile(np.array([[1.0, 0.2], [0.1, 1.0]]), (4, 1, 1)),
            "covariances must be symmetric",
        ),
        (
            "covariances",
            np.tile(np.array([[1.0, 0.0], [0.0, -0.1]]), (4, 1, 1)),
            "covariances must be positive semidefinite",
        ),
    ],
)
def test_invalid_shape_finite_or_covariance_input_fails_closed(
    field: str, value: np.ndarray, message: str
) -> None:
    """Malformed arrays are rejected by the package-owned input contract."""
    config = _config()
    kwargs: dict[str, object] = {
        "actor_id": 4,
        "mode_id": "crossing",
        "actor_radius_m": 0.2,
        "time_offsets_s": np.arange(4, dtype=float) * config.dt_s,
        "mean_positions": np.ones((4, 2), dtype=float),
        "covariances": np.zeros((4, 2, 2), dtype=float),
    }
    kwargs[field] = value

    with pytest.raises(CollisionRiskInputError, match=message):
        TrajectoryModeRiskInput(**kwargs)


def test_negative_radius_and_nonincreasing_grid_fail_closed() -> None:
    """Physical and temporal identity invariants cannot be silently repaired."""
    config = _config()
    common = {
        "actor_id": 4,
        "mode_id": "crossing",
        "time_offsets_s": np.arange(4, dtype=float) * config.dt_s,
        "mean_positions": np.ones((4, 2), dtype=float),
        "covariances": np.zeros((4, 2, 2), dtype=float),
    }
    with pytest.raises(CollisionRiskInputError, match="radius"):
        TrajectoryModeRiskInput(actor_radius_m=-0.1, **common)
    with pytest.raises(CollisionRiskInputError, match="radius"):
        TrajectoryModeRiskInput(actor_radius_m=np.nan, **common)
    with pytest.raises(CollisionRiskInputError, match="strictly increasing"):
        TrajectoryModeRiskInput(
            actor_radius_m=0.2,
            **{**common, "time_offsets_s": np.array([0.0, 0.1, 0.1, 0.3])},
        )


def test_time_union_uses_clipped_sum_of_marginal_grid_events() -> None:
    """The aggregate is exactly ``min(1, sum(per_time_probability))``."""
    config = _config()
    means = np.array([[0.3, 0.0], [1.0, 0.0], [0.3, 0.0], [1.0, 0.0]])
    result = estimate_trajectory_mode_risk(_action(config), _mode(config, means), config)

    assert result.per_time_probability == (1.0, 0.0, 1.0, 0.0)
    assert result.estimated_time_union_bound == min(1.0, sum(result.per_time_probability))
    assert "no temporal-independence assumption" in result.claim_boundary
    assert "continuous-time" in result.claim_boundary
    assert not hasattr(_mode(config, means), "probability")
    assert not hasattr(_mode(config, means), "existence_probability")


def test_time_union_sum_is_not_an_independence_product() -> None:
    """The two-time union estimate sums marginals without temporal independence."""
    config = _config(horizon_steps=1, n_samples=8192, seed=29)
    means = np.zeros((2, 2), dtype=float)
    covariance = np.tile(np.eye(2, dtype=float) * 0.25, (2, 1, 1))
    result = estimate_trajectory_mode_risk(
        _action(config), _mode(config, means, covariance), config
    )

    independence_product = 1.0 - np.prod(1.0 - np.asarray(result.per_time_probability))
    assert result.estimated_time_union_bound < 1.0
    assert result.estimated_time_union_bound == sum(result.per_time_probability)
    assert result.estimated_time_union_bound > independence_product
    assert "no temporal-independence assumption" in result.claim_boundary
