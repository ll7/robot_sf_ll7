"""Tests for uncertainty-aware safety buffers and intrusion metrics (issue #3974)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.uncertainty_safety import (
    ADAPTIVE_CONFORMAL_BUFFERS_SCHEMA,
    INTRUSION_METRICS_SCHEMA,
    SPLIT_CONFORMAL_RADIUS_SCHEMA,
    AdaptiveConformalConfig,
    adaptive_conformal_buffers,
    compute_intrusion_metrics,
    residual_magnitudes,
    split_conformal_radius,
)

# --------------------------------------------------------------------------- residuals


def test_residual_magnitudes_matches_euclidean_norm() -> None:
    """Residual magnitude equals the Euclidean distance between the two point sets."""
    predicted = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    actual = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 0.0]])
    residuals = residual_magnitudes(predicted, actual)
    np.testing.assert_allclose(residuals, [0.0, 3.0, 3.0])


def test_residual_magnitudes_rejects_shape_mismatch() -> None:
    """Mismatched predicted/actual shapes fail closed."""
    with pytest.raises(ValueError, match="same shape"):
        residual_magnitudes(np.zeros((3, 2)), np.zeros((2, 2)))


# ------------------------------------------------------------------- split conformal


def test_split_conformal_radius_finite_sample_index() -> None:
    """Radius is the finite-sample-corrected k-th smallest calibration score."""
    # n=9 residuals 1..9; k = ceil(10 * 0.9) = 9 -> 9th smallest = 9.0.
    scores = np.arange(1.0, 10.0)
    assert split_conformal_radius(scores, coverage_target=0.9) == 9.0


def test_split_conformal_radius_infinite_when_target_unreachable() -> None:
    """When the sample cannot certify the target the radius is +inf."""
    # n=5, k = ceil(6 * 0.9) = 6 > 5 -> cannot certify -> +inf.
    scores = np.arange(1.0, 6.0)
    assert split_conformal_radius(scores, coverage_target=0.9) == math.inf


def test_split_conformal_radius_covers_target_on_holdout() -> None:
    """On exchangeable data the calibrated radius covers held-out samples near the target."""
    rng = np.random.default_rng(1234)
    target = 0.9
    covered = 0
    trials = 400
    for _ in range(trials):
        calibration = np.abs(rng.normal(size=200))
        holdout = abs(float(rng.normal()))
        radius = split_conformal_radius(calibration, coverage_target=target)
        covered += int(holdout <= radius)
    empirical = covered / trials
    # Guarantee is coverage >= target; allow a modest sampling band.
    assert empirical >= target - 0.05
    assert empirical <= target + 0.08


@pytest.mark.parametrize("bad_target", [0.0, 1.0, -0.1, 1.5])
def test_split_conformal_radius_rejects_bad_target(bad_target: float) -> None:
    """Coverage targets outside (0, 1) fail closed."""
    with pytest.raises(ValueError, match="coverage_target"):
        split_conformal_radius(np.arange(1.0, 5.0), coverage_target=bad_target)


def test_split_conformal_radius_rejects_empty() -> None:
    """An empty calibration set fails closed."""
    with pytest.raises(ValueError, match="non-empty"):
        split_conformal_radius(np.array([]), coverage_target=0.9)


def test_split_conformal_schema_constant() -> None:
    """The split-conformal schema identifier is stable and versioned."""
    assert SPLIT_CONFORMAL_RADIUS_SCHEMA == "uncertainty_safety.split_conformal_radius.v1"


# --------------------------------------------------------------- adaptive conformal


def test_adaptive_conformal_emits_from_min_history() -> None:
    """Buffers are emitted from ``min_history`` onward, one per remaining step."""
    residuals = np.arange(1.0, 11.0)
    cfg = AdaptiveConformalConfig(coverage_target=0.9, step_size=0.05, min_history=3)
    result = adaptive_conformal_buffers(residuals, config=cfg)
    assert result.indices[0] == 3
    assert result.indices.tolist() == list(range(3, 10))
    assert result.radii.shape == result.covered.shape == result.alphas.shape
    assert result.schema == ADAPTIVE_CONFORMAL_BUFFERS_SCHEMA


def test_adaptive_conformal_coverage_approaches_target_under_drift() -> None:
    """ACI tracks the target coverage even when the residual scale drifts upward."""
    # This is the SoNIC-style non-stationary setting where a fixed radius would fail.
    rng = np.random.default_rng(7)
    steps = 4000
    scale = np.linspace(1.0, 4.0, steps)
    residuals = np.abs(rng.normal(size=steps)) * scale
    cfg = AdaptiveConformalConfig(coverage_target=0.9, step_size=0.05, window=200, min_history=50)
    result = adaptive_conformal_buffers(residuals, config=cfg)
    assert abs(result.empirical_coverage - 0.9) < 0.03


def test_adaptive_conformal_higher_target_yields_larger_radii() -> None:
    """A higher coverage target yields larger mean radii and no lower empirical coverage."""
    rng = np.random.default_rng(3)
    residuals = np.abs(rng.normal(size=1000))
    low = adaptive_conformal_buffers(
        residuals, config=AdaptiveConformalConfig(coverage_target=0.8, window=200)
    )
    high = adaptive_conformal_buffers(
        residuals, config=AdaptiveConformalConfig(coverage_target=0.95, window=200)
    )
    assert np.mean(high.radii) >= np.mean(low.radii)
    assert high.empirical_coverage >= low.empirical_coverage - 1e-9


def test_adaptive_conformal_rejects_bad_config() -> None:
    """Invalid ACI configuration and non-1-D residuals fail closed."""
    with pytest.raises(ValueError, match="step_size"):
        adaptive_conformal_buffers(np.arange(5.0), config=AdaptiveConformalConfig(step_size=0.0))
    with pytest.raises(ValueError, match="min_history"):
        adaptive_conformal_buffers(np.arange(5.0), config=AdaptiveConformalConfig(min_history=0))
    with pytest.raises(ValueError, match="1-D"):
        adaptive_conformal_buffers(np.zeros((3, 2)))


# ------------------------------------------------------------------ intrusion metrics


def _line(points: list[tuple[float, float]]) -> np.ndarray:
    """Build a ``(T, 2)`` trajectory array from a list of points."""
    return np.asarray(points, dtype=np.float64)


def test_intrusion_metrics_clear_run_has_zero_intrusion() -> None:
    """A run with the robot far from the pedestrian reports zero intrusion everywhere."""
    robot = _line([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)])
    peds = robot.reshape(3, 1, 2) + np.array([100.0, 0.0])
    preds = peds.copy()
    metrics = compute_intrusion_metrics(robot, peds, preds, 0.5, base_radius=1.0)
    assert metrics.current_position_intrusion_time_ratio == 0.0
    assert metrics.predicted_trajectory_intrusion_time_ratio == 0.0
    assert metrics.uncertainty_buffer_intrusion_time_ratio == 0.0
    assert metrics.cumulative_intrusion_depth == 0.0
    assert metrics.max_intrusion_depth == 0.0
    assert metrics.num_steps == 3
    assert metrics.schema == INTRUSION_METRICS_SCHEMA


def test_intrusion_metrics_buffer_widens_intrusion_ratio() -> None:
    """Adding a conformal buffer turns a near-miss into a counted buffer intrusion."""
    # Robot sits 1.2 m from the predicted pedestrian position every step.
    robot = _line([(0.0, 0.0), (0.0, 0.0)])
    peds = np.array([[[1.2, 0.0]], [[1.2, 0.0]]])
    preds = peds.copy()
    base_radius = 1.0

    # Bare radius (1.0) does not reach 1.2 -> no predicted intrusion.
    tight = compute_intrusion_metrics(robot, peds, preds, 0.0, base_radius=base_radius)
    assert tight.predicted_trajectory_intrusion_time_ratio == 0.0
    assert tight.uncertainty_buffer_intrusion_time_ratio == 0.0

    # Adding a 0.5 conformal buffer -> 1.5 reach -> buffer intrusion every step.
    buffered = compute_intrusion_metrics(robot, peds, preds, 0.5, base_radius=base_radius)
    assert buffered.predicted_trajectory_intrusion_time_ratio == 0.0
    assert buffered.uncertainty_buffer_intrusion_time_ratio == 1.0
    # Penetration depth = (base + conformal) - distance = 1.5 - 1.2 = 0.3 per step.
    assert buffered.max_intrusion_depth == pytest.approx(0.3)
    assert buffered.cumulative_intrusion_depth == pytest.approx(0.6)


def test_intrusion_metrics_current_vs_predicted_differ() -> None:
    """Current-position and predicted-position intrusion are measured independently."""
    # Pedestrian currently far but predicted to be right next to the robot.
    robot = _line([(0.0, 0.0)])
    peds = np.array([[[5.0, 0.0]]])
    preds = np.array([[[0.3, 0.0]]])
    metrics = compute_intrusion_metrics(robot, peds, preds, 0.0, base_radius=1.0)
    assert metrics.current_position_intrusion_time_ratio == 0.0
    assert metrics.predicted_trajectory_intrusion_time_ratio == 1.0


def test_intrusion_metrics_per_step_per_ped_radii() -> None:
    """Per-step per-pedestrian ``(T, P)`` radii broadcast correctly."""
    robot = _line([(0.0, 0.0), (0.0, 0.0)])
    peds = np.array([[[2.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [0.0, 2.0]]], dtype=np.float64)
    preds = peds.copy()
    radii = np.array([[1.5, 0.0], [0.0, 1.5]])  # each step one ped gets a wide buffer
    metrics = compute_intrusion_metrics(robot, peds, preds, radii, base_radius=1.0)
    # Each step has one pedestrian whose buffer (1.0+1.5=2.5) reaches distance 2.0.
    assert metrics.uncertainty_buffer_intrusion_time_ratio == 1.0
    assert metrics.max_intrusion_depth == pytest.approx(0.5)


def test_intrusion_metrics_no_pedestrians_is_zero() -> None:
    """A run with zero pedestrians reports zero intrusion but preserves step count."""
    robot = _line([(0.0, 0.0), (1.0, 0.0)])
    empty = np.zeros((2, 0, 2))
    metrics = compute_intrusion_metrics(robot, empty, empty, 0.0, base_radius=1.0)
    assert metrics.uncertainty_buffer_intrusion_time_ratio == 0.0
    assert metrics.cumulative_intrusion_depth == 0.0
    assert metrics.num_steps == 2


def test_intrusion_metrics_rejects_bad_inputs() -> None:
    """Non-positive base radius, negative radii, and shape mismatches fail closed."""
    robot = _line([(0.0, 0.0)])
    peds = np.array([[[1.0, 0.0]]])
    with pytest.raises(ValueError, match="base_radius"):
        compute_intrusion_metrics(robot, peds, peds, 0.0, base_radius=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        compute_intrusion_metrics(robot, peds, peds, -0.5, base_radius=1.0)
    with pytest.raises(ValueError, match="shape"):
        compute_intrusion_metrics(robot, peds, np.zeros((2, 1, 2)), 0.0, base_radius=1.0)
