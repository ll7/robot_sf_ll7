"""Tests for pedestrian trajectory-quality diagnostic primitives."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from robot_sf.benchmark.ped_trajectory_quality import (
    TrajectoryQualityConfig,
    compute_trajectory_quality_distributions,
    ensure_json_safe,
)


def test_trajectory_quality_reports_required_metric_keys() -> None:
    """Straight-line trajectories produce stable JSON-safe diagnostic summaries."""

    positions = np.array(
        [
            [[0.0, 0.0], [0.0, 2.0]],
            [[0.5, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 2.0]],
            [[1.5, 0.0], [0.0, 2.0]],
        ]
    )

    report = compute_trajectory_quality_distributions(positions, dt_s=0.5)

    assert report["status"] == "ok"
    assert report["diagnostic_only"] is True
    assert report["thresholds_applied"] is False
    for key in (
        "speed_mps",
        "acceleration_mps2",
        "curvature_1pm",
        "turning_angle_rad",
        "pairwise_distance_m",
        "stop_frequency_hz",
        "stop_fraction",
    ):
        assert key in report
        assert report[key]["count"] >= 0

    assert report["speed_mps"]["count"] == 6
    assert report["speed_mps"]["mean"] == pytest.approx(0.5)
    assert "q05" in report["speed_mps"]
    assert report["stop_fraction"]["max"] == pytest.approx(1.0)
    json.dumps(ensure_json_safe(report))


def test_trajectory_quality_computes_turning_and_curvature() -> None:
    """A quarter-turn trajectory records nonzero turning angle and curvature."""

    positions = np.array(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 1.0]],
            [[1.0, 2.0]],
        ]
    )

    report = compute_trajectory_quality_distributions(positions, dt_s=1.0)

    assert report["turning_angle_rad"]["max"] == pytest.approx(math.pi / 2.0)
    assert report["curvature_1pm"]["max"] > 0.0


def test_trajectory_quality_handles_empty_and_no_pedestrians() -> None:
    """Empty inputs keep stable keys and explicit status instead of crashing."""

    empty_report = compute_trajectory_quality_distributions(np.empty((0, 2, 2)), dt_s=0.1)
    no_ped_report = compute_trajectory_quality_distributions(np.empty((3, 0, 2)), dt_s=0.1)

    assert empty_report["status"] == "empty"
    assert no_ped_report["status"] == "no_pedestrians"
    assert empty_report["speed_mps"]["status"] == "empty"
    assert no_ped_report["pairwise_distance_m"]["count"] == 0


def test_trajectory_quality_rejects_invalid_shapes_and_timestep() -> None:
    """Input validation fails closed for malformed synthetic trajectories."""

    with pytest.raises(ValueError, match="positions must have shape"):
        compute_trajectory_quality_distributions(np.empty((3, 2)), dt_s=0.1)

    with pytest.raises(ValueError, match="dt_s must be positive"):
        compute_trajectory_quality_distributions(np.empty((3, 2, 2)), dt_s=0.0)

    with pytest.raises(ValueError, match="velocities must have"):
        compute_trajectory_quality_distributions(
            np.empty((3, 2, 2)),
            np.empty((3, 3, 2)),
            dt_s=0.1,
        )


def test_trajectory_quality_respects_pairwise_stride_and_custom_quantiles() -> None:
    """Configuration changes only diagnostic summaries, not pass/fail behavior."""

    positions = np.array(
        [
            [[0.0, 0.0], [3.0, 0.0]],
            [[1.0, 0.0], [3.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0]],
        ]
    )
    config = TrajectoryQualityConfig(pairwise_sample_stride=2, quantiles=(0.25, 0.5, 0.75))

    report = compute_trajectory_quality_distributions(positions, dt_s=1.0, config=config)

    assert report["pairwise_distance_m"]["count"] == 2
    assert "q25" in report["speed_mps"]
    assert "q75" in report["speed_mps"]


def test_empty_trajectory_quality_report_uses_configured_quantiles() -> None:
    """Empty reports keep the same quantile keys as populated reports."""

    config = TrajectoryQualityConfig(quantiles=(0.25, 0.5, 0.75))

    report = compute_trajectory_quality_distributions(
        np.empty((0, 2, 2)),
        dt_s=0.1,
        config=config,
    )

    assert report["status"] == "empty"
    assert "q25" in report["speed_mps"]
    assert "q75" in report["speed_mps"]
    assert "q05" not in report["speed_mps"]
