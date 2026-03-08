"""Tests for compact predictor-derived foresight features."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.predictive_foresight import (
    PredictiveForesightConfig,
    PredictiveForesightEncoder,
)


def _make_obs() -> dict[str, object]:
    return {
        "robot": {
            "position": np.array([1.0, 1.0], dtype=np.float32),
            "heading": np.array([0.0], dtype=np.float32),
            "speed": np.array([0.5, 0.0], dtype=np.float32),
            "radius": np.array([0.3], dtype=np.float32),
        },
        "goal": {
            "current": np.array([4.0, 1.0], dtype=np.float32),
            "next": np.array([4.0, 1.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": np.array([[2.0, 0.6], [2.4, 1.4]], dtype=np.float32),
            "velocities": np.array([[0.1, 0.0], [0.0, -0.1]], dtype=np.float32),
            "radius": np.array([0.3], dtype=np.float32),
            "count": np.array([2.0], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {"timestep": np.array([0.2], dtype=np.float32)},
    }


def test_predictive_foresight_features_are_finite() -> None:
    """Foresight encoder should always emit finite compact features."""
    encoder = PredictiveForesightEncoder(PredictiveForesightConfig(enabled=True))
    features = encoder.encode(_make_obs())
    assert set(features) == {
        "min_clearance",
        "ttc_risk",
        "crossing_count",
        "gap_scores",
        "flow_alignment",
        "uncertainty",
    }
    for value in features.values():
        assert np.all(np.isfinite(value))
