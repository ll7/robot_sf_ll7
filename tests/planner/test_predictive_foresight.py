"""Tests for compact predictor-derived foresight features."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.predictive_foresight import (
    PredictiveForesightConfig,
    PredictiveForesightEncoder,
    predictive_foresight_config_from_source,
    predictive_foresight_spaces,
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


def test_crossing_count_only_counts_crossings_inside_front_corridor() -> None:
    """Crossing count should ignore centerline sign changes that happen outside the corridor."""
    encoder = PredictiveForesightEncoder(
        PredictiveForesightConfig(
            enabled=True,
            front_corridor_length=3.0,
            front_corridor_half_width=1.0,
        )
    )
    future = np.array(
        [
            [
                [1.0, 0.5],
                [1.5, -0.4],
                [2.0, -0.6],
            ],
            [
                [3.5, 0.8],
                [4.0, -0.8],
                [4.5, -0.9],
            ],
        ],
        dtype=np.float32,
    )
    mask = np.array([1.0, 1.0], dtype=np.float32)

    count = encoder._crossing_count(future=future, mask=mask, steps=3)
    assert count == 1.0


def test_predictive_foresight_spaces_follow_active_config() -> None:
    """Observation bounds should track the configured agent and corridor limits."""
    space = predictive_foresight_spaces(
        PredictiveForesightConfig(max_agents=23, front_corridor_half_width=2.5)
    )

    assert space["crossing_count"].high[0] == pytest.approx(23.0)
    assert np.allclose(space["gap_scores"].high, np.array([2.5, 2.5], dtype=np.float32))


def test_predictive_foresight_config_preserves_default_model_id_for_empty_override() -> None:
    """Empty source model ids should not overwrite the default predictor id."""

    class _Source:
        predictive_foresight_enabled = True
        predictive_foresight_model_id = ""

    cfg = predictive_foresight_config_from_source(_Source())
    assert cfg.model_id == PredictiveForesightConfig().model_id


def test_predictive_foresight_gap_and_crossing_features_are_clipped_to_space_bounds() -> None:
    """Encoded features should stay inside the declared observation-space bounds."""
    encoder = PredictiveForesightEncoder(
        PredictiveForesightConfig(
            enabled=True,
            max_agents=3,
            front_corridor_half_width=0.5,
        )
    )

    encoder._adapter = type(
        "_Adapter",
        (),
        {
            "_build_model_input": staticmethod(
                lambda _obs: (
                    np.zeros((1, 4), dtype=np.float32),
                    np.ones((4,), dtype=np.float32),
                    np.zeros((2,), dtype=np.float32),
                    0.0,
                )
            ),
            "_predict_trajectories": staticmethod(
                lambda _state, _mask: np.zeros((4, 2, 2), dtype=np.float32)
            ),
            "_effective_rollout_steps": staticmethod(lambda **_kwargs: 2),
            "_min_predicted_distance": staticmethod(lambda **_kwargs: 100.0),
            "_socnav_fields": staticmethod(
                lambda obs: (obs["robot"], obs["goal"], obs["pedestrians"])
            ),
        },
    )()
    encoder._ttc_risk = lambda **_kwargs: 5000.0  # type: ignore[method-assign]
    encoder._crossing_count = lambda **_kwargs: 99.0  # type: ignore[method-assign]
    encoder._gap_scores = lambda **_kwargs: np.array([2.0, 3.0], dtype=np.float32)  # type: ignore[method-assign]
    encoder._flow_alignment = lambda **_kwargs: 2.0  # type: ignore[method-assign]

    features = encoder.encode(_make_obs())
    space = predictive_foresight_spaces(encoder.config)
    assert space["crossing_count"].contains(features["crossing_count"])
    assert space["gap_scores"].contains(features["gap_scores"])
    assert space["ttc_risk"].contains(features["ttc_risk"])
    assert space["flow_alignment"].contains(features["flow_alignment"])
