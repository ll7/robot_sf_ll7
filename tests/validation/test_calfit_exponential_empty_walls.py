"""Exercise the selected diagnostic force on real empty-wall array bytes."""

from types import SimpleNamespace

import numpy as np

from scripts.validation import pedestrian_validation_10074 as suite


def test_exponential_case_supports_the_suite_free_space_protocol(monkeypatch):
    def probe(state, segments, config, steps, **kwargs):
        force = suite.reused.ObstacleForce.__call__(
            SimpleNamespace(
                get_peds=lambda: np.array([[0.0, 0.0]]),
                get_obstacles=lambda: np.empty((0,)),
                config=config.obstacle_force_config,
            )
        )
        assert force.shape == (1, 2)
        assert np.array_equal(force, np.zeros((1, 2)))
        t = np.arange(201) * 0.1
        positions = np.stack([t, np.zeros_like(t)], axis=-1)[:, None, :]
        speeds = (1.29 * (1 - np.exp(-t / 0.54)))[:, None]
        return positions, speeds[1:], np.array([1.29])

    monkeypatch.setattr(suite, "protocol_simulate", probe)
    suite.run_task(
        (
            "V1",
            1001,
            "native",
            0.25,
            "radius",
            {
                "calfit": True,
                "speed_tier": "literature",
                "wall_candidate": {"family": "exponential_edge", "factor": 5.0, "offset_m": 0.02},
            },
        )
    )
