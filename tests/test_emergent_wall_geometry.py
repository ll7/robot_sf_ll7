"""Fast construction-only contracts for emergent wall input order (#10056)."""

import numpy as np
import pytest

from robot_sf.research.emergent_phenomena import (
    RELEASED_DEFAULT_CALIBRATION,
    ScenarioConfig,
    build_bidirectional_corridor,
    build_high_density_exit,
    build_narrow_doorway,
)


@pytest.mark.parametrize(
    "builder,expected",
    [
        (build_bidirectional_corridor, [(-1, 13, 2, 2), (-1, 13, -2, -2)]),
        (
            build_narrow_doorway,
            [(-1, 13, 2, 2), (-1, 13, -2, -2), (5.25, 5.25, 0.7, 3), (5.25, 5.25, -0.7, -3)],
        ),
        (
            build_high_density_exit,
            [
                (-1, 14, 2, 2),
                (-1, 14, -2, -2),
                (-1, -1, -2, 2),
                (12, 12, 0.8, 3),
                (12, 12, -0.8, -3),
            ],
        ),
    ],
    ids=["custom_corridor", "offset_doorway", "custom_exit_width"],
)
def test_custom_walls_use_pysf_axis_order(builder, expected):
    """Hand-written (x1,x2,y1,y2) walls catch endpoint-order regressions."""
    config = ScenarioConfig(
        name="construction_only",
        length=12,
        half_width=2,
        n_pedestrians=8,
        seed=1001,
        n_steps=1,
        extra={"door_x": 5.25, "door_half_width": 0.7, "exit_half_width": 0.8},
    )
    # Build actual public simulator inputs, without constructing or stepping a simulator.
    _, obstacles, _ = builder(config, RELEASED_DEFAULT_CALIBRATION)
    np.testing.assert_allclose(obstacles, expected, rtol=0, atol=1e-9)
