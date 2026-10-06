"""Read back real simulator walls against independent endpoint geometry (#10056)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pysocialforce as pysf
import pytest

from robot_sf.benchmark.scenario_generator import generate_scenario
from robot_sf.research.emergent_phenomena import (
    RELEASED_DEFAULT_CALIBRATION,
    ScenarioConfig,
    build_bidirectional_corridor,
    build_high_density_exit,
    build_narrow_doorway,
    released_default_config,
)


@pytest.mark.parametrize(
    "builder,expected",
    [
        (build_bidirectional_corridor, [(-1, 2, 13, 2), (-1, -2, 13, -2)]),
        (
            build_narrow_doorway,
            [(-1, 2, 13, 2), (-1, -2, 13, -2), (5, 0.7, 5, 3), (5, -0.7, 5, -3)],
        ),
        (
            build_high_density_exit,
            [
                (-1, 2, 14, 2),
                (-1, -2, 14, -2),
                (-1, -2, -1, 2),
                (12, 0.8, 12, 3),
                (12, -0.8, 12, -3),
            ],
        ),
    ],
)
def test_emergent_simulator_has_intended_walls(builder, expected):
    """Asymmetric coordinates detect permutations, including overridden jambs."""
    config = ScenarioConfig(
        name="geometry_only",
        length=12,
        half_width=2,
        n_pedestrians=8,
        seed=1001,
        n_steps=1,
        extra={"door_x": 5, "door_half_width": 0.7, "exit_half_width": 0.8},
    )
    state, obstacles, _ = builder(config, RELEASED_DEFAULT_CALIBRATION)
    sim = pysf.Simulator(state=state, obstacles=obstacles, config=released_default_config())
    np.testing.assert_allclose(sim.env.obstacles_raw[:, :4], expected, rtol=0, atol=1e-9)
    # Also verify the sampled geometry consumed by obstacle repulsion.
    for samples, (x1, y1, x2, y2) in zip(sim.env.obstacles, expected, strict=True):
        np.testing.assert_allclose(samples[[0, -1]], [(x1, y1), (x2, y2)], rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "layout,expected",
    [
        ("bottleneck", [(5, 0, 5, 2.5), (5, 3.5, 5, 6)]),
        (
            "maze",
            [(10 / 3, 0, 10 / 3, 3.6), (20 / 3, 2.4, 20 / 3, 6), (10 / 3, 3.6, 20 / 3, 3.6)],
        ),
    ],
)
def test_generated_simulator_has_intended_walls(layout, expected):
    """Keep public endpoint tuples and the simulator's actual walls consistent."""
    scenario = generate_scenario({"obstacle": layout}, seed=1001)
    np.testing.assert_allclose(scenario.obstacles, expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        scenario.simulator.env.obstacles_raw[:, :4], expected, rtol=0, atol=1e-9
    )


def test_force_microbenchmark_preserves_sampled_endpoints(monkeypatch):
    """The benchmark's generated walls must retain the sampled start/end points."""
    path = Path(__file__).resolve().parents[2] / "fast-pysf/benchmarks/forces_benchmark.py"
    spec = importlib.util.spec_from_file_location("wallfix_force_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    coordinates = iter([[(1, 1)], [(8, 1)], [(2, 3)], [(5, 7)]])
    monkeypatch.setattr(module.SimSettings, "rand_2d_coords", lambda *_: next(coordinates))
    # Global dynamics sampling is seeded without changing randomness in other tests.
    rng = np.random.default_rng(1001)
    monkeypatch.setattr(np.random, "normal", rng.normal)
    monkeypatch.setattr(np.random, "multivariate_normal", rng.multivariate_normal)
    settings = module.SimSettings(10, 10, 1, 1, 1, [[0.1, 0], [0, 0.1]], 1)
    config = settings.sample()
    sim = pysf.Simulator(config.initial_state, config.groups, config.obstacles)
    np.testing.assert_allclose(sim.env.obstacles_raw[:, :4], [(2, 3, 5, 7)], rtol=0, atol=1e-9)
