"""Protect seed reproducibility and tick-zero accounting for the wall-law matrix."""

import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from scripts.validation import issue_10304_wall_law_matrix as matrix


@pytest.mark.parametrize("variant", ["legacy", "physical_margin10x"])
def test_same_episode_seed_reproduces_robot_placement_and_trial_row(monkeypatch, variant):
    """Ambient RNG state must not change the robot start or any reported field."""
    placements = []
    init_simulators = matrix.init_simulators

    def capture_placement(*args, **kwargs):
        simulators = init_simulators(*args, **kwargs)
        placements.append(np.asarray(simulators[0].robot_pos).copy())
        return simulators

    monkeypatch.setattr(matrix, "init_simulators", capture_placement)
    random.seed(7)
    np.random.seed(7)
    first = matrix._run_one((variant, 1001))
    random.seed(91)
    np.random.seed(91)
    second = matrix._run_one((variant, 1001))

    np.testing.assert_array_equal(placements[0], placements[1])
    assert first == second


@pytest.mark.parametrize(
    ("start", "end", "initial", "new"),
    [([10.0, 13.5], [20.0, 12.0], 0, 1), ([20.0, 12.0], [10.0, 13.5], 1, 0)],
)
def test_overlap_classification_uses_tick_zero_before_in_place_integration(
    monkeypatch, start, end, initial, new
):
    """A live position array can enter or leave a wall without rewriting tick zero."""
    obstacle = SimpleNamespace(iter_polygons=lambda: [box(18, 9, 22, 13)])
    definition = SimpleNamespace(
        obstacles=[obstacle],
        single_pedestrians=[SimpleNamespace(trajectory=[end])],
    )
    config = SimpleNamespace(
        sim_config=SimpleNamespace(episode_step_limit=1),
        map_pool=SimpleNamespace(map_defs={"bottleneck": definition}),
    )
    positions = np.asarray([start], dtype=float)

    def step_once(_action):
        positions[:] = end

    simulator = SimpleNamespace(ped_pos=positions, step_once=step_once)
    monkeypatch.setattr(matrix, "_select_scenario", lambda: {})
    monkeypatch.setattr(matrix, "build_robot_config_from_scenario", lambda *a, **kw: config)
    monkeypatch.setattr(matrix, "init_simulators", lambda *a, **kw: [simulator])

    row = matrix._run_one(("legacy", 1001))

    assert row.initial_overlaps == initial
    assert row.new_overlaps == new
    assert row.any_overlaps == 1


@pytest.mark.parametrize("seed_spec", ["1000", "1201", "1000-1001", "1200-1201"])
def test_cli_rejects_seeds_outside_dev_band_before_dispatch(monkeypatch, tmp_path, seed_spec):
    """Out-of-band seeds must be rejected without starting any episode."""
    output = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", ["matrix", "--seeds", seed_spec, "--output-dir", str(output)])

    def unexpected_dispatch(**_kwargs):
        pytest.fail("out-of-band seed reached the worker pool")

    monkeypatch.setattr(matrix.concurrent.futures, "ProcessPoolExecutor", unexpected_dispatch)

    with pytest.raises(SystemExit) as error:
        matrix.main()
    assert error.value.code == 2
    assert not output.exists()


def test_seed_parser_accepts_both_dev_boundaries():
    """The full authorized band, including both endpoints, stays available."""
    assert matrix._parse_seed_spec("1001-1200") == list(range(1001, 1201))
