"""Focused tests for the emergent-phenomena demonstration harness (issue #5149).

These tests exercise the substrate-faithful scenario builders, the runner, and
the order-parameter metrics on CPU. They do NOT run the full multi-scenario
demonstration (that is exercised by the generation script); they use small,
fast scenarios to keep the suite cheap and to make failure reasons precise.
"""

from __future__ import annotations

import numpy as np
import pysocialforce as pysf
import pytest

from robot_sf.research.emergent_phenomena import (
    DEFAULT_LITERATURE_DESIRED_SPEED,
    DEFAULT_RELEASED_DESIRED_SPEED,
    INITIAL_SPEED_RELEASED,
    LITERATURE_CALIBRATION,
    MAX_SPEED_MULTIPLIER,
    RELEASED_DEFAULT_CALIBRATION,
    ScenarioConfig,
    build_bidirectional_corridor,
    build_high_density_exit,
    build_narrow_doorway,
    doorway_oscillation,
    exit_arching,
    lane_purity,
    lane_segregation_index,
    released_default_config,
    run_emergent_phenomena_demo,
    run_scenario,
)

# --------------------------------------------------------------------------- #
# Builder contract tests
# --------------------------------------------------------------------------- #


def _corridor_config(**overrides) -> ScenarioConfig:
    base = {
        "name": "bidirectional_corridor",
        "length": 12.0,
        "half_width": 2.0,
        "n_pedestrians": 8,
        "seed": 123,
        "n_steps": 20,
    }
    base.update(overrides)
    return ScenarioConfig(**base)


def test_released_default_speed_matches_substrate_derivation():
    """The released-default desired speed must equal the documented slow regime."""
    # fast-pysf: initial_speed=0.5, max_speed_multiplier=1.3 -> 0.65 m/s.
    assert INITIAL_SPEED_RELEASED == pytest.approx(0.5)
    assert MAX_SPEED_MULTIPLIER == pytest.approx(1.3)
    assert DEFAULT_RELEASED_DESIRED_SPEED == pytest.approx(0.65)
    assert DEFAULT_LITERATURE_DESIRED_SPEED == pytest.approx(1.3)


def test_released_default_config_pins_substrate_defaults():
    """released_default_config must reproduce the fast-pysf dataclass defaults."""
    cfg = released_default_config()
    assert cfg.scene_config.enable_group is True
    assert cfg.scene_config.agent_radius == pytest.approx(0.35)
    assert cfg.scene_config.dt_secs == pytest.approx(0.1)
    assert cfg.scene_config.max_speed_multiplier == pytest.approx(1.3)
    assert cfg.social_force_config.factor == pytest.approx(5.1)
    assert cfg.obstacle_force_config.factor == pytest.approx(10.0)


@pytest.mark.parametrize(
    "builder_name,scenario",
    [
        ("corridor", "bidirectional_corridor"),
        ("doorway", "narrow_doorway"),
        ("exit", "high_density_exit"),
    ],
)
def test_builders_produce_valid_state_and_balanced_flows(builder_name, scenario):
    """Each builder must emit a well-shaped state and matching obstacles."""
    builders = {
        "corridor": build_bidirectional_corridor,
        "doorway": build_narrow_doorway,
        "exit": build_high_density_exit,
    }
    cfg = ScenarioConfig(
        name=scenario,
        length=12.0,
        half_width=2.0,
        n_pedestrians=10,
        seed=7,
        n_steps=5,
        extra={"door_x": 6.0, "door_half_width": 0.6, "exit_half_width": 0.6},
    )
    cal = RELEASED_DEFAULT_CALIBRATION
    state, obstacles, dirs = builders[builder_name](cfg, cal)
    # State is (N, 7) with finite values.
    assert state.shape == (10, 7)
    assert np.all(np.isfinite(state))
    # Desired directions are unit-ish along x.
    assert dirs.shape == (10, 2)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-9)
    # Obstacles are non-empty line segments for all walled scenarios.
    assert len(obstacles) >= 2
    # Corridor and doorway have both directions; exit is unidirectional (+x).
    if scenario == "high_density_exit":
        assert np.all(dirs[:, 0] > 0)
    else:
        assert (dirs[:, 0] > 0).sum() > 0
        assert (dirs[:, 0] < 0).sum() > 0


def test_builders_realize_target_desired_speed_via_substrate():
    """Spawn velocity magnitude must be desired/max_speed_multiplier so that the
    substrate's own max_speeds derivation reproduces the target desired speed."""
    cfg = _corridor_config(n_pedestrians=6)
    state, _, _ = build_bidirectional_corridor(cfg, RELEASED_DEFAULT_CALIBRATION)
    init_speeds = np.linalg.norm(state[:, 2:4], axis=1)
    expected = DEFAULT_RELEASED_DESIRED_SPEED / MAX_SPEED_MULTIPLIER
    np.testing.assert_allclose(init_speeds, expected)
    # Sanity: building a real simulator yields max_speeds ~= desired.
    sim = pysf.Simulator(state=state.copy(), obstacles=[], config=released_default_config())
    np.testing.assert_allclose(sim.peds.max_speeds, DEFAULT_RELEASED_DESIRED_SPEED)


# --------------------------------------------------------------------------- #
# Runner tests
# --------------------------------------------------------------------------- #


def test_run_scenario_records_full_trajectory_shape():
    """run_scenario must record (n_steps+1) frames of positions/velocities."""
    cfg = _corridor_config(n_pedestrians=8, n_steps=15)
    result = run_scenario(cfg, RELEASED_DEFAULT_CALIBRATION)
    assert result.trajectory.positions.shape == (16, 8, 2)
    assert result.trajectory.velocities.shape == (16, 8, 2)
    assert result.trajectory.times.shape == (16,)
    assert result.trajectory.times[1] == pytest.approx(0.1)
    assert np.all(np.isfinite(result.trajectory.positions))


def test_run_scenario_computes_finite_order_parameters():
    """Order parameters for each phenomenon must be finite floats."""
    corridor = run_scenario(_corridor_config(n_steps=30), RELEASED_DEFAULT_CALIBRATION)
    for key in ("lane_segregation_index", "lane_purity"):
        assert key in corridor.order_parameters
        val = corridor.order_parameters[key]
        assert isinstance(val, float)
        assert np.isfinite(val)
        assert 0.0 <= val <= 1.0

    doorway_cfg = ScenarioConfig(
        name="narrow_doorway",
        length=10.0,
        half_width=2.0,
        n_pedestrians=8,
        seed=3,
        n_steps=40,
        extra={"door_x": 5.0, "door_half_width": 0.6},
    )
    doorway = run_scenario(doorway_cfg, RELEASED_DEFAULT_CALIBRATION)
    for key in ("oscillation_flips", "throughput_peds_per_sec", "mean_burst_windows"):
        assert key in doorway.order_parameters
        assert np.isfinite(doorway.order_parameters[key])
        assert doorway.order_parameters["throughput_peds_per_sec"] >= 0.0

    exit_cfg = ScenarioConfig(
        name="high_density_exit",
        length=8.0,
        half_width=3.0,
        n_pedestrians=16,
        seed=3,
        n_steps=40,
        extra={"exit_half_width": 0.6},
    )
    exit_run = run_scenario(exit_cfg, RELEASED_DEFAULT_CALIBRATION)
    for key in ("exit_density_ratio", "arch_lateral_spread"):
        assert key in exit_run.order_parameters
        assert np.isfinite(exit_run.order_parameters[key])
        assert exit_run.order_parameters["exit_density_ratio"] >= 0.0


def test_run_scenario_is_deterministic_given_seed():
    """Two identical runs must produce byte-identical trajectories."""
    cfg = _corridor_config(n_pedestrians=8, n_steps=20)
    a = run_scenario(cfg, RELEASED_DEFAULT_CALIBRATION)
    b = run_scenario(cfg, RELEASED_DEFAULT_CALIBRATION)
    np.testing.assert_allclose(a.trajectory.positions, b.trajectory.positions)
    assert a.order_parameters == b.order_parameters


def test_run_scenario_unknown_name_raises():
    """An unknown scenario name must fail closed with a clear error."""
    bad = ScenarioConfig(
        name="nope", length=10.0, half_width=2.0, n_pedestrians=4, seed=1, n_steps=5
    )
    with pytest.raises(ValueError, match="Unknown scenario"):
        run_scenario(bad, RELEASED_DEFAULT_CALIBRATION)


def test_literature_calibration_runs_faster_than_released():
    """Mean realized speed under the literature calibration must exceed released."""
    cfg = _corridor_config(n_pedestrians=8, n_steps=20)
    released = run_scenario(cfg, RELEASED_DEFAULT_CALIBRATION)
    literature = run_scenario(cfg, LITERATURE_CALIBRATION)
    assert literature.max_speeds.mean() > released.max_speeds.mean()


# --------------------------------------------------------------------------- #
# Order-parameter unit tests
# --------------------------------------------------------------------------- #


def _make_record(positions: np.ndarray, dirs_x: np.ndarray, dt: float = 0.1):
    from robot_sf.research.emergent_phenomena import TrajectoryRecord

    n = positions.shape[1]
    velocities = np.zeros_like(positions)
    desired = np.zeros((n, 2))
    desired[:, 0] = dirs_x
    times = np.arange(positions.shape[0]) * dt
    return TrajectoryRecord(
        positions=positions, velocities=velocities, desired_directions=desired, times=times, dt=dt
    )


def _static_record(ys: np.ndarray, dirs_x: np.ndarray, n_frames: int = 6, dt: float = 0.1):
    """Build a trajectory record where peds hold constant y (x=0) for n_frames."""
    n = len(ys)
    # positions shape (T, N, 2): x stays 0, y stays ys.
    positions = np.zeros((n_frames, n, 2), dtype=float)
    positions[:, :, 1] = ys[None, :]
    return _make_record(positions, dirs_x, dt=dt)


def test_lane_segregation_index_perfect_separation_is_high():
    """Perfectly separated +x/-x bands must give a high segregation index."""
    # 10 peds: +x all at y=+1.0, -x all at y=-1.0, held constant over time.
    n_half = 5
    ys = np.concatenate([np.full(n_half, 1.0), np.full(n_half, -1.0)])
    dirs_x = np.concatenate([np.ones(n_half), -np.ones(n_half)])
    rec = _static_record(ys, dirs_x)
    assert lane_segregation_index(rec) > 0.9


def test_lane_segregation_index_mixed_flow_is_low():
    """Perfectly intermixed +x/-x at the same y must give a low index."""
    dirs_x = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    ys = np.zeros_like(dirs_x)  # all same y -> no correlation possible
    rec = _static_record(ys, dirs_x)
    assert lane_segregation_index(rec) <= 0.05


def test_lane_purity_in_range_and_monotone_with_separation():
    """Lane purity must be in [0,1] and increase with separation."""
    dirs_x = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    ys_sep = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    ys_mix = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    p_sep = lane_purity(_static_record(ys_sep, dirs_x))
    p_mix = lane_purity(_static_record(ys_mix, dirs_x))
    assert 0.0 <= p_sep <= 1.0
    assert 0.0 <= p_mix <= 1.0
    assert p_sep > p_mix


def test_doorway_oscillation_counts_flips_and_throughput():
    """A synthetic alternating-crossing trajectory must register flips and flow."""
    # 4 peds at door region; alternate which side advances past door_x=5 over time.
    n_steps = 40
    positions = np.full((n_steps, 4, 2), 5.0)
    # ped0,1 drift +x (cross to x>5); ped2,3 drift -x (cross to x<5) over windows.
    for t in range(n_steps):
        positions[t, 0, 0] = 5.0 - 1.0 + t * 0.2  # crosses to >5 around t=5
        positions[t, 1, 0] = 5.0 - 1.0 + t * 0.2
        positions[t, 2, 0] = 5.0 + 1.0 - t * 0.2  # crosses to <5 around t=5
        positions[t, 3, 0] = 5.0 + 1.0 - t * 0.2
    dirs_x = np.array([1.0, 1.0, -1.0, -1.0])
    rec = _make_record(positions, dirs_x)
    out = doorway_oscillation(rec, door_x=5.0, window_steps=5)
    assert out["throughput_peds_per_sec"] > 0.0
    assert out["oscillation_flips"] >= 0.0


def test_exit_arching_high_local_density_raises_ratio():
    """A static cluster near the exit must give a density ratio > 1."""
    # 20 peds: half stacked within exit_radius of exit_x=10, half spread in bulk.
    rng = np.random.default_rng(0)
    near = np.column_stack(
        [
            np.full(10, 9.5),
            rng.uniform(-1.0, 1.0, 10),
        ]
    )
    far = np.column_stack(
        [
            rng.uniform(0.0, 5.0, 10),
            rng.uniform(-3.0, 3.0, 10),
        ]
    )
    static = np.vstack([near, far])[None, ...]  # (1, N, 2)
    static = np.tile(static, (4, 1, 1)).astype(float)
    dirs_x = np.ones(20)
    rec = _make_record(static, dirs_x)
    out = exit_arching(rec, exit_x=10.0, exit_radius=1.5)
    assert out["exit_density_ratio"] > 1.0


# --------------------------------------------------------------------------- #
# Top-level demo smoke test (cheap: tiny scenarios)
# --------------------------------------------------------------------------- #


def test_run_emergent_phenomena_demo_smoke():
    """The top-level demo must run end-to-end and return a well-formed report."""
    scenarios = [
        ScenarioConfig(
            name="bidirectional_corridor",
            length=10.0,
            half_width=2.0,
            n_pedestrians=6,
            seed=5149,
            n_steps=20,
        ),
        ScenarioConfig(
            name="narrow_doorway",
            length=8.0,
            half_width=2.0,
            n_pedestrians=6,
            seed=5149,
            n_steps=20,
            extra={"door_x": 4.0, "door_half_width": 0.6},
        ),
        ScenarioConfig(
            name="high_density_exit",
            length=6.0,
            half_width=3.0,
            n_pedestrians=12,
            seed=5149,
            n_steps=20,
            extra={"exit_half_width": 0.6},
        ),
    ]
    calibrations = [RELEASED_DEFAULT_CALIBRATION, LITERATURE_CALIBRATION]
    report = run_emergent_phenomena_demo(scenarios=scenarios, calibrations=calibrations)
    # 3 scenarios x 2 calibrations = 6 results.
    assert len(report.results) == 6
    assert report.substrate_version == pysf.__version__
    assert "scene" in report.config_json
    assert "social_force" in report.config_json
    for result in report.results:
        assert result.order_parameters
        assert result.calibration in calibrations
