"""Check the risk model, plant rollout and map-runner binding on development inputs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from scipy.stats import ncx2
from shapely.geometry import Polygon

from robot_sf.benchmark.map_runner.map_runner import build_map_policy
from robot_sf.planner.clearance_geometry import StaticObstacleClearance
from robot_sf.planner.dra_mppi import DRAMPPIAdapter
from robot_sf.planner.dra_mppi_risk import joint_collision_probabilities
from robot_sf.planner.drive_rollout import native_drive_rollout, native_drive_rollout_batch
from robot_sf.robot.differential_drive import DifferentialDriveSettings

_DEV_SEED = 1031


def test_joint_collision_probability_integrates_each_pedestrian_first() -> None:
    """Two individually sub-threshold pedestrians can exceed the joint threshold."""
    radius = 1.4
    sigma = 0.09
    means = np.array([[[1.56, 0.0]], [[0.0, -1.56]]])
    rollout = np.array([[[0.0, 0.0]], [[0.05, 0.02]], [[-0.1, 0.0]]])
    shifts = np.random.default_rng(_DEV_SEED).random((24, 1, 2))
    estimate = np.mean(
        [
            joint_collision_probabilities(
                rollout, means, np.array([sigma]), radius, shift, 160, 125
            )
            for shift in shifts
        ],
        axis=0,
    )
    # For a Gaussian with isotropic covariance, the scaled squared separation
    # follows a noncentral chi-square distribution with two degrees of freedom.
    noncentrality = ((rollout[:, None] - means[None, :]) ** 2).sum(axis=-1) / sigma**2
    marginal = ncx2.cdf((radius / sigma) ** 2, df=2, nc=noncentrality)
    expected = 1.0 - np.prod(1.0 - marginal, axis=1)
    assert (marginal[0] < 0.05).all()
    assert estimate[0, 0] > 0.05
    np.testing.assert_allclose(estimate, expected, atol=0.006, rtol=0.0)


def test_batched_native_drive_matches_scalar_and_brakes() -> None:
    """Batched forecasts preserve the drive's wheel integration and asymmetric braking."""
    drive = DifferentialDriveSettings(max_linear_decel=1.6, max_angular_accel=0.7)
    sequence = np.random.default_rng(_DEV_SEED).normal(0.6, 0.8, size=(5, 8, 2))
    batch = native_drive_rollout_batch(sequence, drive, 0.8, -0.24, 0.1, braking_tail=True)
    for index, controls in enumerate(sequence):
        original = native_drive_rollout(controls, drive, 0.8, -0.24, 0.1)
        for output, expected in zip(batch, original, strict=True):
            np.testing.assert_allclose(output[index, : len(expected)], expected, atol=1e-14, rtol=0)
        assert abs(batch[2][index, -1, 0]) <= 1e-9


def test_batched_clearance_preserves_sweep_and_signed_solid_depth() -> None:
    """Endpoints in free space can have a crossing motion segment."""
    polygon = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
    geometry = StaticObstacleClearance([[(0, 0), (0, 5)]], [polygon])
    previous = np.array([[-1.0, 1.0], [3.0, 1.0], [6.0, 0.0], [1.0, 3.0]])
    positions = np.array([[1.0, 1.0], [3.0, 3.0], [6.0, 0.0], [5.0, 3.0]])
    clear = geometry.rollout_clearance(np.stack((previous, positions), axis=1), radius=1.0)
    np.testing.assert_allclose(clear, [-1.0, -2.0, np.sqrt(8.0) - 1.0, -1.0])


def _environment(half_width=1.3):
    walls = [[(-2.0, y), (10.0, y)] for y in (-half_width, half_width)]
    return SimpleNamespace(
        _get_static_grid_obstacles=lambda: (walls, []),
        simulator=SimpleNamespace(
            robots=[SimpleNamespace(config=DifferentialDriveSettings())],
        ),
    )


def _observation(peds=()) -> dict:
    return {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0, 0.0]},
        "goal": {"current": [8.0, 0.0], "next": [0.0, 0.0]},
        "pedestrians": {"positions": peds, "count": [len(peds)], "radius": [0.4]},
        "sim": {"timestep": [0.1]},
    }


def test_registered_dra_policy_uses_risk_and_restarts_its_warm_start() -> None:
    """The first command on a clear corridor repeats after reset and yields to contact risk."""
    policy, meta = build_map_policy(
        "dra_mppi",
        {"allow_testing_algorithms": True, "random_seed": _DEV_SEED},
        robot_kinematics="differential_drive",
    )
    assert meta["canonical_algorithm"] == "dra_mppi"
    assert isinstance(vars(policy)["_planner_adapter"], DRAMPPIAdapter)
    vars(policy)["_planner_bind_env"](_environment())
    first = policy(_observation())
    assert 0 < first[0] <= 1.5
    assert abs(first[1]) <= 1.0

    vars(policy)["_planner_reset"]()
    near = policy(_observation([(1.54, 0.0)]))
    assert near[0] < first[0]
    vars(policy)["_planner_reset"]()
    assert policy(_observation()) == first


def _corridor_policy():
    source = Path(__file__).resolve().parents[2] / "configs/algos/dra_mppi_r2.yaml"
    policy, _ = build_map_policy(
        "dra_mppi",
        yaml.safe_load(source.read_text()),
        robot_kinematics="differential_drive",
    )
    vars(policy)["_planner_bind_env"](_environment(half_width=2.2))
    return policy


def test_corridor_profile_steers_away_from_a_waiting_walker() -> None:
    """A visible walker ahead changes the early approach on either side of the aisle."""
    policy = _corridor_policy()
    for lateral in (-0.4, 0.4):
        vars(policy)["_planner_reset"]()
        speed, turn = policy(_observation([(6.5, lateral)]))
        assert speed > 0.0
        assert turn * np.sign(lateral) < -0.25


def test_corridor_profile_yields_until_a_risk_feasible_pass_returns() -> None:
    """A central oncoming walker closes every pass, while a clear lane restores motion."""
    policy = _corridor_policy()
    observation = _observation([(2.9, 0.1)])
    observation["robot"]["speed"] = [0.8, 0.0]
    observation["pedestrians"]["velocities"] = [(-1.2, 0.0)]
    speed, turn = policy(observation)
    diagnostics = vars(policy)["_planner_adapter"].diagnostics()
    assert diagnostics["all_rollouts_over_risk_steps"] == 1
    assert speed == 0.0
    assert turn < -0.25
    assert diagnostics["corridor_yield_steps"] == 1
    assert diagnostics["yield_guard_rejections"] == 0
    assert policy(_observation())[0] > 0.0


def test_corridor_braking_guard_rejects_contact_during_the_stop() -> None:
    """Initial separation alone cannot justify a steered stop with a moving walker."""
    policy = _corridor_policy()
    observation = _observation([(1.75, 0.1)])
    observation["robot"]["speed"] = [1.1, 0.0]
    observation["pedestrians"]["velocities"] = [(-1.2, 0.0)]
    assert policy(observation) == (0.0, 0.0)
    assert vars(policy)["_planner_adapter"].diagnostics()["yield_guard_rejections"] == 1


def _crossing_observation(x, pedestrian=None, velocity=None):
    observation = _observation([] if pedestrian is None else [pedestrian])
    observation["robot"]["position"] = [x, 0.0]
    observation["robot"]["speed"] = [1.4, 0.0]
    observation["goal"]["current"] = [12.0, 0.0]
    if velocity is not None:
        observation["pedestrians"]["velocities"] = [velocity]
    return observation


def _approaching_crossing_policy(*, opening=True):
    policy, _ = build_map_policy(
        "dra_mppi",
        {"allow_testing_algorithms": True, "crossing_yield": True, "random_seed": 7023},
        robot_kinematics="differential_drive",
    )
    extents = [(-3.0, 1.0), (6.2, 15.0)] if opening else [(-3.0, 15.0)]
    walls = [[(start, y), (end, y)] for start, end in extents for y in (-2.2, 2.2)]
    vars(policy)["_planner_bind_env"](
        SimpleNamespace(
            _get_static_grid_obstacles=lambda: (walls, []),
            simulator=SimpleNamespace(robots=[SimpleNamespace(config=DifferentialDriveSettings())]),
        )
    )
    policy(_crossing_observation(-0.28))
    policy(_crossing_observation(-0.14))
    return policy


def test_crossing_profile_yields_and_resumes_progress_when_the_gap_opens() -> None:
    """A close transverse walker requires yielding before entering the crossing."""
    policy = _approaching_crossing_policy()
    walker = _crossing_observation(0.0, (1.6, -1.5), (0.0, 0.9))
    assert policy(walker) == (0.0, 0.0)

    clear = _crossing_observation(0.0)
    clear["robot"]["speed"] = [0.0, 0.0]
    assert policy(clear)[0] > 0.0

    corridor = _approaching_crossing_policy(opening=False)
    assert corridor(walker)[0] > 0.0


def test_crossing_profile_uses_a_sample_that_can_brake_behind_a_walker() -> None:
    """An eligible sampled approach keeps moving when the blended command is too fast."""
    policy = _approaching_crossing_policy()
    walker = _crossing_observation(0.0, (2.8, -2.4), (0.0, 0.9))
    assert 0.0 < policy(walker)[0] < 1.2
