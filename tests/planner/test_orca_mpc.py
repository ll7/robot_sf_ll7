"""Independent dynamics, continuous clearance and default-path regression checks."""

import subprocess
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.clearance_geometry import StaticObstacleClearance, swept_relative_clearance
from robot_sf.planner.drive_rollout import native_drive_rollout, native_drive_rollout_batch
from robot_sf.planner.mppi_social import MPPISocialConfig, MPPISocialPlannerAdapter
from robot_sf.planner.orca_mpc import ORCAMPCAdapter, ORCAMPCConfig, _DriveMPCSampler
from robot_sf.robot.differential_drive import DifferentialDriveSettings


def observation(speed=(0.0, 0.0), goal=(10.0, 0.0), pedestrians=(), velocities=()):
    """Return a small real SocNav-schema fixture with no simulator future state."""
    return {
        "robot": {
            "position": np.array([0.0, 0.0]),
            "heading": np.array([0.0]),
            "speed": np.array(speed),
            "radius": np.array([0.2]),
        },
        "goal": {"current": np.array(goal), "next": np.array(goal)},
        "pedestrians": {
            "positions": np.array(pedestrians).reshape(-1, 2),
            "velocities": np.array(velocities).reshape(-1, 2),
            "count": np.array([len(pedestrians)]),
            "radius": np.array([0.2]),
        },
        "sim": {"timestep": np.array([0.1])},
    }


def sampler(*, segments=(), drive=None, **kwargs):
    """Bind a sampler to declared plant settings and exact geometry."""
    result = _DriveMPCSampler(
        ORCAMPCConfig(horizon_steps=10, sample_count=8, iterations=1, **kwargs)
    )
    result.drive = drive or DifferentialDriveSettings(radius=0.2)
    result._static_clearance = StaticObstacleClearance(segments, [])
    return result


@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
def test_batch_rollout_matches_native_plant_with_asymmetric_braking(dt):
    """Batch forecasts follow the existing independent scalar native drive."""
    drive = DifferentialDriveSettings(
        max_linear_accel=0.6, max_linear_decel=1.7, max_angular_accel=0.8
    )
    sequences = np.random.default_rng(1001).uniform([-1, -2], [3, 2], (12, 20, 2))
    positions, headings, velocities = native_drive_rollout_batch(
        sequences, drive, 0.9, -0.3, dt, braking_steps=50
    )
    for index, sequence in enumerate(sequences):
        expected = native_drive_rollout(sequence, drive, 0.9, -0.3, dt)
        for actual, truth in zip(
            (positions[index], headings[index], velocities[index]), expected, strict=True
        ):
            np.testing.assert_allclose(actual[: len(truth)], truth, rtol=0, atol=1e-12)
        assert velocities[index, -1, 0] == 0.0
    changes = np.diff(
        np.concatenate((np.tile([0.9, -0.3], (12, 1, 1)), velocities), axis=1), axis=1
    )
    assert np.max(changes[:, :, 0]) <= 0.6 * dt + 1e-12
    assert np.min(changes[:, :, 0]) >= -1.7 * dt - 1e-12
    assert np.max(np.abs(changes[:, :, 1])) <= 0.8 * dt + 1e-12


def test_swept_crossing_between_clear_endpoints_and_initial_contact():
    """Crossing and time-zero contact are rejected despite clear later samples."""
    paths = np.array([[[0, 0], [2, 0]], [[0, 3], [2, 3]]], float)
    futures = np.array([[[1, -1], [1, 1]]], float)
    np.testing.assert_allclose(
        swept_relative_clearance(paths, futures, 0.4), [-0.4, np.sqrt(5) - 0.4]
    )
    assert swept_relative_clearance(paths[:1], np.array([[[0, 0], [8, 8]]]), 0.4)[0] == -0.4
    assert np.isinf(swept_relative_clearance(paths, np.empty((0, 2, 2)), 0.4)).all()


def test_rejects_wall_in_braking_tail():
    """A safe short horizon cannot admit a command whose stopping tail hits a wall."""
    short = _DriveMPCSampler(ORCAMPCConfig(horizon_steps=1, sample_count=8, iterations=1))
    short.drive = DifferentialDriveSettings(radius=0.2)
    short._static_clearance = StaticObstacleClearance([[(0.9, -2), (0.9, 2)]], [])
    command = short.plan(observation(speed=(1.5, 0.0)))
    assert command == pytest.approx((1.4, 0.0))
    assert short.last["status"] == "no_admissible_braking"


def test_open_space_first_command_is_feasible_and_prior_is_shifted():
    """Search applies its first feasible command and retains its sequence for replanning."""
    planner = sampler()
    planner.seed_command = (2.0, 1.0)
    command = planner.plan(observation(speed=(0.3, -0.2)))
    assert 0.2 - 1e-12 <= command[0] <= 0.4 + 1e-12
    assert -0.3 - 1e-12 <= command[1] <= -0.1 + 1e-12
    assert planner.last["status"] == "admissible_sampled_sequence"
    np.testing.assert_array_equal(command, planner.previous[0])
    prior = planner.previous.copy()
    planner.plan(observation(speed=command))
    np.testing.assert_array_equal(planner.seeds[3], np.vstack((prior[1:], prior[-1:])))


def test_no_admissible_initial_overlap_and_goal_stop_remain_reachable():
    """Unresolved overlap and goal braking never claim an admissible solution."""
    planner = sampler()
    command = planner.plan(observation(speed=(0.7, 0.4), pedestrians=[(0, 0)], velocities=[(0, 0)]))
    assert command == pytest.approx((0.6, 0.3))
    assert planner.last["status"] == "no_admissible_braking"
    command = planner.plan(observation(speed=(0.7, 0.4), goal=(0, 0)))
    assert command == pytest.approx((0.6, 0.3))
    assert planner.last["status"] == "goal_reached"
    assert planner.previous is None


def test_dt_binding_and_native_dependency_fail_closed():
    """Unsupported clock, plant, unbound geometry and explicit fallback fail closed."""
    with pytest.raises(ValueError, match="fallback"):
        ORCAMPCAdapter(allow_fallback=True)
    adapter = ORCAMPCAdapter({"sample_count": 8, "iterations": 1, "horizon_steps": 10})
    with pytest.raises(ValueError, match="differential drive"):
        adapter.bind_env(SimpleNamespace())
    with pytest.raises(RuntimeError, match="binding"):
        adapter.sampler_adapter.plan(observation())
    obs = observation()
    obs["sim"]["timestep"][0] = 0.2
    with pytest.raises(ValueError, match="control period"):
        sampler().plan(obs)


def test_adapter_composition_reset_and_diagnostics(monkeypatch):
    """ORCA seeds the sampler; an episode reset removes old solutions and restarts RNG."""
    adapter = ORCAMPCAdapter({"sample_count": 8, "iterations": 1, "horizon_steps": 10})
    drive = DifferentialDriveSettings(radius=0.2)
    env = SimpleNamespace(
        env_config=SimpleNamespace(robot_config=drive), _get_static_grid_obstacles=lambda: ([], [])
    )
    adapter.bind_env(env)
    monkeypatch.setattr(adapter.primary_adapter, "plan", lambda _obs: (1.2, -0.5))
    adapter.reset(seed=1001)
    first = adapter.plan(observation())
    assert adapter.sampler_adapter.seed_command == (1.2, -0.5)
    assert adapter.diagnostics()["decision_counts"] == {"admissible_sampled_sequence": 1}
    adapter.reset(seed=1001)
    assert adapter.sampler_adapter.previous is None
    assert adapter.diagnostics()["decision_counts"] == {}
    assert adapter.plan(observation()) == first


@pytest.mark.parametrize(
    "payload",
    [
        {"rollout_dt": 0},
        {"hard_clearance": np.nan},
        {"sample_count": True},
        {"sample_count": 4},
        {"progress_escape_enabled": True},
    ],
)
def test_invalid_config_rejected(payload):
    """Invalid search parameters cannot silently degrade the research method."""
    with pytest.raises(ValueError):
        ORCAMPCConfig(**payload)


def test_default_mppi_commands_byte_equal_to_frozen_main():
    """Optional search hooks retain the exact historical RNG arithmetic and commands."""
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "show", "a52d2cb65a1f1d9850568a9b857fb53a71307405:robot_sf/planner/mppi_social.py"],
        cwd=root,
        text=True,
    )
    baseline = types.ModuleType("frozen_mppi")
    import sys

    sys.modules[baseline.__name__] = baseline
    try:
        exec(compile(source, "frozen_mppi.py", "exec"), baseline.__dict__)  # noqa: S102 - pinned local Git source
        old = baseline.MPPISocialPlannerAdapter(
            baseline.MPPISocialConfig(sample_count=8, horizon_steps=3, iterations=1)
        )
        new = MPPISocialPlannerAdapter(
            MPPISocialConfig(sample_count=8, horizon_steps=3, iterations=1)
        )
        for speed in (0.0, 0.2, 0.7):
            for obs in (
                observation(speed=(speed, 0.1)),
                observation(speed=(speed, 0.1), pedestrians=[(2, 1)], velocities=[(-0.2, 0)]),
            ):
                assert np.array(old.plan(obs)).tobytes() == np.array(new.plan(obs)).tobytes()
        assert old._rng.bit_generator.state == new._rng.bit_generator.state
    finally:
        sys.modules.pop(baseline.__name__)


def test_default_orca_source_commands_and_trace_byte_equal_to_main():
    """Native ORCA's source, commands and complete adapter traces retain main bytes."""
    import json
    import sys

    from robot_sf.planner.socnav_base import SocNavPlannerConfig
    from robot_sf.planner.socnav_orca import ORCAPlannerAdapter

    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "show", "a52d2cb65a1f1d9850568a9b857fb53a71307405:robot_sf/planner/socnav_orca.py"],
        cwd=root,
    )
    assert source == (root / "robot_sf/planner/socnav_orca.py").read_bytes()
    baseline = types.ModuleType("frozen_orca")
    sys.modules[baseline.__name__] = baseline
    try:
        exec(compile(source, "frozen_orca.py", "exec"), baseline.__dict__)  # noqa: S102 - pinned local Git source
        config = SocNavPlannerConfig(orca_adapter_trace_enabled=True)
        old = baseline.ORCAPlannerAdapter(config, allow_fallback=False)
        new = ORCAPlannerAdapter(config, allow_fallback=False)
        for heading in (0.0, 0.7, -1.1):
            obs = observation(speed=(0.3, 0.1), pedestrians=[(2, 1)], velocities=[(-0.2, 0)])
            obs["robot"]["heading"][0] = heading
            assert np.asarray(old.plan(obs)).tobytes() == np.asarray(new.plan(obs)).tobytes()
            assert json.dumps(old.adapter_trace(), sort_keys=True) == json.dumps(
                new.adapter_trace(), sort_keys=True
            )
    finally:
        sys.modules.pop(baseline.__name__)


def test_between_timestep_wall_crossing_and_cv_failure():
    """A wall intersecting only a motion interior is excluded, and malformed CV input fails."""
    planner = sampler(
        segments=[[(0.1, -2), (0.1, 2)]], drive=DifferentialDriveSettings(radius=0.01)
    )
    obs = observation(speed=(2.0, 0))
    planner._extract_state(obs)
    planner._initial_sequence(np.tile([2.0, 0], (10, 1)))
    costs = planner._batch_sequence_cost(
        batch=np.tile([2.0, 0], (1, 10, 1)),
        robot_pos=np.zeros(2),
        heading=0.0,
        goal=np.array([10, 0]),
    )
    assert np.isinf(costs[0])
    assert planner.best_sequence is None
    obs["pedestrians"]["positions"] = np.array([[2.0, 0]])
    obs["pedestrians"]["count"] = np.array([1])
    with pytest.raises(ValueError, match="complete finite CV"):
        sampler().plan(obs)
