"""Prove safety at real entry points using sentinels, never held-out episodes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.support.heldout_seed_guard import pytest_collection_modifyitems
from tests.support.seedguard_boundaries import HeldoutSeedError, check_simulation_seed

# seed-holdout: synthetic-fixture begin
# Boundary probes abort before construction/reset/step; no evaluation episode is run.


@pytest.fixture(autouse=True)
def isolate_probe_audit(monkeypatch):
    """Expected rejections in these sentinel probes are not suite violations."""
    monkeypatch.delenv("ROBOT_SF_PYTEST_SEED_AUDIT", raising=False)


def test_factory_rejects_before_rng_or_construction(monkeypatch):
    """Even a runner catching Exception cannot consume the evaluation RNG."""
    from robot_sf.gym_env import environment_factory

    calls = []
    monkeypatch.setattr(environment_factory.random, "seed", calls.append)
    with pytest.raises(HeldoutSeedError, match="environment_factory"):
        environment_factory._apply_global_seed(111)
    assert calls == []


@pytest.mark.parametrize("seed", [140, 50036])
@pytest.mark.parametrize("kind", ["robot", "pedestrian"])
def test_reset_rejects_before_reset_body_or_first_step(kind, seed, monkeypatch):
    """Uninitialized instances prove no reset work is reached; step is a spy."""
    from robot_sf.gym_env.pedestrian_env import PedestrianEnv
    from robot_sf.gym_env.robot_env import RobotEnv

    cls = RobotEnv if kind == "robot" else PedestrianEnv
    env = object.__new__(cls)
    calls = []
    monkeypatch.setattr(cls, "step", lambda *args: calls.append("step"))
    with pytest.raises(HeldoutSeedError, match=f"{cls.__name__}.reset"):
        env.reset(seed=seed)
        env.step(None)
    assert calls == []


@pytest.mark.parametrize(
    "field", ["route_spawn_seed", "desired_speed_seed", "archetype_seed", "response_law_seed"]
)
def test_simulator_rejects_before_population(field):
    """Minimal config cannot construct physics: seed must be checked first."""
    from robot_sf.sim.simulator import _build_pysf_simulation

    with pytest.raises(HeldoutSeedError, match=f"simulator.{field}"):
        _build_pysf_simulation(
            config=SimpleNamespace(**{field: 123}),
            map_def=None,
            robots=[],
            robot_pose_provider=lambda: [],
            peds_have_obstacle_forces=False,
        )


def test_map_runner_rejects_before_environment_factory():
    """Only seed exists, so any access to later runtime fields would fail."""
    from robot_sf.benchmark.map_runner.map_runner_episode import _setup_and_run_step_loop

    with pytest.raises(HeldoutSeedError, match="map_runner.episode"):
        _setup_and_run_step_loop(SimpleNamespace(seed=120))


def test_map_runner_rejects_before_policy_or_noise_setup():
    """The outer episode entry must reject before any episode-dependent setup."""
    from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode

    with pytest.raises(HeldoutSeedError, match="map_runner.episode"):
        run_map_episode({}, 50036)


@pytest.mark.parametrize("seed", [None, 110, 141, 1001, 1030])
def test_other_seeds_are_allowed(seed):
    check_simulation_seed(seed, boundary="probe")


@pytest.mark.parametrize("reason", [None, "", "   ", 42])
def test_marker_requires_nonempty_reason(reason):
    marker = SimpleNamespace(args=(), kwargs={"reason": reason})
    item = SimpleNamespace(nodeid="toy::static", iter_markers=lambda _: [marker])
    with pytest.raises(pytest.UsageError, match="requires reason"):
        pytest_collection_modifyitems([item])


@pytest.mark.heldout_seed_ok(reason="Static policy metadata; no RNG, reset, or simulation")
def test_reasoned_marker_keeps_static_band_data_usable():
    assert len(list(range(111, 141))) == 30


def test_marker_cannot_allow_real_simulation(request):
    request.node.add_marker(pytest.mark.heldout_seed_ok(reason="Sentinel probe with no simulation"))
    with pytest.raises(HeldoutSeedError):
        check_simulation_seed(111, boundary="probe")


def test_child_process_inherits_guard_before_first_step(tmp_path):
    """A child runner aborts before it can create or step an environment."""
    import subprocess
    import sys

    stepped = tmp_path / "stepped"
    code = (
        "from robot_sf.gym_env.environment_factory import make_robot_env, _apply_global_seed\n"
        "assert getattr(_apply_global_seed, '_seedguard_boundary', None) == 'environment_factory'\n"
        "env = make_robot_env(seed=111)\n"
        f"open({str(stepped)!r}, 'w').write('step reached')\n"
        "env.step(None)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "HeldoutSeedError" in result.stderr
    assert "environment_factory" in result.stderr
    assert not stepped.exists()


def test_direct_physics_pedestrian_seed_rejects_before_sampling(monkeypatch):
    """Direct physics construction must obey the same boundary as RobotEnv."""
    import numpy as np
    from pysocialforce.config import SceneConfig
    from pysocialforce.scene import PedState

    def forbidden_rng(*args, **kwargs):
        raise AssertionError("pedestrian sampler reached before guard")

    monkeypatch.setattr(np.random, "default_rng", forbidden_rng)
    config = SceneConfig(desired_speed_mean=1.0, desired_speed_seed=123)
    state = np.array([[0.0, 0.0, 0.5, 0.0, 1.0, 0.0]])
    pedestrian = object.__new__(PedState)
    with pytest.raises(HeldoutSeedError, match="pysocialforce.pedestrian_seed"):
        pedestrian.__init__(state, [], config)
    assert not hasattr(pedestrian, "default_tau")


def test_child_custom_environment_cannot_drop_guard():
    """A deliberately minimal child environment still receives session protection."""
    import subprocess
    import sys

    code = (
        "import os\n"
        "assert os.environ.get('ROBOT_SF_PYTEST_SEED_GUARD') == '1', 'guard not propagated'\n"
        "from tests.support.seedguard_boundaries import check_simulation_seed\n"
        "check_simulation_seed(111, boundary='child sentinel')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "HeldoutSeedError" in result.stderr
    assert "child sentinel" in result.stderr


@pytest.mark.parametrize("flag", ["-I", "-S", "-E", "-IS"])
def test_isolated_child_rejects_before_any_user_step(flag, tmp_path):
    """Isolation flags cannot remove the guard from a Python simulation child."""
    import subprocess
    import sys

    stepped = tmp_path / "stepped"
    code = (
        "import sys\n"
        "assert 'robot_sf' not in sys.modules, 'project imported before pinned source setup'\n"
        "guard = sys.modules['tests.support.seedguard_boundaries']\n"
        "guard.check_simulation_seed(50036, boundary='isolated child')\n"
        f"open({str(stepped)!r}, 'w').write('first step reached')\n"
    )
    result = subprocess.run(
        [sys.executable, flag, "-c", code], env={}, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "HeldoutSeedError" in result.stderr
    assert "isolated child" in result.stderr
    assert not stepped.exists()


def test_session_guard_survives_environment_flag_mutation(monkeypatch):
    """An environment-isolation fixture must not disable the active session."""
    monkeypatch.delenv("ROBOT_SF_PYTEST_SEED_GUARD")
    reached = []
    with pytest.raises(HeldoutSeedError):
        check_simulation_seed(50036, boundary="environment mutation sentinel")
        reached.append("first step")
    assert reached == []


@pytest.mark.heldout_seed_ok(
    reason="Standalone legacy RNG seed-policy check; no simulation body runs"
)
def test_static_seed_policy_marker_allows_rng_but_blocks_simulation():
    """The marker permits seed-policy references while forbidding real episodes."""
    import numpy as np

    from robot_sf.gym_env.robot_env import RobotEnv

    np.random.seed(111)
    assert np.random.get_state()[1][0] == 111
    with pytest.raises(HeldoutSeedError, match="heldout_seed_ok cannot execute"):
        object.__new__(RobotEnv).reset(seed=1001)


def test_standalone_static_rng_only_rejects_at_simulation_boundary():
    """Static arithmetic is allowed; the held-out stream cannot seed physics."""
    import numpy as np

    from robot_sf.sim.simulator import init_simulators

    np.random.seed(50036)
    assert np.random.get_state()[1][0] == 50036
    with pytest.raises(HeldoutSeedError, match="legacy.numpy.random"):
        init_simulators(None, None)


# seed-holdout: synthetic-fixture end


def test_seed_bands_matches_canonical_policy_when_available():
    from tests.support import seedguard_boundaries

    policy = pytest.importorskip("robot_sf.benchmark.seed_bands")
    assert seedguard_boundaries.resolve_held_out_seeds() == frozenset(policy.HELD_OUT_SEEDS)
    assert seedguard_boundaries.HELD_OUT_SEEDS == frozenset(policy.HELD_OUT_SEEDS)


def test_worker_guard_active(tmp_path):
    """Run with --dist=each to prove the boundary in every xdist worker."""
    import json
    import os
    from pathlib import Path

    from robot_sf.gym_env.robot_env import RobotEnv

    assert os.environ.get("ROBOT_SF_PYTEST_SEED_GUARD") == "1"
    assert RobotEnv.reset._seedguard_boundary == "RobotEnv.reset"
    with pytest.raises(HeldoutSeedError):
        object.__new__(RobotEnv).reset(seed=50036)
    proof = os.environ.get("SEEDGUARD_WORKER_PROOF")
    if proof:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        Path(proof, f"{worker}.json").write_text(json.dumps({"worker": worker, "active": True}))
