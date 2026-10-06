"""Production refusal before RNG use, without the pytest interception guard."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark import seed_bands

SENTINEL = 1030


@pytest.fixture
def forbidden_sentinel(monkeypatch):
    """Use a development seed as synthetic policy input; isolate pytest interception."""
    from tests.support import seedguard_boundaries

    monkeypatch.setattr(seed_bands, "HELD_OUT_SEEDS", frozenset({SENTINEL}))
    monkeypatch.setattr(seedguard_boundaries, "_ACTIVE", False)
    monkeypatch.delenv("ROBOT_SF_PYTEST_SEED_GUARD", raising=False)


def reached(*args, **kwargs):
    raise AssertionError("unguarded RNG or simulation dispatch reached")


@pytest.mark.parametrize(
    "boundary", ["reset_rng", "factory", "simulator", "crowd", "dummy", "episode", "classic"]
)
def test_production_refuses_before_rng_or_dispatch(  # noqa: C901 - explicit boundary controls
    forbidden_sentinel, monkeypatch, boundary
):
    if boundary == "reset_rng":
        from robot_sf.gym_env import env_util

        monkeypatch.setattr(env_util.random, "seed", reached)

        def invoke(seed=SENTINEL):
            with env_util.global_reset_seed(seed):
                reached()
    elif boundary == "factory":
        from robot_sf.gym_env import environment_factory

        monkeypatch.setattr(environment_factory.random, "seed", reached)

        def invoke(seed=SENTINEL):
            environment_factory._apply_global_seed(seed)
    elif boundary == "simulator":
        from robot_sf.sim import simulator

        monkeypatch.setattr(simulator.np.random, "SeedSequence", reached)

        def invoke(seed=SENTINEL):
            simulator._build_pysf_simulation(
                map_def=None,
                config=SimpleNamespace(
                    pedestrian_seed=seed,
                    route_spawn_seed=None,
                    archetype_seed=None,
                    response_law_seed=None,
                    desired_speed_seed=None,
                ),
                robots=[],
                peds_have_obstacle_forces=False,
                robot_pose_provider=lambda: [],
            )
    elif boundary == "crowd":
        from gymnasium import Env

        from robot_sf.gym_env.crowd_sim_env import CrowdSimEnv

        monkeypatch.setattr(Env, "reset", reached)

        def invoke(seed=SENTINEL):
            CrowdSimEnv.reset(object.__new__(CrowdSimEnv), seed=seed)
    elif boundary == "dummy":
        from robot_sf.sim.backends import dummy_backend

        monkeypatch.setattr(dummy_backend.np.random, "default_rng", reached)

        def invoke(seed=SENTINEL):
            dummy_backend.DummySimulator(map_def=None, seed=seed)
    elif boundary == "classic":
        from robot_sf.benchmark import scenario_generator

        monkeypatch.setattr(scenario_generator.np.random, "default_rng", reached)

        def invoke(seed=SENTINEL):
            scenario_generator.generate_scenario({}, seed)
    else:
        from robot_sf.benchmark.map_runner import map_runner_episode

        monkeypatch.setattr(map_runner_episode, "_resolve_episode_run_context", reached)

        def invoke(seed=SENTINEL):
            map_runner_episode.run_map_episode(
                scenario={},
                seed=seed,
                policy_builder=reached,
                horizon=1,
                dt=0.1,
                record_forces=False,
                snqi_weights=None,
                snqi_baseline=None,
                algo="social_force",
                scenario_path=Path("scenarios.yaml"),
            )

    with pytest.raises(ValueError, match="held-out simulation seed"):
        invoke()
    with pytest.raises(AssertionError, match="unguarded RNG or simulation dispatch reached"):
        invoke(1001)


def test_map_seed_dispatch_refuses_retired_fallback(forbidden_sentinel):
    from robot_sf.benchmark.map_runner.map_runner_batch_plan import build_seed_jobs

    with pytest.raises(ValueError, match="held-out simulation seed"):
        build_seed_jobs(
            [{"name": "seedless"}],
            suite_seeds={"classic_interactions": [SENTINEL]},
            suite_key="classic_interactions",
        )


def test_direct_campaign_refuses_before_preflight(forbidden_sentinel):
    from robot_sf.benchmark.camera_ready.campaign import run_campaign

    cfg = SimpleNamespace(
        seed_policy=SimpleNamespace(mode="fixed-list", seeds=[SENTINEL]),
        snqi_v2_binding=None,
        snqi_v2_spec=None,
    )
    with pytest.raises(ValueError, match="held-out simulation seed"):
        run_campaign(
            cfg,
            prepare_campaign_preflight=reached,
            run_batch=reached,
            compute_aggregates_with_ci=reached,
            export_publication_bundle=reached,
        )


def test_child_factory_cannot_bypass_production_guard(forbidden_sentinel):
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("ROBOT_SF_PYTEST_SEED_GUARD", None)
    script = """
import os
os.environ.pop("ROBOT_SF_PYTEST_SEED_GUARD", None)
from tests.support import seedguard_boundaries
seedguard_boundaries._ACTIVE = False
from robot_sf.benchmark import seed_bands
from robot_sf.gym_env import environment_factory
seed_bands.HELD_OUT_SEEDS = frozenset({1030})
def reached(*args, **kwargs):
    raise AssertionError("unguarded child RNG reached")
environment_factory.random.seed = reached
try:
    environment_factory._apply_global_seed(1030)
except ValueError as exc:
    assert "held-out simulation seed" in str(exc)
else:
    raise AssertionError("missing production refusal")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_classic_batch_admits_every_expanded_seed_before_output(forbidden_sentinel, monkeypatch):
    from robot_sf.benchmark import runner

    monkeypatch.setattr(runner, "_prepare_batch_setup", reached)
    with pytest.raises(ValueError, match="held-out simulation seed"):
        runner.run_batch([{"repeats": 2}], "unused.jsonl", "unused.json", base_seed=1029)
    with pytest.raises(AssertionError, match="unguarded RNG or simulation dispatch reached"):
        runner.run_batch([{"repeats": 2}], "unused.jsonl", "unused.json", base_seed=1001)


@pytest.mark.parametrize(
    "boundary",
    [
        "build_bidirectional_corridor",
        "build_narrow_doorway",
        "build_high_density_exit",
        "run_scenario",
        "run_native_reference",
        "run_reference_campaign",
    ],
)
def test_research_refuses_before_rng_or_raw_simulator(forbidden_sentinel, monkeypatch, boundary):
    from robot_sf.research import emergent_phenomena as emergent
    from robot_sf.research import lane_formation_reference_guarded as reference

    if boundary.startswith("build_"):
        monkeypatch.setattr(emergent.np.random, "default_rng", reached)

        def invoke(seed):
            getattr(emergent, boundary)(SimpleNamespace(seed=seed), None)
    elif boundary == "run_scenario":
        monkeypatch.setitem(emergent._BUILDERS, "bidirectional_corridor", reached)

        def invoke(seed):
            emergent.run_scenario(SimpleNamespace(seed=seed, name="bidirectional_corridor"), None)
    elif boundary == "run_native_reference":
        monkeypatch.setattr(reference._reference, "build_bidirectional_corridor", reached)

        def invoke(seed):
            reference.run_native_reference(
                protocol=reference.ReferenceProtocol(),
                condition="mixed_sustained_flow",
                seed=seed,
                calibration=emergent.RELEASED_DEFAULT_CALIBRATION,
            )
    else:
        monkeypatch.setattr(reference._reference, "run_native_reference", reached)

        def invoke(seed):
            reference.run_reference_campaign(seeds=[1001, seed])

    with pytest.raises(ValueError, match="held-out simulation seed"):
        invoke(SENTINEL)
    with pytest.raises(AssertionError, match="unguarded RNG or simulation dispatch reached"):
        invoke(1001)


@pytest.mark.parametrize(
    "boundary", ["run_scenario", "run_native_reference", "run_reference_campaign"]
)
def test_research_refuses_nested_desired_speed_seed(forbidden_sentinel, monkeypatch, boundary):
    from robot_sf.research import emergent_phenomena as emergent
    from robot_sf.research import lane_formation_reference_guarded as reference

    config = emergent.released_default_config()
    config.scene_config.desired_speed_seed = SENTINEL
    if boundary == "run_scenario":
        monkeypatch.setitem(emergent._BUILDERS, "bidirectional_corridor", reached)

        def invoke():
            emergent.run_scenario(
                SimpleNamespace(seed=1001, name="bidirectional_corridor"), None, config
            )
    elif boundary == "run_native_reference":
        monkeypatch.setattr(reference._reference, "build_bidirectional_corridor", reached)

        def invoke():
            reference.run_native_reference(
                protocol=reference.ReferenceProtocol(),
                condition="mixed_sustained_flow",
                seed=1001,
                calibration=emergent.RELEASED_DEFAULT_CALIBRATION,
                sim_config=config,
            )
    else:
        monkeypatch.setattr(reference._reference, "run_native_reference", reached)

        def invoke():
            reference.run_reference_campaign(seeds=[1001], sim_config=config)

    with pytest.raises(ValueError, match="held-out simulation seed"):
        invoke()
    config.scene_config.desired_speed_seed = 1002
    with pytest.raises(AssertionError, match="unguarded RNG or simulation dispatch reached"):
        invoke()


def test_historical_inventory_resolution_does_not_admit_execution(forbidden_sentinel):
    from robot_sf.benchmark.map_runner.map_runner_batch_plan import build_seed_jobs
    from robot_sf.benchmark.map_runner.map_runner_identity import _select_seeds

    inventory = {"classic_interactions": [SENTINEL]}
    assert _select_seeds({}, suite_seeds=inventory, suite_key="classic_interactions") == [SENTINEL]
    with pytest.raises(ValueError, match="held-out simulation seed"):
        build_seed_jobs([{}], suite_seeds=inventory, suite_key="classic_interactions")


@pytest.mark.parametrize("boundary", ["classic_episode", "thumbnail", "review_episode"])
def test_refused_seed_preserves_global_rng_state(
    forbidden_sentinel, monkeypatch, boundary, tmp_path
):
    """Refusal must precede real Python, NumPy and torch seeding and policy creation."""
    import random

    import numpy as np
    import torch

    from robot_sf.benchmark import runner, scenario_thumbnails
    from robot_sf.common.seed import set_global_seed

    set_global_seed(1001)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    policy_calls = []
    monkeypatch.setattr(
        runner, "_create_robot_policy", lambda *a, **kw: policy_calls.append(1) or (None, {})
    )
    try:
        if boundary == "review_episode":
            from robot_sf.analysis_workbench import review_execute

            result = review_execute._execute_episode_job(
                {
                    "scenario_id": review_execute._SUPPORTED_FIXTURE_SCENARIO_ID,
                    "source_ref": review_execute._supported_fixture_source_reference(),
                    "seed": SENTINEL,
                    "horizon_steps": 1,
                    "robot_speed_m_s": 1.0,
                    "ped_speed_m_s": 1.0,
                    "ped_start_delay_s": 0.0,
                }
            )
            assert result["status"] == "error"
            assert "held-out simulation seed" in result["error"]
        else:
            with pytest.raises(ValueError, match="held-out simulation seed"):
                if boundary == "classic_episode":
                    runner.run_episode({}, SENTINEL, horizon=1)
                else:
                    scenario_thumbnails.render_scenario_thumbnail(
                        {}, SENTINEL, tmp_path / "unused.png"
                    )
        assert random.getstate() == python_state, "refused seed changed Python RNG state"
        current_numpy = np.random.get_state()
        assert current_numpy[0] == numpy_state[0]
        np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
        assert current_numpy[2:] == numpy_state[2:]
        assert torch.equal(torch.get_rng_state(), torch_state), (
            "refused seed changed torch RNG state"
        )
        assert policy_calls == [], "refused seed constructed a robot policy"
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


@pytest.mark.parametrize("boundary", ["inventory", "profile", "command"])
def test_parameter_screen_admits_all_seeds_before_profile_rng(
    forbidden_sentinel, monkeypatch, boundary, tmp_path
):
    from robot_sf.research import lane_formation_parameter_screen_guarded as screen
    from scripts.validation import run_issue_6969_parameter_screen_guarded as command

    monkeypatch.setattr(screen._screen, "build_space_filling_profiles", reached)

    def invoke(seed):
        if boundary == "command":
            monkeypatch.setattr(
                command.sys,
                "argv",
                [
                    "guarded-screen",
                    "--seeds",
                    f"1001,{seed}",
                    "--output-dir",
                    str(tmp_path / "unused"),
                ],
            )
            command.main()
        elif boundary == "profile":
            screen.run_parameter_screen(seeds=[1001], profile_seed=seed)
        else:
            screen.run_parameter_screen(seeds=[1001, seed])

    with pytest.raises(ValueError, match="held-out simulation seed"):
        invoke(SENTINEL)
    with pytest.raises(AssertionError, match="unguarded RNG or simulation dispatch reached"):
        invoke(1001)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("seed", [True, False, 1001.5, "1001", object()])
def test_seed_guard_rejects_non_integer_seed(seed):
    """Refuse boolean/coerced seeds with a boundary-specific domain error."""
    from robot_sf.benchmark.runtime_seed_guard import check_simulation_seed

    with pytest.raises(
        ValueError, match="simulation seed must be an integer at type admission"
    ) as exc:
        check_simulation_seed(seed, boundary="type admission")
    assert isinstance(exc.value.__cause__, TypeError)


def test_authorization_uses_real_identity_reader(forbidden_sentinel, monkeypatch, tmp_path):
    """An authorization file is verified before its claimed source can admit a seed."""
    import subprocess

    from robot_sf.benchmark import release_protocol, runtime_seed_guard

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text("output/\n", encoding="utf-8")
    identity = tmp_path / "output" / "identity.json"
    identity.parent.mkdir()
    identity.write_text('{"schema_version":"unsupported"}', encoding="utf-8")
    monkeypatch.setattr(release_protocol, "get_repository_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="resolved release identity schema_version is unsupported"):
        runtime_seed_guard.check_simulation_seed(
            SENTINEL, boundary="identity reader", authorization=identity
        )
    with pytest.raises(ValueError, match="held-out simulation seed"):
        runtime_seed_guard.check_simulation_seed(SENTINEL, boundary="after refused identity")


@pytest.fixture
def verified_authorization(forbidden_sentinel, monkeypatch, tmp_path):
    """Isolate verified-manifest transport; keep the guard and release policy real."""
    from robot_sf.benchmark import release_protocol, runtime_seed_guard

    # Metadata only: no freeze checkout, RNG call, environment or simulation.
    identity = SimpleNamespace(
        source_sha="66f402ba176b13e45210d0da0b2cf20fcdc0cc02",
        resolved_seeds=[SENTINEL],
        release_kind="benchmark-data",
    )
    path = tmp_path / "verified-identity.json"
    verified_paths = []
    policy_calls = []
    real_policy = release_protocol.sealed_seed_execution_problem

    def verify(requested_path):
        verified_paths.append(requested_path)
        assert requested_path == path
        return identity

    def policy(manifest, seeds, *, source_commit):
        policy_calls.append((manifest, seeds, source_commit))
        return real_policy(manifest, seeds, source_commit=source_commit)

    monkeypatch.setattr(release_protocol, "verify_resolved_release_identity", verify)
    monkeypatch.setattr(release_protocol, "sealed_seed_execution_problem", policy)
    monkeypatch.setattr(seed_bands, "EVAL_SEEDS_0_0_8", (SENTINEL,))
    return runtime_seed_guard, release_protocol, identity, path, verified_paths, policy_calls


def test_authorization_requires_immutable_source(verified_authorization):
    """A verified development identity cannot authorize held-out execution."""
    guard, _, identity, path, verified_paths, policy_calls = verified_authorization
    identity.source_sha = "a" * 40
    with pytest.raises(
        ValueError, match="sealed authorization requires the immutable freeze source"
    ):
        guard.check_simulation_seed(SENTINEL, boundary="source admission", authorization=path)
    assert verified_paths == [path]
    assert policy_calls == [], "wrong source reached release execution admission"


def test_authorization_preserves_real_release_policy_refusal(verified_authorization, monkeypatch):
    """A frozen-source claim cannot override the real development-release refusal."""
    guard, protocol, identity, path, verified_paths, policy_calls = verified_authorization
    identity.release_kind = "development_rehearsal"
    monkeypatch.setattr(protocol, "EVAL_SEEDS_0_0_8", (SENTINEL,))
    with pytest.raises(ValueError, match="development rehearsal cannot be a sealed release"):
        guard.check_simulation_seed(SENTINEL, boundary="policy admission", authorization=path)
    assert verified_paths == [path]
    assert policy_calls == [(identity, (SENTINEL,), "66f402ba176b13e45210d0da0b2cf20fcdc0cc02")]


def test_authorization_is_scoped_to_verified_seed_inventory(verified_authorization):
    """A metadata-only admitted seed does not grant permission to later calls."""
    guard, _, identity, path, verified_paths, policy_calls = verified_authorization
    assert guard.validate_sealed_authorization(path) is identity
    assert (
        guard.check_simulation_seed(SENTINEL, boundary="scoped admission", authorization=path)
        is None
    )
    assert verified_paths == [path, path]
    assert policy_calls == [
        (identity, (SENTINEL,), "66f402ba176b13e45210d0da0b2cf20fcdc0cc02"),
        (identity, (SENTINEL,), "66f402ba176b13e45210d0da0b2cf20fcdc0cc02"),
    ]
    with pytest.raises(ValueError, match="held-out simulation seed"):
        guard.check_simulation_seed(SENTINEL, boundary="no global admission")
    assert verified_paths == [path, path]


@pytest.mark.parametrize("missing_binding", ["evaluation_band", "identity_inventory"])
def test_authorization_requires_both_seed_bindings(
    verified_authorization, monkeypatch, missing_binding
):
    """Even a valid source/policy needs both evaluation and identity membership."""
    guard, _, identity, path, verified_paths, policy_calls = verified_authorization
    if missing_binding == "evaluation_band":
        monkeypatch.setattr(seed_bands, "EVAL_SEEDS_0_0_8", (1001,))
    else:
        identity.resolved_seeds = [1001]
    with pytest.raises(ValueError, match="held-out simulation seed"):
        guard.check_simulation_seed(SENTINEL, boundary="seed membership", authorization=path)
    assert verified_paths == [path]
    assert policy_calls == [
        (identity, tuple(identity.resolved_seeds), "66f402ba176b13e45210d0da0b2cf20fcdc0cc02")
    ]
