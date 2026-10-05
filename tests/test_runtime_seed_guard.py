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
    from robot_sf.research import lane_formation_reference as reference

    if boundary.startswith("build_"):
        monkeypatch.setattr(emergent.np.random, "default_rng", reached)

        def invoke(seed):
            getattr(emergent, boundary)(SimpleNamespace(seed=seed), None)
    elif boundary == "run_scenario":
        monkeypatch.setitem(emergent._BUILDERS, "bidirectional_corridor", reached)

        def invoke(seed):
            emergent.run_scenario(SimpleNamespace(seed=seed, name="bidirectional_corridor"), None)
    elif boundary == "run_native_reference":
        monkeypatch.setattr(reference, "build_bidirectional_corridor", reached)

        def invoke(seed):
            reference.run_native_reference(
                protocol=reference.ReferenceProtocol(),
                condition="mixed_sustained_flow",
                seed=seed,
                calibration=emergent.RELEASED_DEFAULT_CALIBRATION,
            )
    else:
        monkeypatch.setattr(reference, "run_native_reference", reached)

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
    from robot_sf.research import lane_formation_reference as reference

    config = emergent.released_default_config()
    config.scene_config.desired_speed_seed = SENTINEL
    if boundary == "run_scenario":
        monkeypatch.setitem(emergent._BUILDERS, "bidirectional_corridor", reached)

        def invoke():
            emergent.run_scenario(
                SimpleNamespace(seed=1001, name="bidirectional_corridor"), None, config
            )
    elif boundary == "run_native_reference":
        monkeypatch.setattr(reference, "build_bidirectional_corridor", reached)

        def invoke():
            reference.run_native_reference(
                protocol=reference.ReferenceProtocol(),
                condition="mixed_sustained_flow",
                seed=1001,
                calibration=emergent.RELEASED_DEFAULT_CALIBRATION,
                sim_config=config,
            )
    else:
        monkeypatch.setattr(reference, "run_native_reference", reached)

        def invoke():
            reference.run_reference_campaign(seeds=[1001], sim_config=config)

    with pytest.raises(ValueError, match="held-out simulation seed"):
        invoke()
    config.scene_config.desired_speed_seed = 1002
    with pytest.raises(AssertionError, match="unguarded RNG or simulation dispatch reached"):
        invoke()
