"""#10190 contracts: override activation, dev-only admission, paired arithmetic."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.runtime_seed_guard import check_simulation_seed
from robot_sf.sim.pedestrian_speed_tiers import sample_desired_pedestrian_speeds
from robot_sf.training.scenario_loader import build_robot_config_from_scenario

ROOT = Path(__file__).resolve().parents[2]


def test_measured_profile_reaches_production_substrate():
    """Catch ignored/rejected overrides and geometry/speed reset in the real builder."""
    profile = {
        "ped_radius": 0.25,
        "ped_force_radius": 0.25,
        "desired_speed_mean": 1.29,
        "desired_speed_std": 0.19,
        "desired_speed_truncated": True,
    }
    config = build_robot_config_from_scenario(
        {"simulation_config": profile}, scenario_path=ROOT / "test.yaml"
    ).sim_config
    assert config.ped_radius == 0.25
    assert config.ped_force_radius == 0.25
    assert config.desired_speed_mean == 1.29
    assert config.desired_speed_std == 0.19
    from robot_sf.benchmark.pedestrian_manipulation_checks import build_probe
    from robot_sf.nav.map_config import SinglePedestrianDefinition

    actors = [
        SinglePedestrianDefinition(id=str(i), start=(5, 4 * i + 4), goal=(110, 4 * i + 4))
        for i in range(32)
    ]
    sim, _, radius = build_probe(profile, 1001, actors, [])
    assert radius == 0.25
    assert sim.peds.agent_radius == 0.25
    before = sim.peds.max_speeds.copy()
    assert np.mean(before) == pytest.approx(1.29, abs=0.10)
    assert np.std(before) > 0.1
    sim.step(2)
    np.testing.assert_array_equal(sim.peds.max_speeds, before)


@pytest.mark.parametrize("mean", [1.29, 0.65])
def test_exact_speed_profile_overrides(mean):
    """Catch tier rounding or loss of the requested .19 spread."""
    settings = build_robot_config_from_scenario(
        {
            "simulation_config": {
                "desired_speed_mean": mean,
                "desired_speed_std": 0.19,
                "desired_speed_seed": 1001,
                "desired_speed_truncated": True,
            }
        },
        scenario_path=ROOT / "test.yaml",
    ).sim_config
    assert (
        settings.desired_speed_mean,
        settings.desired_speed_std,
        settings.desired_speed_seed,
        settings.desired_speed_truncated,
    ) == (mean, 0.19, 1001, True)


@pytest.mark.parametrize(
    "seed", [0, 1, 110, 111, 140, 141, 311, 1000, 1031, 9001, 9002, None, True, 1001.0]
)
def test_dev_only_seed_guard(seed):
    """Catch admitting an arbitrary non-held-out seed or coercing a non-integer."""
    with pytest.raises(ValueError, match="seed"):
        check_simulation_seed(seed, boundary="pedsens-test", dev_only=True)


@pytest.mark.parametrize("seed", [1001, 1030])
def test_dev_boundary_seeds_accepted(seed):
    """Catch off-by-one exclusions at either end of the dev interval."""
    check_simulation_seed(seed, boundary="pedsens-test", dev_only=True)


def test_truncation_has_no_clipped_boundary_mass():
    """Catch using censored/clipped normals for the opt-in truncated distribution."""
    speeds = sample_desired_pedestrian_speeds(512, 0, 1, seed=1001, truncate=True)
    assert np.all((speeds > 0) & (speeds < 3))
    assert np.mean(speeds) == pytest.approx(0.79, abs=0.10)


def fixture_rows():
    """Hand fixture: tau-b=-.5, A success=-1/contact=+1, C paired time=+2."""
    rows = []
    for factor in ("legacy", "radius"):
        for planner in ("A", "B", "C"):
            for seed in (1001, 1002):
                success = int(planner in {"C", "A" if factor == "legacy" else "B"})
                time = (10 if seed == 1001 else 20) + (2 if factor == "radius" else 0)
                rows.append(
                    {
                        "factor": factor,
                        "planner": planner,
                        "scenario": "tiny",
                        "seed": seed,
                        "values": {
                            "success": success,
                            "collision": int(factor == "radius" and planner == "A"),
                            "near_miss": 0,
                            "time_to_goal": time if success else None,
                        },
                    }
                )
    return rows


def test_summary_hand_fixture():
    """Catch incorrect tie handling, delta sign, unpaired CIs and missing-time zero fill."""
    study = importlib.import_module("robot_sf.benchmark.pedestrian_sensitivity")
    summary = study.summarize(fixture_rows(), samples=40)
    factor = summary["factors"]["radius"]
    assert factor["kendall_tau"]["estimate"] == pytest.approx(-0.5)
    assert factor["kendall_tau"]["ci"] == pytest.approx([-0.5, -0.5])
    assert factor["deltas"]["A"]["success"]["ci"] == [-1, -1]
    assert factor["deltas"]["A"]["collision"]["estimate"] == 1
    assert factor["deltas"]["C"]["time_to_goal"]["ci"] == [2, 2]
    assert factor["deltas"]["C"]["time_to_goal"]["paired_n"] == 2
    assert factor["deltas"]["A"]["time_to_goal"]["estimate"] is None
    assert factor["deltas"]["A"]["time_to_goal"]["ci"] is None
    assert len(summary["new_failures"]) == 2


def test_summary_refuses_unpaired_grid():
    """Catch silently dropping absent episodes and reporting a biased paired estimate."""
    from robot_sf.benchmark.pedestrian_sensitivity import summarize

    with pytest.raises(ValueError, match="incomplete paired"):
        summarize(fixture_rows()[:-1], samples=10)


def test_combined_profile_dependency_is_not_silently_ignored():
    """Catch executing combined as speed/radius only when wall selector is absent."""
    from robot_sf.benchmark.pedestrian_sensitivity import profile_scenario
    from robot_sf.sim.sim_config import SimulationSettings

    profiles = yaml.safe_load(
        (ROOT / "configs/benchmarks/pedestrian_sensitivity_10190.yaml").read_text()
    )["profiles"]
    assert profiles["combined"]["ped_radius"] == 0.25
    assert profiles["combined"]["desired_speed_mean"] == 1.29
    assert profiles["combined"]["obstacle_force_profile"] == "body_edge_v3"
    if "obstacle_force_profile" not in SimulationSettings.__dataclass_fields__:
        with pytest.raises(ValueError, match="PR #10073"):
            profile_scenario({}, profiles["combined"], 1001)
    else:
        bound = profile_scenario({}, profiles["combined"], 1001)
        config = build_robot_config_from_scenario(bound, scenario_path=ROOT / "test.yaml")
        assert str(config.sim_config.obstacle_force_profile) == "body_edge_v3"
        from robot_sf.benchmark.pedestrian_manipulation_checks import build_probe
        from robot_sf.nav.map_config import SinglePedestrianDefinition

        sim, _, radius = build_probe(
            profiles["combined"],
            1001,
            [SinglePedestrianDefinition(id="combined", start=(5, 4), goal=(110, 4))],
            [],
        )
        assert radius == 0.25
        assert sim.peds.agent_radius == 0.25
        assert str(sim.config.obstacle_force_config.law_version) == "body_edge_exponential_v3"
        assert sim.config.obstacle_force_config.factor == 10.0


def test_study_wall_force_depends_on_pedestrian_radius():
    """At a fixed wall distance, only the larger pedestrian is within force range."""
    from pysocialforce.forces import ObstacleForce

    from robot_sf.benchmark.pedestrian_manipulation_checks import build_probe
    from robot_sf.nav.map_config import SinglePedestrianDefinition
    from robot_sf.nav.obstacle import Obstacle

    profiles = yaml.safe_load(
        (ROOT / "configs/benchmarks/pedestrian_sensitivity_10190.yaml").read_text()
    )["profiles"]
    actor = SinglePedestrianDefinition(id="walker", start=(7.55, 3.0), goal=(12.0, 3.0))
    wall = Obstacle([(8, 0), (8.1, 0), (8.1, 6), (8, 6)])
    forces = []
    laws = []
    for factor, radius in [("wall", 0.35), ("combined", 0.25)]:
        sim, _, _ = build_probe(profiles[factor], 1001, [actor], [wall], width=16, height=6)
        assert sim.peds.agent_radius == radius
        np.testing.assert_array_equal(sim.peds.state[0, :2], actor.start)
        force = next(force for force in sim.forces if isinstance(force, ObstacleForce))
        forces.append(force()[0])
        laws.append(force.config.law_version)

    assert forces[0][0] < 0
    np.testing.assert_array_equal(forces[1], [0.0, 0.0])
    assert laws == ["body_edge_exponential_v3", "body_edge_exponential_v3"]


def test_legacy_profile_retains_default_kernel_and_geometry():
    """Catch an accidental default-radius or desired-speed change."""
    from robot_sf.benchmark.pedestrian_manipulation_checks import build_probe
    from robot_sf.nav.map_config import SinglePedestrianDefinition

    sim, _, radius = build_probe(
        {}, 1001, [SinglePedestrianDefinition(id="legacy", start=(5, 4), goal=(110, 4))], []
    )
    assert radius == 0.4
    assert sim.peds.agent_radius == 0.35
    assert sim.peds.max_speeds[0] == pytest.approx(0.65)


def test_native_goal_seconds_reconstructed_without_failure_imputation():
    """Catch omitted raw seconds silently becoming missing even for successful rows."""
    from robot_sf.benchmark.pedestrian_sensitivity import metric_values

    record = {
        "metrics": {
            "success": True,
            "collisions": 0,
            "near_misses": 1,
            "time_to_goal_norm_success_only": 0.616,
        },
        "effective_budget_steps": 500,
        "scenario_params": {"run_dt": 0.1},
    }
    assert metric_values(record)["time_to_goal"] == pytest.approx(30.8)
    record["metrics"]["success"] = False
    record["metrics"].pop("time_to_goal_norm_success_only")
    assert metric_values(record)["time_to_goal"] is None


@pytest.mark.parametrize("runtime_fallback", [False, True], ids=["finite", "nonfinite"])
def test_sacadrl_runtime_stop_is_recorded_as_fallback(tmp_path, monkeypatch, runtime_fallback):
    """A runtime stop bars pairing even when the fail-fast planner can load its model."""
    import json
    from types import SimpleNamespace

    from robot_sf.planner.socnav_sacadrl import SACADRLPlannerAdapter
    from scripts.benchmark import run_pedestrian_sensitivity as driver

    cfg, release, matrix, scenarios, planners = driver.prepare(
        driver.DEFAULT_CONFIG, check_dependencies=False
    )
    scenario = next(
        scenario for scenario in scenarios if scenario["name"] == cfg["smoke"]["scenario"]
    )
    planner = next(planner for planner in planners if planner["key"] == "sacadrl")
    predictions = iter([np.nan] if runtime_fallback else [])

    def build_model(adapter):
        assert adapter._allow_fallback is False
        return SimpleNamespace(
            predict=lambda _obs: np.array([[next(predictions, 1.0)]]),
            actions=np.array([[1.0, 0.0]]),
        )

    monkeypatch.setattr(SACADRLPlannerAdapter, "_build_model", build_model)
    job = ("legacy", planner, scenario, 1001, {}, matrix, tmp_path, release)
    if runtime_fallback:
        with pytest.raises(
            ValueError, match=r"non-native episode refused:.*nonfinite_model_output"
        ):
            driver.run_slot(job)
    else:
        paired_row = driver.run_slot(job)
        assert paired_row["execution_mode"] == "adapter"

    episode = tmp_path / "legacy" / "sacadrl" / f"{scenario['name']}--1001.jsonl"
    metadata = json.loads(episode.read_text())["algorithm_metadata"]
    checkpoint = metadata["planner_runtime"]["checkpoint_provenance"]
    assert checkpoint["load_succeeded"] is True
    assert checkpoint["fallback_triggered"] is runtime_fallback
    assert metadata["planner_kinematics"]["execution_mode"] == (
        "fallback" if runtime_fallback else "adapter"
    )
    if runtime_fallback:
        assert checkpoint["fallback_reason"] == "nonfinite_model_output"
        assert metadata["status"] == "fallback"
        assert metadata["fallback_reason"] == "nonfinite_model_output"


def test_plan_inventory_and_pre_dispatch_seed_guard():
    """Catch leaking release-eval seeds into the dev matrix or running during planning."""
    from scripts.benchmark.run_pedestrian_sensitivity import DEFAULT_CONFIG, prepare

    cfg, _, _, scenarios, planners = prepare(DEFAULT_CONFIG, check_dependencies=False)
    assert len(scenarios) == 48
    assert len(planners) == 14
    assert cfg["selected_seeds"] == list(range(1001, 1031))
    with pytest.raises(ValueError, match="non-development seed"):
        prepare(DEFAULT_CONFIG, seeds=[9001], check_dependencies=False)


def test_default_settings_bytes_and_profile_roundtrip():
    """Catch default hash drift or losing opt-in controls during dataclass replace."""
    import hashlib
    import json
    from dataclasses import replace

    from robot_sf.sim.sim_config import SimulationSettings

    digest = hashlib.sha256(
        json.dumps(
            SimulationSettings().to_dict(), sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()
    # Independently measured on origin/main 61cc91877a159e1d08b321543bd860042520b9ed.
    assert digest == "0d054279f4a6e090d40e1fa54accaf58e955e624a3309472b44107f8790bb78d"
    profile = replace(SimulationSettings(ped_force_radius=0.25, desired_speed_truncated=True))
    assert profile.to_dict()["ped_force_radius"] == 0.25
    assert profile.to_dict()["desired_speed_truncated"] is True


def test_reduced_workstation_plan(tmp_path):
    """Bind the released inputs to an exact 300-slot plan before dispatch."""
    from scripts.benchmark.run_pedestrian_sensitivity import main

    scenarios = [
        "classic_bottleneck_medium",
        "classic_doorway_medium",
        "classic_head_on_corridor_low",
        "classic_urban_crossing_medium",
        "francis2023_perpendicular_traffic",
    ]
    out = tmp_path / "plan"
    assert (
        main(
            [
                "--mode",
                "plan",
                "--out",
                str(out),
                "-n",
                "12",
                "--scenarios",
                *scenarios,
                "--planners",
                "goal",
                "social_force",
                "--seeds",
                "1001",
                "1002",
                "1003",
                "1004",
                "1005",
            ]
        )
        == 0
    )
    manifest = yaml.safe_load((out / "manifest.json").read_text())
    assert manifest["status"] == "planned"
    assert manifest["workers"] == 12
    assert manifest["episode_count"] == 300
    assert set(manifest["scenarios"]) == set(scenarios)
    assert [planner["key"] for planner in manifest["planners"]] == ["goal", "social_force"]
    assert manifest["config"]["selected_seeds"] == [1001, 1002, 1003, 1004, 1005]
    assert manifest["config"]["selected_factors"] == [
        "legacy",
        "radius",
        "speed",
        "spread",
        "wall",
        "combined",
    ]
    assert list(out.iterdir()) == [out / "manifest.json"]


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (["--scenarios", "missing"], "unknown scenario"),
        (["--scenarios", "classic_doorway_medium", "classic_doorway_medium"], "unique scenario"),
        (["--planners", "goal", "missing"], "unknown planner"),
        (["--planners", "goal"], "at least two planners"),
    ],
)
def test_selection_errors_refuse_output(tmp_path, selection, message):
    """A typo, duplicate or unrankable roster must not expand into a different run."""
    from scripts.benchmark.run_pedestrian_sensitivity import main

    out = tmp_path / "plan"
    with pytest.raises(ValueError, match=message):
        main(["--out", str(out), *selection])
    assert not out.exists()
