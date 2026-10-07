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
    from robot_sf.benchmark.pedestrian_manipulation_checks import build_probe
    from robot_sf.nav.map_config import SinglePedestrianDefinition

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
    "seed", [0, 1, 110, 111, 140, 141, 311, 1000, 1031, 50036, 59019, None, True, 1001.0]
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
    assert profiles["combined"]["obstacle_force_profile"] == "gradient_v3"
    if "obstacle_force_profile" not in SimulationSettings.__dataclass_fields__:
        with pytest.raises(ValueError, match="PR #10073"):
            profile_scenario({}, profiles["combined"], 1001)
    else:
        bound = profile_scenario({}, profiles["combined"], 1001)
        config = build_robot_config_from_scenario(bound, scenario_path=ROOT / "test.yaml")
        assert str(config.sim_config.obstacle_force_profile) == "gradient_v3"


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
