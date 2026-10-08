"""Real authored group scenarios must perform their named membership transitions."""

from pathlib import Path

import pytest

from robot_sf.sim.simulator import init_simulators
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

ROOT = Path(__file__).resolve().parents[2]


def build_group_scenario(role, seed=1001):
    """Load the checked-in scenario through production loading and physics."""
    path = ROOT / f"configs/scenarios/single/francis2023_{role}_group.yaml"
    scenario = load_scenarios(path)[0]
    config = build_robot_config_from_scenario(scenario, scenario_path=path)
    config.sim_config.pedestrian_seed = seed
    return init_simulators(
        config, next(iter(config.map_pool.map_defs.values())), random_start_pos=False
    )[0]


def memberships(sim):
    """Ignore empty retired group containers and compare pedestrian members."""
    return {frozenset(group) for group in sim.groups.groups.values() if group}


@pytest.mark.parametrize(
    ("role", "expected"),
    [("join", {frozenset({0, 1}), frozenset({2})}), ("leave", {frozenset({0, 1, 2})})],
)
def test_authored_roles_start_with_real_group(role, expected):
    """Both named scenarios have the intended group at reset, also after a new episode."""
    sim = build_group_scenario(role)
    assert memberships(sim) == expected
    sim.step_once([(0.0, 0.0)])
    sim.reset_state()
    assert memberships(sim) == expected
    assert {frozenset(group) for group in sim.pysf_sim.peds.groups if group} == expected


def test_authored_join_completes_under_social_repulsion():
    """The real joiner can reach its threshold while social-force physics is active."""
    sim = build_group_scenario("join")
    for _ in range(400):
        sim.step_once([(0.0, 0.0)])
    assert memberships(sim) == {frozenset({0, 1, 2})}
    sim.reset_state()
    assert memberships(sim) == {frozenset({0, 1}), frozenset({2})}
    assert {frozenset(group) for group in sim.pysf_sim.peds.groups if group} == {
        frozenset({0, 1}),
        frozenset({2}),
    }


def test_authored_leave_separates_from_retained_group(monkeypatch):
    """The leaver exits its authored group while the two anchors remain together."""
    sim = build_group_scenario("leave")
    force_memberships = []
    compute_forces = sim.pysf_sim.compute_forces

    def capture_force_membership():
        force_memberships.append({frozenset(group) for group in sim.pysf_sim.peds.groups if group})
        return compute_forces()

    monkeypatch.setattr(sim.pysf_sim, "compute_forces", capture_force_membership)
    sim.step_once([(0.0, 0.0)])
    assert memberships(sim) == {frozenset({0}), frozenset({1, 2})}
    assert force_memberships == [{frozenset({0}), frozenset({1, 2})}]


@pytest.mark.parametrize(
    "override",
    [
        {"initial_group_id": ""},
        {"initial_group_id": 1},
        {"join_distance_m": -0.1},
        {"join_distance_m": float("nan")},
        {"join_distance_m": True},
    ],
)
def test_authored_membership_overrides_reject_invalid_settings(override):
    """Bad authoring must fail at the real loader rather than silently disable grouping."""
    path = ROOT / "configs/scenarios/single/francis2023_join_group.yaml"
    scenario = load_scenarios(path)[0]
    scenario["single_pedestrians"][2].update(override)
    with pytest.raises(ValueError, match="initial_group_id|join_distance_m"):
        build_robot_config_from_scenario(scenario, scenario_path=path)
