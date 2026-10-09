"""Exercise authored groups through the scenario loader and real pedestrian physics."""

from pathlib import Path

import pytest

from robot_sf.sim.simulator import init_simulators
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios


def _simulator(kind):
    path = Path(f"configs/scenarios/single/francis2023_{kind}_group.yaml")
    scenario = dict(load_scenarios(path)[0])
    scenario["seeds"] = [1001]
    config = build_robot_config_from_scenario(scenario, scenario_path=path)
    config.sim_config.pedestrian_seed = 1001
    map_def = next(iter(config.map_pool.map_defs.values()))
    simulator = init_simulators(config, map_def, num_robots=1, random_start_pos=False)[0]
    simulator.reset_state()
    return simulator


@pytest.mark.parametrize("kind,expected", [("join", [1, 2]), ("leave", [3])])
def test_authored_group_exists_at_reset(kind, expected):
    """The named group scenarios start with the intended multi-person group."""
    simulator = _simulator(kind)
    assert sorted(len(g) for g in simulator.groups.groups_as_lists if g) == expected


def test_join_completes_under_social_repulsion_and_reset_restores_membership():
    """A physically approaching joiner joins, and the next episode starts unjoined."""
    simulator = _simulator("join")
    for _ in range(400):
        simulator.step_once([(0.0, 0.0)])
    assert simulator.groups.group_by_ped_id[2] == simulator.groups.group_by_ped_id[0]
    assert sorted(len(g) for g in simulator.groups.groups_as_lists if g) == [3]
    simulator.reset_state()
    assert sorted(len(g) for g in simulator.groups.groups_as_lists if g) == [1, 2]
    assert simulator.groups.group_by_ped_id[0] == simulator.groups.group_by_ped_id[1]
    assert simulator.groups.group_by_ped_id[2] != simulator.groups.group_by_ped_id[0]


def test_leave_splits_the_authored_group_and_reset_restores_membership():
    """The departing member leaves a real group, which is restored on episode reset."""
    simulator = _simulator("leave")
    assert sorted(len(g) for g in simulator.groups.groups_as_lists if g) == [3]
    simulator.step_once([(0.0, 0.0)])
    assert simulator.groups.group_by_ped_id[0] != simulator.groups.group_by_ped_id[1]
    assert simulator.groups.group_by_ped_id[1] == simulator.groups.group_by_ped_id[2]
    simulator.reset_state()
    assert sorted(len(g) for g in simulator.groups.groups_as_lists if g) == [3]
