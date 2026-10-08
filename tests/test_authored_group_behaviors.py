"""Exercise authored join/leave scenarios through real pedestrian physics (#10028)."""

from pathlib import Path

import numpy as np
import pytest

from robot_sf.common.seed import set_global_seed
from robot_sf.ped_npc.ped_behavior import SinglePedestrianBehavior
from robot_sf.sim.simulator import init_simulators
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios


@pytest.fixture
def group_simulator():
    """Build either scenario with one development seed and no moving robot."""

    def build(role):
        path = Path(f"configs/scenarios/single/francis2023_{role}_group.yaml").resolve()
        scenario = load_scenarios(path)[0]
        set_global_seed(1001)
        config = build_robot_config_from_scenario(scenario, scenario_path=path)
        config.route_spawn_seed = 1001
        map_def = next(iter(config.map_pool.map_defs.values()))
        sim = init_simulators(config, map_def, num_robots=1, random_start_pos=False)[0]
        sim.reset_state()
        return sim

    return build


def _members(sim, ped_id):
    """Return actual runtime members, ignoring retired empty groups."""
    return sim.groups.groups[sim.groups.group_by_ped_id[ped_id]]


def test_join_scenario_starts_with_two_group_anchors(group_simulator):
    """The anchors share a physics group; the joiner begins outside it."""
    sim = group_simulator("join")
    assert _members(sim, 0) == {0, 1}
    assert _members(sim, 2) == {2}
    assert {0, 1} in [set(group) for group in sim.pysf_sim.peds.groups]


def test_join_completes_under_social_repulsion_and_repeats_after_reset(group_simulator):
    """A real join must finish without starting within the arrival threshold."""
    sim = group_simulator("join")
    behavior = next(b for b in sim.peds_behaviors if isinstance(b, SinglePedestrianBehavior))
    joiner = behavior._runtimes[2]
    for _episode in range(2):
        assert np.linalg.norm(sim.ped_pos[2] - sim.ped_pos[:2].mean(axis=0)) > 5.0
        for _ in range(200):
            sim.step_once([(0.0, 0.0)])
        assert _members(sim, 2) == {0, 1, 2}
        assert joiner.joined_group_id == sim.groups.group_by_ped_id[0]
        sim.reset_state()
        assert _members(sim, 0) == {0, 1}
        assert _members(sim, 2) == {2}


def test_leave_scenario_leaves_a_real_group_and_restores_it_on_reset(group_simulator):
    """Leaving splits a three-person group and reset restores the authored episode."""
    sim = group_simulator("leave")
    original_group_count = len(sim.groups.groups)
    for _episode in range(3):
        assert _members(sim, 0) == {0, 1, 2}
        sim.step_once([(0.0, 0.0)])
        assert _members(sim, 0) == {0}
        assert _members(sim, 1) == {1, 2}
        sim.reset_state()
        assert {0, 1, 2} in [set(group) for group in sim.pysf_sim.peds.groups]
    assert len(sim.groups.groups) <= original_group_count + 1


def test_join_uses_authored_radius_instead_of_waypoint_threshold(group_simulator):
    """A join accepts the group's nearby shared space, but only within its authored radius."""
    sim = group_simulator("join")
    behavior = next(b for b in sim.peds_behaviors if isinstance(b, SinglePedestrianBehavior))
    anchor_group = sim.groups.group_by_ped_id[0]
    center = np.asarray(sim.groups.group_centroid(anchor_group))
    sim.ped_pos[2] = center + (0.81, 0.0)
    behavior.step()
    assert sim.groups.group_by_ped_id[2] != anchor_group
    sim.ped_pos[2] = center + (0.79, 0.0)
    behavior.step()
    assert sim.groups.group_by_ped_id[2] == anchor_group
