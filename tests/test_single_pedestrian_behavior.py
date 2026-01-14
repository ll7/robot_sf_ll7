"""Tests for runtime single-pedestrian behavior (waypoints, waits, roles)."""

import numpy as np

from robot_sf.nav.map_config import PedestrianWaitRule, SinglePedestrianDefinition
from robot_sf.ped_npc.ped_behavior import SinglePedestrianBehavior
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates


def test_single_pedestrian_waits_advance_waypoints():
    """Verify waits pause waypoint advancement before proceeding to the next waypoint."""
    ped = SinglePedestrianDefinition(
        id="ped1",
        start=(0.0, 0.0),
        trajectory=[(1.0, 0.0), (2.0, 0.0)],
        wait_at=[PedestrianWaitRule(waypoint_index=0, wait_s=2.0)],
    )
    ped_states = np.zeros((1, 7))
    ped_states[0, 0:2] = (1.0, 0.0)
    ped_states[0, 4:6] = (1.0, 0.0)
    states = PedestrianStates(lambda: ped_states)
    groups = PedestrianGroupings(states)
    behavior = SinglePedestrianBehavior(
        states,
        groups,
        [ped],
        single_offset=0,
        time_step_s=1.0,
        goal_proximity_threshold=0.1,
    )

    behavior.step()
    assert np.allclose(ped_states[0, 4:6], [1.0, 0.0])
    assert np.allclose(ped_states[0, 2:4], [0.0, 0.0])

    behavior.step()
    assert np.allclose(ped_states[0, 4:6], [1.0, 0.0])

    behavior.step()
    assert np.allclose(ped_states[0, 4:6], [2.0, 0.0])


def test_single_pedestrian_follow_role_targets_robot():
    """Verify follow role updates the pedestrian goal relative to the robot pose."""
    ped = SinglePedestrianDefinition(id="ped1", start=(0.0, 0.0), role="follow")
    ped_states = np.zeros((1, 7))
    ped_states[0, 0:2] = (0.0, 0.0)
    states = PedestrianStates(lambda: ped_states)
    groups = PedestrianGroupings(states)
    behavior = SinglePedestrianBehavior(
        states,
        groups,
        [ped],
        single_offset=0,
        time_step_s=0.1,
    )
    behavior.set_robot_pose_provider(lambda: [((2.0, 0.0), 0.0)])

    behavior.step()
    assert np.allclose(ped_states[0, 4:6], [1.0, 0.0])


def test_single_pedestrian_join_and_leave_group():
    """Verify join and leave roles update group membership as expected."""
    peds = [
        SinglePedestrianDefinition(id="leader", start=(0.0, 0.0)),
        SinglePedestrianDefinition(
            id="joiner",
            start=(0.1, 0.0),
            role="join",
            role_target_id="leader",
        ),
        SinglePedestrianDefinition(id="leaver", start=(0.2, 0.0), role="leave"),
    ]
    ped_states = np.zeros((3, 7))
    ped_states[:, 0:2] = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)]
    states = PedestrianStates(lambda: ped_states)
    groups = PedestrianGroupings(states)
    leader_group = groups.new_group({0, 2})

    behavior = SinglePedestrianBehavior(
        states,
        groups,
        peds,
        single_offset=0,
        time_step_s=0.1,
        goal_proximity_threshold=0.5,
    )

    behavior.step()
    assert groups.group_by_ped_id[1] == leader_group
    assert groups.group_by_ped_id[2] != leader_group
