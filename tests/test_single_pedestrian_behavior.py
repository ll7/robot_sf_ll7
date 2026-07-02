"""Tests for runtime single-pedestrian behavior (waypoints, waits, roles)."""

import numpy as np
import pytest

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


def test_single_pedestrian_start_delay_releases_to_trajectory():
    """Verify start_delay_s holds a pedestrian before releasing its trajectory."""
    ped = SinglePedestrianDefinition(
        id="ped1",
        start=(0.0, 0.0),
        trajectory=[(1.0, 0.0), (2.0, 0.0)],
        start_delay_s=2.0,
    )
    ped_states = np.zeros((1, 7))
    ped_states[0, 0:2] = (0.0, 0.0)
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
    assert np.allclose(ped_states[0, 4:6], [0.0, 0.0])
    assert np.allclose(ped_states[0, 2:4], [0.0, 0.0])

    behavior.step()
    assert np.allclose(ped_states[0, 4:6], [1.0, 0.0])


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


def _proximity_hold_behavior(robot_poses, *, time_step_s=1.0, hold_timeout_s=6.0):
    """Build a single-pedestrian behavior configured with a proximity-released hold.

    The pedestrian starts already at its curb hold waypoint ``(1.0, 0.0)`` with a crossing
    target of ``(2.0, 0.0)`` used as the proximity reference point.

    Returns:
        tuple: The behavior controller and its backing ``ped_states`` array.
    """
    ped = SinglePedestrianDefinition(
        id="crosser",
        start=(0.0, 0.0),
        trajectory=[(1.0, 0.0), (2.0, 0.0)],
        hold_until_robot_within_m=5.5,
        hold_ref_point=(2.0, 0.0),
        hold_timeout_s=hold_timeout_s,
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
        time_step_s=time_step_s,
        goal_proximity_threshold=0.1,
    )
    behavior.set_robot_pose_provider(lambda: robot_poses)
    return behavior, ped_states


def test_proximity_hold_holds_while_robot_is_far():
    """A pedestrian with a proximity hold stays at the curb while the robot is far away."""
    behavior, ped_states = _proximity_hold_behavior([((100.0, 100.0), 0.0)])

    for _ in range(3):
        behavior.step()
        # Held in place: zero velocity and goal pinned to the curb waypoint.
        assert np.allclose(ped_states[0, 2:4], [0.0, 0.0])
        assert np.allclose(ped_states[0, 4:6], [1.0, 0.0])

    assert behavior.hold_release_reasons() == {"crosser": None}


def test_proximity_hold_releases_when_robot_is_within_radius():
    """The crossing is released when a robot enters the hold radius of the reference point."""
    behavior, ped_states = _proximity_hold_behavior([((2.0, 0.0), 0.0)])

    behavior.step()

    # Released immediately: the pedestrian advances toward the crossing target.
    assert np.allclose(ped_states[0, 4:6], [2.0, 0.0])
    assert behavior.hold_release_reasons() == {"crosser": "robot_proximity"}


def test_proximity_hold_release_times_out_fail_open():
    """The hold releases after hold_timeout_s even if no robot ever approaches (fail-open)."""
    behavior, ped_states = _proximity_hold_behavior(
        [((100.0, 100.0), 0.0)],
        time_step_s=1.0,
        hold_timeout_s=2.0,
    )

    behavior.step()  # elapsed 0 -> hold, accrue 1s
    assert np.allclose(ped_states[0, 4:6], [1.0, 0.0])
    behavior.step()  # elapsed 1 -> hold, accrue 2s
    assert np.allclose(ped_states[0, 4:6], [1.0, 0.0])
    behavior.step()  # elapsed 2 >= timeout -> release, advance to crossing target
    assert np.allclose(ped_states[0, 4:6], [2.0, 0.0])
    assert behavior.hold_release_reasons() == {"crosser": "timeout"}


def test_proximity_hold_reset_reengages_hold():
    """Resetting the behavior re-arms the proximity hold for a new episode."""
    behavior, _ped_states = _proximity_hold_behavior([((2.0, 0.0), 0.0)])

    behavior.step()
    assert behavior.hold_release_reasons() == {"crosser": "robot_proximity"}

    behavior.reset()
    assert behavior.hold_release_reasons() == {"crosser": None}


def test_proximity_hold_disabled_when_ref_point_off_trajectory():
    """A reference point that is not on the trajectory disables the hold (fail-open)."""
    ped = SinglePedestrianDefinition(
        id="crosser",
        start=(0.0, 0.0),
        trajectory=[(1.0, 0.0), (2.0, 0.0)],
        hold_until_robot_within_m=5.5,
        hold_ref_point=(9.0, 9.0),
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
    behavior.set_robot_pose_provider(lambda: [((100.0, 100.0), 0.0)])

    behavior.step()
    # No hold engaged: the pedestrian advances to the next waypoint immediately.
    assert np.allclose(ped_states[0, 4:6], [2.0, 0.0])


def test_proximity_hold_requires_trajectory():
    """A proximity hold without a trajectory is rejected at definition time."""
    with pytest.raises(ValueError, match="requires a trajectory"):
        SinglePedestrianDefinition(
            id="crosser",
            start=(0.0, 0.0),
            goal=(5.0, 0.0),
            hold_until_robot_within_m=5.5,
            hold_ref_point=(5.0, 0.0),
        )


def test_hold_ref_point_requires_hold_radius():
    """hold_ref_point without hold_until_robot_within_m is rejected as a misconfiguration."""
    with pytest.raises(ValueError, match="require hold_until_robot_within_m"):
        SinglePedestrianDefinition(
            id="crosser",
            start=(0.0, 0.0),
            trajectory=[(1.0, 0.0), (2.0, 0.0)],
            hold_ref_point=(2.0, 0.0),
        )


def test_hold_radius_must_be_positive():
    """A non-positive hold radius is rejected."""
    with pytest.raises(ValueError, match="hold_until_robot_within_m must be > 0"):
        SinglePedestrianDefinition(
            id="crosser",
            start=(0.0, 0.0),
            trajectory=[(1.0, 0.0), (2.0, 0.0)],
            hold_until_robot_within_m=0.0,
            hold_ref_point=(2.0, 0.0),
        )
