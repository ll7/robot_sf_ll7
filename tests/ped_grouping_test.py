"""Tests for PedestrianGroupings group creation, mutation, and centroid behavior."""

import numpy as np

from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates


def intersect(s1: set, s2: set) -> set:
    """Return the elements present in both sets.

    Args:
        s1: First set.
        s2: Second set.

    Returns:
        Set intersection of s1 and s2.
    """
    return {e for e in s1 if e in s2}


def contains_all(s: set, comp: set) -> bool:
    """Return whether s contains every element of comp.

    Args:
        s: Set to search.
        comp: Elements that must all be present.

    Returns:
        True when comp is a subset of s.
    """
    return len(intersect(s, comp)) >= len(comp)


def contains_none(s: set, comp: set) -> bool:
    """Return whether s and comp share no elements.

    Args:
        s: Set to search.
        comp: Elements that must all be absent.

    Returns:
        True when the sets are disjoint.
    """
    return len(intersect(s, comp)) == 0


def set_except(s1: set, s2: set) -> set:
    """Return the elements of s1 that are not in s2.

    Args:
        s1: Source set.
        s2: Elements to exclude.

    Returns:
        Set difference s1 minus s2.
    """
    return {e for e in s1 if e not in s2}


def init_groups():
    """Build a six-pedestrian grouping with groups {0, 1, 2} and {3, 4}."""
    pysf_data = np.array(
        [
            # group of 3 pedestrians
            [0, 1, 0, 0, 10, 10],
            [0.5, 1, 0, 0, 10, 10],
            [1, 1, 0, 0, 10, 10],
            # group of 2 pedestrians
            [2, 3, 0, 0, 10, 1],
            [3, 2, 0, 0, 10, 1],
            # standalone pedestrian
            [5, 6, 0, 0, 7, 5],
        ],
    )
    states = PedestrianStates(lambda: pysf_data)
    groups = PedestrianGroupings(states)
    groups.new_group({0, 1, 2})
    groups.new_group({3, 4})
    return groups


def test_can_create_group_from_unassigned_pedestrians():
    """new_group stores the given pedestrian ids under the returned group id."""
    ped_ids = {0, 1, 2}
    groups = PedestrianGroupings(None)  # type: ignore
    gid = groups.new_group(ped_ids)
    assert groups.groups[gid] == ped_ids


def test_can_create_group_from_assigned_pedestrians():
    """Re-grouping moves membership: the old group empties and the new one holds the ids."""
    ped_ids = {0, 1, 2}
    groups = PedestrianGroupings(None)  # type: ignore
    old_gid = groups.new_group(ped_ids)
    new_gid = groups.new_group(ped_ids)
    assert groups.groups[old_gid] == set()
    assert groups.groups[new_gid] == ped_ids


def test_groups_as_lists_reuses_snapshot_until_grouping_changes():
    """Group list conversion should avoid repeated list materialization when stable."""
    groups = init_groups()

    first = groups.groups_as_lists
    second = groups.groups_as_lists

    assert second is first


def test_groups_as_lists_cache_invalidates_when_grouping_changes():
    """Join/leave style grouping mutations should refresh the cached list snapshot."""
    groups = init_groups()
    first = groups.groups_as_lists

    groups.add_to_group(5, 0)
    after_join = groups.groups_as_lists

    assert after_join is not first
    assert 5 in after_join[0]

    groups.new_group({5})
    after_leave = groups.groups_as_lists

    assert after_leave is not after_join
    assert after_leave[-1] == [5]


def test_can_remove_entire_group():
    """remove_group leaves none of the former members in that group."""
    removed_gid = 0
    groups = init_groups()
    ped_ids_removed = groups.groups[removed_gid]
    groups.remove_group(removed_gid)
    assert contains_none(groups.groups[removed_gid], ped_ids_removed)


def test_can_redirect_group_towards_new_goal():
    """redirect_group sets the group goal to the supplied offset target."""
    redirected_gid = 0
    groups = init_groups()
    old_goal = groups.goal_of_group(redirected_gid)
    new_goal = old_goal[0] + 1, old_goal[1] + 1
    groups.redirect_group(redirected_gid, new_goal)
    assert groups.goal_of_group(redirected_gid) == new_goal


def test_group_centroid_matches_member_position_mean():
    """Verify centroid contract because route behaviors use it every simulation step."""
    groups = init_groups()

    assert groups.group_centroid(0) == (0.5, 1.0)
    assert groups.group_centroid(1) == (2.5, 2.5)
