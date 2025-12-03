"""TODO docstring. Document this module."""

import numpy as np

from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates


def intersect(s1: set, s2: set) -> set:
    """TODO docstring. Document this function.

    Args:
        s1: TODO docstring.
        s2: TODO docstring.

    Returns:
        TODO docstring.
    """
    return {e for e in s1 if e in s2}


def contains_all(s: set, comp: set) -> bool:
    """TODO docstring. Document this function.

    Args:
        s: TODO docstring.
        comp: TODO docstring.

    Returns:
        TODO docstring.
    """
    return len(intersect(s, comp)) >= len(comp)


def contains_none(s: set, comp: set) -> bool:
    """TODO docstring. Document this function.

    Args:
        s: TODO docstring.
        comp: TODO docstring.

    Returns:
        TODO docstring.
    """
    return len(intersect(s, comp)) == 0


def set_except(s1: set, s2: set) -> set:
    """TODO docstring. Document this function.

    Args:
        s1: TODO docstring.
        s2: TODO docstring.

    Returns:
        TODO docstring.
    """
    return {e for e in s1 if e not in s2}


def init_groups():
    """TODO docstring. Document this function."""
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
    """TODO docstring. Document this function."""
    ped_ids = {0, 1, 2}
    groups = PedestrianGroupings(None)  # type: ignore
    gid = groups.new_group(ped_ids)
    assert groups.groups[gid] == ped_ids


def test_can_create_group_from_assigned_pedestrians():
    """TODO docstring. Document this function."""
    ped_ids = {0, 1, 2}
    groups = PedestrianGroupings(None)  # type: ignore
    old_gid = groups.new_group(ped_ids)
    new_gid = groups.new_group(ped_ids)
    assert groups.groups[old_gid] == set()
    assert groups.groups[new_gid] == ped_ids


# def test_can_reassign_pedestrians_to_existing_group():
#     ped_ids, old_gid, target_gid = {0, 1, 2}, 0, 1
#     groups = init_groups()
#     groups.reassign_pedestrians(target_gid, ped_ids)
#     assert contains_none(groups.groups[old_gid], ped_ids)
#     assert contains_all(groups.groups[target_gid], ped_ids)


def test_can_remove_entire_group():
    """TODO docstring. Document this function."""
    removed_gid = 0
    groups = init_groups()
    ped_ids_removed = groups.groups[removed_gid]
    groups.remove_group(removed_gid)
    assert contains_none(groups.groups[removed_gid], ped_ids_removed)


def test_can_redirect_group_towards_new_goal():
    """TODO docstring. Document this function."""
    redirected_gid = 0
    groups = init_groups()
    old_goal = groups.goal_of_group(redirected_gid)
    new_goal = old_goal[0] + 1, old_goal[1] + 1
    groups.redirect_group(redirected_gid, new_goal)
    assert groups.goal_of_group(redirected_gid) == new_goal
