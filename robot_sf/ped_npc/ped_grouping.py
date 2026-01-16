"""Pedestrian grouping utilities and state accessors."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from robot_sf.common.types import Vec2D


@dataclass
class PedestrianStates:
    """
    A class that represents the states of pedestrians.

    Attributes
    ----------
    pysf_states : Callable[[], np.ndarray]
        A function that returns the current states of the pedestrians.

    Properties
    ----------
    num_peds : int
        The number of pedestrians.
    ped_positions : np.ndarray
        The positions of the pedestrians.
    """

    pysf_states: Callable[[], np.ndarray]

    @property
    def num_peds(self) -> int:
        """
        Get the number of pedestrians.

        Returns
        -------
        int
            The number of pedestrians.
        """
        return self.pysf_states().shape[0]

    @property
    def ped_positions(self) -> np.ndarray:
        """
        Get the positions of the pedestrians.

        Returns
        -------
        np.ndarray
            The positions of the pedestrians.
        """
        return self.pysf_states()[:, 0:2]

    @property
    def ped_velocities(self) -> np.ndarray:
        """Get the velocities of the pedestrians.

        Returns
        -------
        np.ndarray
            The velocities of the pedestrians.
        """
        return self.pysf_states()[:, 2:4]

    def redirect(self, ped_id: int, new_goal: Vec2D):
        """
        Redirect a pedestrian to a new goal.

        Parameters
        ----------
        ped_id : int
            The ID of the pedestrian.
        new_goal : Vec2D
            The new goal of the pedestrian.
        """
        self.pysf_states()[ped_id, 4:6] = new_goal

    def set_velocity(self, ped_id: int, new_velocity: Vec2D):
        """
        Set the velocity of a pedestrian.

        Parameters
        ----------
        ped_id : int
            The ID of the pedestrian.
        new_velocity : Vec2D
            The new (vx, vy) velocity.
        """
        self.pysf_states()[ped_id, 2:4] = new_velocity

    def reposition(self, ped_id: int, new_pos: Vec2D):
        """
        Reposition a pedestrian to a new position.

        Parameters
        ----------
        ped_id : int
            The ID of the pedestrian.
        new_pos : Vec2D
            The new position of the pedestrian.
        """
        self.pysf_states()[ped_id, 0:2] = new_pos

    def goal_of(self, ped_id: int) -> Vec2D:
        """
        Get the goal of a pedestrian.

        Parameters
        ----------
        ped_id : int
            The ID of the pedestrian.

        Returns
        -------
        Vec2D
            The goal of the pedestrian.
        """
        pos_x, pos_y = self.pysf_states()[ped_id, 4:6]
        return (pos_x, pos_y)

    def pos_of(self, ped_id: int) -> Vec2D:
        """
        Get the position of a pedestrian.

        Parameters
        ----------
        ped_id : int
            The ID of the pedestrian.

        Returns
        -------
        Vec2D
            The position of the pedestrian.
        """
        pos_x, pos_y = self.pysf_states()[ped_id, 0:2]
        return (pos_x, pos_y)

    def pos_of_many(self, ped_ids: set[int]) -> np.ndarray:
        """
        Get the positions of multiple pedestrians.

        Parameters
        ----------
        ped_ids : Set[int]
            The IDs of the pedestrians.

        Returns
        -------
        np.ndarray
            The positions of the pedestrians.
        """
        return self.pysf_states()[list(ped_ids), 0:2]


@dataclass
class PedestrianGroupings:
    """
    A class that represents the groupings of pedestrians.

    Attributes
    ----------
    states : PedestrianStates
        The states of the pedestrians.
    groups : Dict[int, Set[int]]
        The groups of pedestrians, represented as a dictionary where the keys are group
        IDs and the values are sets of pedestrian IDs. Default is an empty dictionary.
    group_by_ped_id : Dict[int, int]
        A dictionary that maps pedestrian IDs to group IDs. Default is an empty dictionary.
    """

    states: PedestrianStates
    groups: dict[int, set[int]] = field(default_factory=dict)
    group_by_ped_id: dict[int, int] = field(default_factory=dict)

    @property
    def groups_as_lists(self) -> list[list[int]]:
        # info: this facilitates slicing over numpy arrays
        #       for some reason, numpy cannot slide over indices provided as set ...
        """Return pedestrian group ids as lists (numpy-friendly)."""
        return [list(ped_ids) for ped_ids in self.groups.values()]

    @property
    def group_ids(self) -> set[int]:
        # info: ignore empty groups
        """Return ids of non-empty groups."""
        return {k for k in self.groups if len(self.groups[k]) > 0}

    def group_centroid(self, group_id: int) -> Vec2D:
        """Return the centroid position of a group."""
        group = self.groups[group_id]
        positions = self.states.pos_of_many(group)
        c_x, c_y = np.mean(positions, axis=0)
        return (c_x, c_y)

    def group_size(self, group_id: int) -> int:
        """Return the number of pedestrians in a group."""
        return len(self.groups[group_id])

    def goal_of_group(self, group_id: int) -> Vec2D:
        """Return the goal of an arbitrary member of the group."""
        any_ped_id_of_group = next(iter(self.groups[group_id]))
        return self.states.goal_of(any_ped_id_of_group)

    def new_group(self, ped_ids: set[int]) -> int:
        """Create a new group from the given pedestrian ids.

        Returns:
            The new group id.
        """
        new_gid = max(self.groups.keys()) + 1 if self.groups.keys() else 0
        self.groups[new_gid] = ped_ids.copy()
        for ped_id in ped_ids:
            if ped_id in self.group_by_ped_id:
                old_gid = self.group_by_ped_id[ped_id]
                self.groups[old_gid].remove(ped_id)
            self.group_by_ped_id[ped_id] = new_gid
        return new_gid

    def add_to_group(self, ped_id: int, group_id: int) -> None:
        """Add a pedestrian to an existing group, removing them from any prior group."""
        if ped_id in self.group_by_ped_id:
            old_gid = self.group_by_ped_id[ped_id]
            if old_gid == group_id:
                return
            self.groups[old_gid].discard(ped_id)
        if group_id not in self.groups:
            self.groups[group_id] = set()
        self.groups[group_id].add(ped_id)
        self.group_by_ped_id[ped_id] = group_id

    def ensure_group_for_ped(self, ped_id: int) -> int:
        """Ensure a pedestrian belongs to a group, creating a single-member group if needed.

        Returns:
            int: The group id the pedestrian belongs to.
        """
        if ped_id in self.group_by_ped_id:
            return self.group_by_ped_id[ped_id]
        return self.new_group({ped_id})

    def remove_group(self, group_id: int):
        """Remove a group by reassigning members to single-member groups."""
        ped_ids = deepcopy(self.groups[group_id])
        for ped_id in ped_ids:
            self.new_group({ped_id})
        self.groups[group_id].clear()

    def redirect_group(self, group_id: int, new_goal: Vec2D):
        """Redirect all members of a group to a new goal."""
        for ped_id in self.groups[group_id]:
            self.states.redirect(ped_id, new_goal)

    def reposition_group(self, group_id: int, new_positions: list[Vec2D]):
        """Reposition group members using a list of new positions."""
        for ped_id, new_pos in zip(self.groups[group_id], new_positions, strict=False):
            self.states.reposition(ped_id, new_pos)
