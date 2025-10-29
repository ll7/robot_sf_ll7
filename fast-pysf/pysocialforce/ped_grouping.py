"""
ped_grouping.py
Handles the
- PedstrianStates
- PedestrianGroupings
"""
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np

Vec2D = Tuple[float, float]


@dataclass
class PedestrianStates:
    """
    Represents the states of pedestrians in a simulation.

    Attributes:
        raw_states (np.ndarray): The raw states of the pedestrians.

    Methods:
        num_peds() -> int: Returns the number of pedestrians.
        ped_positions() -> np.ndarray: Returns the positions of all pedestrians.
        redirect(ped_id: int, new_goal: Vec2D): Redirects a pedestrian to a new goal.
        reposition(ped_id: int, new_pos: Vec2D): Repositions a pedestrian to a new position.
        goal_of(ped_id: int) -> Vec2D: Returns the goal position of a pedestrian.
        pos_of(ped_id: int) -> Vec2D: Returns the current position of a pedestrian.
        pos_of_many(ped_ids: Set[int]) -> np.ndarray: Returns the positions of multiple pedestrians.
    """
    raw_states: np.ndarray

    @property
    def num_peds(self) -> int:
        """
        Get the number of pedestrians in the simulation.

        Returns:
            int: The number of pedestrians.
        """
        return self.raw_states.shape[0]

    @property
    def ped_positions(self) -> np.ndarray:
        """
        Get the positions of the pedestrians.

        Returns:
            np.ndarray: An array containing the x and y coordinates of the pedestrians.
        """
        return self.raw_states[:, 0:2]

    @property
    def ped_velocities(self) -> np.ndarray:
        """
        Returns the velocities of the pedestrians.

        Returns:
            np.ndarray: An array containing the velocities of the pedestrians.
        """
        return self.raw_states[:, 2:4]

    def redirect(self, ped_id: int, new_goal: Vec2D):
        """
        Redirects the specified pedestrian to a new goal position.

        Args:
            ped_id (int): The ID of the pedestrian to redirect.
            new_goal (Vec2D): The new goal position for the pedestrian.
        """
        self.raw_states[ped_id, 4:6] = new_goal

    def reposition(self, ped_id: int, new_pos: Vec2D):
        """
        Repositions a pedestrian to a new position.

        Args:
            ped_id (int): The ID of the pedestrian to reposition.
            new_pos (Vec2D): The new position to set for the pedestrian.
        """
        self.raw_states[ped_id, 0:2] = new_pos

    def goal_of(self, ped_id: int) -> Vec2D:
        """
        Returns the goal position of a pedestrian.

        Args:
            ped_id (int): The ID of the pedestrian.

        Returns:
            Vec2D: The goal position of the pedestrian.
        """
        pos_x, pos_y = self.raw_states[ped_id, 4:6]
        return (pos_x, pos_y)

    def pos_of(self, ped_id: int) -> Vec2D:
        """
        Returns the position of the pedestrian with the given ID.

        Args:
            ped_id (int): The ID of the pedestrian.

        Returns:
            Vec2D: The position of the pedestrian as a 2D vector.
        """
        pos_x, pos_y = self.raw_states[ped_id, 0:2]
        return (pos_x, pos_y)

    def pos_of_many(self, ped_ids: Set[int]) -> np.ndarray:
        """
        Returns the positions of multiple pedestrians.

        Args:
            ped_ids (Set[int]): A set of pedestrian IDs.

        Returns:
            np.ndarray: An array containing the positions of the specified pedestrians.
        """
        return self.raw_states[list(ped_ids), 0:2]


@dataclass
class PedestrianGroupings:
    """
    A class representing groupings of pedestrians.

    Attributes:
        states (PedestrianStates): The states of the pedestrians.
        groups (Dict[int, Set[int]]): A dictionary mapping group IDs to sets of pedestrian IDs.
        group_by_ped_id (Dict[int, int]): A dictionary mapping pedestrian IDs to group IDs.

    Methods:
        groups_as_lists() -> List[List[int]]: Returns the groups as lists of pedestrian IDs.
        group_ids() -> Set[int]: Returns the set of group IDs.
        group_centroid(group_id: int) -> Vec2D: Returns the centroid of a group.
        group_size(group_id: int) -> int: Returns the size of a group.
        goal_of_group(group_id: int) -> Vec2D: Returns the goal of a group.
        new_group(ped_ids: Set[int]) -> int: Creates a new group with the given pedestrian IDs.
        remove_group(group_id: int): Removes a group.
        redirect_group(group_id: int, new_goal: Vec2D): Redirects the pedestrians in a group to a new goal.
        reposition_group(group_id: int, new_positions: List[Vec2D]): Repositions the pedestrians in a group to new positions.
    """
    states: PedestrianStates
    groups: Dict[int, Set[int]] = field(default_factory=dict)
    group_by_ped_id: Dict[int, int] = field(default_factory=dict)

    @property
    def groups_as_lists(self) -> List[List[int]]:
        """
        Returns the groups as lists of pedestrian IDs.

        Returns:
            List[List[int]]: The groups as lists of pedestrian IDs.
        """
        # info: this facilitates slicing over numpy arrays
        #       for some reason, numpy cannot slide over indices provided as set ...
        return [list(ped_ids) for ped_ids in self.groups.values()]

    @property
    def group_ids(self) -> Set[int]:
        """
        Returns the set of group IDs.

        Returns:
            Set[int]: The set of group IDs.
        """
        # info: ignore empty groups
        return {k for k in self.groups if len(self.groups[k]) > 0}

    def group_centroid(self, group_id: int) -> Vec2D:
        """
        Returns the centroid of a group.

        Args:
            group_id (int): The ID of the group.

        Returns:
            Vec2D: The centroid of the group.
        """
        group = self.groups[group_id]
        positions = self.states.pos_of_many(group)
        c_x, c_y = np.mean(positions, axis=0)
        return (c_x, c_y)

    def group_size(self, group_id: int) -> int:
        """
        Returns the size of a group.

        Args:
            group_id (int): The ID of the group.

        Returns:
            int: The size of the group.
        """
        return len(self.groups[group_id])

    def goal_of_group(self, group_id: int) -> Vec2D:
        """
        Returns the goal of a group.

        Args:
            group_id (int): The ID of the group.

        Returns:
            Vec2D: The goal of the group.
        """
        any_ped_id_of_group = next(iter(self.groups[group_id]))
        return self.states.goal_of(any_ped_id_of_group)

    def new_group(self, ped_ids: Set[int]) -> int:
        """
        Creates a new group with the given pedestrian IDs.

        Args:
            ped_ids (Set[int]): The set of pedestrian IDs.

        Returns:
            int: The ID of the new group.
        """
        new_gid = max(self.groups.keys()) + 1 if self.groups.keys() else 0
        self.groups[new_gid] = ped_ids.copy()
        for ped_id in ped_ids:
            if ped_id in self.group_by_ped_id:
                old_gid = self.group_by_ped_id[ped_id]
                self.groups[old_gid].remove(ped_id)
            self.group_by_ped_id[ped_id] = new_gid
        return new_gid

    def remove_group(self, group_id: int):
        """
        Removes a group.

        Args:
            group_id (int): The ID of the group.
        """
        ped_ids = deepcopy(self.groups[group_id])
        for ped_id in ped_ids:
            self.new_group({ped_id})
        self.groups[group_id].clear()

    def redirect_group(self, group_id: int, new_goal: Vec2D):
        """
        Redirects the pedestrians in a group to a new goal.

        Args:
            group_id (int): The ID of the group.
            new_goal (Vec2D): The new goal position.
        """
        for ped_id in self.groups[group_id]:
            self.states.redirect(ped_id, new_goal)

    def reposition_group(self, group_id: int, new_positions: List[Vec2D]):
        """
        Repositions the pedestrians in a group to new positions.

        Args:
            group_id (int): The ID of the group.
            new_positions (List[Vec2D]): The new positions of the pedestrians.
        """
        for ped_id, new_pos in zip(self.groups[group_id], new_positions):
            self.states.reposition(ped_id, new_pos)
