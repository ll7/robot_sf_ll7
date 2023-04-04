from math import dist
from typing import List, Set, Dict, Tuple, Protocol, Callable
from dataclasses import dataclass, field

import numpy as np

from robot_sf.ped_spawn_generator import ZonePointsGenerator

Vec2D = Tuple[float, float]


@dataclass
class PedestrianStates(Protocol):
    pysf_states: Callable[[], np.ndarray]

    @property
    def num_peds(self) -> int:
        raise NotImplementedError()

    @property
    def ped_positions(self) -> np.ndarray:
        raise NotImplementedError()

    def redirect(self, ped_id: int, new_goal: Vec2D):
        raise NotImplementedError()

    def goal_of(self, ped_id: int) -> Vec2D:
        raise NotImplementedError()

    def pos_of(self, ped_id: int) -> Vec2D:
        raise NotImplementedError()

    def pos_of_many(self, ped_ids: Set[int]) -> np.ndarray:
        raise NotImplementedError()


@dataclass
class PySFPedestrianStates:
    pysf_states: Callable[[], np.ndarray]

    @property
    def num_peds(self) -> int:
        return self.pysf_states().shape[0]

    @property
    def ped_positions(self) -> np.ndarray:
        return self.pysf_states()[:, 0:2]

    def redirect(self, ped_id: int, new_goal: Vec2D):
        self.pysf_states()[ped_id, 4:6] = new_goal

    def goal_of(self, ped_id: int) -> Vec2D:
        pos_x, pos_y = self.pysf_states()[ped_id, 4:6]
        return (pos_x, pos_y)

    def pos_of(self, ped_id: int) -> Vec2D:
        pos_x, pos_y = self.pysf_states()[ped_id, 0:2]
        return (pos_x, pos_y)

    def pos_of_many(self, ped_ids: Set[int]) -> np.ndarray:
        return self.pysf_states()[list(ped_ids), 0:2]


@dataclass
class PedestrianGroupings:
    states: PedestrianStates
    groups: Dict[int, Set[int]] = field(default_factory=dict)
    group_by_ped_id: Dict[int, int] = field(default_factory=dict)

    # TODO: remove group reassignments, feature isn't required

    @property
    def group_ids(self) -> Set[int]:
        # info: pedestrian reassignments can cause groups
        #       to become empty -> ignore empty groups
        return {k for k in self.groups if len(self.groups[k]) > 0}

    @property
    def standalone_ped_ids(self) -> Set[int]:
        return {id for id in range(self.states.num_peds) \
                if id not in self.group_by_ped_id}

    def group_centroid(self, group_id: int) -> Vec2D:
        group = self.groups[group_id]
        positions = self.states.pos_of_many(group)
        c_x, c_y = np.mean(positions, axis=0)
        return (c_x, c_y)

    def is_standalone(self, ped_id: int) -> bool:
        return ped_id not in self.group_by_ped_id

    def goal_of_group(self, group_id: int) -> Vec2D:
        any_ped_id_of_group = next(iter(self.groups[group_id]))
        return self.states.goal_of(any_ped_id_of_group)

    def new_group(self, ped_ids: Set[int]) -> int:
        new_gid = max(self.groups.keys()) + 1 if self.groups.keys() else 0
        self.groups[new_gid] = ped_ids.copy()
        for ped_id in ped_ids:
            if not self.is_standalone(ped_id):
                old_gid = self.group_by_ped_id[ped_id]
                self.groups[old_gid].remove(ped_id)
            self.group_by_ped_id[ped_id] = new_gid
        return new_gid

    def reassign_pedestrians(self, target_group_id: int, ped_ids: Set[int]):
        for ped_id in ped_ids:
            self.reassign_pedestrian(target_group_id, ped_id)

    def reassign_pedestrian(self, target_group_id: int, ped_id: int):
        if not self.is_standalone(ped_id):
            old_group = self.groups[self.group_by_ped_id[ped_id]]
            old_group.remove(ped_id)
        self.groups[target_group_id].add(ped_id)
        self.group_by_ped_id[ped_id] = target_group_id

    def remove_group(self, group_id: int):
        for ped_id in self.groups[group_id]:
            if not self.is_standalone(ped_id):
                self.group_by_ped_id.pop(ped_id)
        self.groups[group_id].clear()

    def redirect_group(self, group_id: int, new_goal: Vec2D):
        for ped_id in self.groups[group_id]:
            self.states.redirect(ped_id, new_goal)

    def redirect_pedestrian(self, ped_id: int, new_goal: Vec2D):
        # info: throwing an exception would crash the whole simulation,
        #       rather do something consistent than aborting the training
        if not self.is_standalone(ped_id):
            self.redirect_group(self.group_by_ped_id[ped_id], new_goal)
        else:
            self.states.redirect(ped_id, new_goal)


@dataclass
class GroupRedirectBehavior:
    groups: PedestrianGroupings
    zone_assignments: Dict[int, int]
    spawn_zones: List[ZonePointsGenerator]
    goal_proximity_threshold: float = 1
    pick_new_goal: Callable[[int], Vec2D] = field(init=False)

    def __post_init__(self):
        self.pick_new_goal = lambda pid: \
            self.spawn_zones[self.zone_assignments[pid]].generate(1)[0][0]

    def redirect_groups_if_at_goal(self):
        for gid in self.groups.group_ids:
            centroid = self.groups.group_centroid(gid)
            goal = self.groups.goal_of_group(gid)
            dist_to_goal = dist(centroid, goal)
            if dist_to_goal < self.goal_proximity_threshold:
                any_pid = next(iter(self.groups.groups[gid]))
                new_goal = self.pick_new_goal(any_pid)
                self.groups.redirect_group(gid, new_goal)

        for pid in self.groups.standalone_ped_ids:
            pos = self.groups.states.pos_of(pid)
            goal = self.groups.states.goal_of(pid)
            dist_to_goal = dist(pos, goal)
            if dist_to_goal < self.goal_proximity_threshold:
                new_goal = self.pick_new_goal(pid)
                self.groups.redirect_pedestrian(pid, new_goal)

    def pick_new_goals(self):
        for gid in self.groups.group_ids:
            any_pid = next(iter(self.groups.groups[gid]))
            new_goal = self.pick_new_goal(any_pid)
            self.groups.redirect_group(gid, new_goal)

        for pid in self.groups.standalone_ped_ids:
            new_goal = self.pick_new_goal(pid)
            self.groups.redirect_pedestrian(pid, new_goal)
