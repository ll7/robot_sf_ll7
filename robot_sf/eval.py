from dataclasses import dataclass, field
from typing import List
from enum import IntEnum
from collections import deque
from statistics import mean


class EnvOutcome(IntEnum):
    REACHED_GOAL = 0
    TIMEOUT = 1
    PEDESTRIAN_COLLISION = 2
    OBSTACLE_COLLISION = 3
    ROBOT_COLLISION = 4
    ROBOT_OBSTACLE_COLLISION = 5
    ROBOT_PEDESTRIAN_COLLISION = 6


@dataclass
class EnvMetrics:
    route_outcomes: List[EnvOutcome] = field(default_factory=list)
    intermediate_goal_outcomes: List[EnvOutcome] = field(default_factory=list)
    cache_size: int = 10

    @property
    def total_routes(self) -> int:
        return max(len(self.route_outcomes), 1)

    @property
    def total_intermediate_goals(self) -> int:
        return max(len(self.intermediate_goal_outcomes), 1)

    @property
    def pedestrian_collisions(self) -> int:
        return len([o for o in self.route_outcomes if o == EnvOutcome.PEDESTRIAN_COLLISION])

    @property
    def obstacle_collisions(self) -> int:
        return len([o for o in self.route_outcomes if o == EnvOutcome.OBSTACLE_COLLISION])

    @property
    def exceeded_timesteps(self) -> int:
        return len([o for o in self.route_outcomes if o == EnvOutcome.TIMEOUT])

    @property
    def completed_routes(self) -> int:
        return len([o for o in self.route_outcomes if o == EnvOutcome.REACHED_GOAL])

    @property
    def reached_intermediate_goals(self) -> int:
        return len([o for o in self.intermediate_goal_outcomes if o == EnvOutcome.REACHED_GOAL])

    @property
    def route_completion_rate(self) -> float:
        return self.completed_routes / self.total_routes

    @property
    def interm_goal_completion_rate(self) -> float:
        return self.reached_intermediate_goals / self.total_intermediate_goals

    @property
    def timeout_rate(self) -> float:
        return self.exceeded_timesteps / self.total_routes

    @property
    def obstacle_collision_rate(self) -> float:
        return self.obstacle_collisions / self.total_routes

    @property
    def pedestrian_collision_rate(self) -> float:
        return self.pedestrian_collisions / self.total_routes

    def update(self, meta: dict):
        is_end_of_interm_goal = meta["is_pedestrian_collision"] or meta["is_obstacle_collision"] \
            or meta["is_robot_at_goal"] or meta["is_timesteps_exceeded"]
        is_end_of_route = meta["is_pedestrian_collision"] or meta["is_obstacle_collision"] \
            or meta["is_route_complete"] or meta["is_timesteps_exceeded"]

        if is_end_of_interm_goal:
            self._on_next_intermediate_outcome(meta)
        if is_end_of_route:
            self._on_next_route_outcome(meta)

    def _on_next_intermediate_outcome(self, meta: dict):
        if meta["is_pedestrian_collision"]:
            outcome = EnvOutcome.PEDESTRIAN_COLLISION
        elif meta["is_obstacle_collision"]:
            outcome = EnvOutcome.OBSTACLE_COLLISION
        elif meta["is_robot_at_goal"]:
            outcome = EnvOutcome.REACHED_GOAL
        elif meta["is_timesteps_exceeded"]:
            outcome = EnvOutcome.TIMEOUT
        else:
            raise NotImplementedError("unknown environment outcome")

        if len(self.intermediate_goal_outcomes) > self.cache_size:
            self.intermediate_goal_outcomes.pop(0)
        self.intermediate_goal_outcomes.append(outcome)

    def _on_next_route_outcome(self, meta: dict):
        if meta["is_pedestrian_collision"]:
            outcome = EnvOutcome.PEDESTRIAN_COLLISION
        elif meta["is_obstacle_collision"]:
            outcome = EnvOutcome.OBSTACLE_COLLISION
        elif meta["is_route_complete"]:
            outcome = EnvOutcome.REACHED_GOAL
        elif meta["is_timesteps_exceeded"]:
            outcome = EnvOutcome.TIMEOUT
        else:
            raise NotImplementedError("unknown environment outcome")

        if len(self.route_outcomes) > self.cache_size:
            self.route_outcomes.pop(0)
        self.route_outcomes.append(outcome)


@dataclass
class VecEnvMetrics:
    metrics: List[EnvMetrics]

    @property
    def route_completion_rate(self) -> float:
        return sum(m.route_completion_rate for m in self.metrics) / len(self.metrics)

    @property
    def interm_goal_completion_rate(self) -> float:
        return sum(m.interm_goal_completion_rate for m in self.metrics) / len(self.metrics)

    @property
    def timeout_rate(self) -> float:
        return sum(m.timeout_rate for m in self.metrics) / len(self.metrics)

    @property
    def obstacle_collision_rate(self) -> float:
        return sum(m.obstacle_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def pedestrian_collision_rate(self) -> float:
        return sum(m.pedestrian_collision_rate for m in self.metrics) / len(self.metrics)

    def update(self, metas: List[dict]):
        for metric, meta in zip(self.metrics, metas):
            metric.update(meta)


@dataclass
class PedEnvMetrics:
    route_outcomes: deque[EnvOutcome] = field(default_factory=lambda: deque(maxlen=10))
    avg_distance: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    route_distances: list[float] = field(default_factory=list)

    @property
    def total_routes(self) -> int:
        return max(len(self.route_outcomes), 1)

    @property
    def pedestrian_collisions(self) -> int:
        return len([o for o in self.route_outcomes if o == EnvOutcome.PEDESTRIAN_COLLISION])

    @property
    def obstacle_collisions(self) -> int:
        return len([o for o in self.route_outcomes if o == EnvOutcome.OBSTACLE_COLLISION])

    @property
    def exceeded_timesteps(self) -> int:
        return len([o for o in self.route_outcomes if o == EnvOutcome.TIMEOUT])

    @property
    def robot_collisions(self) -> int:
        return len([o for o in self.route_outcomes if o == EnvOutcome.ROBOT_COLLISION])

    @property
    def robot_at_goal(self) -> int:
        return len([o for o in self.route_outcomes if o == EnvOutcome.REACHED_GOAL])

    @property
    def robot_obstacle_collisions(self) -> int:
        return len([o for o in self.route_outcomes if o == EnvOutcome.ROBOT_OBSTACLE_COLLISION])

    @property
    def robot_pedestrian_collisions(self) -> int:
        return len([o for o in self.route_outcomes if o == EnvOutcome.ROBOT_PEDESTRIAN_COLLISION])

    @property
    def timeout_rate(self) -> float:
        return self.exceeded_timesteps / self.total_routes

    @property
    def obstacle_collision_rate(self) -> float:
        return self.obstacle_collisions / self.total_routes

    @property
    def pedestrian_collision_rate(self) -> float:
        return self.pedestrian_collisions / self.total_routes

    @property
    def robot_collision_rate(self) -> float:
        return self.robot_collisions / self.total_routes

    @property
    def robot_at_goal_rate(self) -> float:
        return self.robot_at_goal / self.total_routes

    @property
    def robot_obstacle_collision_rate(self) -> float:
        return self.robot_obstacle_collisions / self.total_routes

    @property
    def robot_pedestrian_collision_rate(self) -> float:
        return self.robot_pedestrian_collisions / self.total_routes

    @property
    def route_end_distance(self) -> float:
        return mean(self.avg_distance) if self.avg_distance else 0.0

    def update(self, meta: dict):
        self.route_distances.append(meta["distance_to_robot"])
        is_end_of_route = meta["is_pedestrian_collision"] or meta["is_obstacle_collision"] or \
            meta["is_robot_collision"] or meta["is_timesteps_exceeded"] or \
            meta["is_robot_at_goal"] or meta["is_robot_obstacle_collision"] or \
            meta["is_robot_pedestrian_collision"]
        if not is_end_of_route:
            return

        # If Robot collides with ego_pedestrian, the outcome is ROBOT_COLLISION
        # If Robot collides with pysf pedestrian, the outcome is ROBOT_PEDESTRIAN_COLLISION
        if meta["is_pedestrian_collision"]:
            outcome = EnvOutcome.PEDESTRIAN_COLLISION
        elif meta["is_obstacle_collision"]:
            outcome = EnvOutcome.OBSTACLE_COLLISION
        elif meta["is_robot_collision"]:
            outcome = EnvOutcome.ROBOT_COLLISION
        elif meta["is_timesteps_exceeded"]:
            outcome = EnvOutcome.TIMEOUT
        elif meta["is_robot_at_goal"]:
            outcome = EnvOutcome.REACHED_GOAL
        elif meta["is_robot_obstacle_collision"]:
            outcome = EnvOutcome.ROBOT_OBSTACLE_COLLISION
        elif meta["is_robot_pedestrian_collision"]:
            outcome = EnvOutcome.ROBOT_PEDESTRIAN_COLLISION
        else:
            raise NotImplementedError("unknown environment outcome")

        self.route_outcomes.append(outcome)
        self.avg_distance.append(mean(self.route_distances))
        self.route_distances.clear()


@dataclass
class PedVecEnvMetrics:
    metrics: List[PedEnvMetrics]

    @property
    def timeout_rate(self) -> float:
        return sum(m.timeout_rate for m in self.metrics) / len(self.metrics)

    @property
    def obstacle_collision_rate(self) -> float:
        return sum(m.obstacle_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def pedestrian_collision_rate(self) -> float:
        return sum(m.pedestrian_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_collision_rate(self) -> float:
        return sum(m.robot_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_at_goal_rate(self) -> float:
        return sum(m.robot_at_goal_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_obstacle_collision_rate(self) -> float:
        return sum(m.robot_obstacle_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_pedestrian_collision_rate(self) -> float:
        return sum(m.robot_pedestrian_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def route_end_distance(self) -> float:
        return sum(m.route_end_distance for m in self.metrics) / len(self.metrics)

    def update(self, metas: List[dict]):
        for metric, meta in zip(self.metrics, metas):
            metric.update(meta)
