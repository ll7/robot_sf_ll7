"""Evaluation helpers describing environment outcomes and cumulative metrics.

This module defines lightweight summaries of mission progress, collisions, and
completion rates for both single-agent and vectorized environments. ``EnvMetrics``
and its companions ingest meta data published by episodes, so clients such as
evaluators, planners, or benchmarks can compute failure rates over rolling caches.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from statistics import mean


class EnvOutcome(IntEnum):
    """TODO docstring. Document this class."""

    REACHED_GOAL = 0
    TIMEOUT = 1
    PEDESTRIAN_COLLISION = 2
    OBSTACLE_COLLISION = 3
    ROBOT_COLLISION = 4
    ROBOT_OBSTACLE_COLLISION = 5
    ROBOT_PEDESTRIAN_COLLISION = 6


@dataclass
class EnvMetrics:
    """TODO docstring. Document this class."""

    route_outcomes: list[EnvOutcome] = field(default_factory=list)
    intermediate_goal_outcomes: list[EnvOutcome] = field(default_factory=list)
    cache_size: int = 10

    @property
    def total_routes(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return max(len(self.route_outcomes), 1)

    @property
    def total_intermediate_goals(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return max(len(self.intermediate_goal_outcomes), 1)

    @property
    def pedestrian_collisions(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.route_outcomes if o == EnvOutcome.PEDESTRIAN_COLLISION])

    @property
    def obstacle_collisions(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.route_outcomes if o == EnvOutcome.OBSTACLE_COLLISION])

    @property
    def exceeded_timesteps(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.route_outcomes if o == EnvOutcome.TIMEOUT])

    @property
    def completed_routes(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.route_outcomes if o == EnvOutcome.REACHED_GOAL])

    @property
    def reached_intermediate_goals(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.intermediate_goal_outcomes if o == EnvOutcome.REACHED_GOAL])

    @property
    def route_completion_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.completed_routes / self.total_routes

    @property
    def interm_goal_completion_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.reached_intermediate_goals / self.total_intermediate_goals

    @property
    def timeout_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.exceeded_timesteps / self.total_routes

    @property
    def obstacle_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.obstacle_collisions / self.total_routes

    @property
    def pedestrian_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.pedestrian_collisions / self.total_routes

    def update(self, meta: dict):
        """TODO docstring. Document this function.

        Args:
            meta: TODO docstring.
        """
        is_end_of_interm_goal = (
            meta["is_pedestrian_collision"]
            or meta["is_obstacle_collision"]
            or meta["is_robot_at_goal"]
            or meta["is_timesteps_exceeded"]
        )
        is_end_of_route = (
            meta["is_pedestrian_collision"]
            or meta["is_obstacle_collision"]
            or meta["is_route_complete"]
            or meta["is_timesteps_exceeded"]
        )

        if is_end_of_interm_goal:
            self._on_next_intermediate_outcome(meta)
        if is_end_of_route:
            self._on_next_route_outcome(meta)

    def _on_next_intermediate_outcome(self, meta: dict):
        """TODO docstring. Document this function.

        Args:
            meta: TODO docstring.
        """
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
        """TODO docstring. Document this function.

        Args:
            meta: TODO docstring.
        """
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
    """TODO docstring. Document this class."""

    metrics: list[EnvMetrics]

    @property
    def route_completion_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.route_completion_rate for m in self.metrics) / len(self.metrics)

    @property
    def interm_goal_completion_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.interm_goal_completion_rate for m in self.metrics) / len(self.metrics)

    @property
    def timeout_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.timeout_rate for m in self.metrics) / len(self.metrics)

    @property
    def obstacle_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.obstacle_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def pedestrian_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.pedestrian_collision_rate for m in self.metrics) / len(self.metrics)

    def update(self, metas: list[dict]):
        """TODO docstring. Document this function.

        Args:
            metas: TODO docstring.
        """
        for metric, meta in zip(self.metrics, metas, strict=False):
            metric.update(meta)


@dataclass
class PedEnvMetrics:
    """TODO docstring. Document this class."""

    route_outcomes: deque[EnvOutcome] = field(default_factory=lambda: deque(maxlen=10))
    avg_distance: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    route_distances: list[float] = field(default_factory=list)

    @property
    def total_routes(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return max(len(self.route_outcomes), 1)

    @property
    def pedestrian_collisions(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.route_outcomes if o == EnvOutcome.PEDESTRIAN_COLLISION])

    @property
    def obstacle_collisions(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.route_outcomes if o == EnvOutcome.OBSTACLE_COLLISION])

    @property
    def exceeded_timesteps(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.route_outcomes if o == EnvOutcome.TIMEOUT])

    @property
    def robot_collisions(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.route_outcomes if o == EnvOutcome.ROBOT_COLLISION])

    @property
    def robot_at_goal(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.route_outcomes if o == EnvOutcome.REACHED_GOAL])

    @property
    def robot_obstacle_collisions(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.route_outcomes if o == EnvOutcome.ROBOT_OBSTACLE_COLLISION])

    @property
    def robot_pedestrian_collisions(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return len([o for o in self.route_outcomes if o == EnvOutcome.ROBOT_PEDESTRIAN_COLLISION])

    @property
    def timeout_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.exceeded_timesteps / self.total_routes

    @property
    def obstacle_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.obstacle_collisions / self.total_routes

    @property
    def pedestrian_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.pedestrian_collisions / self.total_routes

    @property
    def robot_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.robot_collisions / self.total_routes

    @property
    def robot_at_goal_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.robot_at_goal / self.total_routes

    @property
    def robot_obstacle_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.robot_obstacle_collisions / self.total_routes

    @property
    def robot_pedestrian_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.robot_pedestrian_collisions / self.total_routes

    @property
    def route_end_distance(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return mean(self.avg_distance) if self.avg_distance else 0.0

    def update(self, meta: dict):
        """TODO docstring. Document this function.

        Args:
            meta: TODO docstring.
        """
        self.route_distances.append(meta["distance_to_robot"])

        if not self._is_end_of_route(meta):
            return

        outcome = self._determine_outcome(meta)
        self._finalize_route_outcome(outcome)

    def _is_end_of_route(self, meta: dict) -> bool:
        """Check if the current step marks the end of a route."""
        end_conditions = [
            "is_pedestrian_collision",
            "is_obstacle_collision",
            "is_robot_collision",
            "is_timesteps_exceeded",
            "is_robot_at_goal",
            "is_route_complete",
            "is_robot_obstacle_collision",
            "is_robot_pedestrian_collision",
        ]
        return any(meta.get(condition, False) for condition in end_conditions)

    def _determine_outcome(self, meta: dict) -> EnvOutcome:
        """Determine the outcome based on the meta information.

        Note: Collision outcomes take precedence over goal/timeout outcomes.
        This ensures that if both a collision and goal are flagged simultaneously,
        the collision is reported as the outcome.
        """
        # Check collisions first (highest priority)
        collision_checks = [
            ("is_robot_collision", EnvOutcome.ROBOT_COLLISION),
            ("is_robot_pedestrian_collision", EnvOutcome.ROBOT_PEDESTRIAN_COLLISION),
            ("is_robot_obstacle_collision", EnvOutcome.ROBOT_OBSTACLE_COLLISION),
            ("is_pedestrian_collision", EnvOutcome.PEDESTRIAN_COLLISION),
            ("is_obstacle_collision", EnvOutcome.OBSTACLE_COLLISION),
        ]

        for condition, outcome in collision_checks:
            if meta.get(condition, False):
                return outcome

        # Then check timeout (medium priority)
        if meta.get("is_timesteps_exceeded", False):
            return EnvOutcome.TIMEOUT

        # Finally check goal conditions (lowest priority)
        if meta.get("is_robot_at_goal", False):
            return EnvOutcome.REACHED_GOAL
        if meta.get("is_route_complete", False):
            return EnvOutcome.REACHED_GOAL

        raise NotImplementedError("unknown environment outcome")

    def _finalize_route_outcome(self, outcome: EnvOutcome):
        """Finalize the route outcome and update metrics."""
        self.route_outcomes.append(outcome)
        self.avg_distance.append(mean(self.route_distances))
        self.route_distances.clear()


@dataclass
class PedVecEnvMetrics:
    """TODO docstring. Document this class."""

    metrics: list[PedEnvMetrics]

    @property
    def timeout_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.timeout_rate for m in self.metrics) / len(self.metrics)

    @property
    def obstacle_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.obstacle_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def pedestrian_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.pedestrian_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.robot_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_at_goal_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.robot_at_goal_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_obstacle_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.robot_obstacle_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_pedestrian_collision_rate(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.robot_pedestrian_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def route_end_distance(self) -> float:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return sum(m.route_end_distance for m in self.metrics) / len(self.metrics)

    def update(self, metas: list[dict]):
        """TODO docstring. Document this function.

        Args:
            metas: TODO docstring.
        """
        for metric, meta in zip(self.metrics, metas, strict=False):
            metric.update(meta)
