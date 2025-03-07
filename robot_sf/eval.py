"""
Evaluation metrics for robot-sf.

This module provides classes for tracking and calculating various metrics in
robot-sf. Metrics include collision rates, goal completion rates,
and timeout rates.

Classes:
- EnvOutcome: Enum defining possible episode outcomes
- EnvMetrics: Metrics tracking for standard robot environments
- VecEnvMetrics: Metrics aggregation across multiple environments
- PedEnvMetrics: Metrics tracking for pedestrian-centric environments
- PedVecEnvMetrics: Metrics aggregation for multiple pedestrian environments
"""

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from statistics import mean
from typing import List


class EnvOutcome(IntEnum):
    """
    Possible outcomes of an environment episode.

    These values represent the different ways an episode can end,
    allowing for analysis of success and failure modes.
    """

    REACHED_GOAL = 0  # Agent successfully reached its goal
    TIMEOUT = 1  # Episode exceeded maximum allowed timesteps
    PEDESTRIAN_COLLISION = 2  # Agent collided with a pedestrian
    OBSTACLE_COLLISION = 3  # Agent collided with an obstacle
    ROBOT_COLLISION = 4  # Specific to pedestrian envs: pedestrian collided with robot
    ROBOT_OBSTACLE_COLLISION = 5  # Robot collided with an obstacle
    ROBOT_PEDESTRIAN_COLLISION = 6  # Robot collided with a pedestrian


@dataclass
class EnvMetrics:
    """
    Tracks and calculates metrics for robot navigation environments.

    This class maintains a history of episode outcomes and computes
    success/failure rates. It's designed for robot-centric environments
    where the agent controls a robot navigating among pedestrians.

    Attributes:
        route_outcomes: List of outcomes for complete routes
        intermediate_goal_outcomes: List of outcomes for intermediate waypoints
        cache_size: Maximum number of outcomes to store (acts as a sliding window)
    """

    route_outcomes: List[EnvOutcome] = field(default_factory=list)
    intermediate_goal_outcomes: List[EnvOutcome] = field(default_factory=list)
    cache_size: int = 10

    @property
    def total_routes(self) -> int:
        """Total number of route outcomes recorded, minimum 1 to avoid division by zero."""
        return max(len(self.route_outcomes), 1)

    @property
    def total_intermediate_goals(self) -> int:
        """Total number of intermediate goal outcomes recorded, minimum 1."""
        return max(len(self.intermediate_goal_outcomes), 1)

    @property
    def pedestrian_collisions(self) -> int:
        """Number of episodes ending in pedestrian collisions."""
        return len([o for o in self.route_outcomes if o == EnvOutcome.PEDESTRIAN_COLLISION])

    @property
    def obstacle_collisions(self) -> int:
        """Number of episodes ending in obstacle collisions."""
        return len([o for o in self.route_outcomes if o == EnvOutcome.OBSTACLE_COLLISION])

    @property
    def exceeded_timesteps(self) -> int:
        """Number of episodes ending due to timeout."""
        return len([o for o in self.route_outcomes if o == EnvOutcome.TIMEOUT])

    @property
    def completed_routes(self) -> int:
        """Number of successfully completed routes."""
        return len([o for o in self.route_outcomes if o == EnvOutcome.REACHED_GOAL])

    @property
    def reached_intermediate_goals(self) -> int:
        """Number of successfully reached intermediate goals."""
        return len([o for o in self.intermediate_goal_outcomes if o == EnvOutcome.REACHED_GOAL])

    @property
    def route_completion_rate(self) -> float:
        """Fraction of routes successfully completed."""
        return self.completed_routes / self.total_routes

    @property
    def interm_goal_completion_rate(self) -> float:
        """Fraction of intermediate goals successfully reached."""
        return self.reached_intermediate_goals / self.total_intermediate_goals

    @property
    def timeout_rate(self) -> float:
        """Fraction of episodes ending due to timeout."""
        return self.exceeded_timesteps / self.total_routes

    @property
    def obstacle_collision_rate(self) -> float:
        """Fraction of episodes ending due to obstacle collisions."""
        return self.obstacle_collisions / self.total_routes

    @property
    def pedestrian_collision_rate(self) -> float:
        """Fraction of episodes ending due to pedestrian collisions."""
        return self.pedestrian_collisions / self.total_routes

    def update(self, meta: dict):
        """
        Update metrics based on the current episode metadata.

        This method checks if intermediate goals or complete routes have
        been reached or failed, and updates metrics accordingly.

        Args:
            meta: A dictionary of metadata from the environment step
        """
        # Check if this is the end of an intermediate goal
        is_end_of_interm_goal = (
            meta["is_pedestrian_collision"]
            or meta["is_obstacle_collision"]
            or meta["is_robot_at_goal"]
            or meta["is_timesteps_exceeded"]
        )

        # Check if this is the end of a complete route
        is_end_of_route = (
            meta["is_pedestrian_collision"]
            or meta["is_obstacle_collision"]
            or meta["is_route_complete"]
            or meta["is_timesteps_exceeded"]
        )

        # Update the appropriate metrics
        if is_end_of_interm_goal:
            self._on_next_intermediate_outcome(meta)
        if is_end_of_route:
            self._on_next_route_outcome(meta)

    def _on_next_intermediate_outcome(self, meta: dict):
        """
        Process the outcome of reaching an intermediate goal.

        Args:
            meta: A dictionary of metadata from the environment step

        Raises:
            NotImplementedError: If the outcome cannot be determined
        """
        # Determine the outcome of this intermediate goal
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

        # Maintain the sliding window of outcomes
        if len(self.intermediate_goal_outcomes) > self.cache_size:
            self.intermediate_goal_outcomes.pop(0)
        self.intermediate_goal_outcomes.append(outcome)

    def _on_next_route_outcome(self, meta: dict):
        """
        Process the outcome of completing a route.

        Args:
            meta: A dictionary of metadata from the environment step

        Raises:
            NotImplementedError: If the outcome cannot be determined
        """
        # Determine the outcome of this route
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

        # Maintain the sliding window of outcomes
        if len(self.route_outcomes) > self.cache_size:
            self.route_outcomes.pop(0)
        self.route_outcomes.append(outcome)


@dataclass
class VecEnvMetrics:
    """
    Aggregates metrics across multiple environments.

    Used for vectorized environments where multiple environments run in parallel.
    Metrics are aggregated by averaging across all environments.

    Attributes:
        metrics: List of EnvMetrics objects, one per environment
    """

    metrics: List[EnvMetrics]

    @property
    def route_completion_rate(self) -> float:
        """Average route completion rate across all environments."""
        return sum(m.route_completion_rate for m in self.metrics) / len(self.metrics)

    @property
    def interm_goal_completion_rate(self) -> float:
        """Average intermediate goal completion rate across all environments."""
        return sum(m.interm_goal_completion_rate for m in self.metrics) / len(self.metrics)

    @property
    def timeout_rate(self) -> float:
        """Average timeout rate across all environments."""
        return sum(m.timeout_rate for m in self.metrics) / len(self.metrics)

    @property
    def obstacle_collision_rate(self) -> float:
        """Average obstacle collision rate across all environments."""
        return sum(m.obstacle_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def pedestrian_collision_rate(self) -> float:
        """Average pedestrian collision rate across all environments."""
        return sum(m.pedestrian_collision_rate for m in self.metrics) / len(self.metrics)

    def update(self, metas: List[dict]):
        """
        Update metrics for all environments based on the current episode metadata.

        Args:
            metas: A list of metadata dictionaries, one per environment
        """
        for metric, meta in zip(self.metrics, metas):
            metric.update(meta)


@dataclass
class PedEnvMetrics:
    """
    Tracks and calculates metrics for pedestrian-centric environments.

    This class is used when the agent controls a pedestrian rather than a robot,
    with different success and failure conditions. It also tracks distances
    between the pedestrian and robot.

    Attributes:
        route_outcomes: Deque of outcomes for complete routes (fixed max size)
        avg_distance: Deque of average distances at route completion
        route_distances: List of distances recorded during current route
    """

    route_outcomes: deque[EnvOutcome] = field(default_factory=lambda: deque(maxlen=10))
    avg_distance: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    route_distances: list[float] = field(default_factory=list)

    @property
    def total_routes(self) -> int:
        """Total number of route outcomes recorded, minimum 1 to avoid division by zero."""
        return max(len(self.route_outcomes), 1)

    @property
    def pedestrian_collisions(self) -> int:
        """Number of episodes ending in pedestrian collisions."""
        return len([o for o in self.route_outcomes if o == EnvOutcome.PEDESTRIAN_COLLISION])

    @property
    def obstacle_collisions(self) -> int:
        """Number of episodes ending in obstacle collisions."""
        return len([o for o in self.route_outcomes if o == EnvOutcome.OBSTACLE_COLLISION])

    @property
    def exceeded_timesteps(self) -> int:
        """Number of episodes ending due to timeout."""
        return len([o for o in self.route_outcomes if o == EnvOutcome.TIMEOUT])

    @property
    def robot_collisions(self) -> int:
        """Number of episodes ending due to the ego pedestrian colliding with robot."""
        return len([o for o in self.route_outcomes if o == EnvOutcome.ROBOT_COLLISION])

    @property
    def robot_at_goal(self) -> int:
        """Number of episodes where the robot reached its goal."""
        return len([o for o in self.route_outcomes if o == EnvOutcome.REACHED_GOAL])

    @property
    def robot_obstacle_collisions(self) -> int:
        """Number of episodes ending due to the robot colliding with obstacles."""
        return len([o for o in self.route_outcomes if o == EnvOutcome.ROBOT_OBSTACLE_COLLISION])

    @property
    def robot_pedestrian_collisions(self) -> int:
        """Number of episodes ending due to the robot colliding with NPC pedestrians."""
        return len([o for o in self.route_outcomes if o == EnvOutcome.ROBOT_PEDESTRIAN_COLLISION])

    @property
    def timeout_rate(self) -> float:
        """Fraction of episodes ending due to timeout."""
        return self.exceeded_timesteps / self.total_routes

    @property
    def obstacle_collision_rate(self) -> float:
        """Fraction of episodes ending due to obstacle collisions."""
        return self.obstacle_collisions / self.total_routes

    @property
    def pedestrian_collision_rate(self) -> float:
        """Fraction of episodes ending due to pedestrian collisions."""
        return self.pedestrian_collisions / self.total_routes

    @property
    def robot_collision_rate(self) -> float:
        """Fraction of episodes ending due to ego pedestrian colliding with robot."""
        return self.robot_collisions / self.total_routes

    @property
    def robot_at_goal_rate(self) -> float:
        """Fraction of episodes where the robot reached its goal."""
        return self.robot_at_goal / self.total_routes

    @property
    def robot_obstacle_collision_rate(self) -> float:
        """Fraction of episodes ending due to robot colliding with obstacles."""
        return self.robot_obstacle_collisions / self.total_routes

    @property
    def robot_pedestrian_collision_rate(self) -> float:
        """Fraction of episodes ending due to robot colliding with NPC pedestrians."""
        return self.robot_pedestrian_collisions / self.total_routes

    @property
    def route_end_distance(self) -> float:
        """Average distance at route end across recorded episodes."""
        return mean(self.avg_distance) if self.avg_distance else 0.0

    def update(self, meta: dict):
        """
        Update metrics based on the current episode metadata.

        Tracks distances between pedestrian and robot, and records
        outcomes when a route ends.

        Args:
            meta: A dictionary of metadata from the environment step

        Raises:
            NotImplementedError: If the outcome cannot be determined
        """
        # Always track distance to robot
        self.route_distances.append(meta["distance_to_robot"])

        # Check if this is the end of a route
        is_end_of_route = (
            meta["is_pedestrian_collision"]
            or meta["is_obstacle_collision"]
            or meta["is_robot_collision"]
            or meta["is_timesteps_exceeded"]
            or meta["is_robot_at_goal"]
            or meta["is_robot_obstacle_collision"]
            or meta["is_robot_pedestrian_collision"]
        )
        if not is_end_of_route:
            return

        # Determine the outcome of this route
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

        # Record the outcome and average distance
        self.route_outcomes.append(outcome)
        self.avg_distance.append(mean(self.route_distances))
        self.route_distances.clear()


@dataclass
class PedVecEnvMetrics:
    """
    Aggregates metrics across multiple pedestrian environments.

    Used for vectorized pedestrian-centric environments where multiple
    environments run in parallel. Metrics are aggregated by averaging
    across all environments.

    Attributes:
        metrics: List of PedEnvMetrics objects, one per environment
    """

    metrics: List[PedEnvMetrics]

    @property
    def timeout_rate(self) -> float:
        """Average timeout rate across all environments."""
        return sum(m.timeout_rate for m in self.metrics) / len(self.metrics)

    @property
    def obstacle_collision_rate(self) -> float:
        """Average obstacle collision rate across all environments."""
        return sum(m.obstacle_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def pedestrian_collision_rate(self) -> float:
        """Average pedestrian collision rate across all environments."""
        return sum(m.pedestrian_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_collision_rate(self) -> float:
        """Average rate of ego pedestrian colliding with robot."""
        return sum(m.robot_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_at_goal_rate(self) -> float:
        """Average rate of robot reaching its goal."""
        return sum(m.robot_at_goal_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_obstacle_collision_rate(self) -> float:
        """Average rate of robot colliding with obstacles."""
        return sum(m.robot_obstacle_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def robot_pedestrian_collision_rate(self) -> float:
        """Average rate of robot colliding with NPC pedestrians."""
        return sum(m.robot_pedestrian_collision_rate for m in self.metrics) / len(self.metrics)

    @property
    def route_end_distance(self) -> float:
        return sum(m.route_end_distance for m in self.metrics) / len(self.metrics)

    def update(self, metas: List[dict]):
        """
        Update metrics for all environments based on the current episode metadata.

        Args:
            metas: A list of metadata dictionaries, one per environment
        """
        for metric, meta in zip(self.metrics, metas):
            metric.update(meta)
