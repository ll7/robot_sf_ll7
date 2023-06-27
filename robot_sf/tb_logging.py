from dataclasses import dataclass, field
from typing import Optional, List
from enum import IntEnum

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat, SummaryWriter


class EnvOutcome(IntEnum):
    REACHED_GOAL=0
    TIMEOUT=1
    PEDESTRIAN_COLLISION=2
    OBSTACLE_COLLISION=3


@dataclass
class EnvMetrics:
    route_outcomes: List[EnvOutcome] = field(default_factory=list)
    intermediate_goal_outcomes: List[EnvOutcome] = field(default_factory=list)
    cache_size: int = 100

    @property
    def total_routes(self) -> int:
        return len(self.route_outcomes)

    @property
    def total_intermediate_goals(self) -> int:
        return len(self.intermediate_goal_outcomes)

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

        self.intermediate_goal_outcomes.append(outcome)
        if len(self.intermediate_goal_outcomes) > self.cache_size:
            self.intermediate_goal_outcomes.pop(0)

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

        self.route_outcomes.append(outcome)
        if len(self.route_outcomes) > self.cache_size:
            self.route_outcomes.pop(0)


@dataclass
class VecEnvMetrics:
    metrics: List[EnvMetrics]

    @property
    def route_completion_rate(self) -> float:
        return sum([m.route_completion_rate for m in self.metrics]) / len(self.metrics)

    @property
    def interm_goal_completion_rate(self) -> float:
        return sum([m.interm_goal_completion_rate for m in self.metrics]) / len(self.metrics)

    @property
    def timeout_rate(self) -> float:
        return sum([m.timeout_rate for m in self.metrics]) / len(self.metrics)

    @property
    def obstacle_collision_rate(self) -> float:
        return sum([m.obstacle_collision_rate for m in self.metrics]) / len(self.metrics)

    @property
    def pedestrian_collision_rate(self) -> float:
        return sum([m.pedestrian_collision_rate for m in self.metrics]) / len(self.metrics)

    def update(self, metas: List[dict]):
        for metric, meta in zip(self.metrics, metas):
            metric.update(meta)


class DrivingMetricsCallback(BaseCallback):

    def __init__(self, num_envs: int):
        super(DrivingMetricsCallback, self).__init__()
        self.writer: Optional[SummaryWriter] = None
        self.metrics = VecEnvMetrics([EnvMetrics() for _ in range(num_envs)])

    @property
    def meta_dicts(self) -> List[dict]:
        return [m["meta"] for m in self.locals["infos"]]

    @property
    def is_logging_step(self) -> bool:
        return self.n_calls % self._log_freq == 0

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls

        if self.logger is not None:
            output_formats = self.logger.output_formats
            tb_formatter: TensorBoardOutputFormat = next(
                filter(lambda f: isinstance(f, TensorBoardOutputFormat), output_formats), None)
            self.writer = tb_formatter.writer if tb_formatter else None

        if self.writer is None:
            print("WARNING: failed to initialize tensorboard environment metrics!")

    def _on_step(self) -> bool:
        self.metrics.update(self.meta_dicts)

        if self.writer is not None and self.is_logging_step:
            self.writer.add_scalar("metrics/route_completion_rate",
                                   self.metrics.route_completion_rate, self.num_timesteps)
            self.writer.add_scalar("metrics/interm_goal_completion_rate",
                                   self.metrics.interm_goal_completion_rate, self.num_timesteps)
            self.writer.add_scalar("metrics/timeout_rate",
                                   self.metrics.timeout_rate, self.num_timesteps)
            self.writer.add_scalar("metrics/obstacle_collision_rate",
                                   self.metrics.obstacle_collision_rate, self.num_timesteps)
            self.writer.add_scalar("metrics/pedestrian_collision_rate",
                                   self.metrics.pedestrian_collision_rate, self.num_timesteps)
            self.writer.flush()
        return True # info: don't request early abort
