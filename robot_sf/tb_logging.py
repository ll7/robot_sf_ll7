"""TensorBoard logging callbacks for training metrics.

Provides callbacks for logging navigation and pedestrian metrics during
StableBaselines3 training runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.eval import EnvMetrics, PedEnvMetrics, PedVecEnvMetrics, VecEnvMetrics

if TYPE_CHECKING:
    from stable_baselines3.common.logger import SummaryWriter, TensorBoardOutputFormat


def _init_classes() -> dict[str, Any]:  # noqa: C901
    from stable_baselines3.common.callbacks import BaseCallback  # noqa: PLC0415
    from stable_baselines3.common.logger import (  # noqa: PLC0415
        TensorBoardOutputFormat,
    )

    class BaseMetricsCallback(BaseCallback):
        """Base callback for logging metrics to TensorBoard during training."""

        def __init__(self):
            super().__init__()
            self.writer: SummaryWriter | None = None
            self._log_freq = 1000

        @property
        def meta_dicts(self) -> list[dict]:
            return [m["meta"] for m in self.locals["infos"]]

        @property
        def is_logging_step(self) -> bool:
            return self.n_calls % self._log_freq == 0

        def _on_training_start(self):
            if self.logger is not None:
                output_formats = self.logger.output_formats
                tb_formatter: TensorBoardOutputFormat | None = next(
                    (f for f in output_formats if isinstance(f, TensorBoardOutputFormat)),
                    None,
                )
                self.writer = tb_formatter.writer if tb_formatter is not None else None

            if self.writer is None:
                pass

        def _on_step(self) -> bool:
            raise NotImplementedError

    class DrivingMetricsCallback(BaseMetricsCallback):
        """Callback for logging robot navigation metrics during training."""

        def __init__(self, num_envs: int):
            super().__init__()
            self.metrics = VecEnvMetrics([EnvMetrics() for _ in range(num_envs)])

        def _on_step(self) -> bool:
            self.metrics.update(self.meta_dicts)

            if self.writer is not None and self.is_logging_step:
                self.writer.add_scalar(
                    "metrics/route_completion_rate",
                    self.metrics.route_completion_rate,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/interm_goal_completion_rate",
                    self.metrics.interm_goal_completion_rate,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/timeout_rate",
                    self.metrics.timeout_rate,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/obstacle_collision_rate",
                    self.metrics.obstacle_collision_rate,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/pedestrian_collision_rate",
                    self.metrics.pedestrian_collision_rate,
                    self.num_timesteps,
                )
                self.writer.flush()
            return True

    class AdversarialPedestrianMetricsCallback(BaseMetricsCallback):
        """Callback for logging adversarial pedestrian metrics during training."""

        def __init__(self, num_envs: int):
            super().__init__()
            self.metrics = PedVecEnvMetrics([PedEnvMetrics() for _ in range(num_envs)])

        def _on_step(self) -> bool:
            self.metrics.update(self.meta_dicts)

            if self.writer is not None and self.is_logging_step:
                self.writer.add_scalar(
                    "metrics/timeout_rate",
                    self.metrics.timeout_rate,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/obstacle_collision_rate",
                    self.metrics.obstacle_collision_rate,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/pedestrian_collision_rate",
                    self.metrics.pedestrian_collision_rate,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/robot_collision_rate",
                    self.metrics.robot_collision_rate,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/robot_at_goal_rate",
                    self.metrics.robot_at_goal_rate,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/robot_obstacle_collision_rate",
                    self.metrics.robot_obstacle_collision_rate,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/robot_pedestrian_collision_rate",
                    self.metrics.robot_pedestrian_collision_rate,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/avg_distance_to_robot",
                    self.metrics.route_end_distance,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/avg_ego_ped_speed_at_collision",
                    self.metrics.avg_ego_ped_speed_at_collision,
                    self.num_timesteps,
                )
                self.writer.add_scalar(
                    "metrics/avg_collision_impact_angle_rad",
                    self.metrics.avg_collision_impact_angle_rad_at_collision,
                    self.num_timesteps,
                )
                self.writer.flush()
            return True

    return {
        "BaseMetricsCallback": BaseMetricsCallback,
        "DrivingMetricsCallback": DrivingMetricsCallback,
        "AdversarialPedestrianMetricsCallback": AdversarialPedestrianMetricsCallback,
        "AdversialPedestrianMetricsCallback": AdversarialPedestrianMetricsCallback,
    }


_cache: dict[str, Any] | None = None
_LAZY_NAMES = {
    "BaseMetricsCallback",
    "DrivingMetricsCallback",
    "AdversarialPedestrianMetricsCallback",
    "AdversialPedestrianMetricsCallback",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_NAMES:
        global _cache
        if _cache is None:
            _cache = _init_classes()
            for lazy_name, value in _cache.items():
                if value.__name__ == lazy_name:
                    value.__qualname__ = lazy_name
                globals()[lazy_name] = value
        return _cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
