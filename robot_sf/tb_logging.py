"""Module tb_logging auto-generated docstring."""

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import SummaryWriter, TensorBoardOutputFormat

from robot_sf.eval import EnvMetrics, PedEnvMetrics, PedVecEnvMetrics, VecEnvMetrics


class BaseMetricsCallback(BaseCallback):
    """BaseMetricsCallback class."""

    def __init__(self):
        """Init.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        super().__init__()
        self.writer: SummaryWriter | None = None
        self._log_freq = 1000  # log every 1000 calls

    @property
    def meta_dicts(self) -> list[dict]:
        """Meta dicts.

        Returns:
            list[dict]: list of dict.
        """
        return [m["meta"] for m in self.locals["infos"]]

    @property
    def is_logging_step(self) -> bool:
        """Is logging step.

        Returns:
            bool: Boolean flag.
        """
        return self.n_calls % self._log_freq == 0

    def _on_training_start(self):
        """On training start.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        if self.logger is not None:
            output_formats = self.logger.output_formats
            tb_formatter: TensorBoardOutputFormat | None = next(
                (f for f in output_formats if isinstance(f, TensorBoardOutputFormat)),
                None,
            )
            self.writer = tb_formatter.writer if tb_formatter is not None else None

        if self.writer is None:
            pass

    # Define an abstract method for _on_step() if needed
    def _on_step(self) -> bool:
        """On step.

        Returns:
            bool: Boolean flag.
        """
        raise NotImplementedError


class DrivingMetricsCallback(BaseMetricsCallback):
    """DrivingMetricsCallback class."""

    def __init__(self, num_envs: int):
        """Init.

        Args:
            num_envs: number of environments.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        super().__init__()
        self.metrics = VecEnvMetrics([EnvMetrics() for _ in range(num_envs)])

    def _on_step(self) -> bool:
        """On step.

        Returns:
            bool: Boolean flag.
        """
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
        return True  # info: don't request early abort


class AdversialPedestrianMetricsCallback(BaseMetricsCallback):
    """AdversialPedestrianMetricsCallback class."""

    def __init__(self, num_envs: int):
        """Init.

        Args:
            num_envs: number of environments.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        super().__init__()
        self.metrics = PedVecEnvMetrics([PedEnvMetrics() for _ in range(num_envs)])

    def _on_step(self) -> bool:
        """On step.

        Returns:
            bool: Boolean flag.
        """
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
            self.writer.flush()
        return True  # info: don't request early abort
