"""Tests for TensorBoard training-metrics callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from stable_baselines3.common.logger import TensorBoardOutputFormat

from robot_sf.tb_logging import (
    AdversarialPedestrianMetricsCallback,
    AdversialPedestrianMetricsCallback,
    BaseMetricsCallback,
    DrivingMetricsCallback,
)

if TYPE_CHECKING:
    from pathlib import Path


class _FakeLogger:
    """Minimal SB3 logger stub exposing output formats."""

    def __init__(self, output_formats: list[object]) -> None:
        self.output_formats = output_formats


class _FakeModel:
    """Minimal SB3 model stub exposing the logger property target."""

    def __init__(self, output_formats: list[object]) -> None:
        self.logger = _FakeLogger(output_formats)


class _RecordingWriter:
    """Collect TensorBoard scalar writes for assertions."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_calls = 0

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def flush(self) -> None:
        self.flush_calls += 1


class _DrivingMetricsStub:
    """Expose the metric attributes consumed by DrivingMetricsCallback."""

    route_completion_rate = 0.1
    interm_goal_completion_rate = 0.2
    timeout_rate = 0.3
    obstacle_collision_rate = 0.4
    pedestrian_collision_rate = 0.5

    def __init__(self) -> None:
        self.updated_with: list[dict] | None = None

    def update(self, meta_dicts: list[dict]) -> None:
        self.updated_with = meta_dicts


class _PedMetricsStub:
    """Expose the metric attributes consumed by AdversarialPedestrianMetricsCallback."""

    timeout_rate = 0.1
    obstacle_collision_rate = 0.2
    pedestrian_collision_rate = 0.3
    robot_collision_rate = 0.4
    robot_at_goal_rate = 0.5
    robot_obstacle_collision_rate = 0.6
    robot_pedestrian_collision_rate = 0.7
    route_end_distance = 1.5
    avg_ego_ped_speed_at_collision = 0.8
    avg_collision_impact_angle_rad_at_collision = 0.9

    def __init__(self) -> None:
        self.updated_with: list[dict] | None = None

    def update(self, meta_dicts: list[dict]) -> None:
        self.updated_with = meta_dicts


def test_base_metrics_callback_extracts_meta_and_uses_tensorboard_formatter(
    tmp_path: Path,
) -> None:
    """Initialize the TensorBoard writer from a matching SB3 output formatter."""
    callback = BaseMetricsCallback()
    callback.locals = {"infos": [{"meta": {"episode": 1}}, {"meta": {"episode": 2}}]}
    callback.n_calls = 1000

    tb_formatter = TensorBoardOutputFormat(str(tmp_path))
    callback.model = _FakeModel([object(), tb_formatter])

    try:
        assert callback.meta_dicts == [{"episode": 1}, {"episode": 2}]
        assert callback.is_logging_step is True

        callback._on_training_start()

        assert callback.writer is tb_formatter.writer
    finally:
        tb_formatter.close()


def test_base_metrics_callback_handles_missing_tensorboard_and_is_abstract() -> None:
    """Keep writer unset without a TensorBoard formatter and require subclass override."""
    callback = BaseMetricsCallback()
    callback.model = _FakeModel([object()])
    callback.n_calls = 1

    callback._on_training_start()

    assert callback.writer is None
    assert callback.is_logging_step is False
    with pytest.raises(NotImplementedError):
        callback._on_step()


def test_driving_metrics_callback_logs_expected_scalars() -> None:
    """Log driving metrics when the callback reaches a logging step."""
    callback = DrivingMetricsCallback(num_envs=2)
    callback.metrics = _DrivingMetricsStub()
    callback.writer = _RecordingWriter()
    callback.locals = {"infos": [{"meta": {"timeout": False}}, {"meta": {"timeout": True}}]}
    callback.n_calls = 1000
    callback.num_timesteps = 4321

    result = callback._on_step()

    assert result is True
    assert callback.metrics.updated_with == [{"timeout": False}, {"timeout": True}]
    assert callback.writer.scalars == [
        ("metrics/route_completion_rate", 0.1, 4321),
        ("metrics/interm_goal_completion_rate", 0.2, 4321),
        ("metrics/timeout_rate", 0.3, 4321),
        ("metrics/obstacle_collision_rate", 0.4, 4321),
        ("metrics/pedestrian_collision_rate", 0.5, 4321),
    ]
    assert callback.writer.flush_calls == 1


def test_driving_metrics_callback_skips_writer_when_not_logging_step() -> None:
    """Update metrics every step but emit no scalars off the logging cadence."""
    callback = DrivingMetricsCallback(num_envs=1)
    callback.metrics = _DrivingMetricsStub()
    callback.writer = _RecordingWriter()
    callback.locals = {"infos": [{"meta": {"timeout": False}}]}
    callback.n_calls = 999
    callback.num_timesteps = 12

    result = callback._on_step()

    assert result is True
    assert callback.metrics.updated_with == [{"timeout": False}]
    assert callback.writer.scalars == []
    assert callback.writer.flush_calls == 0


def test_adversarial_pedestrian_metrics_callback_logs_extended_collision_metrics() -> None:
    """Log the pedestrian-specific collision and kinematics metrics."""
    callback = AdversarialPedestrianMetricsCallback(num_envs=2)
    callback.metrics = _PedMetricsStub()
    callback.writer = _RecordingWriter()
    callback.locals = {"infos": [{"meta": {"episode": 1}}, {"meta": {"episode": 2}}]}
    callback.n_calls = 1000
    callback.num_timesteps = 987

    result = callback._on_step()

    assert result is True
    assert callback.metrics.updated_with == [{"episode": 1}, {"episode": 2}]
    assert callback.writer.scalars == [
        ("metrics/timeout_rate", 0.1, 987),
        ("metrics/obstacle_collision_rate", 0.2, 987),
        ("metrics/pedestrian_collision_rate", 0.3, 987),
        ("metrics/robot_collision_rate", 0.4, 987),
        ("metrics/robot_at_goal_rate", 0.5, 987),
        ("metrics/robot_obstacle_collision_rate", 0.6, 987),
        ("metrics/robot_pedestrian_collision_rate", 0.7, 987),
        ("metrics/avg_distance_to_robot", 1.5, 987),
        ("metrics/avg_ego_ped_speed_at_collision", 0.8, 987),
        ("metrics/avg_collision_impact_angle_rad", 0.9, 987),
    ]
    assert callback.writer.flush_calls == 1


def test_adversarial_pedestrian_metrics_callback_legacy_alias_remains_available() -> None:
    """Preserve the legacy callback name while switching callers to the corrected spelling."""
    assert AdversialPedestrianMetricsCallback is AdversarialPedestrianMetricsCallback
