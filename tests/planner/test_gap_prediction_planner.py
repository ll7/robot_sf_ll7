"""Tests for the gap-aware predictive planner."""

from __future__ import annotations

from robot_sf.planner.gap_prediction import (
    GapAwarePredictionAdapter,
    GapPredictionConfig,
    build_gap_prediction_config,
)
from robot_sf.planner.socnav import SocNavPlannerConfig
from robot_sf.planner.stream_gap import StreamGapPlannerConfig


class _StubHead:
    def __init__(self, command: tuple[float, float]) -> None:
        self.command = command

    def plan(self, observation: dict[str, object]) -> tuple[float, float]:
        del observation
        return self.command


def test_gap_prediction_stops_when_gap_head_blocks() -> None:
    """Gap veto should stop the predictive head when the corridor is blocked."""
    adapter = GapAwarePredictionAdapter(
        GapPredictionConfig(
            stream_gap=StreamGapPlannerConfig(wait_speed=0.0, approach_speed=0.4),
            predictive=SocNavPlannerConfig(),
        )
    )
    adapter._gap = _StubHead((0.0, 0.3))
    adapter._prediction = _StubHead((0.9, 0.1))
    assert adapter.plan({}) == (0.0, 0.3)


def test_gap_prediction_caps_predictive_speed_in_approach_mode() -> None:
    """Approach mode should keep predictive steering but cap speed."""
    adapter = GapAwarePredictionAdapter(
        GapPredictionConfig(
            stream_gap=StreamGapPlannerConfig(wait_speed=0.0, approach_speed=0.4),
            predictive=SocNavPlannerConfig(),
            approach_speed_cap=0.35,
        )
    )
    adapter._gap = _StubHead((0.3, 0.2))
    adapter._prediction = _StubHead((0.8, 0.1))
    assert adapter.plan({}) == (0.35, 0.1)


def test_gap_prediction_uses_predictive_command_when_gap_is_open() -> None:
    """Open-gap mode should pass through the predictive command."""
    adapter = GapAwarePredictionAdapter(
        GapPredictionConfig(
            stream_gap=StreamGapPlannerConfig(wait_speed=0.0, approach_speed=0.4),
            predictive=SocNavPlannerConfig(),
        )
    )
    adapter._gap = _StubHead((0.9, 0.0))
    adapter._prediction = _StubHead((0.7, -0.2))
    assert adapter.plan({}) == (0.7, -0.2)


def test_build_gap_prediction_config_preserves_predictive_and_gap_fields() -> None:
    """Builder should preserve both predictive-root and nested gap config values."""
    cfg = build_gap_prediction_config(
        {
            "predictive_goal_weight": 9.1,
            "approach_speed_cap": 0.31,
            "stream_gap": {"commit_speed": 0.88},
        }
    )
    assert abs(cfg.predictive.predictive_goal_weight - 9.1) < 1e-9
    assert abs(cfg.approach_speed_cap - 0.31) < 1e-9
    assert abs(cfg.stream_gap.commit_speed - 0.88) < 1e-9
