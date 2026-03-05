"""Tests for predictive MPPI planner wiring and deterministic behavior."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.predictive_mppi import (
    PredictiveMPPIAdapter,
    build_predictive_mppi_config,
)


def _obs(
    *, robot=(0.0, 0.0), heading=0.0, goal=(2.0, 0.0), ped_positions=None, ped_velocities=None
) -> dict[str, object]:
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([0.2], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
            "radius": 0.25,
        },
    }


class _StubPredictor:
    def __init__(self, future: np.ndarray, *, anchor: tuple[float, float] = (0.3, 0.0)) -> None:
        self.future = future
        self.anchor = anchor
        self.config = type("Cfg", (), {"predictive_rollout_dt": 0.2})

    def _socnav_fields(self, observation: dict[str, object]) -> tuple[dict, dict, dict]:
        return observation["robot"], observation["goal"], observation["pedestrians"]  # type: ignore[index]

    def _as_1d_float(self, values: object, *, pad: int | None = None) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(values, dtype=float))
        if pad is not None and arr.size < pad:
            arr = np.pad(arr, (0, pad - arr.size), constant_values=0.0)
        return arr

    def _build_model_input(
        self, observation: dict[str, object]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        ped = observation["pedestrians"]  # type: ignore[index]
        count = int(np.asarray(ped["count"], dtype=float)[0])  # type: ignore[index]
        return (
            np.zeros((count, 4), dtype=np.float32),
            np.ones((count,), dtype=np.float32),
            np.zeros(2, dtype=float),
            0.0,
        )

    def _predict_trajectories(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del state, mask
        return self.future

    def _effective_rollout_steps(self, *, future_peds: np.ndarray, mask: np.ndarray) -> int:
        del mask
        return int(future_peds.shape[1])

    def _risk_speed_cap_ratio(self, *, future_peds: np.ndarray, mask: np.ndarray) -> float:
        del future_peds, mask
        return 1.0

    def _min_obstacle_clearance(self, point: np.ndarray, observation: dict[str, object]) -> float:
        del point, observation
        return 10.0

    def _path_penalty(
        self,
        *,
        robot_pos: np.ndarray,
        direction: np.ndarray,
        observation: dict[str, object],
        base_distance: float,
        num_samples: int,
    ) -> tuple[float, float]:
        del robot_pos, direction, observation, base_distance, num_samples
        return 0.0, 0.0

    def plan(self, observation: dict[str, object]) -> tuple[float, float]:
        del observation
        return self.anchor


def test_predictive_mppi_is_deterministic_for_fixed_seed() -> None:
    """Two planners with identical seed and predictor should return same action."""
    cfg = build_predictive_mppi_config({"random_seed": 7, "sample_count": 24, "iterations": 2})
    planner_a = PredictiveMPPIAdapter(cfg, allow_fallback=True)
    planner_b = PredictiveMPPIAdapter(cfg, allow_fallback=True)
    future = np.zeros((2, 8, 2), dtype=np.float32)
    future[0, :, :] = np.array([0.9, 0.2], dtype=np.float32)
    future[1, :, :] = np.array([1.1, -0.2], dtype=np.float32)
    planner_a._predictor = _StubPredictor(future, anchor=(0.35, 0.0))
    planner_b._predictor = _StubPredictor(future, anchor=(0.35, 0.0))
    observation = _obs(ped_positions=[(0.9, 0.2), (1.1, -0.2)])
    assert planner_a.plan(observation) == planner_b.plan(observation)


def test_predictive_mppi_stops_at_goal() -> None:
    """Planner should stop immediately when already within goal tolerance."""
    cfg = build_predictive_mppi_config({"goal_tolerance": 0.3})
    planner = PredictiveMPPIAdapter(cfg, allow_fallback=True)
    planner._predictor = _StubPredictor(np.zeros((0, 8, 2), dtype=np.float32))
    assert planner.plan(_obs(goal=(0.1, 0.0))) == (0.0, 0.0)


def test_predictive_mppi_falls_back_to_stop_for_immediate_conflict() -> None:
    """Unsafe immediate sequences should be rejected in favor of stop."""
    cfg = build_predictive_mppi_config(
        {
            "random_seed": 7,
            "sample_count": 24,
            "iterations": 2,
            "hard_ped_clearance": 0.6,
            "first_step_ped_clearance": 0.7,
        }
    )
    planner = PredictiveMPPIAdapter(cfg, allow_fallback=True)
    future = np.zeros((1, 8, 2), dtype=np.float32)
    future[0, :, :] = np.array([0.35, 0.0], dtype=np.float32)
    planner._predictor = _StubPredictor(future, anchor=(0.4, 0.0))
    linear, angular = planner.plan(_obs(ped_positions=[(0.35, 0.0)]))
    assert linear == 0.0
    assert abs(angular) <= cfg.max_angular_speed


def test_build_predictive_mppi_config_preserves_root_and_predictive_fields() -> None:
    """Builder should preserve both MPPI-root and predictive-root configuration values."""
    cfg = build_predictive_mppi_config(
        {
            "random_seed": 13,
            "sample_count": 88,
            "goal_progress_weight": 7.7,
            "predictive_goal_weight": 9.1,
            "predictive_model_id": "predictive_proxy_selected_v2",
        }
    )
    assert cfg.random_seed == 13
    assert cfg.sample_count == 88
    assert abs(cfg.goal_progress_weight - 7.7) < 1e-9
    assert abs(cfg.socnav.predictive_goal_weight - 9.1) < 1e-9
    assert cfg.socnav.predictive_model_id == "predictive_proxy_selected_v2"
    assert abs(cfg.hard_ped_clearance - 0.62) < 1e-9
