"""Tests for predictive planner rollout collection dataset features."""

from __future__ import annotations

import numpy as np

from scripts.training import collect_predictive_planner_data as collect


def _frame(
    *,
    robot_pos: tuple[float, float] = (0.0, 0.0),
    robot_heading: float = 0.0,
    robot_speed: tuple[float, float] = (0.0, 0.0),
    goal_current: tuple[float, float] = (1.0, 0.0),
    ped_positions: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
) -> collect.Frame:
    positions_source = [(1.0, 0.0)] if ped_positions is None else ped_positions
    if ped_velocities is None:
        velocities_source = [(0.1, 0.0)] * len(positions_source)
    else:
        velocities_source = ped_velocities
    positions = np.asarray(positions_source, dtype=np.float32).reshape(-1, 2)
    velocities = np.asarray(velocities_source, dtype=np.float32).reshape(-1, 2)
    return collect.Frame(
        robot_pos=np.asarray(robot_pos, dtype=np.float32),
        robot_heading=float(robot_heading),
        robot_speed=np.asarray(robot_speed, dtype=np.float32),
        goal_current=np.asarray(goal_current, dtype=np.float32),
        ped_positions_world=positions,
        ped_velocities_world=velocities,
        ped_count=int(positions.shape[0]),
    )


def test_extract_frame_reads_robot_speed_and_goal_from_flat_observation() -> None:
    """Flat observations should expose the extra ego-conditioning features."""
    obs = {
        "robot_position": [1.0, 2.0],
        "robot_heading": [0.5],
        "robot_speed": [0.4, -0.2],
        "goal_current": [4.0, 6.0],
        "pedestrians_positions": [2.0, 3.0, 5.0, 7.0],
        "pedestrians_velocities": [0.0, 0.1, -0.2, 0.3],
        "pedestrians_count": [2],
    }

    frame = collect._extract_frame(obs, max_agents=4)

    assert np.allclose(frame.robot_pos, np.array([1.0, 2.0], dtype=np.float32))
    assert frame.robot_heading == 0.5
    assert np.allclose(frame.robot_speed, np.array([0.4, -0.2], dtype=np.float32))
    assert np.allclose(frame.goal_current, np.array([4.0, 6.0], dtype=np.float32))
    assert frame.ped_count == 2


def test_frames_to_samples_defaults_to_legacy_state_dim() -> None:
    """Without ego conditioning the collector should preserve the 4D baseline contract."""
    frames = [
        _frame(
            ped_positions=[(1.0, 0.0)],
            ped_velocities=[(0.1, 0.0)],
        ),
        _frame(
            ped_positions=[(1.2, 0.0)],
            ped_velocities=[(0.1, 0.0)],
        ),
    ]

    state, target, mask, target_mask = collect._frames_to_samples(
        frames,
        max_agents=3,
        horizon_steps=1,
        ego_conditioning=False,
    )

    assert state.shape == (1, 3, 4)
    assert target.shape == (1, 3, 1, 2)
    assert mask.shape == (1, 3)
    assert target_mask.shape == (1, 3, 1)
    assert np.allclose(state[0, 0], np.array([1.0, 0.0, 0.1, 0.0], dtype=np.float32))


def test_frames_to_samples_adds_ego_conditioning_features() -> None:
    """Ego-conditioned mode should append speed, goal direction, and goal distance."""
    frames = [
        _frame(
            robot_speed=(0.5, -0.1),
            goal_current=(3.0, 4.0),
            ped_positions=[(1.0, 0.0)],
            ped_velocities=[(0.1, 0.2)],
        ),
        _frame(
            robot_speed=(0.5, -0.1),
            goal_current=(3.0, 4.0),
            ped_positions=[(1.2, 0.2)],
            ped_velocities=[(0.1, 0.2)],
        ),
    ]

    state, _target, mask, target_mask = collect._frames_to_samples(
        frames,
        max_agents=2,
        horizon_steps=1,
        ego_conditioning=True,
    )

    assert state.shape == (1, 2, 9)
    assert np.allclose(state[0, 0, 0:4], np.array([1.0, 0.0, 0.1, 0.2], dtype=np.float32))
    assert np.allclose(state[0, 0, 4:6], np.array([0.5, -0.1], dtype=np.float32))
    assert np.allclose(state[0, 0, 6:8], np.array([0.6, 0.8], dtype=np.float32), atol=1e-6)
    assert state[0, 0, 8] == 5.0
    assert mask[0, 0] == 1.0
    assert target_mask[0, 0, 0] == 1.0


def test_parse_args_accepts_ego_conditioning_flag(monkeypatch) -> None:
    """CLI parser should expose the ego-conditioning toggle for standalone collection."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_predictive_planner_data.py",
            "--episodes",
            "2",
            "--ego-conditioning",
        ],
    )

    args = collect.parse_args()

    assert args.episodes == 2
    assert args.ego_conditioning is True
