"""End-to-end checkpoint-backed model-based action selection for issue #4013.

The existing tests prove the learned predictor *loads* a trained checkpoint
(`tests/planner/test_learned_short_horizon_trainer.py`) and that the MPC adapter
wires the predictor's ``predict`` into the prediction-MPC constraints
(`tests/planner/test_learned_short_horizon_predictor.py`). Neither exercises the
issue #4013 Definition-of-Done clause "model-based action selection runs on a
smoke scenario": the adapter's ``plan`` was never called against a *trained*
checkpoint to confirm a bounded unicycle command is actually emitted without
fallback.

These tests close that gap. They train a tiny CPU checkpoint, build the
``learned_prediction_mpc`` adapter on that checkpoint (fail-closed, no untrained
smoke, no constant-velocity fallback), and assert:

* the predictor reports ``evidence_tier=checkpoint_loaded`` (real learned weights);
* ``plan`` emits a finite, bounded, goal-directed command in open space;
* ``plan`` still emits a bounded command when a pedestrian sits in the path
  (model-based avoidance runs rather than crashing);
* a configured-but-missing checkpoint fails closed instead of silently degrading.

Claim boundary: diagnostic-only. This proves the checkpoint-backed model-based
selection *path executes*; it is not benchmark, navigation-quality, or
paper-facing evidence, and it is not the paired 3-arm smoke comparison that
remains open on issue #4013.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.learned_prediction_mpc import build_learned_prediction_mpc_adapter
from robot_sf.planner.learned_short_horizon_trainer import (
    ShortHorizonTrainerConfig,
    train_short_horizon_predictor,
)

# Small predictor geometry shared by the trainer and the inference adapter so the
# checkpoint state dict loads without a shape mismatch. These must stay in sync.
_MAX_PEDS = 3
_HORIZON = 2
_HIDDEN = 16


def _obs(
    *,
    goal: tuple[float, float] = (3.0, 0.0),
    ped_positions: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Build a compact SocNav observation payload for the adapter ``plan`` call."""

    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray([0.0, 0.0], dtype=float),
            "heading": np.asarray([0.0], dtype=float),
            "speed": np.asarray([0.0], dtype=float),
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
            "radius": np.asarray([0.25], dtype=float),
        },
    }


def _train_checkpoint(tmp_path) -> str:
    """Train a fast tiny CPU checkpoint and return its path."""

    config = ShortHorizonTrainerConfig(
        max_pedestrians=_MAX_PEDS,
        horizon_steps=_HORIZON,
        hidden_dim=_HIDDEN,
        num_samples=64,
        epochs=60,
        seed=4013,
        output_dir=str(tmp_path / "short_horizon"),
    )
    result = train_short_horizon_predictor(config)
    assert result.checkpoint_path.exists()
    return str(result.checkpoint_path)


def _checkpoint_algo_config(checkpoint_path: str) -> dict[str, object]:
    """Build the fail-closed checkpoint-backed adapter config."""

    return {
        "allow_testing_algorithms": True,
        "allow_untrained_smoke": False,
        "fallback_to_constant_velocity": False,
        "checkpoint_path": checkpoint_path,
        "max_pedestrians": _MAX_PEDS,
        "horizon_steps": _HORIZON,
        "hidden_dim": _HIDDEN,
        "rollout_dt": 0.2,
        "solver_max_iterations": 16,
    }


def test_checkpoint_backed_adapter_loads_without_fallback(tmp_path) -> None:
    """Adapter built on a trained checkpoint reports checkpoint-loaded, not fallback."""

    pytest.importorskip("torch")
    checkpoint_path = _train_checkpoint(tmp_path)
    planner = build_learned_prediction_mpc_adapter(_checkpoint_algo_config(checkpoint_path))

    diagnostics = planner.diagnostics()
    assert diagnostics["predictor"]["backend"] == "learned_short_horizon"
    assert diagnostics["predictor"]["evidence_tier"] == "checkpoint_loaded"
    assert diagnostics["predictor"]["diagnostic_only"] is False

    futures = planner._future_predictor.predict(_obs(), horizon_steps=_HORIZON, dt=0.2)
    assert futures.source == "checkpoint_loaded"
    assert np.all(np.isfinite(futures.positions_world))


def test_checkpoint_backed_model_based_selection_runs_in_open_space(tmp_path) -> None:
    """``plan`` emits a finite, bounded, goal-directed command from learned weights."""

    pytest.importorskip("torch")
    checkpoint_path = _train_checkpoint(tmp_path)
    planner = build_learned_prediction_mpc_adapter(_checkpoint_algo_config(checkpoint_path))

    linear, angular = planner.plan(_obs(goal=(3.0, 0.0)))

    assert np.isfinite(linear)
    assert np.isfinite(angular)
    # Open-space command should advance toward the goal within control bounds.
    assert linear > 0.0
    assert linear <= planner.config.max_linear_speed
    assert abs(angular) <= planner.config.max_angular_speed
    # The plan came from the checkpoint-loaded predictor, not a fallback.
    assert planner._future_predictor.diagnostics()["last_source"] == "checkpoint_loaded"


def test_checkpoint_backed_model_based_selection_runs_with_pedestrian_in_path(tmp_path) -> None:
    """Model-based avoidance runs and stays bounded when a pedestrian blocks the path."""

    pytest.importorskip("torch")
    checkpoint_path = _train_checkpoint(tmp_path)
    planner = build_learned_prediction_mpc_adapter(_checkpoint_algo_config(checkpoint_path))

    linear, angular = planner.plan(
        _obs(
            goal=(3.0, 0.0),
            ped_positions=[(1.0, 0.0)],
            ped_velocities=[(0.0, 0.0)],
        )
    )

    assert np.isfinite(linear)
    assert np.isfinite(angular)
    assert 0.0 <= linear <= planner.config.max_linear_speed
    assert abs(angular) <= planner.config.max_angular_speed


def test_checkpoint_backed_adapter_fails_closed_on_missing_checkpoint(tmp_path) -> None:
    """A configured-but-missing checkpoint fails closed instead of degrading silently."""

    pytest.importorskip("torch")
    missing = str(tmp_path / "does_not_exist.pt")
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        build_learned_prediction_mpc_adapter(_checkpoint_algo_config(missing))
