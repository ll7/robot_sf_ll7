"""Unit tests for PPO planner helper and fallback paths."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig
from robot_sf.baselines.social_force import Observation


def _planner_config(**overrides):
    payload = {
        "model_path": "/nonexistent/model.zip",
        "fallback_to_goal": True,
    }
    payload.update(overrides)
    return PPOPlannerConfig(**payload)


def _obs() -> Observation:
    return Observation(
        dt=0.1,
        robot={"position": [1.0, 2.0], "velocity": [0.2, 0.1], "goal": [4.0, 6.0]},
        agents=[
            {"position": [2.0, 2.0]},
            {"position": [10.0, 10.0]},
        ],
        obstacles=[],
    )


def test_parse_config_rejects_invalid_type():
    """Planner should reject unsupported config payload types."""
    with pytest.raises(TypeError):
        PPOPlanner(123)  # type: ignore[arg-type]


def test_step_dict_mode_fallback_unicycle_sets_status_and_reason():
    """Dict obs mode should fallback to goal action when model inference is unavailable."""
    planner = PPOPlanner(_planner_config(obs_mode="dict", action_space="unicycle", v_max=1.5))
    action = planner.step({"robot_position": [0.0, 0.0], "goal_current": [2.0, 0.0]})
    assert set(action) == {"v", "omega"}
    assert action["v"] == pytest.approx(1.5)
    assert action["omega"] == pytest.approx(0.0)
    assert planner.get_metadata()["status"] == "fallback"
    assert planner.get_metadata()["fallback_reason"] in {"model_missing", "prediction_failed"}


def test_build_model_obs_dict_without_model_passthrough():
    """Without loaded model, dict observations should be returned as arrays."""
    planner = PPOPlanner(_planner_config(obs_mode="dict"))
    converted = planner._build_model_obs_dict({"a": [1, 2], "b": [3.0]})
    assert converted["a"].shape == (2,)
    assert converted["b"].shape == (1,)


def test_build_model_obs_dict_reshapes_to_model_space():
    """Dict observations should reshape to the loaded model's declared space shape."""
    planner = PPOPlanner(_planner_config(obs_mode="dict"))
    planner._model = SimpleNamespace(
        observation_space=SimpleNamespace(
            spaces={"occupancy_grid": SimpleNamespace(shape=(2, 2), dtype=np.float32)},
        ),
    )
    converted = planner._build_model_obs_dict({"occupancy_grid": [0, 1, 2, 3]})
    assert converted["occupancy_grid"].shape == (2, 2)
    assert converted["occupancy_grid"].dtype == np.float32


def test_build_model_obs_dict_raises_on_missing_key():
    """Planner should fail when required dict keys are missing."""
    planner = PPOPlanner(_planner_config(obs_mode="dict"))
    planner._model = SimpleNamespace(
        observation_space=SimpleNamespace(
            spaces={"required": SimpleNamespace(shape=(1,), dtype=np.float32)},
        ),
    )
    with pytest.raises(ValueError, match="Missing required dict observation keys"):
        planner._build_model_obs_dict({"other": [1.0]})


def test_build_model_obs_dict_raises_on_size_mismatch():
    """Planner should fail when value size cannot match expected shape."""
    planner = PPOPlanner(_planner_config(obs_mode="dict"))
    planner._model = SimpleNamespace(
        observation_space=SimpleNamespace(
            spaces={"x": SimpleNamespace(shape=(2, 2), dtype=np.float32)},
        ),
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        planner._build_model_obs_dict({"x": [1.0, 2.0, 3.0]})


def test_predict_action_success_and_error_paths():
    """Predict helper should return squeezed array on success and None on errors."""
    planner = PPOPlanner(_planner_config())
    planner._model = SimpleNamespace(
        predict=lambda *_args, **_kwargs: (np.array([0.2, -0.4]), None),
    )
    out = planner._predict_action(np.array([1.0, 2.0]))
    assert np.allclose(out, np.array([0.2, -0.4]))

    planner._model = SimpleNamespace(
        predict=lambda *_args, **_kwargs: (_ for _ in ()).throw(IndexError("bad")),
    )
    assert planner._predict_action(np.array([1.0, 2.0])) is None


def test_build_model_obs_image_requires_payload():
    """Image mode should require obs.robot['image'] and return it when provided."""
    planner = PPOPlanner(_planner_config(obs_mode="image"))
    with pytest.raises(ValueError, match="Image observation requested"):
        planner._build_model_obs(_obs())

    obs = _obs()
    obs.robot["image"] = np.ones((4, 4, 3), dtype=np.uint8)
    image = planner._build_model_obs(obs)
    assert image.shape == (4, 4, 3)


def test_vectorize_and_action_mapping_paths():
    """Vectorization and action mapping should work for both action spaces."""
    planner = PPOPlanner(_planner_config(obs_mode="vector", nearest_k=2, action_space="velocity"))
    vec = planner._build_model_obs(_obs())
    assert vec.shape == (8,)

    velocity_action = planner._action_vec_to_dict_from_array(np.array([10.0, 0.0]))
    assert velocity_action["vx"] <= planner.config.v_max + 1e-6

    planner_unicycle = PPOPlanner(
        _planner_config(action_space="unicycle", v_max=1.0, omega_max=0.5)
    )
    uni_action = planner_unicycle._action_vec_to_dict_from_array(np.array([2.0, 2.0]))
    assert uni_action == {"v": 1.0, "omega": 0.5}


def test_fallback_actions_and_metadata():
    """Fallback helpers should generate stop/action commands and metadata."""
    planner = PPOPlanner(_planner_config(action_space="unicycle", v_max=1.0))
    stop = planner._fallback_action_dict({"robot_position": [0.0, 0.0], "goal_current": [0.0, 0.0]})
    assert stop == {"v": 0.0, "omega": 0.0}

    move = planner._fallback_action(_obs())
    assert set(move) == {"v", "omega"}
    assert move["v"] >= 0.0

    planner.reset(seed=7)
    planner.close()
    planner._fallback_reason = "prediction_failed"
    meta = planner.get_metadata()
    assert meta["algorithm"] == "ppo"
    assert meta["fallback_reason"] == "prediction_failed"
