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


def test_build_model_obs_dict_backfills_predictive_features(monkeypatch):
    """Planner should synthesize predictor-derived keys when the model expects them."""
    planner = PPOPlanner(_planner_config(obs_mode="dict", predictive_foresight_enabled=True))
    planner._model = SimpleNamespace(
        observation_space=SimpleNamespace(
            spaces={
                "predictive_min_clearance": SimpleNamespace(shape=(1,), dtype=np.float32),
                "predictive_gap_scores": SimpleNamespace(shape=(2,), dtype=np.float32),
            },
        ),
    )
    seen: dict[str, object] = {}

    class _DummyEncoder:
        def encode(self, obs):
            seen.update(obs)
            return {
                "min_clearance": np.array([1.5], dtype=np.float32),
                "gap_scores": np.array([0.4, 0.6], dtype=np.float32),
            }

    planner._predictive_foresight = _DummyEncoder()
    converted = planner._build_model_obs_dict(
        {
            "robot_position": [0.0, 0.0],
            "robot_heading": [0.0],
            "goal_current": [1.0, 0.0],
            "pedestrians_positions": [[1.0, 0.5]],
            "pedestrians_velocities": [[0.1, 0.0]],
            "pedestrians_count": [1.0],
        }
    )
    assert converted["predictive_min_clearance"][0] == pytest.approx(1.5)
    assert converted["predictive_gap_scores"].shape == (2,)
    assert set(seen) >= {"robot", "goal", "pedestrians", "map", "sim"}


def test_build_model_obs_dict_preserves_existing_predictive_features() -> None:
    """Planner should not overwrite predictive_* keys already present in the observation."""
    planner = PPOPlanner(_planner_config(obs_mode="dict", predictive_foresight_enabled=True))
    planner._model = SimpleNamespace(
        observation_space=SimpleNamespace(
            spaces={"predictive_min_clearance": SimpleNamespace(shape=(1,), dtype=np.float32)},
        ),
    )

    class _DummyEncoder:
        def encode(self, _obs):
            return {"min_clearance": np.array([9.0], dtype=np.float32)}

    planner._predictive_foresight = _DummyEncoder()
    converted = planner._build_model_obs_dict(
        {"predictive_min_clearance": np.array([1.25], dtype=np.float32)}
    )
    assert converted["predictive_min_clearance"][0] == pytest.approx(1.25)


def test_configure_rebuilds_predictive_foresight_encoder(monkeypatch) -> None:
    """configure() should rebuild foresight encoder when predictive settings change."""
    built: list[tuple[str, int]] = []

    class _DummyEncoder:
        def __init__(self, config):
            built.append((config.model_id, config.horizon_steps))

    monkeypatch.setattr("robot_sf.baselines.ppo.PredictiveForesightEncoder", _DummyEncoder)
    planner = PPOPlanner(
        _planner_config(
            predictive_foresight_enabled=True,
            predictive_foresight_model_id="predictive_a",
            predictive_foresight_horizon_steps=8,
        )
    )
    planner.configure(
        _planner_config(
            predictive_foresight_enabled=True,
            predictive_foresight_model_id="predictive_b",
            predictive_foresight_horizon_steps=12,
        )
    )
    assert built == [("predictive_a", 8), ("predictive_b", 12)]


def test_get_metadata_redacts_predictive_checkpoint_path() -> None:
    """Planner metadata should not expose local predictive checkpoint paths."""
    planner = PPOPlanner(
        _planner_config(
            model_path="/tmp/models/policy/model.zip",
            predictive_foresight_checkpoint_path="/tmp/models/predictive/model.pt",
        )
    )

    metadata = planner.get_metadata()
    assert metadata["config"]["model_path"] == "model.zip"
    assert metadata["config"]["predictive_foresight_checkpoint_path"] == "model.pt"


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


def test_load_model_resolves_registry_model_id(monkeypatch, tmp_path):
    """Registry-backed model ids should resolve before PPO.load is called."""
    resolved_model = tmp_path / "model.zip"
    resolved_model.write_text("checkpoint", encoding="utf-8")
    called = {}

    def _fake_resolve(model_id: str):
        called["model_id"] = model_id
        return resolved_model

    monkeypatch.setattr("robot_sf.baselines.ppo.resolve_model_path", _fake_resolve)
    monkeypatch.setattr(
        "robot_sf.baselines.ppo.PPO",
        SimpleNamespace(load=lambda path, **kwargs: {"path": path, **kwargs}),
    )
    planner = PPOPlanner(_planner_config(model_id="ppo_demo", model_path="unused.zip"))
    assert called["model_id"] == "ppo_demo"
    assert planner._model["path"] == str(resolved_model)


def test_load_model_resolution_failure_falls_back(monkeypatch):
    """Registry resolution failures should trigger fallback_to_goal when enabled."""
    monkeypatch.setattr(
        "robot_sf.baselines.ppo.resolve_model_path",
        lambda _model_id: (_ for _ in ()).throw(KeyError("missing")),
    )
    planner = PPOPlanner(_planner_config(model_id="ppo_demo", model_path="unused.zip"))
    assert planner.get_metadata()["status"] == "fallback"
    assert planner.get_metadata()["fallback_reason"] == "model_resolution_failed"
