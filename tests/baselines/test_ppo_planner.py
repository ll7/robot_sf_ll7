"""Unit tests for PPO planner helper and fallback paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from gymnasium import spaces

from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig
from robot_sf.baselines.social_force import Observation


def _planner_config(**overrides):
    """Build a fallback-enabled PPO planner config with optional overrides."""
    payload = {
        "model_path": "/nonexistent/model.zip",
        "fallback_to_goal": True,
    }
    payload.update(overrides)
    return PPOPlannerConfig(**payload)


def _obs() -> Observation:
    """Build a compact social-force observation for PPO helper tests."""
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


def test_planner_rejects_local_output_model_path_even_with_fallback() -> None:
    """Direct PPO construction should share the benchmark local-artifact boundary."""
    with pytest.raises(ValueError, match="local-only model artifact"):
        PPOPlanner(
            {
                "model_path": "output/model_cache/ppo/model.zip",
                "fallback_to_goal": True,
            }
        )


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


def test_build_model_obs_dict_flattens_structured_socnav_observation() -> None:
    """PPO dict mode should align structured SocNav observations to flat SB3 keys."""
    planner = PPOPlanner(_planner_config(obs_mode="dict"))
    planner._model = SimpleNamespace(
        observation_space=SimpleNamespace(
            spaces={
                "robot_position": SimpleNamespace(shape=(2,), dtype=np.float32),
                "robot_speed": SimpleNamespace(shape=(2,), dtype=np.float32),
                "goal_current": SimpleNamespace(shape=(2,), dtype=np.float32),
                "pedestrians_positions": SimpleNamespace(shape=(1, 2), dtype=np.float32),
                "sim_timestep": SimpleNamespace(shape=(1,), dtype=np.float32),
            },
        ),
    )

    converted = planner._build_model_obs_dict(
        {
            "robot": {
                "position": np.array([1.0, 2.0], dtype=np.float32),
                "velocity_xy": np.array([0.3, 0.4], dtype=np.float32),
            },
            "goal": {"current": np.array([5.0, 6.0], dtype=np.float32)},
            "pedestrians": {"positions": np.array([[2.0, 3.0]], dtype=np.float32)},
            "sim": {"timestep": np.array([0.1], dtype=np.float32)},
        }
    )

    assert converted["robot_position"].tolist() == pytest.approx([1.0, 2.0])
    assert converted["robot_speed"].tolist() == pytest.approx([0.3, 0.4])
    assert converted["goal_current"].tolist() == pytest.approx([5.0, 6.0])
    assert converted["pedestrians_positions"].shape == (1, 2)
    assert converted["sim_timestep"].tolist() == pytest.approx([0.1])


def test_step_dict_mode_flattens_runtime_dict_for_box_checkpoint() -> None:
    """BC checkpoints trained with FlattenObservation should receive flat Box inputs."""
    planner = PPOPlanner(_planner_config(obs_mode="dict", action_space="unicycle"))
    captured: dict[str, np.ndarray | bool] = {}

    class _Model:
        """Flat-Box model stub that records prediction observations."""

        observation_space = spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)

        def predict(self, obs, deterministic: bool = True):
            """Record the flattened observation and return a unicycle action."""
            captured["obs"] = np.asarray(obs)
            captured["deterministic"] = deterministic
            return np.array([0.4, -0.1], dtype=np.float32), None

    planner._model = _Model()
    planner._status = "ok"
    planner.bind_env(
        SimpleNamespace(
            observation_space=spaces.Dict(
                {
                    "robot_position": spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(2,),
                        dtype=np.float32,
                    ),
                    "sim_timestep": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                }
            )
        )
    )

    action = planner.step(
        {
            "robot_position": np.array([1.0, 2.0], dtype=np.float32),
            "sim_timestep": np.array([0.2], dtype=np.float32),
        }
    )

    assert action == {"v": pytest.approx(0.4), "omega": pytest.approx(-0.1)}
    assert np.asarray(captured["obs"]).shape == (3,)
    assert np.asarray(captured["obs"]).dtype == np.float32


def test_bind_env_rejects_flat_box_checkpoint_shape_mismatch() -> None:
    """Runtime binding should fail before benchmark execution on adapter mismatch."""
    planner = PPOPlanner(_planner_config(obs_mode="dict"))
    planner._model = SimpleNamespace(
        observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="does not flatten to the checkpoint Box shape"):
        planner.bind_env(
            SimpleNamespace(
                observation_space=spaces.Dict(
                    {
                        "robot_position": spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=(2,),
                            dtype=np.float32,
                        )
                    }
                )
            )
        )


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
        """Predictive encoder stub that records the normalized observation."""

        def encode(self, obs):
            """Return deterministic predictive features for model backfill."""
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
        """Predictive encoder stub that should not overwrite existing fields."""

        def encode(self, _obs):
            """Return a sentinel feature value for overwrite detection."""
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
        """Predictive encoder constructor stub that records rebuild settings."""

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
        """Resolve registry ids to the temporary checkpoint path."""
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


def test_issue_791_portable_baseline_uses_registry_and_auto_device(monkeypatch, tmp_path):
    """The portable issue-791 baseline should load through model_id on CPU-safe settings."""
    config_path = Path("configs/baselines/ppo_issue_791_eval_aligned_large_capacity_portable.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved_model = tmp_path / "model.zip"
    resolved_model.write_text("checkpoint", encoding="utf-8")
    called = {}

    def _fake_resolve(model_id: str):
        """Resolve the portable baseline model id to the temporary checkpoint."""
        called["model_id"] = model_id
        return resolved_model

    monkeypatch.setattr("robot_sf.baselines.ppo.resolve_model_path", _fake_resolve)
    monkeypatch.setattr(
        "robot_sf.baselines.ppo.PPO",
        SimpleNamespace(load=lambda path, **kwargs: {"path": path, **kwargs}),
    )

    planner = PPOPlanner(config)

    assert "model_path" not in config
    assert config["predictive_foresight_device"] == "auto"
    assert called["model_id"] == config["model_id"]
    assert planner._model["path"] == str(resolved_model)


def test_issue_791_cpu_baseline_forces_cpu_inference(monkeypatch, tmp_path):
    """The issue-1554 CPU-safe baseline must not load PPO or foresight on CUDA."""
    config_path = Path("configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved_model = tmp_path / "model.zip"
    resolved_model.write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr(
        "robot_sf.baselines.ppo.resolve_model_path", lambda _model_id: resolved_model
    )
    monkeypatch.setattr(
        "robot_sf.baselines.ppo.PPO",
        SimpleNamespace(load=lambda path, **kwargs: {"path": path, **kwargs}),
    )

    planner = PPOPlanner(config)

    assert config["device"] == "cpu"
    assert config["predictive_foresight_device"] == "cpu"
    assert planner._model["device"] == "cpu"


def test_load_model_resolution_failure_falls_back(monkeypatch):
    """Registry resolution failures should trigger fallback_to_goal when enabled."""
    monkeypatch.setattr(
        "robot_sf.baselines.ppo.resolve_model_path",
        lambda _model_id: (_ for _ in ()).throw(KeyError("missing")),
    )
    planner = PPOPlanner(_planner_config(model_id="ppo_demo", model_path="unused.zip"))
    assert planner.get_metadata()["status"] == "fallback"
    assert planner.get_metadata()["fallback_reason"] == "model_resolution_failed"
