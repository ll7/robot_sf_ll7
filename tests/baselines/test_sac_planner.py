"""Tests for SAC benchmark adapter observation handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

import robot_sf.baselines.sac as sac_mod
from robot_sf.baselines.sac import SACPlanner
from robot_sf.baselines.social_force import Observation


class _DummySACModel:
    """Test double that records the latest observation passed to predict()."""

    def __init__(self, *, action: np.ndarray, observation_space: gym_spaces.Space | None = None):
        self.action = np.asarray(action, dtype=np.float32)
        self.observation_space = observation_space
        self.last_obs = None

    def predict(self, obs, deterministic: bool = True):
        self.last_obs = obs
        return self.action, None


def _make_observation() -> Observation:
    """Build a structured observation with one pedestrian for vector-mode tests."""
    return Observation(
        dt=0.1,
        robot={
            "position": np.array([1.0, 1.0], dtype=np.float32),
            "velocity": np.array([0.2, -0.1], dtype=np.float32),
            "goal": np.array([4.0, 4.0], dtype=np.float32),
            "heading": 0.0,
        },
        agents=[
            {"position": np.array([2.0, 3.0], dtype=np.float32)},
        ],
    )


def _planner_without_model(*, relative_obs: bool = True) -> SACPlanner:
    planner = SACPlanner(
        {
            "model_path": "output/models/sac/does_not_exist.zip",
            "fallback_to_goal": True,
            "relative_obs": relative_obs,
        }
    )
    planner._model = None
    return planner


def test_planner_loads_from_dict_config_and_reports_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Dict config input should parse, load, and serialize cleanly."""
    model_path = tmp_path / "sac_model.zip"
    model_path.write_text("stub", encoding="utf-8")
    fake_model = _DummySACModel(action=np.array([0.4, -0.2], dtype=np.float32))

    class _FakeSAC:
        @staticmethod
        def load(*_args, **_kwargs):
            assert Path(_args[0]) == model_path
            assert _kwargs["device"] == "cpu"
            return fake_model

    monkeypatch.setattr(sac_mod, "SAC", _FakeSAC)

    planner = SACPlanner(
        {
            "model_path": str(model_path),
            "device": "cpu",
            "obs_mode": "vector",
            "action_space": "unicycle",
        },
        seed=17,
    )

    assert planner._model is fake_model
    assert planner.get_metadata()["status"] == "ok"
    assert planner.get_metadata()["config"]["model_path"] == model_path.name
    planner.reset(seed=9)
    assert planner._seed == 9
    planner.close()
    assert planner._model is None


def test_planner_handles_missing_sb3_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Stable-Baselines3 should fail open only when fallback is enabled."""
    monkeypatch.setattr(sac_mod, "SAC", None)

    planner = SACPlanner(
        {
            "model_path": "output/models/sac/does_not_exist.zip",
            "fallback_to_goal": True,
        }
    )

    meta = planner.get_metadata()
    assert meta["status"] == "fallback"
    assert meta["fallback_reason"] == "sb3_missing"


def test_planner_missing_model_fails_closed_without_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing checkpoint should raise when fallback_to_goal is disabled."""
    missing_path = tmp_path / "missing.zip"

    class _FakeSAC:
        @staticmethod
        def load(*_args, **_kwargs):  # pragma: no cover - not reached
            raise AssertionError("load should not be called for a missing file")

    monkeypatch.setattr(sac_mod, "SAC", _FakeSAC)
    monkeypatch.setattr(sac_mod, "resolve_model_path", lambda _model_id: missing_path)
    monkeypatch.setattr(
        sac_mod,
        "raise_fatal_with_remedy",
        lambda message, remedy: (_ for _ in ()).throw(RuntimeError(message)),
    )

    with pytest.raises(RuntimeError, match="SAC model file not found"):
        SACPlanner({"model_id": "missing", "fallback_to_goal": False})


def test_planner_load_failure_can_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Checkpoint load failure should degrade to fallback navigation when enabled."""
    model_path = tmp_path / "broken.zip"
    model_path.write_text("stub", encoding="utf-8")

    class _FakeSAC:
        @staticmethod
        def load(*_args, **_kwargs):
            raise ValueError("bad checkpoint")

    monkeypatch.setattr(sac_mod, "SAC", _FakeSAC)

    planner = SACPlanner({"model_path": str(model_path), "fallback_to_goal": True})
    meta = planner.get_metadata()
    assert meta["status"] == "fallback"
    assert meta["fallback_reason"] == "model_load_failed"


def test_planner_load_failure_fails_closed_when_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Checkpoint load failure should raise when fallback_to_goal is disabled."""
    model_path = tmp_path / "broken.zip"
    model_path.write_text("stub", encoding="utf-8")

    class _FakeSAC:
        @staticmethod
        def load(*_args, **_kwargs):
            raise ValueError("bad checkpoint")

    monkeypatch.setattr(sac_mod, "SAC", _FakeSAC)
    monkeypatch.setattr(
        sac_mod,
        "raise_fatal_with_remedy",
        lambda message, remedy: (_ for _ in ()).throw(RuntimeError(message)),
    )

    with pytest.raises(RuntimeError, match="Failed to load SAC model"):
        SACPlanner({"model_path": str(model_path), "fallback_to_goal": False})


def test_step_vector_mode_uses_model_prediction_and_fallback_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Vector mode should vectorize structured obs, then fall back when the model disappears."""
    model_path = tmp_path / "vector.zip"
    model_path.write_text("stub", encoding="utf-8")
    fake_model = _DummySACModel(action=np.array([3.5, 1.4], dtype=np.float32))

    class _FakeSAC:
        @staticmethod
        def load(*_args, **_kwargs):
            return fake_model

    monkeypatch.setattr(sac_mod, "SAC", _FakeSAC)

    planner = SACPlanner(
        {
            "model_path": str(model_path),
            "obs_mode": "vector",
            "action_space": "unicycle",
            "action_semantics": "absolute",  # test that absolute mode clamps v and omega
            "nearest_k": 1,
            "fallback_to_goal": True,
        }
    )

    action = planner.step(_make_observation())
    assert action == {"v": 2.0, "omega": 1.0}  # clamped: v_max=2.0, omega_max=1.0
    assert isinstance(fake_model.last_obs, np.ndarray)
    assert fake_model.last_obs.shape == (6,)

    planner._model = None
    fallback = planner.step(_make_observation())
    assert fallback["v"] == pytest.approx(2.0)
    assert fallback["omega"] == pytest.approx(0.7853981633974483)


def test_step_dict_mode_applies_transform_and_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dict mode should preserve the checkpoint contract and honour the ego transform."""
    model_path = tmp_path / "dict.zip"
    model_path.write_text("stub", encoding="utf-8")
    spaces = gym_spaces.Dict(
        {
            "robot_position": gym_spaces.Box(-50.0, 50.0, shape=(2,), dtype=np.float32),
            "robot_heading": gym_spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
            "goal_current": gym_spaces.Box(-50.0, 50.0, shape=(2,), dtype=np.float32),
            "goal_next": gym_spaces.Box(-50.0, 50.0, shape=(2,), dtype=np.float32),
            "pedestrians_positions": gym_spaces.Box(
                low=np.full((2, 2), -50.0, dtype=np.float32),
                high=np.full((2, 2), 50.0, dtype=np.float32),
                dtype=np.float32,
            ),
            "robot_speed": gym_spaces.Box(-5.0, 5.0, shape=(2,), dtype=np.float32),
        }
    )
    fake_model = _DummySACModel(
        action=np.array([0.25, -0.5], dtype=np.float32), observation_space=spaces
    )

    class _FakeSAC:
        @staticmethod
        def load(*_args, **_kwargs):
            return fake_model

    monkeypatch.setattr(sac_mod, "SAC", _FakeSAC)

    planner = SACPlanner(
        {
            "model_path": str(model_path),
            "obs_mode": "dict",
            "obs_transform": "ego",
            "relative_obs": False,
            "action_space": "velocity",
            "fallback_to_goal": True,
        }
    )

    action = planner.step(
        {
            "robot_position": [10.0, 5.0],
            "robot_heading": [np.pi / 2],
            "goal_current": [13.0, 5.0],
            "goal_next": [15.0, 5.0],
            "pedestrians_positions": [[11.0, 5.0], [10.0, 7.0]],
            "robot_velocity_xy": [0.4, -0.1],
        }
    )

    assert action == {"vx": 0.25, "vy": -0.5}
    assert np.allclose(
        fake_model.last_obs["robot_position"], np.array([0.0, 0.0], dtype=np.float32)
    )
    assert np.allclose(
        fake_model.last_obs["goal_current"],
        np.array([0.0, -3.0], dtype=np.float32),
        atol=1e-5,
    )
    assert np.allclose(
        fake_model.last_obs["goal_next"],
        np.array([0.0, -5.0], dtype=np.float32),
        atol=1e-5,
    )
    assert np.allclose(
        fake_model.last_obs["pedestrians_positions"][0],
        np.array([0.0, -1.0], dtype=np.float32),
        atol=1e-6,
    )
    assert np.allclose(
        fake_model.last_obs["pedestrians_positions"][1],
        np.array([2.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    assert np.allclose(fake_model.last_obs["robot_speed"], np.array([0.4, -0.1], dtype=np.float32))


def test_build_model_obs_dict_raises_on_missing_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing dict observation keys should fail loudly before prediction."""
    model_path = tmp_path / "dict_missing.zip"
    model_path.write_text("stub", encoding="utf-8")
    spaces = gym_spaces.Dict(
        {
            "robot_position": gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "goal_current": gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "goal_next": gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        }
    )
    fake_model = _DummySACModel(
        action=np.array([0.0, 0.0], dtype=np.float32), observation_space=spaces
    )

    class _FakeSAC:
        @staticmethod
        def load(*_args, **_kwargs):
            return fake_model

    monkeypatch.setattr(sac_mod, "SAC", _FakeSAC)

    planner = SACPlanner(
        {"model_path": str(model_path), "obs_mode": "dict", "fallback_to_goal": True}
    )
    with pytest.raises(ValueError, match="Missing required dict observation keys"):
        planner._build_model_obs_dict({"robot_position": [0.0, 0.0], "goal_current": [1.0, 1.0]})


def test_build_model_obs_dict_raises_on_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Shape mismatches should be rejected before hitting the SB3 model."""
    model_path = tmp_path / "dict_shape.zip"
    model_path.write_text("stub", encoding="utf-8")
    spaces = gym_spaces.Dict(
        {
            "robot_position": gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "goal_current": gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        }
    )
    fake_model = _DummySACModel(
        action=np.array([0.0, 0.0], dtype=np.float32), observation_space=spaces
    )

    class _FakeSAC:
        @staticmethod
        def load(*_args, **_kwargs):
            return fake_model

    monkeypatch.setattr(sac_mod, "SAC", _FakeSAC)

    planner = SACPlanner(
        {"model_path": str(model_path), "obs_mode": "dict", "fallback_to_goal": True}
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        planner._build_model_obs_dict(
            {"robot_position": [0.0, 0.0, 0.0], "goal_current": [1.0, 1.0]}
        )


def test_build_model_obs_dict_applies_relative_socnav_transform_without_model() -> None:
    """Inference adapter should match the relative observation contract used in training."""
    planner = _planner_without_model(relative_obs=True)

    converted = planner._build_model_obs_dict(
        {
            "robot_position": [10.0, 5.0],
            "goal_current": [13.0, 7.0],
            "goal_next": [15.0, 9.0],
            "pedestrians_positions": [[11.0, 6.0], [0.0, 0.0]],
            "robot_speed": [0.25, 0.0],
        }
    )

    assert np.allclose(converted["robot_position"], np.array([0.0, 0.0], dtype=np.float32))
    assert np.allclose(converted["goal_current"], np.array([3.0, 2.0], dtype=np.float32))
    assert np.allclose(converted["goal_next"], np.array([5.0, 4.0], dtype=np.float32))
    assert np.allclose(
        converted["pedestrians_positions"][0],
        np.array([1.0, 1.0], dtype=np.float32),
    )
    assert np.allclose(
        converted["pedestrians_positions"][1],
        np.array([0.0, 0.0], dtype=np.float32),
    )


def test_build_model_obs_dict_can_leave_absolute_socnav_transform_disabled() -> None:
    """The relative transform should stay opt-out for compatibility experiments."""
    planner = _planner_without_model(relative_obs=False)

    converted = planner._build_model_obs_dict(
        {
            "robot_position": [10.0, 5.0],
            "goal_current": [13.0, 7.0],
        }
    )

    assert np.allclose(converted["robot_position"], np.array([10.0, 5.0]))
    assert np.allclose(converted["goal_current"], np.array([13.0, 7.0]))


def test_build_model_obs_dict_can_rotate_socnav_positions_to_ego_frame() -> None:
    """Inference adapter ego transform should match training-side rotation semantics."""
    planner = _planner_without_model(relative_obs=False)
    planner.config.obs_transform = "ego"

    converted = planner._build_model_obs_dict(
        {
            "robot_position": [10.0, 5.0],
            "robot_heading": [np.pi / 2],
            "goal_current": [13.0, 5.0],
            "pedestrians_positions": [[11.0, 5.0], [10.0, 7.0]],
        }
    )

    assert np.allclose(
        converted["robot_position"], np.array([0.0, 0.0], dtype=np.float32), atol=1e-6
    )
    assert np.allclose(
        converted["goal_current"], np.array([0.0, -3.0], dtype=np.float32), atol=1e-6
    )
    assert np.allclose(
        converted["pedestrians_positions"][0],
        np.array([0.0, -1.0], dtype=np.float32),
        atol=1e-6,
    )
    assert np.allclose(
        converted["pedestrians_positions"][1],
        np.array([2.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
