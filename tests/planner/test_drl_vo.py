"""Tests for the DRL-VO baseline adapter and benchmark registration."""

import numpy as np
import pytest

import robot_sf.baselines.drl_vo as drl_vo_module
from robot_sf.baselines import get_baseline, list_baselines
from robot_sf.baselines.drl_vo import DrlVoPlanner, DrlVoPlannerConfig
from robot_sf.baselines.social_force import Observation
from robot_sf.benchmark.runner import _create_robot_policy


class _FakeTensor:
    """Minimal tensor test double for DRL-VO prediction paths."""

    def __init__(self, value):
        self.value = value

    def dim(self):
        return 1

    def unsqueeze(self, _dim):
        return self


class _FakeCuda:
    """Minimal CUDA availability shim for fake torch."""

    @staticmethod
    def is_available():
        return False


class _FakeTorch:
    """Minimal torch test double for DRL-VO loader and inference branches."""

    Tensor = _FakeTensor
    cuda = _FakeCuda()
    load_result = None
    load_error = None

    @classmethod
    def tensor(cls, value, **_kwargs):
        return _FakeTensor(value)

    @classmethod
    def load(cls, _path, **_kwargs):
        if cls.load_error is not None:
            raise cls.load_error
        return cls.load_result


_FakeTorch.float32 = "float32"


def _obs(goal=None, *, heading=0.0, agents=None) -> Observation:
    """Build a compact DRL-VO observation for focused unit tests."""
    return Observation(
        dt=0.1,
        robot={
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [1.0, 0.0] if goal is None else goal,
            "heading": heading,
            "radius": 0.3,
        },
        agents=[] if agents is None else agents,
    )


def test_drl_vo_is_registered_in_baselines() -> None:
    """DRL-VO should be discoverable through the baseline registry."""
    available = list_baselines()
    assert "drl_vo" in available
    planner_cls = get_baseline("drl_vo")
    assert planner_cls is DrlVoPlanner


def test_drl_vo_fallback_to_goal_action() -> None:
    """The DRL-VO adapter should fall back to goal-seeking when no model is available."""
    planner = DrlVoPlanner(DrlVoPlannerConfig(fallback_to_goal=True), seed=0)
    obs = Observation(
        dt=0.1,
        robot={"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [1.0, 0.0], "radius": 0.3},
        agents=[],
    )
    action = planner.step(obs)
    assert isinstance(action, dict)
    assert set(action.keys()) == {"vx", "vy"}
    assert action["vx"] > 0.0
    assert abs(action["vy"]) < 1e-6


def test_drl_vo_unicycle_fallback_steers_toward_goal() -> None:
    """Unicycle fallback should turn toward the goal instead of always driving straight."""
    planner = DrlVoPlanner(
        DrlVoPlannerConfig(
            fallback_to_goal=True,
            action_space="unicycle",
            omega_max=0.5,
        ),
        seed=0,
    )
    obs = Observation(
        dt=0.1,
        robot={
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [1.0, 0.0],
            "heading": np.pi / 2,
            "radius": 0.3,
        },
        agents=[],
    )

    action = planner.step(obs)
    assert action["v"] > 0.0
    assert action["omega"] == pytest.approx(-0.5)


def test_drl_vo_predict_passes_deterministic_flag() -> None:
    """Model predict calls should receive DrlVoPlannerConfig.deterministic when supported."""
    if drl_vo_module.torch is None:
        pytest.skip("PyTorch is required for tensor-backed DRL-VO prediction")

    class _PredictModel:
        def __init__(self) -> None:
            self.deterministic = None

        def predict(self, _tensor, *, deterministic: bool):
            self.deterministic = deterministic
            return [0.1, 0.2]

    model = _PredictModel()
    planner = DrlVoPlanner(
        DrlVoPlannerConfig(
            fallback_to_goal=True,
            deterministic=False,
        ),
        seed=0,
    )
    planner._model = model
    obs = Observation(
        dt=0.1,
        robot={"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [1.0, 0.0], "radius": 0.3},
        agents=[],
    )

    action = planner.step(obs)
    assert action == {"vx": 0.1, "vy": 0.2}
    assert model.deterministic is False


def test_drl_vo_runner_integration_policy() -> None:
    """The benchmark runner should be able to instantiate and execute a DRL-VO policy."""
    policy_fn, metadata = _create_robot_policy("drl_vo", None, seed=1)
    assert callable(policy_fn)
    assert metadata["algorithm"] == "drl_vo"

    velocity = policy_fn(
        np.array([0.0, 0.0], dtype=float),
        np.array([0.0, 0.0], dtype=float),
        np.array([2.0, 0.0], dtype=float),
        np.zeros((0, 2), dtype=float),
        0.1,
    )
    assert isinstance(velocity, np.ndarray)
    assert velocity.shape == (2,)
    assert velocity[0] >= 0.0


def test_drl_vo_missing_model_path_falls_back(tmp_path) -> None:
    """When the DRL-VO checkpoint is missing, the planner should fall back to goal motion."""
    planner = DrlVoPlanner(
        {
            "model_path": str(tmp_path / "missing_model.pt"),
            "fallback_to_goal": True,
        },
        seed=123,
    )
    obs = Observation(
        dt=0.1,
        robot={"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [0.2, 0.0], "radius": 0.3},
        agents=[],
    )
    action = planner.step(obs)
    assert action["vx"] > 0.0
    assert action["vy"] == pytest.approx(0.0)


def test_drl_vo_model_resolution_failure_falls_back(monkeypatch) -> None:
    """Model-id resolution errors should enter the graceful fallback path."""
    monkeypatch.setattr(drl_vo_module, "torch", _FakeTorch)
    monkeypatch.setattr(
        drl_vo_module,
        "resolve_model_path",
        lambda _model_id: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    planner = DrlVoPlanner({"model_id": "missing", "fallback_to_goal": True}, seed=0)
    assert planner.get_metadata()["fallback_reason"] == "model_resolution_failed"


def test_drl_vo_checkpoint_load_failure_falls_back(monkeypatch, tmp_path) -> None:
    """Checkpoint load errors should enter the graceful fallback path."""
    model_path = tmp_path / "broken.pt"
    model_path.touch()
    monkeypatch.setattr(drl_vo_module, "torch", _FakeTorch)
    _FakeTorch.load_result = None
    _FakeTorch.load_error = RuntimeError("broken checkpoint")

    planner = DrlVoPlanner({"model_path": str(model_path), "fallback_to_goal": True}, seed=0)
    assert planner.get_metadata()["fallback_reason"] == "model_load_failed"
    _FakeTorch.load_error = None


def test_drl_vo_invalid_checkpoint_falls_back(monkeypatch, tmp_path) -> None:
    """Non-callable checkpoint payloads should enter the invalid-model fallback path."""
    model_path = tmp_path / "invalid.pt"
    model_path.touch()
    monkeypatch.setattr(drl_vo_module, "torch", _FakeTorch)
    _FakeTorch.load_result = object()
    _FakeTorch.load_error = None

    planner = DrlVoPlanner({"model_path": str(model_path), "fallback_to_goal": True}, seed=0)
    assert planner.get_metadata()["fallback_reason"] == "invalid_model"


def test_drl_vo_callable_model_prediction_and_lifecycle(monkeypatch, tmp_path) -> None:
    """Callable checkpoint payloads should load and support prediction/lifecycle methods."""
    model_path = tmp_path / "callable.pt"
    model_path.touch()
    monkeypatch.setattr(drl_vo_module, "torch", _FakeTorch)
    _FakeTorch.load_result = lambda _tensor: ([0.2, -0.1], "state")
    _FakeTorch.load_error = None

    planner = DrlVoPlanner(
        {"model_path": str(model_path), "device": "cpu", "fallback_to_goal": True},
        seed=0,
    )
    assert planner._torch_device() == "cpu"
    assert planner.step(_obs()) == {"vx": 0.2, "vy": -0.1}

    planner.reset(seed=42)
    assert planner._seed == 42
    planner.configure({"model_path": str(model_path), "fallback_to_goal": True})
    assert planner.get_metadata()["status"] == "ok"
    planner.close()
    assert planner._model is None


def test_drl_vo_parses_dict_and_unicycle_actions(monkeypatch, tmp_path) -> None:
    """Action parsing should preserve dict and vector payloads for both command spaces."""
    monkeypatch.setattr(drl_vo_module, "torch", _FakeTorch)
    _FakeTorch.load_result = None
    _FakeTorch.load_error = None
    planner = DrlVoPlanner({"model_path": str(tmp_path / "missing.pt"), "fallback_to_goal": True})

    assert planner._parse_model_action({"vx": 0.3, "vy": -0.4}) == {"vx": 0.3, "vy": -0.4}
    planner.config.action_space = "unicycle"
    assert planner._parse_model_action({"v": 0.5, "omega": -0.2}) == {
        "v": 0.5,
        "omega": -0.2,
    }
    assert planner._parse_model_action([0.6, 0.1]) == {"v": 0.6, "omega": 0.1}
    with pytest.raises(ValueError, match="unsupported action format"):
        planner._parse_model_action([1.0, 2.0, 3.0])


def test_drl_vo_rejects_invalid_config_and_observation() -> None:
    """Invalid config and observation payloads should fail with clear TypeErrors."""
    with pytest.raises(TypeError, match="Invalid config type"):
        DrlVoPlanner("bad")  # type: ignore[arg-type]

    planner = DrlVoPlanner(DrlVoPlannerConfig(fallback_to_goal=True), seed=0)
    with pytest.raises(TypeError, match="Unsupported observation type"):
        planner.step(object())  # type: ignore[arg-type]


def test_drl_vo_zero_distance_fallback_stops() -> None:
    """Goal fallback should stop when the robot is already at the goal."""
    planner = DrlVoPlanner(DrlVoPlannerConfig(fallback_to_goal=True), seed=0)
    assert planner.step(_obs(goal=[0.0, 0.0])) == {"vx": 0.0, "vy": 0.0}

    planner.config.action_space = "unicycle"
    assert planner.step(_obs(goal=[0.0, 0.0])) == {"v": 0.0, "omega": 0.0}


def test_drl_vo_model_input_sorts_nearest_agents() -> None:
    """Model input should keep the nearest agents before farther observation entries."""
    planner = DrlVoPlanner(DrlVoPlannerConfig(fallback_to_goal=True, nearest_k=2), seed=0)
    model_input = planner._build_model_input(
        _obs(
            agents=[
                {"position": [5.0, 0.0], "velocity": [0.5, 0.0]},
                {"position": [1.0, 0.0], "velocity": [0.1, 0.0]},
                {"position": [2.0, 0.0], "velocity": [0.2, 0.0]},
            ]
        )
    )
    assert model_input.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, 2.0, 0.0, 0.2, 0.0]


def test_drl_vo_allows_benchmark_opt_in_flag() -> None:
    """Benchmark configs may include allow_testing_algorithms without breaking DRL-VO init."""
    planner = DrlVoPlanner(
        {
            "allow_testing_algorithms": True,
            "fallback_to_goal": True,
        },
        seed=42,
    )
    assert planner.config.fallback_to_goal is True
    assert planner.config.action_space == "velocity"
