"""Focused coverage for the extracted SA-CADRL planner-family module."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner import socnav
from robot_sf.planner import socnav_sacadrl as sacadrl


def _observation(*, goal: tuple[float, float] = (4.0, 0.0), pedestrians=None) -> dict:
    """Build a compact nested SocNav observation for adapter unit tests."""
    positions = np.asarray(pedestrians if pedestrians is not None else [], dtype=float)
    if positions.size == 0:
        positions = np.zeros((0, 2), dtype=float)
    return {
        "robot": {
            "position": np.array([0.0, 0.0]),
            "heading": np.array([0.0]),
            "radius": np.array([0.3]),
        },
        "goal": {"current": np.asarray(goal, dtype=float)},
        "pedestrians": {
            "positions": positions,
            "velocities": np.zeros_like(positions),
            "count": np.array([positions.shape[0]]),
            "radius": np.array([0.3]),
        },
        "sim": {"timestep": np.array([0.1])},
    }


def test_actions_and_session_config_cover_cpu_and_non_cpu_paths() -> None:
    """The extracted action table and TensorFlow config preserve both device modes."""
    assert sacadrl._sacadrl_actions().shape == (11, 2)

    class FakeTensorFlow:
        def GPUOptions(self, *, allow_growth):
            return {"allow_growth": allow_growth}

        def ConfigProto(self, **kwargs):
            return SimpleNamespace(**kwargs)

    tf_module = FakeTensorFlow()
    cpu_config = sacadrl._sacadrl_session_config(tf_module, device=" /device:CPU:0 ")
    assert cpu_config.device_count == {"GPU": 0}
    assert cpu_config.gpu_options == {"allow_growth": True}

    gpu_config = sacadrl._sacadrl_session_config(tf_module, device="/gpu:0")
    assert not hasattr(gpu_config, "device_count")


def test_tensorflow_model_wrapper_crops_pads_and_predicts(  # noqa: C901
    monkeypatch, tmp_path: Path
) -> None:
    """The model wrapper keeps checkpoint inference and input-shape adaptation intact."""

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeGraph:
        def as_default(self):
            return Context()

        def get_tensor_by_name(self, name):
            class FakeTensor:
                shape = (None, 4)

                def __init__(self, tensor_name):
                    self.name = tensor_name

            return FakeTensor(name)

    class FakeSession:
        def __init__(self, **_kwargs):
            self.runs = []

        def run(self, tensor, *, feed_dict=None):
            self.runs.append((tensor, feed_dict))
            return np.ones((1, 2))

    class FakeTensorFlow:
        Graph = FakeGraph

        def device(self, _device):
            return Context()

        def GPUOptions(self, *, allow_growth):
            return SimpleNamespace(allow_growth=allow_growth)

        def ConfigProto(self, **kwargs):
            return SimpleNamespace(**kwargs)

        def Session(self, **kwargs):
            return FakeSession(**kwargs)

        def global_variables_initializer(self):
            return "initializer"

        class train:
            @staticmethod
            def import_meta_graph(_path, *, clear_devices):
                assert clear_devices is True
                return SimpleNamespace(restore=lambda *_args: None)

    monkeypatch.setattr(sacadrl._socnav, "tf", FakeTensorFlow())
    model = sacadrl._SACADRLModel(tmp_path / "checkpoint")
    assert model.actions.shape == (11, 2)
    assert model._crop(np.ones((1, 6))).shape == (1, 4)
    assert model._crop(np.ones((1, 2))).shape == (1, 4)
    assert model._crop(np.ones((1, 4))).shape == (1, 4)
    assert model.predict(np.ones((1, 4))).shape == (1, 2)


@pytest.mark.parametrize(
    ("positions", "velocities"),
    [
        ([], []),
        ([1.0, 2.0], [0.1, 0.2]),
        ([1.0], [0.1]),
        ([[1.0, 2.0, 3.0]], [[0.1, 0.2, 0.3]]),
        ([[1.0]], [[0.1]]),
        ([[[1.0, 2.0]]], [[[0.1, 0.2]]]),
    ],
)
def test_normalization_handles_supported_malformed_shapes(positions, velocities) -> None:
    """The adapter retains its defensive position and velocity normalization."""
    adapter = sacadrl.SACADRLPlannerAdapter(allow_fallback=True)
    normalized = adapter._normalize_pedestrians(
        {"positions": positions, "velocities": velocities, "count": [1], "radius": [0.3]}
    )
    assert normalized[0].ndim == 2
    assert normalized[0].shape[1] == 2
    assert normalized[1].shape[1] == 2


def test_adapter_builds_network_input_and_agent_states(monkeypatch) -> None:
    """Goal-frame, velocity, ordering, and network-input helpers preserve dimensions."""
    config = sacadrl.SocNavPlannerConfig(sacadrl_sorting_method="closest_last")
    adapter = sacadrl.SACADRLPlannerAdapter(config=config, allow_fallback=True)
    observation = _observation(pedestrians=[[1.0, 0.2], [2.0, -0.1]])

    distance, parallel, orthogonal, heading = adapter._compute_goal_frame(
        np.zeros(2), 0.0, np.zeros(2)
    )
    assert distance == 0.0
    assert np.allclose(parallel, [1.0, 0.0])
    assert np.allclose(orthogonal, [0.0, 1.0])
    assert heading == 0.0

    velocities = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(adapter._ego_to_global_velocities(0.0, velocities), velocities)
    assert adapter._ego_to_global_velocities(0.0, np.zeros((0, 2))).shape == (0, 2)

    states, count = adapter._build_other_agents_states(
        observation["pedestrians"]["positions"],
        velocities,
        np.zeros(2),
        0.3,
        0.3,
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    )
    assert states.shape == (config.sacadrl_max_other_agents, 7)
    assert count == 2.0
    empty_states, empty_count = adapter._build_other_agents_states(
        np.zeros((0, 2)),
        np.zeros((0, 2)),
        np.zeros(2),
        0.3,
        0.3,
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    )
    assert empty_states.shape == states.shape
    assert empty_count == 0.0

    vector, pref_speed, distance = adapter._build_network_input(observation)
    assert vector.shape == (1, 5 + 7 * config.sacadrl_max_other_agents)
    assert pref_speed == config.sacadrl_pref_speed
    assert distance > 0.0

    monkeypatch.setattr(adapter, "_get_safe_heading", lambda *_args: (np.array([1.0, 0.0]), 0.0))
    assert adapter._heuristic_plan(observation)[0] >= 0.0
    assert adapter._heuristic_plan(_observation())[0] >= 0.0

    closest_last = sacadrl.SACADRLPlannerAdapter(
        config=sacadrl.SocNavPlannerConfig(
            sacadrl_max_other_agents=2, sacadrl_sorting_method="closest_last"
        ),
        allow_fallback=True,
    )
    selected_states, _selected_count = closest_last._build_other_agents_states(
        np.array([[1.0, 0.0], [2.0, 0.0], [10.0, 0.0]]),
        np.zeros((3, 2)),
        np.zeros(2),
        0.3,
        0.3,
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    )
    assert selected_states[:, -1].tolist() == pytest.approx([1.4, 0.4])


def test_checkpoint_resolution_hashes_bundle_and_fails_closed(tmp_path: Path, monkeypatch) -> None:
    """Checkpoint resolution retains suffix handling, provenance hashing, and fail-closed errors."""
    prefix = tmp_path / "model"
    (tmp_path / "model.meta").write_bytes(b"meta")
    (tmp_path / "model.index").write_bytes(b"index")
    (tmp_path / "model.data-00000-of-00001").write_bytes(b"data")
    adapter = sacadrl.SACADRLPlannerAdapter(
        config=sacadrl.SocNavPlannerConfig(sacadrl_checkpoint_path=str(prefix) + ".meta")
    )
    assert adapter._resolve_checkpoint_prefix() == prefix
    provenance = adapter.diagnostics()["checkpoint_provenance"]
    assert provenance["checkpoint_sha256"]
    assert provenance["hash_source"] == "computed_tensorflow_checkpoint_bundle"

    dotted_prefix = tmp_path / "model.v1.ckpt-100"
    (tmp_path / "model.v1.ckpt-100.meta").write_bytes(b"meta")
    (tmp_path / "model.v1.ckpt-100.index").write_bytes(b"index")
    (tmp_path / "model.v1.ckpt-100.data-00000-of-00001").write_bytes(b"data")
    dotted = sacadrl.SACADRLPlannerAdapter(
        config=sacadrl.SocNavPlannerConfig(sacadrl_checkpoint_path=str(dotted_prefix) + ".meta")
    )
    assert dotted._resolve_checkpoint_prefix() == dotted_prefix

    implicit = sacadrl.SACADRLPlannerAdapter()
    monkeypatch.setattr(sacadrl, "resolve_model_path", lambda _model_id: prefix)
    assert implicit._resolve_checkpoint_prefix() == prefix

    for missing in ("model.meta", "model.index", "model.data-00000-of-00001"):
        path = tmp_path / missing
        contents = path.read_bytes()
        path.unlink()
        with pytest.raises(FileNotFoundError):
            sacadrl.SACADRLPlannerAdapter(
                config=sacadrl.SocNavPlannerConfig(sacadrl_checkpoint_path=str(prefix))
            )._resolve_checkpoint_prefix()
        path.write_bytes(contents)


def test_adapter_model_lifecycle_and_model_backed_plan(monkeypatch) -> None:
    """Success, fallback, cached-error, and model-backed plan paths remain explicit."""
    adapter = sacadrl.SACADRLPlannerAdapter(allow_fallback=True)
    marker = object()
    monkeypatch.setattr(adapter, "_build_model", lambda: marker)
    assert adapter._ensure_model() is marker
    assert adapter._ensure_model() is marker

    failed = sacadrl.SACADRLPlannerAdapter(allow_fallback=True)
    monkeypatch.setattr(
        failed, "_build_model", lambda: (_ for _ in ()).throw(RuntimeError("missing"))
    )
    assert failed._ensure_model() is None
    assert failed._ensure_model() is None
    assert failed.diagnostics()["checkpoint_provenance"]["load_status"] == "fallback"

    strict = sacadrl.SACADRLPlannerAdapter(allow_fallback=False)
    attempts = 0

    def fail_strictly():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("missing")

    monkeypatch.setattr(strict, "_build_model", fail_strictly)
    with pytest.raises(RuntimeError, match="missing"):
        strict._ensure_model()
    assert attempts == 1
    with pytest.raises(RuntimeError, match="missing"):
        strict._ensure_model()

    class FakeModel:
        actions = np.array([[0.5, -0.1], [1.0, 0.2]])

        def predict(self, _obs):
            return np.array([[0.1, 0.9]])

    planned = sacadrl.SACADRLPlannerAdapter(allow_fallback=False)

    def fake_model():
        return FakeModel()

    monkeypatch.setattr(planned, "_ensure_model", fake_model)
    action = planned.plan(_observation())
    assert action == (1.0, planned.config.max_angular_speed)
    assert planned.plan(_observation(goal=(0.1, 0.0))) == (0.0, 0.0)

    fallback = sacadrl.SACADRLPlannerAdapter(allow_fallback=True)
    monkeypatch.setattr(fallback, "_ensure_model", lambda: None)
    assert fallback.plan(_observation(goal=(0.1, 0.0))) == (0.0, 0.0)


def test_facade_wildcard_import_includes_lazy_public_exports() -> None:
    """Lazy public symbols remain visible through facade introspection and wildcard import."""
    assert "SACADRLPlannerAdapter" in dir(socnav)
    assert "make_sacadrl_policy" in dir(socnav)
    assert "SACADRLPlannerAdapter" in socnav.__all__
    assert "make_sacadrl_policy" in socnav.__all__
    assert socnav.SACADRLPlannerAdapter is sacadrl.SACADRLPlannerAdapter
    assert socnav.make_sacadrl_policy is sacadrl.make_sacadrl_policy
