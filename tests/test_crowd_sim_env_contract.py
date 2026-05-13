"""Contract tests for the crowd-only simulation environment."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from robot_sf.gym_env import crowd_sim_env
from robot_sf.gym_env.crowd_sim_env import CrowdSimEnv, CrowdSimulationConfig
from robot_sf.gym_env.env_config import SimulationSettings
from robot_sf.gym_env.environment_factory import make_crowd_sim_env


class FakePedestrianStates:
    """Minimal pedestrian state container compatible with CrowdSimEnv."""

    def __init__(self, num_peds: int = 2):
        """Create a deterministic pedestrian state matrix.

        The row count stays configurable so resets can swap between fake maps.
        """
        rows = []
        for index in range(num_peds):
            base = float(index * 2)
            rows.append(
                [
                    1.0 + base,
                    2.0 + base,
                    0.1 + base / 10,
                    0.2 + base / 10,
                    5.0 + base,
                    6.0 + base,
                ]
            )
        self._states = np.array(rows, dtype=float)

    @property
    def num_peds(self) -> int:
        """Return the number of fake pedestrians."""
        return self._states.shape[0]

    def pysf_states(self) -> np.ndarray:
        """Return the mutable fake PySocialForce state matrix."""
        return self._states


class FakeSimulator:
    """Small simulator double that records zero-robot stepping."""

    def __init__(self, **kwargs):
        """Capture simulator construction kwargs for assertions."""
        self.kwargs = kwargs
        num_peds = int(getattr(kwargs["map_def"], "ped_count", 2))
        self.pysf_state = FakePedestrianStates(num_peds=num_peds)
        self.last_ped_forces = np.zeros((0, 2), dtype=float)
        self.last_actions = []

    @property
    def ped_pos(self) -> np.ndarray:
        """Return current pedestrian positions."""
        return self.pysf_state.pysf_states()[:, 0:2]

    def step_once(self, actions):
        """Advance fake pedestrians and record the action list."""
        self.last_actions = actions
        self.pysf_state.pysf_states()[:, 0:2] += 1.0
        self.last_ped_forces = np.ones(
            (self.pysf_state.num_peds, 2),
            dtype=float,
        )


class FakeMapPool:
    """Map-pool double exposing the methods used by CrowdSimEnv."""

    def __init__(self):
        """Create deterministic fake map entries with different pedestrian counts."""
        self.map = SimpleNamespace(obstacles=[], ped_count=2)
        self.alt_map = SimpleNamespace(obstacles=[], ped_count=1)
        self.map_defs = {"fake": self.map, "alt": self.alt_map}

    def choose_random_map(self):
        """Return the fake map."""
        return self.map

    def get_map(self, map_id: str):
        """Return a fake map by id."""
        return self.map_defs[map_id]


def _config(**kwargs) -> CrowdSimulationConfig:
    """Build a deterministic crowd simulation config for tests."""
    base = {
        "sim_config": SimulationSettings(
            sim_time_in_secs=0.2,
            time_per_step_in_secs=0.1,
        ),
        "map_pool": FakeMapPool(),
        "map_id": "fake",
    }
    base.update(kwargs)
    return CrowdSimulationConfig(**base)


def test_crowd_sim_env_steps_without_robot_action(monkeypatch):
    """CrowdSimEnv should step pedestrians automatically with no robot actions."""
    monkeypatch.setattr(crowd_sim_env, "Simulator", FakeSimulator)

    env = CrowdSimEnv(_config())
    obs, info = env.reset()
    next_obs, reward, terminated, truncated, next_info = env.step(action=np.array([1.0]))

    assert obs["positions"].shape == (2, 2)
    assert next_obs["positions"].tolist() == [[2.0, 3.0], [4.0, 5.0]]
    assert next_obs["forces"].tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["map_id"] == "fake"
    assert next_info["num_pedestrians"] == 2
    assert env.sim.last_actions == []


def test_make_crowd_sim_env_exposes_factory(monkeypatch):
    """Factory should construct the crowd env and preserve the applied seed."""
    monkeypatch.setattr(crowd_sim_env, "Simulator", FakeSimulator)

    env = make_crowd_sim_env(config=_config(), seed=123)

    assert isinstance(env, CrowdSimEnv)
    assert env.applied_seed == 123


def test_make_crowd_sim_env_seed_controls_constructor_map_selection(monkeypatch):
    """Factory seeding should make constructor-time random map selection reproducible."""
    monkeypatch.setattr(crowd_sim_env, "Simulator", FakeSimulator)

    class ManyMapPool(FakeMapPool):
        """Map pool with enough entries to expose unseeded constructor RNG drift."""

        def __init__(self):
            self.map_defs = {
                f"fake-{index:02d}": SimpleNamespace(obstacles=[], ped_count=1)
                for index in range(25)
            }

    def selected_map_ids(seed: int) -> list[str | None]:
        ids = []
        for _ in range(3):
            env = make_crowd_sim_env(
                config=CrowdSimulationConfig(
                    sim_config=SimulationSettings(
                        sim_time_in_secs=0.2,
                        time_per_step_in_secs=0.1,
                    ),
                    map_pool=ManyMapPool(),
                    map_id=None,
                ),
                seed=seed,
            )
            try:
                ids.append(env.map_id)
            finally:
                env.close()
        return ids

    seed_123_ids = selected_map_ids(123)
    seed_456_ids = selected_map_ids(456)

    assert len(set(seed_123_ids)) == 1
    assert len(set(seed_456_ids)) == 1
    assert seed_123_ids[0] != seed_456_ids[0]


def test_crowd_sim_env_render_rgb_array_uses_lazy_view(monkeypatch):
    """RGB rendering should lazily create a view and return the captured frame."""
    monkeypatch.setattr(crowd_sim_env, "Simulator", FakeSimulator)
    frame = np.ones((3, 4, 3), dtype=np.uint8)

    class FakeView:
        """Renderer double that captures one frame per render call."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.frames = []
            self.screen = frame
            self.target_fps = None
            self.closed = False

        def render(self, _state, target_fps: float):
            """Capture one fake frame."""
            self.target_fps = target_fps
            self.frames.append(frame)

        def exit_simulation(self):
            """Close fake renderer."""
            self.closed = True

    monkeypatch.setattr(crowd_sim_env, "SimulationView", FakeView)

    env = CrowdSimEnv(_config(render_mode="rgb_array", video_fps=12.0))
    rendered = env.render()

    assert rendered is frame
    assert env._sim_ui.kwargs["focus_on_robot"] is False
    assert env._sim_ui.target_fps == 12.0


def test_crowd_sim_env_rejects_unsupported_render_mode(monkeypatch):
    """Invalid render modes should fail closed at construction."""
    monkeypatch.setattr(crowd_sim_env, "Simulator", FakeSimulator)

    try:
        CrowdSimEnv(_config(render_mode="unsupported"))
    except ValueError as exc:
        assert "Unsupported render_mode" in str(exc)
    else:
        raise AssertionError("CrowdSimEnv accepted an unsupported render mode")


def test_crowd_sim_env_noop_render_without_rendering(monkeypatch):
    """Headless crowd simulation should not construct a renderer."""
    monkeypatch.setattr(crowd_sim_env, "Simulator", FakeSimulator)

    env = CrowdSimEnv(_config())

    assert env.render() is None
    assert env._sim_ui is None


def test_crowd_sim_env_records_jsonl_and_closes(monkeypatch, tmp_path):
    """Compact recording should capture reset and step events without a robot."""
    monkeypatch.setattr(crowd_sim_env, "Simulator", FakeSimulator)
    recording_path = tmp_path / "crowd.jsonl"

    env = CrowdSimEnv(_config(recording_enabled=True, recording_path=str(recording_path)))
    env.reset(seed=7)
    env.step()

    assert env.recording_path == recording_path
    records = [json.loads(line) for line in recording_path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["reset", "step"]
    assert records[0]["info"]["map_id"] == "fake"
    assert records[1]["info"] == {}
    assert records[-1]["observation"]["positions"] == [[2.0, 3.0], [4.0, 5.0]]

    env.close()
    assert env._recording_file is None


def test_crowd_sim_env_observation_space_stays_static_across_resets(monkeypatch):
    """CrowdSimEnv should keep a fixed observation shape across map resets."""
    monkeypatch.setattr(crowd_sim_env, "Simulator", FakeSimulator)

    env = CrowdSimEnv(
        _config(
            sim_config=SimulationSettings(
                sim_time_in_secs=0.2,
                time_per_step_in_secs=0.1,
                max_total_pedestrians=3,
            )
        )
    )

    initial_shape = env.observation_space["positions"].shape
    obs, _ = env.reset(options={"map_id": "alt"})

    assert initial_shape == (3, 2)
    assert env.observation_space["positions"].shape == (3, 2)
    assert obs["positions"].shape == (3, 2)
    assert obs["positions"][0].tolist() == [1.0, 2.0]
    assert obs["positions"][1:].tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_crowd_sim_env_screen_fallback_and_existing_view_reset(monkeypatch):
    """Renderer fallback should use the screen and keep map state current after reset."""
    monkeypatch.setattr(crowd_sim_env, "Simulator", FakeSimulator)
    frame = np.full((2, 2, 3), 5, dtype=np.uint8)

    class FakeView:
        """Renderer double that intentionally does not capture frame history."""

        def __init__(self, **kwargs):
            """Store renderer construction arguments."""
            self.kwargs = kwargs
            self.frames = []
            self.screen = frame
            self.map_def = None
            self.obstacles = None
            self.target_fps = None
            self.closed = False

        def render(self, _state, target_fps: float):
            """Store the requested target FPS without appending a frame."""
            self.target_fps = target_fps

        def exit_simulation(self):
            """Mark renderer closure."""
            self.closed = True

    monkeypatch.setattr(crowd_sim_env, "SimulationView", FakeView)

    env = CrowdSimEnv(
        _config(
            map_id=None,
            render_mode="rgb_array",
            sim_config=SimulationSettings(
                sim_time_in_secs=0.2,
                time_per_step_in_secs=0.1,
                max_total_pedestrians=2,
            ),
        )
    )
    rendered = env.render()
    env.reset(options={"map_id": "fake"})
    env.close()

    assert np.array_equal(rendered, frame)
    assert env.map_id == "fake"
    assert env._target_fps() == env.metadata["render_fps"]
    assert env._sim_ui is None


def test_crowd_sim_env_video_path_renders_on_step(monkeypatch):
    """Video capture should trigger rendering during automatic pedestrian steps."""
    monkeypatch.setattr(crowd_sim_env, "Simulator", FakeSimulator)

    env = CrowdSimEnv(_config(video_path="crowd.mp4"))
    calls = []
    env.render = lambda: calls.append("render")
    env.step()

    assert calls == ["render"]
