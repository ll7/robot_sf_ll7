"""Focused helper coverage for ``robot_sf.benchmark.runner``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import robot_sf.benchmark.runner as runner_mod


class _DummyPlanner:
    """Minimal planner stub for baseline-loader tests."""

    def __init__(self, config: dict[str, object], *, seed: int) -> None:
        self.config = config
        self.seed = seed


class _FakeClip:
    """Minimal moviepy-like clip stub for synthetic-video tests."""

    def __init__(self, frames: list[np.ndarray], *, fps: int) -> None:
        self.frames = frames
        self.fps = fps
        self.closed = False

    def write_videofile(self, path: str, codec: str, fps: int) -> None:
        """Write a sentinel MP4 payload for successful encode tests."""
        Path(path).write_bytes(b"fake-mp4")

    def close(self) -> None:
        """Record clip closure after encoding."""
        self.closed = True


class _FakePeds:
    """Minimal pedestrian state for force-sampling runner tests."""

    def __init__(self) -> None:
        self._positions = [
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            np.array([[0.1, 0.0], [1.1, 0.0]], dtype=float),
        ]
        self._index = 0

    def pos(self) -> np.ndarray:
        """Return current fake pedestrian positions."""
        return self._positions[self._index]

    def advance(self) -> None:
        """Advance to the next fake pedestrian frame."""
        self._index = min(self._index + 1, len(self._positions) - 1)


class _FakeSimulator:
    """Tiny simulator surface used by ``_simulate_episode_with_policy``."""

    def __init__(self) -> None:
        self.peds = _FakePeds()

    def step(self) -> None:
        """Advance fake pedestrian state."""
        self.peds.advance()


class _FakeScenario:
    """Scenario result stand-in with the runner attributes under test."""

    def __init__(self) -> None:
        self.simulator = _FakeSimulator()
        self.obstacles: list[tuple[float, float, float, float]] = []


class _BatchOnlyFastPysfWrapper:
    """Fake wrapper that proves the runner uses batch force sampling."""

    calls: list[np.ndarray] = []

    def __init__(self, simulator: _FakeSimulator) -> None:
        self.simulator = simulator

    def get_forces_at(self, point: np.ndarray) -> np.ndarray:
        """Fail if the old per-point call path is used."""
        raise AssertionError("runner should sample force snapshots with get_forces_at_points")

    def get_forces_at_points(self, points: np.ndarray) -> np.ndarray:
        """Return deterministic force rows for the supplied points."""
        points_arr = np.asarray(points, dtype=float)
        self.calls.append(points_arr.copy())
        return points_arr + np.array([10.0, 20.0])


def test_load_baseline_planner_covers_import_and_config_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline loader should fail clearly on import/config issues and build valid planners."""
    monkeypatch.setattr(runner_mod, "Observation", object)
    monkeypatch.setattr(runner_mod, "_BASELINE_IMPORT_ERROR", "missing deps")
    monkeypatch.setattr(runner_mod, "get_baseline", None)

    with pytest.raises(RuntimeError, match="missing deps"):
        runner_mod._load_baseline_planner("dummy", None, seed=7)

    monkeypatch.setattr(runner_mod, "get_baseline", lambda algo: _DummyPlanner)
    monkeypatch.setattr(runner_mod, "list_baselines", lambda: ["dummy", "other"])

    with pytest.raises(FileNotFoundError, match="Algorithm config file not found"):
        runner_mod._load_baseline_planner("dummy", str(tmp_path / "missing.yaml"), seed=7)

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Algorithm config must be a mapping"):
        runner_mod._load_baseline_planner("dummy", str(bad_config), seed=7)

    def _raise_unknown(_: str) -> type[_DummyPlanner]:
        """Simulate a baseline registry miss."""
        raise KeyError("unknown")

    monkeypatch.setattr(runner_mod, "get_baseline", _raise_unknown)
    with pytest.raises(ValueError, match="Unknown algorithm 'mystery'"):
        runner_mod._load_baseline_planner("mystery", None, seed=7)

    good_config = tmp_path / "good.yaml"
    good_config.write_text("speed: 1.5\nlabel: demo\n", encoding="utf-8")
    monkeypatch.setattr(runner_mod, "get_baseline", lambda algo: _DummyPlanner)
    planner, observation_cls, config = runner_mod._load_baseline_planner(
        "dummy",
        str(good_config),
        seed=11,
    )

    assert isinstance(planner, _DummyPlanner)
    assert planner.seed == 11
    assert planner.config == {"speed": 1.5, "label": "demo"}
    assert observation_cls is object
    assert config == {"speed": 1.5, "label": "demo"}


def test_load_scenario_matrix_and_small_helpers_cover_default_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario loading and small geometry helpers should handle common edge cases."""
    matrix_stream = tmp_path / "stream.yaml"
    matrix_stream.write_text("id: first\n---\nid: second\n", encoding="utf-8")
    assert runner_mod.load_scenario_matrix(matrix_stream) == [{"id": "first"}, {"id": "second"}]

    single_doc = tmp_path / "single.yaml"
    single_doc.write_text("id: manifest\n", encoding="utf-8")
    monkeypatch.setattr(
        runner_mod,
        "load_scenarios",
        lambda path, base_dir: [{"id": "loaded", "base_dir": str(base_dir)}],
    )
    loaded = runner_mod.load_scenario_matrix(single_doc)
    assert loaded == [{"id": "loaded", "base_dir": str(single_doc)}]

    monkeypatch.setattr(runner_mod.yaml, "safe_load_all", lambda _: iter(()))
    with pytest.raises(ValueError, match="is empty"):
        runner_mod.load_scenario_matrix(single_doc)

    stationary = runner_mod._simple_robot_policy(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert np.array_equal(stationary, np.zeros(2))

    toward_goal = runner_mod._simple_robot_policy(
        np.array([0.0, 0.0]),
        np.array([3.0, 4.0]),
        speed=1.0,
    )
    assert toward_goal == pytest.approx(np.array([0.6, 0.8]))

    start_default, goal_default = runner_mod._prepare_robot_points(None, None)
    assert start_default == pytest.approx(np.array([0.3, 3.0]))
    assert goal_default == pytest.approx(np.array([9.7, 3.0]))

    start_custom, goal_custom = runner_mod._prepare_robot_points([1.0, 2.0], [3.0, 4.0])
    assert start_custom == pytest.approx(np.array([1.0, 2.0]))
    assert goal_custom == pytest.approx(np.array([3.0, 4.0]))

    stacked = runner_mod._stack_or_zero(
        [np.array([1.0, 2.0])],
        stack_fn=np.stack,
        empty_shape=(0, 2),
    )
    assert stacked.shape == (1, 2)

    empty = runner_mod._stack_or_zero([], stack_fn=np.stack, empty_shape=(0, 2))
    assert empty.shape == (0, 2)

    with pytest.raises(AssertionError, match="empty_shape should have zero"):
        runner_mod._stack_or_zero([], stack_fn=np.stack, empty_shape=(1, 2))


def test_emit_video_skip_appends_notes_when_logging_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Video skip notes should still be recorded when Loguru emission fails."""
    record = {"notes": "existing"}

    def _raise_type_error(*args: object, **kwargs: object) -> None:
        """Simulate Loguru warning failure."""
        raise TypeError("logger unavailable")

    monkeypatch.setattr(runner_mod.logger, "warning", _raise_type_error)

    runner_mod._emit_video_skip(
        record=record,
        episode_id="ep-1",
        scenario_id="scenario-a",
        seed=3,
        renderer="synthetic",
        reason="encode-failed",
        steps=None,
        error="boom",
    )

    assert record["notes"] == "existing; video skipped (synthetic): encode-failed error=boom"


def test_simulate_episode_force_snapshots_use_batch_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorded force snapshots should route through FastPysfWrapper batch sampling."""
    _BatchOnlyFastPysfWrapper.calls = []
    scenario = _FakeScenario()
    monkeypatch.setattr(runner_mod, "generate_scenario", lambda params, seed: scenario)
    monkeypatch.setattr(runner_mod, "FastPysfWrapper", _BatchOnlyFastPysfWrapper)

    def _stationary_policy(
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
        ped_positions: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        return np.zeros(2, dtype=float)

    *_, peds_pos_traj, ped_forces_traj, _obstacles, _goal, _reached = (
        runner_mod._simulate_episode_with_policy(
            {},
            seed=7,
            robot_policy=_stationary_policy,
            horizon=1,
            dt=0.1,
            robot_start=None,
            robot_goal=None,
            record_forces=True,
        )
    )

    assert len(_BatchOnlyFastPysfWrapper.calls) == 2
    assert _BatchOnlyFastPysfWrapper.calls[0] == pytest.approx(peds_pos_traj[0])
    assert _BatchOnlyFastPysfWrapper.calls[1] == pytest.approx(peds_pos_traj[1])
    assert ped_forces_traj[0] == pytest.approx(peds_pos_traj[0] + np.array([10.0, 20.0]))
    assert ped_forces_traj[1] == pytest.approx(peds_pos_traj[1] + np.array([10.0, 20.0]))


def test_try_encode_synthetic_video_handles_missing_moviepy_and_unwritable_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic video helper should report missing dependencies and bad output paths."""
    trajectory = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]

    monkeypatch.setattr(runner_mod, "ImageSequenceClip", None)
    video, skip_info = runner_mod._try_encode_synthetic_video(
        trajectory,
        episode_id="ep-missing",
        scenario_id="scenario-a",
        out_dir=tmp_path / "videos",
    )
    assert video is None
    assert skip_info == {"reason": "moviepy-missing", "renderer": "synthetic", "steps": 2}

    monkeypatch.setattr(runner_mod, "ImageSequenceClip", _FakeClip)

    def _raise_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        """Simulate an unwritable video output directory."""
        raise OSError("blocked")

    monkeypatch.setattr(Path, "mkdir", _raise_mkdir)
    video, skip_info = runner_mod._try_encode_synthetic_video(
        trajectory,
        episode_id="ep-unwritable",
        scenario_id="scenario-b",
        out_dir=tmp_path / "bad-videos",
    )
    assert video is None
    assert skip_info is not None
    assert skip_info["reason"] == "unwritable-path"
    assert "blocked" in str(skip_info["error"])


@pytest.mark.parametrize(
    ("raised", "expected_reason"),
    [(OSError("disk full"), "write-failed"), (RuntimeError("codec exploded"), "encode-failed")],
)
def test_try_encode_synthetic_video_reports_encoder_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
    expected_reason: str,
) -> None:
    """Synthetic video helper should convert encoder exceptions into structured skip info."""

    class _ExplodingClip(_FakeClip):
        """Clip stub that raises during video writing."""

        def write_videofile(self, path: str, codec: str, fps: int) -> None:
            """Raise the parametrized encoder exception."""
            raise raised

    monkeypatch.setattr(runner_mod, "ImageSequenceClip", _ExplodingClip)
    video, skip_info = runner_mod._try_encode_synthetic_video(
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        episode_id="ep-error",
        scenario_id="scenario-c",
        out_dir=tmp_path / "videos",
    )

    assert video is None
    assert skip_info is not None
    assert skip_info["reason"] == expected_reason
    assert str(raised) in str(skip_info["error"])


def test_video_perf_helper_covers_warning_and_enforce_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Perf-budget helper should warn on hard breaches and raise on enforced soft breaches."""
    record = {"episode_id": "ep-1", "scenario_id": "scenario-a", "seed": 9}
    video = {"renderer": "synthetic"}

    monkeypatch.setenv("ROBOT_SF_TEST_OVERRIDE_OVERHEAD_RATIO", "0.8")
    monkeypatch.setenv("ROBOT_SF_VIDEO_OVERHEAD_SOFT", "0.1")
    monkeypatch.setenv("ROBOT_SF_VIDEO_OVERHEAD_HARD", "0.5")
    monkeypatch.delenv("ROBOT_SF_PERF_ENFORCE", raising=False)
    runner_mod._annotate_and_check_video_perf(record, video, 0.0, 0.0, 1.0)

    assert record["video"]["overhead_budget_status"] == "hard_breach"
    assert record["video"]["overhead_budget_enforced"] is False

    monkeypatch.setenv("ROBOT_SF_TEST_OVERRIDE_OVERHEAD_RATIO", "0.2")
    monkeypatch.setenv("ROBOT_SF_VIDEO_OVERHEAD_HARD", "1.0")
    monkeypatch.setenv("ROBOT_SF_PERF_ENFORCE", "1")
    with pytest.raises(RuntimeError, match="video overhead soft breach"):
        runner_mod._annotate_and_check_video_perf(record, video, 0.0, 0.0, 1.0)

    assert record["video"]["encode_seconds"] >= 0.0
    assert record["video"]["overhead_ratio"] == pytest.approx(0.2)
    assert record["video"]["overhead_budget_status"] == "soft_breach"
    assert record["video"]["overhead_budget_enforced"] is True


def test_video_perf_helper_ignores_empty_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty test override env values should fall back to the computed timing ratio."""
    record = {"episode_id": "ep-3", "scenario_id": "scenario-c", "seed": 5}
    video = {"renderer": "synthetic"}

    monkeypatch.setenv("ROBOT_SF_TEST_OVERRIDE_OVERHEAD_RATIO", "")
    monkeypatch.setenv("ROBOT_SF_VIDEO_OVERHEAD_SOFT", "2.0")
    monkeypatch.setenv("ROBOT_SF_VIDEO_OVERHEAD_HARD", "3.0")
    monkeypatch.delenv("ROBOT_SF_PERF_ENFORCE", raising=False)

    runner_mod._annotate_and_check_video_perf(record, video, 0.0, 1.0, 2.0)

    assert record["video"]["encode_seconds"] == pytest.approx(1.0)
    assert record["video"]["overhead_ratio"] == pytest.approx(0.5)


def test_maybe_encode_video_swallow_value_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Video encoding wrapper should swallow non-runtime encoder errors to keep batches robust."""

    def _raise_value_error(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        """Simulate a non-runtime encoder state error."""
        raise ValueError("bad encoder state")

    monkeypatch.setattr(runner_mod, "_try_encode_synthetic_video", _raise_value_error)
    record = {"episode_id": "ep-2", "scenario_id": "scenario-b", "seed": 4}

    runner_mod._maybe_encode_video(
        record=record,
        robot_pos_traj=[np.array([0.0, 0.0])],
        videos_dir=str(tmp_path / "videos"),
        video_enabled=True,
        video_renderer="synthetic",
        perf_start=0.0,
    )

    assert "video" not in record
