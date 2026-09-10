"""Tests for the trajectory-visualization fixture and headless smoke path (#8734)."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from examples.advanced import trajectory_viz_fixture as fixture
from examples.advanced.trajectory_viz_fixture import (
    REASON_EMPTY_RECORDING,
    REASON_LEGACY_AMBIGUOUS_SHAPE,
    REASON_MALFORMED_PICKLE,
    REASON_MISSING_FILE,
    REASON_NON_FINITE_VALUES,
    REASON_OK,
    REASON_PATH_ESCAPE,
)
from robot_sf.render.playback_recording import load_states


def test_fixture_roundtrips_through_canonical_reader(tmp_path: Path) -> None:
    """The generated fixture uses the canonical (states, map_def) tuple schema."""
    recording = fixture.build_fixture_recording(tmp_path / "fixture.pkl", steps=5)
    states, map_def = load_states(str(recording))
    assert len(states) == 5
    assert [state.timestep for state in states] == [0, 1, 2, 3, 4]
    assert map_def.width == 1.0


def test_headless_run_is_deterministic(tmp_path: Path) -> None:
    """Two headless runs produce identical summaries including output digests."""
    recording = fixture.build_fixture_recording(tmp_path / "fixture.pkl")
    first, reason = fixture.run_headless(recording, tmp_path / "out-a", max_frames=3)
    assert reason == REASON_OK
    second, reason = fixture.run_headless(recording, tmp_path / "out-b", max_frames=3)
    assert reason == REASON_OK
    assert first is not None and second is not None
    first.pop("summary_path")
    second.pop("summary_path")
    assert first == second
    assert first["frame_count"] == 12
    assert first["actor_count"] == {"robot": 1, "pedestrians": [2]}
    summary_file = tmp_path / "out-a" / "summary.json"
    assert json.loads(summary_file.read_text(encoding="utf-8"))["frame_count"] == 12


def test_headless_run_writes_only_to_requested_dir(tmp_path: Path) -> None:
    """Frames and summary land under the caller-owned directory."""
    recording = fixture.build_fixture_recording(tmp_path / "fixture.pkl")
    out_dir = tmp_path / "owned"
    summary, reason = fixture.run_headless(recording, out_dir, max_frames=2)
    assert reason == REASON_OK
    assert summary is not None
    assert sorted(path.name for path in (out_dir / "frames").glob("*.png"))
    assert (out_dir / "summary.json").is_file()


def test_missing_recording_fails_closed(tmp_path: Path) -> None:
    """A nonexistent recording maps to the missing_file reason code."""
    summary, reason = fixture.run_headless(tmp_path / "absent.pkl", tmp_path / "out")
    assert summary is None
    assert reason == REASON_MISSING_FILE


def test_empty_recording_fails_closed(tmp_path: Path) -> None:
    """An empty file maps to the empty_recording reason code."""
    recording = tmp_path / "empty.pkl"
    recording.write_bytes(b"")
    summary, reason = fixture.run_headless(recording, tmp_path / "out")
    assert summary is None
    assert reason == REASON_EMPTY_RECORDING


def test_malformed_pickle_fails_closed(tmp_path: Path) -> None:
    """Random bytes map to the malformed_pickle reason code, not a crash."""
    recording = tmp_path / "junk.pkl"
    recording.write_bytes(b"\x00\x01not-a-pickle")
    summary, reason = fixture.run_headless(recording, tmp_path / "out")
    assert summary is None
    assert reason == REASON_MALFORMED_PICKLE


def test_legacy_ambiguous_shape_fails_closed(tmp_path: Path) -> None:
    """A valid pickle with the wrong payload shape maps to legacy_ambiguous_shape."""
    recording = tmp_path / "legacy.pkl"
    with recording.open("wb") as handle:
        pickle.dump({"states": []}, handle)
    summary, reason = fixture.run_headless(recording, tmp_path / "out")
    assert summary is None
    assert reason == REASON_LEGACY_AMBIGUOUS_SHAPE


def test_non_finite_values_fail_closed(tmp_path: Path) -> None:
    """NaN payloads fail closed instead of rendering garbage frames."""
    from robot_sf.render.sim_view import _empty_map_definition

    states = fixture.build_fixture_states(steps=3)
    states[1].pedestrian_positions = np.full((2, 2), np.nan)
    recording = tmp_path / "nan.pkl"
    with recording.open("wb") as handle:
        pickle.dump((states, _empty_map_definition()), handle)
    summary, reason = fixture.run_headless(recording, tmp_path / "out")
    assert summary is None
    assert reason == REASON_NON_FINITE_VALUES


def test_out_dir_escape_fails_closed(tmp_path: Path) -> None:
    """An output directory outside repo/tmp/cwd is refused."""
    recording = fixture.build_fixture_recording(tmp_path / "fixture.pkl")
    summary, reason = fixture.run_headless(recording, "/etc/robot-sf-evil")
    assert summary is None
    assert reason == REASON_PATH_ESCAPE


def _load_example_module():
    """Import the digit-prefixed example script by file location."""
    import importlib.util
    import sys

    example_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "advanced"
        / "14_trajectory_visualization.py"
    )
    module_name = "example_14_trajectory_visualization"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_example_main_headless_smoke(tmp_path: Path) -> None:
    """The example entry point exits 0 on the fixture smoke path."""
    example = _load_example_module()
    out_dir = tmp_path / "example-out"
    assert (
        example.main(["--fixture", "--headless", "--out-dir", str(out_dir), "--max-frames", "2"])
        == 0
    )
    assert (out_dir / "summary.json").is_file()


def test_example_main_refuses_missing_recording(tmp_path: Path) -> None:
    """The example entry point exits 2 for a refused recording path."""
    example = _load_example_module()
    assert (
        example.main(
            [
                str(tmp_path / "absent.pkl"),
                "--headless",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
