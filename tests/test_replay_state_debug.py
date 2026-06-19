"""Tests for replay state-debug CLI extraction."""

import json

import numpy as np

from robot_sf.render.jsonl_recording import JSONLRecorder
from robot_sf.render.sim_view import VisualizableSimState
from scripts.tools.debug_replay_state import _summarize_vector_array, main


def _record_simple_episode(
    *,
    output_dir,
    episode_id: int,
    steps: int,
    base_x: float,
) -> None:
    """Write a deterministic JSONL episode fixture for CLI tests."""

    recorder = JSONLRecorder(
        output_dir=str(output_dir),
        suite="debug_cli",
        scenario="state_select",
        algorithm="test",
        seed=11,
    )
    recorder.current_episode_id = episode_id
    recorder.start_episode()

    for step in range(steps):
        state = VisualizableSimState(
            timestep=step,
            robot_action=None,
            robot_pose=((base_x + step, base_x + step * 0.5), 0.0),
            pedestrian_positions=np.array(
                [
                    [base_x + 10 + step, base_x + 20 + step],
                    [base_x + 20 + step, base_x + 30 + step],
                ]
            ),
            ray_vecs=np.array([[1.0, 2.0], [3.0, 4.0]]),
            ped_actions=np.array([]),
        )
        recorder.record_step(state)

    recorder.end_episode()


def test_debug_replay_state_cli_selects_episode_step_and_agent(monkeypatch, tmp_path, capsys):
    """CLI should target the requested episode, frame, and agent."""

    _record_simple_episode(output_dir=tmp_path, episode_id=3, steps=2, base_x=1.0)
    _record_simple_episode(output_dir=tmp_path, episode_id=7, steps=3, base_x=11.0)

    monkeypatch.setattr(
        "sys.argv",
        [
            "debug_replay_state.py",
            str(tmp_path),
            "--episode-id",
            "7",
            "--step",
            "1",
            "--agent-id",
            "1",
            "--output-mode",
            "json",
        ],
    )

    assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["episode"]["id"] == 7
    assert payload["frame"]["index"] == 1
    assert payload["ray_summary"]["count"] == 2
    assert payload["selected_agent"]["kind"] == "pedestrian"
    assert payload["selected_agent"]["index"] == 1
    assert payload["selected_agent"]["pose"] == {"x": 32.0, "y": 42.0}
    assert payload["pedestrians"]["count"] == 2


def test_debug_replay_state_cli_handles_invalid_manifest_root(monkeypatch, tmp_path, capsys):
    """CLI should return a handled error for non-object manifest JSON."""

    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "debug_replay_state.py",
            str(manifest),
        ],
    )

    assert main() == 2
    captured = capsys.readouterr()
    assert "Invalid manifest JSON" in captured.err


def test_vector_summary_omits_null_values():
    """Null vector-like fields should not report as present sensors."""

    assert _summarize_vector_array(None) is None
