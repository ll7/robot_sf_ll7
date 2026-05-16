"""Tests for optional trajectory debug export."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.tools import rerun_debug_export

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "tools" / "rerun_debug_export.py"


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write JSONL records for debug-export fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _run_export(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the debug export CLI."""
    return subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _episode_record() -> dict[str, object]:
    """Return a tiny synthetic episode with trajectory, action, and metric annotations."""
    return {
        "episode_id": "episode-1",
        "scenario_id": "crossing_dense_004",
        "seed": 111,
        "status": "failed",
        "termination_reason": "collision",
        "metrics": {"min_ttc": 0.8, "clearance": 0.25},
        "trajectory_data": [
            {
                "step": 0,
                "time_s": 0.0,
                "robot": {"x": 1.0, "y": 2.0, "theta": 0.1},
                "pedestrians": {"ped-1": {"x": 3.0, "y": 4.0}},
                "action": [0.5, 0.1],
                "metrics": {"min_ttc": 1.5, "clearance": 0.7},
            },
            {
                "step": 1,
                "time_s": 0.4,
                "robot": {"x": 1.2, "y": 2.1, "theta": 0.2},
                "pedestrians": [{"id": "ped-1", "x": 2.8, "y": 3.8}],
                "selected_action": {"linear": 0.4, "angular": -0.1},
                "metrics": {"min_ttc": 0.8, "clearance": 0.25},
            },
        ],
    }


def test_json_debug_export_converts_episode_timeline(tmp_path: Path) -> None:
    """JSON export should preserve poses, actions, terminal event, and annotations."""
    source = tmp_path / "episodes.jsonl"
    output = tmp_path / "debug_timeline.json"
    _write_jsonl(source, [_episode_record()])

    result = _run_export("--source", str(source), "--output", str(output), "--format", "json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "robot-sf-debug-timeline.v1"
    assert payload["summary"] == {"episodes": 1, "frames": 2}
    episode = payload["episodes"][0]
    assert episode["episode_id"] == "episode-1"
    assert episode["terminal_event"] == "collision"
    assert episode["frames"][0]["robot"] == {"x": 1.0, "y": 2.0, "theta": 0.1}
    assert episode["frames"][0]["pedestrians"] == [
        {"entity_id": "ped-1", "x": 3.0, "y": 4.0, "theta": None}
    ]
    assert episode["frames"][0]["action"] == [0.5, 0.1]
    assert episode["frames"][1]["action"] == {"linear": 0.4, "angular": -0.1}
    assert episode["frames"][1]["annotations"] == {"clearance": 0.25, "min_ttc": 0.8}


def test_rerun_export_fails_with_actionable_optional_dependency_message(
    tmp_path: Path,
) -> None:
    """Rerun format should fail clearly when the optional package is unavailable."""
    source = tmp_path / "episodes.jsonl"
    output = tmp_path / "debug_timeline.rrd"
    _write_jsonl(source, [_episode_record()])

    result = _run_export("--source", str(source), "--output", str(output), "--format", "rerun")

    assert result.returncode == 2
    assert "Rerun export requires the optional 'rerun-sdk' package" in result.stderr
    assert not output.exists()


def test_jsonl_loader_reports_malformed_json_line(tmp_path: Path) -> None:
    """Malformed JSONL input should fail closed with the offending line number."""
    source = tmp_path / "episodes.jsonl"
    source.write_text(json.dumps(_episode_record()) + "\n{oops\n", encoding="utf-8")
    output = tmp_path / "debug_timeline.json"

    result = _run_export("--source", str(source), "--output", str(output), "--format", "json")

    assert result.returncode == 1
    assert f"{source}:2 is not valid JSON" in result.stderr
    assert not output.exists()


def test_rerun_export_saves_after_logging_and_disambiguates_missing_episode_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rerun export should save after log calls and avoid episode-id collisions."""
    source = tmp_path / "episodes.jsonl"
    output = tmp_path / "debug_timeline.rrd"
    first = _episode_record()
    second = _episode_record()
    first.pop("episode_id")
    second.pop("episode_id")
    _write_jsonl(source, [first, second])

    events: list[tuple[str, object]] = []

    fake_rerun = SimpleNamespace(
        init=lambda *args, **kwargs: events.append(("init", args)),
        save=lambda path: events.append(("save", path)),
        set_time_seconds=lambda *args: events.append(("time_seconds", args)),
        set_time_sequence=lambda *args: events.append(("time_sequence", args)),
        log=lambda path, payload: events.append(("log", path)),
        Points2D=lambda points, radii=None: {"points": points, "radii": radii},
    )
    monkeypatch.setitem(sys.modules, "rerun", fake_rerun)

    assert rerun_debug_export.write_rerun_debug_export(source=source, output=output) == output

    assert events[-1] == ("save", str(output))
    logged_paths = [payload for event, payload in events if event == "log"]
    assert "episode_0/robot" in logged_paths
    assert "episode_1/robot" in logged_paths
