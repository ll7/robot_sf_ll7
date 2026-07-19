"""Focused contract tests for the tracked geometry-aware native SIPP command."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.map_runner_native_command import (
    _render_request,
    build_native_command_policy,
    native_command_metadata_for_record,
)
from scripts.benchmark import sipp_native_command

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/benchmark/sipp_native_command.py"
CONFIG = REPO_ROOT / "configs/algos/sipp_lattice_native_command.yaml"


def test_render_request_carries_flat_live_dynamic_state() -> None:
    """The command request includes the live fields used by strict SIPP search."""
    payload = json.loads(
        _render_request(
            {
                "robot_position": [1.0, 2.0],
                "robot_heading": [0.25],
                "robot_speed": [0.5, 0.0],
                "robot_angular_velocity": [0.1],
                "goal_current": [3.0, 4.0],
                "goal_next": [5.0, 6.0],
                "pedestrians_positions": [[7.0, 8.0], [0.0, 0.0]],
                "pedestrians_velocities": [[0.2, 0.3], [0.0, 0.0]],
                "pedestrians_count": [1],
            }
        )
    )
    assert payload["robot"]["speed"] == [0.5, 0.0]
    assert payload["robot"]["angular_velocity"] == [0.1]
    assert payload["goal"]["next"] == [5.0, 6.0]
    assert payload["pedestrians"] == {
        "positions": [[7.0, 8.0], [0.0, 0.0]],
        "velocities": [[0.2, 0.3], [0.0, 0.0]],
        "count": [1],
    }


def test_native_metadata_uses_logical_planner_identity_and_fallback_state() -> None:
    """Transport identity stays native-command while analyzer identity is real SIPP."""
    _policy, metadata = build_native_command_policy(
        "native_command",
        {
            "planner_variant": "sipp_lattice",
            "native_command": {"command": [sys.executable, "-c", "pass"]},
        },
        scenario_id="classic_head_on_corridor_low",
        seed=111,
        horizon=500,
        dt=0.1,
    )
    assert metadata["algorithm"] == "sipp_lattice"
    assert metadata["fallback_or_degraded"] is False

    metadata["_native_run_state"]["planner_diagnostics"]["fallback_count"] = 1
    is_native, _deadlock, diagnostics = native_command_metadata_for_record(metadata)
    assert is_native is True
    assert diagnostics["fallback_count"] == 1
    assert metadata["fallback_or_degraded"] is True


def _request(obstacles: list[list[list[float]]]) -> dict[str, object]:
    geometry: dict[str, object] = {
        "schema_version": "native-command-static-geometry.v1",
        "scenario_id": "classic_head_on_corridor_low",
        "obstacle_segments": obstacles,
        "boundary_segments": [
            [[-5.0, -5.0], [5.0, -5.0]],
            [[5.0, -5.0], [5.0, 5.0]],
            [[5.0, 5.0], [-5.0, 5.0]],
            [[-5.0, 5.0], [-5.0, -5.0]],
        ],
    }
    geometry["sha256"] = hashlib.sha256(
        json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0]},
        "goal": {"current": [2.0, 0.0], "next": [2.0, 0.0]},
        "pedestrians": {"positions": [], "velocities": []},
        "static_geometry": geometry,
    }


def _run(request: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG)],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=False,
        cwd=REPO_ROOT,
        timeout=15,
    )


def test_static_geometry_changes_native_sipp_command_or_fails_closed() -> None:
    """A new blocking wall changes the real SIPP command with all dynamic state fixed."""
    free = _run(_request([]))
    blocked = _run(_request([[[0.05, -1.0], [0.05, 1.0]]]))
    assert free.returncode == 0, free.stderr
    assert blocked.returncode == 0, blocked.stderr
    free_payload = json.loads(free.stdout)
    blocked_payload = json.loads(blocked.stdout)
    free_command = free_payload["linear_velocity"], free_payload["angular_velocity"]
    blocked_command = (
        blocked_payload["linear_velocity"],
        blocked_payload["angular_velocity"],
    )
    assert blocked_command != free_command or blocked_command == (0.0, 0.0)


def test_native_sipp_rejects_missing_static_geometry() -> None:
    """The tracked command must not silently plan without static map input."""
    request = _request([])
    del request["static_geometry"]
    result = _run(request)
    assert result.returncode == 2
    assert "static_geometry" in result.stderr


def test_native_sipp_emits_structured_failure_for_unexpected_request_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unexpected request-time failures must preserve the fail-closed protocol."""

    class UnexpectedPlanner:
        """Raise an exception outside the expected request-validation hierarchy."""

        def plan(self, observation: dict[str, object]) -> tuple[float, float]:
            raise AssertionError("unexpected planner fault")

        def diagnostics(self) -> dict[str, object]:
            return {"last_decision": {}}

    monkeypatch.setattr(sipp_native_command, "_load_config", lambda _path: {})
    monkeypatch.setattr(
        sipp_native_command,
        "build_sipp_lattice_search_adapter",
        lambda _config: UnexpectedPlanner(),
    )
    monkeypatch.setattr(
        sipp_native_command.sys, "stdin", io.StringIO(json.dumps(_request([])) + "\n")
    )

    assert sipp_native_command.run(CONFIG) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload == {"error": "unexpected planner fault", "status": "invalid_request"}
