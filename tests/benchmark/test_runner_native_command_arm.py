"""Tests for the issue #5887 native-command runner arm and deadlock/stall metric.

These are CPU-only unit/integration tests. They spawn a tiny inline Python
native-command stub (via ``sys.executable``) that echoes a JSON velocity
command, so no external planner binary or GPU is required.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from robot_sf.benchmark import runner as runner_mod
from robot_sf.benchmark.metrics import compute_deadlock_stall
from robot_sf.benchmark.runner import run_episode
from robot_sf.benchmark.schema_validator import load_schema, validate_episode

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"

# Inline native-command stub: reads one JSON state frame on stdin, writes one
# JSON velocity command on stdout. P-controller toward the goal in unicycle vw.
_NATIVE_STUB_SRC = (
    "import sys, json\n"
    "req = json.loads(sys.stdin.readline())\n"
    "pos = req['robot_pos']; goal = req['robot_goal']\n"
    "dx = goal[0]-pos[0]; dy = goal[1]-pos[1]\n"
    "d = (dx*dx + dy*dy) ** 0.5\n"
    "if d < 1e-6:\n"
    "    print(json.dumps({'v':0.0,'omega':0.0}))\n"
    "else:\n"
    "    print(json.dumps({'v': min(1.0, d), 'omega': 0.0}))\n"
    "sys.stdout.flush()\n"
)

_NATIVE_PERSISTENT_STUB_SRC = (
    "import sys, json\n"
    "for raw in sys.stdin:\n"
    "    req = json.loads(raw)\n"
    "    pos = req['robot_pos']; goal = req['robot_goal']\n"
    "    dx = goal[0]-pos[0]; dy = goal[1]-pos[1]\n"
    "    d = (dx*dx + dy*dy) ** 0.5\n"
    "    print(json.dumps({'v': min(1.0, d), 'omega': 0.0}))\n"
    "    sys.stdout.flush()\n"
)


def _native_command_spec(*, persistent: bool, timeout_s: float = 2.0) -> dict[str, object]:
    """Return a scenario ``native_command`` block invoking the inline stub."""
    argv = [
        sys.executable,
        "-c",
        _NATIVE_PERSISTENT_STUB_SRC if persistent else _NATIVE_STUB_SRC,
    ]
    return {
        "id": "nc_test",
        "num_pedestrians": 0,
        "algo": "native_command",
        "native_command": {
            "argv": argv,
            "env": {"NC_TEST": "1"},
            "timeout_s": timeout_s,
            "persistent": persistent,
        },
    }


def test_native_command_provenance_captures_command_and_hash():
    """Provenance must pin argv + env keys + a content hash of the invocation."""
    spec = _native_command_spec(persistent=False)
    prov = runner_mod._native_command_provenance(
        spec["native_command"]["argv"], spec["native_command"]["env"]
    )
    assert prov["protocol"] == runner_mod.NATIVE_COMMAND_PROTOCOL_VERSION
    assert prov["argv"] == spec["native_command"]["argv"]
    assert prov["env_keys"] == ["NC_TEST"]
    assert len(prov["command_hash_sha256"]) == 64
    assert prov["binary_path"] == sys.executable
    assert len(prov["binary_hash_sha256"]) == 64
    # Hash is stable across identical invocations.
    again = runner_mod._native_command_provenance(
        spec["native_command"]["argv"], spec["native_command"]["env"]
    )
    assert again["command_hash_sha256"] == prov["command_hash_sha256"]


def test_native_command_per_episode_runs_and_validates(tmp_path: Path):
    """A per-episode native-command episode produces a schema-valid record."""
    scenario = _native_command_spec(persistent=False)
    record = run_episode(
        scenario, seed=111, algo="native_command", horizon=40, dt=0.1, record_forces=False
    )
    schema = load_schema(SCHEMA_PATH)
    validate_episode(record, schema)

    am = record["algorithm_metadata"]
    assert am["planner_kinematics"]["execution_mode"] == "native"
    assert am.get("status") == "ok"
    assert "planner_diagnostics" in am
    diag = am["planner_diagnostics"]
    for key in (
        "expansion_limit_hits",
        "runtime_bound_exits",
        "fallback_count",
        "commitment_invalidations",
    ):
        assert isinstance(diag[key], int) and diag[key] >= 0
    assert isinstance(diag["planner_step_runtime_seconds"], list)
    assert len(diag["planner_step_runtime_seconds"]) > 0
    # Deadlock flag is a boolean present in metrics.
    assert isinstance(record["metrics"]["deadlock"], bool)
    assert record["metrics"]["deadlock_stall"]["schema_version"] == "deadlock-stall.v1"
    # Command contract provenance recorded.
    assert am["command_contract"]["command_hash_sha256"]
    assert am["command_mode"] == "per_episode_process"


def test_native_command_persistent_process_runs(tmp_path: Path):
    """Persistent-process mode keeps one child alive and still records runtime."""
    scenario = _native_command_spec(persistent=True)
    record = run_episode(
        scenario, seed=222, algo="native_command", horizon=30, dt=0.1, record_forces=False
    )
    schema = load_schema(SCHEMA_PATH)
    validate_episode(record, schema)
    am = record["algorithm_metadata"]
    assert am["command_mode"] == "persistent_process"
    assert am["planner_kinematics"]["execution_mode"] == "native"
    assert len(am["planner_diagnostics"]["planner_step_runtime_seconds"]) == 30


def test_native_command_persistent_timeout_is_bounded():
    """A persistent child that never responds cannot block the episode indefinitely."""
    policy = runner_mod._NativeCommandPolicy(
        argv=[sys.executable, "-c", "import sys, time; time.sleep(5)"],
        env={},
        timeout_s=0.05,
        persistent=True,
    )
    started = time.monotonic()
    try:
        command = policy.step(np.zeros(2), np.zeros(2), np.ones(2), np.empty((0, 2)), 0.1)
    finally:
        policy.close()
    assert time.monotonic() - started < 1.0
    assert command.tolist() == [0.0, 0.0]
    assert policy.diagnostics()["runtime_bound_exits"] == 1
    assert policy.diagnostics()["fallback_count"] == 1


def test_native_command_fallback_on_nonzero_exit():
    """A nonzero-exit native command falls back to zero velocity and records it."""
    bad_spec = {
        "id": "nc_bad",
        "num_pedestrians": 0,
        "algo": "native_command",
        "native_command": {
            "argv": [sys.executable, "-c", "import sys; sys.exit(3)"],
            "env": {},
            "timeout_s": 2.0,
            "persistent": False,
        },
    }
    record = run_episode(
        bad_spec, seed=9, algo="native_command", horizon=10, dt=0.1, record_forces=False
    )
    diag = record["algorithm_metadata"]["planner_diagnostics"]
    # Every step fell back (nonzero exit -> fallback command).
    assert diag["fallback_count"] == 10
    assert diag["last_exit_code"] == 3
    assert len(diag["planner_step_runtime_seconds"]) == 10
    # Record still schema-valid (fallback is a legal status).
    validate_episode(record, load_schema(SCHEMA_PATH))


def test_native_command_timeout_fallback():
    """A command that exceeds the step timeout falls back and records runtime-bound exit."""
    slow_spec = {
        "id": "nc_slow",
        "num_pedestrians": 0,
        "algo": "native_command",
        "native_command": {
            "argv": [sys.executable, "-c", "import time; time.sleep(5)"],
            "env": {},
            "timeout_s": 0.2,
            "persistent": False,
        },
    }
    record = run_episode(
        slow_spec, seed=5, algo="native_command", horizon=3, dt=0.1, record_forces=False
    )
    diag = record["algorithm_metadata"]["planner_diagnostics"]
    assert diag["runtime_bound_exits"] == 3
    assert diag["fallback_count"] == 3
    validate_episode(record, load_schema(SCHEMA_PATH))


def test_compute_deadlock_stall_detects_no_progress():
    """A stalled trajectory (no goal progress) is flagged as deadlock."""
    import numpy as np

    n = 60
    # Robot barely moves; distance to goal never decreases meaningfully.
    pos = np.zeros((n, 2), dtype=float)
    pos[:, 0] = np.linspace(0.0, 0.05, n)  # tiny drift, well under progress_eps window
    goal = np.array([10.0, 0.0], dtype=float)
    ep = type("_Ep", (), {})()
    ep.robot_pos = pos
    ep.goal = goal
    ep.dt = 0.1
    ep.reached_goal_step = None
    result = compute_deadlock_stall(ep, window_steps=15, progress_eps_m=0.05)
    assert result["deadlock"] is True
    assert result["stall_window_count"] > 0
    assert result["schema_version"] == "deadlock-stall.v1"


def test_compute_deadlock_stall_clear_when_moving():
    """A trajectory that makes steady progress toward the goal is not stalled."""
    import numpy as np

    n = 60
    pos = np.zeros((n, 2), dtype=float)
    pos[:, 0] = np.linspace(0.0, 9.5, n)
    goal = np.array([10.0, 0.0], dtype=float)
    ep = type("_Ep", (), {})()
    ep.robot_pos = pos
    ep.goal = goal
    ep.dt = 0.1
    ep.reached_goal_step = n - 1
    result = compute_deadlock_stall(ep, window_steps=15, progress_eps_m=0.05)
    assert result["deadlock"] is False
    assert result["stall_window_count"] == 0


def test_native_row_accepted_by_issue_5416_analyzer_diagnostics():
    """The #5416 analyzer probe accepts the native-command diagnostics block.

    This is the exact contract the preregistration probe requires: a non-empty
    planner_step_runtime_seconds sequence plus the four non-negative integer
    counters. The frozen roster/scenario/seed keying (the preregistration
    campaign) is intentionally out of scope for this slice.
    """
    from scripts.analysis import analyze_issue_5416_sipp_four_geometry as analyzer

    scenario = _native_command_spec(persistent=False)
    record = run_episode(
        scenario, seed=111, algo="native_command", horizon=40, dt=0.1, record_forces=False
    )
    am = record["algorithm_metadata"]
    # The analyzer's per-row diagnostic parser reads planner_diagnostics from the
    # algorithm_metadata block.
    diag_row = {"algorithm_metadata": {"planner_diagnostics": am["planner_diagnostics"]}}
    parsed_diag, diag_errors = analyzer._diagnostics(diag_row)
    assert diag_errors == []
    assert parsed_diag is not None
    assert len(parsed_diag["planner_step_runtime_seconds"]) == 40
    # And the deadlock boolean is present and typed in metrics.
    measurement, measurement_errors = analyzer._measurement(record)
    assert measurement_errors == []
    assert measurement["deadlock"] is False


def test_native_command_missing_argv_fails_closed():
    """A native_command block without argv must fail closed, not spawn blindly."""
    with pytest.raises(ValueError):
        runner_mod._NativeCommandPolicy(argv=[], env={}, timeout_s=1.0, persistent=False)


def test_native_command_parser_accepts_holonomic_response():
    """The standard runner accepts the documented world-frame velocity response."""
    policy = runner_mod._NativeCommandPolicy(
        argv=[sys.executable], env={}, timeout_s=1.0, persistent=False
    )
    command = policy._parse_response('{"vx": 0.5, "vy": -0.25}')
    assert command.tolist() == [0.5, -0.25]


def test_native_command_parser_rejects_nonfinite_response():
    """NaN/Inf responses fail closed instead of entering the simulator."""
    policy = runner_mod._NativeCommandPolicy(
        argv=[sys.executable], env={}, timeout_s=1.0, persistent=False
    )
    with pytest.raises(ValueError, match="finite"):
        policy._parse_response('{"v": NaN, "omega": 0.0}')
