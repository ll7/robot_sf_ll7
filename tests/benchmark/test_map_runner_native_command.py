"""Tests for the issue #5887 native-command execution arm and deadlock/stall metric."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from robot_sf.benchmark import map_runner_native_command as nc
from robot_sf.benchmark.map_runner_native_command import (
    NativeCommandContractError,
    NativeCommandStepError,
    _NoProgressDeadlockDetector,
    build_native_command_policy,
)

_FAKE_PLANNER = str(Path(__file__).resolve().parent / "fixtures" / "native_command_fake_planner.py")


# ---------------------------------------------------------------------------
# Command-contract parsing
# ---------------------------------------------------------------------------


class TestNativeCommandContract:
    """Validate argv + env contract parsing and template substitution."""

    def test_binary_hash_requires_nonempty_command(self) -> None:
        with pytest.raises(NativeCommandContractError, match="non-empty argv"):
            nc._resolve_binary_hash([])

    def test_from_config_requires_command(self) -> None:
        with pytest.raises(NativeCommandContractError):
            nc.NativeCommandSpec.from_config({})

    def test_from_config_defaults_to_per_episode(self) -> None:
        spec = nc.NativeCommandSpec.from_config({"command": ["/bin/true"]})
        assert spec.mode == "per_episode"
        assert spec.step_timeout_sec == nc._DEFAULT_STEP_TIMEOUT_SEC

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(NativeCommandContractError):
            nc.NativeCommandSpec.from_config({"command": ["/bin/true"], "mode": "frobnicate"})

    def test_nonpositive_step_timeout_rejected(self) -> None:
        with pytest.raises(NativeCommandContractError, match="must be positive"):
            nc.NativeCommandSpec.from_config({"command": ["/bin/true"], "step_timeout_sec": 0.0})

    def test_persistent_mode_accepted(self) -> None:
        spec = nc.NativeCommandSpec.from_config({"command": ["/bin/true"], "mode": "persistent"})
        assert spec.mode == "persistent"

    def test_standard_contract_aliases_are_accepted(self) -> None:
        spec = nc.NativeCommandSpec.from_config(
            {"argv": ["/bin/true"], "persistent": True, "timeout_s": 0.25}
        )
        assert spec.command == ["/bin/true"]
        assert spec.mode == "persistent"
        assert spec.step_timeout_sec == 0.25

    def test_template_substitution(self) -> None:
        spec = nc.NativeCommandSpec.from_config(
            {
                "command": ["planner", "--scenario", "{scenario_id}", "--seed", "{seed}"],
                "env": {"SIPP_SEED": "{seed}"},
            }
        ).resolve_templates(scenario_id="corridor", seed=111, horizon=500, dt=0.1)
        assert spec.command[2] == "corridor"
        assert spec.command[4] == "111"
        assert spec.env["SIPP_SEED"] == "111"

    def test_binary_hash_is_deterministic(self) -> None:
        spec = nc.NativeCommandSpec.from_config({"command": [_FAKE_PLANNER]})
        assert len(spec.binary_hash) == 64
        assert spec.binary_label.endswith("native_command_fake_planner.py")


# ---------------------------------------------------------------------------
# Deadlock / stall detector
# ---------------------------------------------------------------------------


class TestNoProgressDeadlockDetector:
    """Parameterized no-progress-over-window detector behavior."""

    def test_progress_resets_stall(self) -> None:
        detector = _NoProgressDeadlockDetector(window_steps=3, progress_threshold_m=0.05)
        for distance in (10.0, 8.0, 5.0, 1.0):
            detector.update(distance)
        assert detector.active is False

    def test_no_progress_over_window_activates(self) -> None:
        detector = _NoProgressDeadlockDetector(window_steps=3, progress_threshold_m=0.05)
        for distance in (10.0, 9.97, 9.99, 9.98):
            detector.update(distance)
        assert detector.active is True

    def test_partial_window_does_not_activate_stall(self) -> None:
        detector = _NoProgressDeadlockDetector(window_steps=3, progress_threshold_m=0.05)
        for distance in (10.0, 9.99, 9.99):
            detector.update(distance)
        assert detector.active is False

        detector.update(9.98)
        assert detector.active is True

    def test_field_shape(self) -> None:
        detector = _NoProgressDeadlockDetector(window_steps=3, progress_threshold_m=0.05)
        field = detector.as_field()
        assert field["schema_version"] == "native-command-deadlock.v1"
        assert field["window_steps"] == 3
        assert field["progress_threshold_m"] == 0.05

    def test_window_steps_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            _NoProgressDeadlockDetector(window_steps=0)

    def test_progress_threshold_must_be_nonnegative(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            _NoProgressDeadlockDetector(progress_threshold_m=-0.01)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("", "empty response"),
        ("[]", "JSON object"),
        ("{}", "missing a recognized linear/angular command pair"),
    ],
)
def test_native_command_response_rejects_unusable_payloads(response: str, message: str) -> None:
    with pytest.raises(NativeCommandStepError, match=message):
        nc._parse_response(response)


def test_native_command_response_rejects_nonfinite_values() -> None:
    with pytest.raises(NativeCommandStepError, match="finite"):
        nc._parse_response('{"linear_velocity": NaN, "angular_velocity": 0.0}')


def test_native_command_metadata_rejects_non_mapping() -> None:
    assert nc.native_command_metadata_for_record(None) == (False, {}, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Native-command policy builder
# ---------------------------------------------------------------------------


class TestNativeCommandPolicyBuilder:
    """Policy builder must expose native execution mode and diagnostics."""

    def test_metadata_is_native_with_diagnostics(self) -> None:
        policy, meta = build_native_command_policy(
            "native_command",
            {"command": [_FAKE_PLANNER]},
            scenario_id="corridor",
            seed=111,
            horizon=500,
            dt=0.1,
            robot_kinematics="differential_drive",
        )
        kinematics = meta["planner_kinematics"]
        assert kinematics["execution_mode"] == "native"
        diag = meta["planner_diagnostics"]
        for key in (
            "expansion_limit_hits",
            "runtime_bound_exits",
            "fallback_count",
            "commitment_invalidations",
        ):
            assert diag[key] == 0
        assert diag["planner_step_runtime_seconds"] == []
        assert callable(policy)

    def test_per_episode_plan_returns_command(self) -> None:
        policy, _meta = build_native_command_policy(
            "native_command",
            {"command": ["python3", _FAKE_PLANNER], "mode": "per_episode"},
            scenario_id="corridor",
            seed=111,
            horizon=500,
            dt=0.1,
        )
        obs = {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [5.0, 0.0]},
        }
        linear, angular = policy(obs)
        assert isinstance(linear, float) and isinstance(angular, float)
        run_state = _meta["_native_run_state"]
        assert run_state["planner_diagnostics"]["planner_step_runtime_seconds"]
        assert run_state["deadlock_field"]["active"] is False

    def test_invalid_json_response_raises(self, tmp_path: Path) -> None:
        bad_script = tmp_path / "bad.py"
        bad_script.write_text("import sys\nprint('not json')\n")
        policy, _meta = build_native_command_policy(
            "native_command",
            {"command": ["python3", str(bad_script)], "mode": "per_episode"},
            scenario_id="corridor",
            seed=111,
            horizon=500,
            dt=0.1,
        )
        assert policy({"robot": {}, "goal": {}}) == (0.0, 0.0)
        assert _meta["_native_run_state"]["planner_diagnostics"]["fallback_count"] == 1

    def test_persistent_plan_reuses_process(self) -> None:
        policy, _ = build_native_command_policy(
            "native_command",
            {"command": ["python3", _FAKE_PLANNER], "mode": "persistent"},
            scenario_id="corridor",
            seed=111,
            horizon=500,
            dt=0.1,
        )
        obs = {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [5.0, 0.0]},
        }

        # Reset must spawn the process
        policy._planner_reset(seed=111)
        planner = policy._native_planner
        proc1 = planner._process
        assert proc1 is not None
        assert proc1.poll() is None

        # First plan step
        linear, _ = policy(obs)
        assert isinstance(linear, float)
        assert planner._process is proc1

        # Second plan step
        _ = policy(obs)
        assert planner._process is proc1  # Process must be kept alive and reused

        policy._planner_close()
        assert planner._process is None

    def test_persistent_plan_single_spawn_across_five_steps(self) -> None:
        """A long-lived (loops-over-stdin) persistent planner spawns exactly once across >=5 steps.

        The literal regression guard requested by issue #5957: the pre-fix persistent
        branch respawned the child every step, so the new ``process_spawns`` diagnostic
        must read exactly 1 for a healthy persistent run, no matter how many steps run.
        """
        policy, _ = build_native_command_policy(
            "native_command",
            {"command": ["python3", _FAKE_PLANNER], "mode": "persistent"},
            scenario_id="corridor",
            seed=111,
            horizon=500,
            dt=0.1,
        )
        obs = {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [5.0, 0.0]},
        }
        policy._planner_reset(seed=111)
        planner = policy._native_planner

        try:
            num_steps = 6  # >= 5 as required by the issue
            for _ in range(num_steps):
                linear, angular = policy(obs)
                assert isinstance(linear, float)
                assert isinstance(angular, float)

            diag = planner.diagnostics
            # One persistent child reused across every step — the whole point of #5957.
            assert diag["process_spawns"] == 1
            # And it genuinely served all steps with no fallback.
            assert diag["fallback_count"] == 0
            assert diag["runtime_bound_exits"] == 0
        finally:
            policy._planner_close()

    def test_per_episode_plan_spawns_once_per_step(self) -> None:
        """Per-episode mode legitimately launches one fresh child per step.

        Contrast case giving ``process_spawns`` meaning: per-episode mode must record
        one spawn per step, so the counter reflects the chosen launch contract rather
        than being pinned to 1 universally.
        """
        policy, _ = build_native_command_policy(
            "native_command",
            {"command": ["python3", _FAKE_PLANNER], "mode": "per_episode"},
            scenario_id="corridor",
            seed=111,
            horizon=500,
            dt=0.1,
        )
        obs = {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [5.0, 0.0]},
        }
        policy._planner_reset(seed=111)
        planner = policy._native_planner

        try:
            num_steps = 4
            for _ in range(num_steps):
                policy(obs)

            diag = planner.diagnostics
            # Per-episode: one spawn per step, as designed.
            assert diag["process_spawns"] == num_steps
        finally:
            policy._planner_close()

    def test_per_episode_failed_launch_not_counted_as_spawn(self) -> None:
        """A per-episode launch that never starts must not be counted as a spawn.

        Companion to the "count only successful spawns" rule applied to the
        persistent path: ``process_spawns`` records how many subprocesses were
        actually launched, so an ``OSError`` launch failure (e.g. a missing
        binary) must leave the counter at zero even though ``_plan_per_episode``
        was invoked.
        """
        policy, _ = build_native_command_policy(
            "native_command",
            {"command": ["/nonexistent/native_planner_binary"], "mode": "per_episode"},
            scenario_id="corridor",
            seed=111,
            horizon=500,
            dt=0.1,
        )
        obs = {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [5.0, 0.0]},
        }
        planner = policy._native_planner

        try:
            with pytest.raises(NativeCommandContractError, match="failed to launch native command"):
                policy(obs)
            # The launch never happened, so it must not be counted as a spawn.
            assert planner.diagnostics["process_spawns"] == 0
        finally:
            policy._planner_close()

    def test_persistent_plan_timeout(self, tmp_path: Path) -> None:
        import sys

        # A stub that sleeps
        slow_script = tmp_path / "slow.py"
        slow_script.write_text("import sys, time\nfor line in sys.stdin:\n    time.sleep(5)\n")
        policy, _ = build_native_command_policy(
            "native_command",
            {
                "command": [sys.executable, str(slow_script)],
                "mode": "persistent",
                "step_timeout_sec": 0.2,
            },
            scenario_id="corridor",
            seed=111,
            horizon=500,
            dt=0.1,
        )
        policy._planner_reset(seed=111)
        planner = policy._native_planner
        proc = planner._process
        assert proc is not None

        # The map-runner policy converts a bounded native step failure to zero velocity.
        assert policy({"robot": {"position": [0.0, 0.0]}, "goal": {"current": [5.0, 0.0]}}) == (
            0.0,
            0.0,
        )

        # Check diagnostics
        diag = planner.diagnostics
        assert diag["runtime_bound_exits"] == 1
        assert diag["fallback_count"] == 1
        assert planner._process is None  # Stopped on timeout


# ---------------------------------------------------------------------------
# End-to-end smoke through the real map-runner batch
# ---------------------------------------------------------------------------


def test_native_command_smoke_produces_schema_valid_row(tmp_path: Path) -> None:
    """Run one native-command episode end-to-end and assert the new diagnostics fields."""
    from robot_sf.benchmark.map_runner import run_map_batch

    map_file = (
        Path(__file__).resolve().parents[2] / "maps" / "svg_maps" / "planner_sanity_open.svg"
    ).as_posix()
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        "scenarios:\n"
        f"- name: native_smoke\n"
        f"  map_file: {map_file}\n"
        "  simulation_config:\n"
        "    max_episode_steps: 120\n"
        "    ped_density: 0.0\n"
        "  single_pedestrians: []\n"
        "  robot_config: {}\n"
        "  metadata:\n"
        "    archetype: sanity_simple\n"
        "    behavior: none\n"
        "    flow: none\n"
        "    purpose: native_command_smoke\n"
        "  seeds:\n"
        "  - 111\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "episodes.jsonl"
    algo_config_path = tmp_path / "algo.yaml"
    algo_config_path.write_text(
        json.dumps({"command": ["python3", _FAKE_PLANNER], "mode": "per_episode"}),
        encoding="utf-8",
    )

    summary = run_map_batch(
        str(scenario),
        str(out_path),
        schema_path="robot_sf/benchmark/schemas/episode.schema.v1.json",
        algo="native_command",
        algo_config_path=str(algo_config_path),
        horizon=120,
        dt=0.1,
        workers=1,
        resume=False,
        benchmark_profile="experimental",
    )
    assert summary["written"] >= 1

    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "no episode rows were written"
    record = rows[0]
    # New additive diagnostics fields present and schema-shaped.
    assert isinstance(record["metrics"]["deadlock"], bool)
    assert record["algorithm_metadata"]["planner_kinematics"]["execution_mode"] == "native"
    diag = record["algorithm_metadata"]["planner_diagnostics"]
    for key in (
        "expansion_limit_hits",
        "runtime_bound_exits",
        "fallback_count",
        "commitment_invalidations",
    ):
        assert isinstance(diag[key], int) and diag[key] >= 0
    assert (
        isinstance(diag["planner_step_runtime_seconds"], list)
        and diag["planner_step_runtime_seconds"]
    )
    assert (
        record["algorithm_metadata"]["native_command"]["deadlock"]["schema_version"]
        == "native-command-deadlock.v1"
    )
    from scripts.analysis.analyze_issue_5416_sipp_four_geometry import _diagnostics

    diagnostics, errors = _diagnostics(record)
    assert diagnostics is not None
    assert errors == []
    assert record["algorithm_metadata"]["native_command"]["binary_hash"]


def test_native_command_persistent_timeout_is_bounded(tmp_path: Path) -> None:
    """Persistent map-runner reads have a hard timeout instead of unbounded readline."""
    slow_script = tmp_path / "slow_native.py"
    slow_script.write_text(
        "import time\ntime.sleep(5)\n",
        encoding="utf-8",
    )
    policy, meta = build_native_command_policy(
        "native_command",
        {"argv": ["python3", str(slow_script)], "persistent": True, "timeout_s": 0.05},
        scenario_id="slow",
        seed=1,
        horizon=2,
        dt=0.1,
    )
    try:
        started = time.monotonic()
        assert policy({"robot": {}, "goal": {}}) == (0.0, 0.0)
        assert time.monotonic() - started < 1.0
    finally:
        policy._planner_close()  # type: ignore[attr-defined]
    diag = meta["_native_run_state"]["planner_diagnostics"]
    assert diag["runtime_bound_exits"] == 1
    assert diag["fallback_count"] == 1


def test_native_row_passes_5416_analyzer_native_gate(tmp_path: Path) -> None:
    """A native-command row satisfies the issue #5416 analyzer's native/deadlock/diagnostics gate.

    The analyzer's ``_parse_row`` excludes rows whose execution_mode is not native or whose
    deadlock/planner_diagnostics are missing. This test asserts a produced native-command row
    passes those gates so the analyzer probe can return eligible rows > 0 once the row is
    keyed to the frozen matrix.
    """
    from scripts.analysis.analyze_issue_5416_sipp_four_geometry import _diagnostics, _measurement

    record = {
        "version": "v1",
        "horizon": 500,
        "metrics": {
            "deadlock": False,
            "ped_collision_count": 0,
            "obstacle_collision_count": 0,
            "time_to_goal_norm": 0.5,
            "path_efficiency": 0.9,
        },
        "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
        "algorithm_metadata": {
            "status": "ok",
            "fallback_or_degraded": False,
            "planner_kinematics": {"execution_mode": "native"},
            "planner_diagnostics": {
                "expansion_limit_hits": 0,
                "runtime_bound_exits": 0,
                "fallback_count": 0,
                "commitment_invalidations": 0,
                "planner_step_runtime_seconds": [0.01, 0.02],
            },
        },
        "integrity": {"contradictions": []},
    }
    measurement, m_errors = _measurement(record)
    assert measurement is not None
    assert measurement["deadlock"] is False
    assert m_errors == []
    diagnostics, d_errors = _diagnostics(record)
    assert diagnostics is not None
    assert d_errors == []
    assert diagnostics["expansion_limit_hits"] == 0


def test_5416_analyzer_probe_returns_eligible_rows(tmp_path: Path) -> None:
    """The issue #5416 analyzer probe returns eligible rows > 0 on a compliant native row.

    Acceptance criterion #2 of issue #5887: the analyzer probe that previously returned
    partial / 0 eligible rows must return eligible rows > 0 once the native-command arm
    produces schema-valid native rows. We synthesize one row keyed to the frozen matrix
    (scenario_id/seed/planner from the preregistration packet) carrying the native
    execution mode, ``metrics.deadlock``, and ``planner_diagnostics`` the issue requires.
    """
    from scripts.analysis.analyze_issue_5416_sipp_four_geometry import build_analysis

    episode = tmp_path / "native_episode.jsonl"
    row = {
        "version": "v1",
        "scenario_id": "classic_head_on_corridor_low",
        "seed": 111,
        "horizon": 500,
        "planner_id": "sipp_lattice",
        "algo": "native_command",
        "metrics": {
            "deadlock": False,
            "ped_collision_count": 0,
            "obstacle_collision_count": 0,
            "time_to_goal_norm": 0.4,
            "path_efficiency": 0.92,
        },
        "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
        "algorithm_metadata": {
            "status": "ok",
            "fallback_or_degraded": False,
            "planner_kinematics": {"execution_mode": "native"},
            "planner_diagnostics": {
                "expansion_limit_hits": 3,
                "runtime_bound_exits": 0,
                "fallback_count": 0,
                "commitment_invalidations": 1,
                "planner_step_runtime_seconds": [0.01, 0.02, 0.015],
            },
        },
        "integrity": {"contradictions": []},
        "result_provenance": {
            "schema_version": "benchmark_row_provenance.v1",
            "scenario_id": "classic_head_on_corridor_low",
            "seed": 111,
            "config_hash": "abc123",
            "repo_commit": "deadbeef",
            "simulator_settings": {"horizon": 500, "dt": 0.1},
        },
    }
    episode.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = build_analysis(
        episode_paths=[str(episode)],
        output_dir=str(tmp_path / "analysis"),
        packet_path="configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml",
    )
    assert report["matrix"]["eligible_rows"] >= 1
    assert report["matrix"]["excluded_rows"] == 0
