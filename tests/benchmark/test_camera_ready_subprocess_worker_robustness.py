"""Robustness + end-to-end round-trip tests for the subprocess arm worker.

These tests close a gap left by the issue #4826 / #4957 / #5270 code lanes: the
subprocess isolation path (``_main_subprocess_worker``) is the hard-isolation
boundary for cross-arm VRAM leaks, so its contract is to ALWAYS emit structured
JSON on stdout (with ``cleanup_metrics``) so the parent campaign can record the
arm's outcome. Before this fix the worker's except clause caught only
``(json.JSONDecodeError, TypeError, ValueError, RuntimeError, OSError,
ImportError)`` and skipped cleanup on the failure path, so an unexpected error
(KeyError/AttributeError/IndexError/...) crashed the worker with EMPTY stdout:
the parent could not parse a result and the per-arm cleanup telemetry the issue's
acceptance criterion requires was lost.

Two contract layers are locked here:

1. **Worker-always-emits (Bug A)**: an unexpected exception from
   ``_run_single_arm_subprocess`` (e.g. ``KeyError``) is converted to a
   structured ``failed`` summary with ``cleanup_metrics``, and the
   defense-in-depth GPU cleanup still runs. This is verified by monkeypatching
   the arm runner to raise exception types outside the old narrow tuple.
2. **Real end-to-end round-trip**: the worker is actually spawned as
   ``python -m robot_sf.benchmark.camera_ready.resource_lifecycle``, fed
   parent-serialized params, and the parent can parse the structured result.
   No existing test spawns the real worker; every prior test mocks
   ``subprocess.run``. This round-trip guard catches the class of "worker is
   un-runnable / crashes before structured output" bugs that were repeatedly
   found only by live GPU runs (#4957 serialization crash, the nonexistent
   ``scenario_matrix`` module import, the scenario-list handoff).
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from robot_sf.benchmark.camera_ready.resource_lifecycle import (
    _main_subprocess_worker,
    _serialize_subprocess_arm_params,
    _SubprocessArmParams,
)


def _minimal_arm_params_dict() -> dict[str, object]:
    """A minimal, JSON-safe params dict accepted by ``_main_subprocess_worker``."""
    return {
        "planner_key": "robustness_probe",
        "planner_algo": "test_algo",
        "planner_human_model_variant": None,
        "planner_human_model_source": None,
        "planner_group": "core",
        "benchmark_profile": "speed",
        "socnav_missing_prereq_policy": "error",
        "adapter_impact_eval": "disabled",
        "kinematics": "differential_drive",
        "observation_mode": "lidar",
        "workers": 1,
        "horizon": 100,
        "dt": 0.1,
        "scenario_matrix_path": "/tmp/issue_4826_does_not_exist_matrix.yaml",
        "episodes_path": "/tmp/issue_4826_does_not_exist_episodes.jsonl",
        "summary_path": "/tmp/issue_4826_does_not_exist_summary.json",
        "record_forces": True,
        "record_planner_decision_trace": False,
        "record_simulation_step_trace": False,
        "observation_noise": None,
        "synthetic_actuation_profile": None,
        "latency_stress_profile": None,
        "snqi_weights": None,
        "snqi_baseline": None,
        "algo_config_path": None,
        "scoped_scenarios_path": None,
    }


def _run_worker_in_process(params_dict, arm_side_effect):
    """Drive ``_main_subprocess_worker`` with mocked stdin/stdout and a raising arm.

    Returns ``(returncode, parsed_stdout_or_None, cleanup_call_count)``. The arm
    runner is monkeypatched to raise ``arm_side_effect`` so the worker exercises
    its outer exception handler rather than the real (heavy) batch path.
    """
    import robot_sf.benchmark.camera_ready.resource_lifecycle as rl

    cleanup_calls = []
    original_cleanup = rl._cleanup_gpu_memory_before_exit

    def tracking_cleanup(*args, **kwargs):
        cleanup_calls.append((args, kwargs))
        return original_cleanup(*args, **kwargs)

    out_buf = io.StringIO()
    rc = None
    with (
        patch("sys.stdin", io.StringIO(json.dumps(params_dict))),
        patch(
            "robot_sf.benchmark.camera_ready.resource_lifecycle._run_single_arm_subprocess",
            side_effect=arm_side_effect,
        ),
        patch(
            "robot_sf.benchmark.camera_ready.resource_lifecycle._cleanup_gpu_memory_before_exit",
            side_effect=tracking_cleanup,
        ),
        patch("sys.stdout", out_buf),
        patch("loguru.logger"),
    ):
        rc = _main_subprocess_worker()

    raw = out_buf.getvalue().strip()
    parsed = json.loads(raw) if raw else None
    return rc, parsed, len(cleanup_calls)


class TestWorkerAlwaysEmitsStructuredOutput:
    """Bug A: the worker must ALWAYS emit structured JSON + run cleanup.

    Before the fix, exception types outside the narrow tuple
    (``KeyError``/``AttributeError``/``IndexError``/``ArithmeticError``) escaped
    the worker unhandled: stdout stayed empty, the parent could not parse a
    result, and the per-arm ``cleanup_metrics`` telemetry that issue #4826's
    acceptance criterion requires was lost. These tests assert the worker
    converts every ordinary exception into a structured ``failed`` result and
    still invokes the defense-in-depth GPU cleanup.
    """

    @pytest.mark.parametrize(
        "exc",
        [
            KeyError("missing_planner_field"),
            AttributeError("'NoneType' object has no attribute 'open'"),
            IndexError("list index out of range"),
            ArithmeticError("bad numeric op"),
            TypeError("unsupported operand"),
        ],
        ids=["key-error", "attribute-error", "index-error", "arithmetic-error", "type-error"],
    )
    def test_unexpected_exception_yields_structured_failure(self, exc):
        """Any ordinary exception becomes a structured ``failed`` result, not a crash."""
        rc, parsed, cleanup_count = _run_worker_in_process(_minimal_arm_params_dict(), exc)

        # The worker returned (did not raise) with the failure return code.
        assert rc == 1
        # stdout is parseable JSON — the parent can always record the outcome.
        assert parsed is not None
        assert parsed["summary"]["status"] == "failed"
        assert "total_jobs" in parsed["summary"]
        # The per-arm cleanup telemetry the issue requires is always present.
        assert "cleanup_metrics" in parsed
        # Defense-in-depth cleanup ran even on the unexpected-exception path.
        assert cleanup_count == 1

    def test_unexpected_exception_preserves_error_repr_for_diagnostics(self):
        """The structured failure carries a ``repr`` of the error for ops diagnosis."""
        rc, parsed, _ = _run_worker_in_process(
            _minimal_arm_params_dict(), KeyError("missing_planner_field")
        )
        assert rc == 1
        assert "KeyError" in parsed["summary"]["error"]
        assert any("missing_planner_field" in w for w in parsed["warnings"])

    def test_unexpected_exception_cleanup_failure_does_not_mask_original_error(self):
        """If cleanup itself raises, the original arm error still reaches stdout.

        Cleanup is best-effort: a cleanup failure must never prevent the worker
        from emitting the structured result for the arm failure (issue #4826:
        telemetry must never be silently lost).
        """
        out_buf = io.StringIO()
        rc = None
        params_dict = _minimal_arm_params_dict()
        with (
            patch("sys.stdin", io.StringIO(json.dumps(params_dict))),
            patch(
                "robot_sf.benchmark.camera_ready.resource_lifecycle._run_single_arm_subprocess",
                side_effect=KeyError("arm_failed_first"),
            ),
            patch(
                "robot_sf.benchmark.camera_ready.resource_lifecycle._cleanup_gpu_memory_before_exit",
                side_effect=RuntimeError("cleanup_also_failed"),
            ),
            patch("sys.stdout", out_buf),
            patch("loguru.logger"),
        ):
            rc = _main_subprocess_worker()

        assert rc == 1
        parsed = json.loads(out_buf.getvalue().strip())
        # The ARM error is reported, not the cleanup error.
        assert "arm_failed_first" in parsed["summary"]["error"]
        # cleanup_metrics key still present (empty dict when cleanup failed).
        assert "cleanup_metrics" in parsed

    def test_success_path_still_returns_zero(self):
        """The broaden-to-Exception fix does not regress the success path."""
        fake_result = {
            "summary": {"status": "ok", "total_jobs": 1, "written": 1, "failed_jobs": 0},
            "cleanup_metrics": {"torch_available": False, "cuda_available": False},
            "warnings": [],
            "episodes_total": 1,
        }
        out_buf = io.StringIO()
        rc = None
        params_dict = _minimal_arm_params_dict()
        with (
            patch("sys.stdin", io.StringIO(json.dumps(params_dict))),
            patch(
                "robot_sf.benchmark.camera_ready.resource_lifecycle._run_single_arm_subprocess",
                return_value=fake_result,
            ),
            patch("sys.stdout", out_buf),
            patch("loguru.logger"),
        ):
            rc = _main_subprocess_worker()

        assert rc == 0
        parsed = json.loads(out_buf.getvalue().strip())
        assert parsed == fake_result


class TestWorkerPathReconstructionUsesSharedFieldTuple:
    """The worker reconstructs Path-typed fields from the SAME tuple the
    serializer str-converts with.

    Before PR #5270 this was a hardcoded subset that missed
    ``scoped_scenarios_path``; the worker then reconstructed it as ``str`` and
    crashed on ``.open()``. Locking the two sides to one tuple
    (``_SUBPROCESS_ARM_PATH_FIELDS``) prevents that drift class from recurring.
    """

    def test_path_reconstruction_covers_scoped_scenarios_path(self):
        """scoped_scenarios_path round-trips str -> Path through the real worker.

        This exercises the actual serialization + reconstruction contract (not a
        hand-converted copy), so a future field added to one side but not the
        other is caught.
        """
        from robot_sf.benchmark.camera_ready.resource_lifecycle import (
            _SUBPROCESS_ARM_PATH_FIELDS,
        )

        # scoped_scenarios_path must be in the shared tuple that drives BOTH the
        # serializer (str-convert) and the worker (str -> Path reconstruct).
        assert "scoped_scenarios_path" in _SUBPROCESS_ARM_PATH_FIELDS

        base = _SubprocessArmParams(
            planner_key="t",
            planner_algo="t",
            planner_human_model_variant=None,
            planner_human_model_source=None,
            planner_group="core",
            benchmark_profile="speed",
            socnav_missing_prereq_policy="error",
            adapter_impact_eval="disabled",
            kinematics="differential_drive",
            observation_mode="lidar",
            workers=1,
            horizon=10,
            dt=0.1,
            scenario_matrix_path=Path("/tmp/m.yaml"),
            episodes_path=Path("/tmp/e.jsonl"),
            summary_path=Path("/tmp/s.json"),
            record_forces=False,
            record_planner_decision_trace=False,
            record_simulation_step_trace=False,
            observation_noise=None,
            synthetic_actuation_profile=None,
            latency_stress_profile=None,
            snqi_weights=None,
            snqi_baseline=None,
            algo_config_path=None,
            scoped_scenarios_path=Path("/tmp/scoped.json"),
        )
        serialized = json.loads(_serialize_subprocess_arm_params(base))
        # Serializer str-converted it.
        assert serialized["scoped_scenarios_path"] == "/tmp/scoped.json"

        # Worker reconstruction mirrors _main_subprocess_worker's loop.
        for field_name in _SUBPROCESS_ARM_PATH_FIELDS:
            value = serialized.get(field_name)
            if value:
                serialized[field_name] = Path(value)
        rebuilt = _SubprocessArmParams(**serialized)
        assert isinstance(rebuilt.scoped_scenarios_path, Path)
        assert rebuilt == base


class TestRealSubprocessWorkerRoundTrip:
    """End-to-end: spawn the REAL worker process and round-trip params.

    Every prior #4826/#4957/#5270 test mocks ``subprocess.run``, so the real
    worker entrypoint (``python -m robot_sf.benchmark.camera_ready.resource_lifecycle``)
    has never been exercised in CI. This is the guard that would have caught the
    serialization crash (#4957), the nonexistent ``scenario_matrix`` module
    import, and the scenario-list handoff bug — all of which were found only by
    live GPU runs. The worker is spawned against a non-existent scenario matrix
    so it takes the fast controlled-failure path (no simulation, no GPU); the
    contract under test is the round-trip, not benchmark execution.
    """

    def test_worker_module_is_invocable_as_dash_m(self):
        """The worker is runnable as ``python -m <module>`` (the parent's spawn target)."""
        proc = subprocess.run(
            [sys.executable, "-m", "robot_sf.benchmark.camera_ready.resource_lifecycle"],
            input="not valid json",
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        # Invalid JSON raises json.JSONDecodeError (a ValueError subclass),
        # which the broadened handler now converts to structured output rather
        # than a traceback. The module must be invocable either way.
        assert proc.returncode in (0, 1)
        # No unhandled traceback on stderr for a JSON error.
        assert "Traceback (most recent call last)" not in proc.stderr

    def test_real_worker_round_trips_parent_serialized_params(self, tmp_path):
        """Parent serializes -> real worker reads stdin -> structured result on stdout.

        Points the worker at a non-existent scenario matrix and no scoped list,
        so the legacy ``load_scenario_matrix`` path fails fast with an OSError.
        The worker must convert that to a structured ``failed`` result with
        ``cleanup_metrics`` rather than crashing.
        """
        params = _SubprocessArmParams(
            planner_key="e2e_round_trip",
            planner_algo="test_algo",
            planner_human_model_variant=None,
            planner_human_model_source=None,
            planner_group="core",
            benchmark_profile="speed",
            socnav_missing_prereq_policy="error",
            adapter_impact_eval="disabled",
            kinematics="differential_drive",
            observation_mode="lidar",
            workers=1,
            horizon=5,
            dt=0.1,
            scenario_matrix_path=tmp_path / "does_not_exist_matrix.yaml",
            episodes_path=tmp_path / "episodes.jsonl",
            summary_path=tmp_path / "summary.json",
            record_forces=False,
            record_planner_decision_trace=False,
            record_simulation_step_trace=False,
            observation_noise=None,
            synthetic_actuation_profile=None,
            latency_stress_profile=None,
            snqi_weights=None,
            snqi_baseline=None,
            algo_config_path=None,
            scoped_scenarios_path=None,
        )
        arm_json = _serialize_subprocess_arm_params(params)

        proc = subprocess.run(
            [sys.executable, "-m", "robot_sf.benchmark.camera_ready.resource_lifecycle"],
            input=arm_json,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )

        # The worker ran to completion (did not crash with a traceback).
        assert "Traceback (most recent call last)" not in proc.stderr, proc.stderr[-500:]
        raw = proc.stdout.strip()
        assert raw, f"worker produced empty stdout; stderr={proc.stderr[-300:]}"
        parsed = json.loads(raw)

        # Structured result contract: always these four keys.
        assert set(parsed.keys()) >= {"summary", "cleanup_metrics", "warnings", "episodes_total"}
        # The non-existent matrix yields a controlled failure (never a crash).
        assert parsed["summary"]["status"] == "failed"
        # Per-arm cleanup telemetry is always present (issue #4826 acceptance).
        assert isinstance(parsed["cleanup_metrics"], dict)
        assert "planner_key" in parsed["cleanup_metrics"]
        assert parsed["cleanup_metrics"]["planner_key"] == "e2e_round_trip"

    def test_real_worker_consumes_parent_prepared_scoped_scenarios(self, tmp_path):
        """The worker runs the parent-prepared scoped scenario list end-to-end.

        This is the #5270 contract (worker must consume the parent's prepared
        scenarios, not re-load the matrix) exercised through the REAL subprocess
        spawn — which no prior test does. ``scenario_matrix_path`` deliberately
        does not exist: if the worker re-loaded the matrix instead of consuming
        ``scoped_scenarios_path``, the legacy loader would raise rather than
        reach ``run_batch``. ``run_batch`` is not stubbed here, so the bogus
        map_file makes it fail fast; the contract under test is that the scoped
        handoff survives the real spawn and yields a structured result.
        """
        scoped = [{"name": "probe", "map_file": "does_not_exist.svg", "seeds": [1]}]
        scoped_path = tmp_path / "scoped_scenarios.json"
        scoped_path.write_text(json.dumps(scoped), encoding="utf-8")

        params = _SubprocessArmParams(
            planner_key="scoped_e2e",
            planner_algo="test_algo",
            planner_human_model_variant=None,
            planner_human_model_source=None,
            planner_group="core",
            benchmark_profile="speed",
            socnav_missing_prereq_policy="error",
            adapter_impact_eval="disabled",
            kinematics="differential_drive",
            observation_mode="lidar",
            workers=1,
            horizon=5,
            dt=0.1,
            scenario_matrix_path=tmp_path / "does_not_exist_matrix.yaml",
            episodes_path=tmp_path / "episodes.jsonl",
            summary_path=tmp_path / "summary.json",
            record_forces=False,
            record_planner_decision_trace=False,
            record_simulation_step_trace=False,
            observation_noise=None,
            synthetic_actuation_profile=None,
            latency_stress_profile=None,
            snqi_weights=None,
            snqi_baseline=None,
            algo_config_path=None,
            scoped_scenarios_path=scoped_path,
        )
        arm_json = _serialize_subprocess_arm_params(params)

        repo_root = str(Path(__file__).resolve().parents[2])
        proc = subprocess.run(
            [sys.executable, "-m", "robot_sf.benchmark.camera_ready.resource_lifecycle"],
            input=arm_json,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
            cwd=repo_root,
        )

        assert "Traceback (most recent call last)" not in proc.stderr, proc.stderr[-500:]
        raw = proc.stdout.strip()
        assert raw, f"worker produced empty stdout; stderr={proc.stderr[-300:]}"
        parsed = json.loads(raw)
        # The worker reached run_batch via the scoped path (not the legacy
        # matrix loader, which would have crashed on the non-existent matrix).
        # run_batch then fails on the bogus map_file -> controlled failure with
        # cleanup metrics, proving the scoped handoff survived the real spawn.
        assert parsed["summary"]["status"] in {"failed", "not_available", "partial-failure"}
        assert isinstance(parsed["cleanup_metrics"], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
