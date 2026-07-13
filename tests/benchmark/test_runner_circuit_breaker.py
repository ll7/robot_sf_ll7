"""Circuit breaker tests for consecutive-identical-failure abort in batch runner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.circuit_breaker import normalize_circuit_breaker_threshold
from robot_sf.benchmark.map_runner_batch_runner import _serial_execute_map_jobs
from robot_sf.benchmark.runner import (
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    _error_signature,
    _run_batch_sequential,
)

_RUNNER_MOD = "robot_sf.benchmark.runner"


def _dummy_schema() -> dict:
    """Return a minimal JSON schema that won't block validation."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {},
    }


def _dummy_scenario(scenario_id: str = "test-scenario") -> dict:
    """Return a minimal scenario dict."""
    return {"id": scenario_id}


def _dummy_fixed_params() -> dict:
    """Return a minimal fixed_params dict."""
    return {
        "horizon": 10,
        "dt": 0.1,
        "record_forces": False,
        "snqi_weights": None,
        "snqi_baseline": None,
        "algo": "simple_policy",
        "algo_config_path": None,
        "video_enabled": False,
        "video_renderer": "none",
        "videos_dir": None,
        "experimental_ped_impact": False,
        "ped_impact_radius_m": 2.0,
        "ped_impact_window_steps": 5,
        "provenance": None,
    }


def _valid_record(seed: int) -> dict:
    """Return a minimal valid episode record that passes schema validation."""
    return {
        "episode_id": f"test-episode-{seed}",
        "scenario_id": "test-scenario",
        "seed": seed,
        "version": "v1",
        "scenario_params": _dummy_scenario(),
        "metrics": {"success": 1.0, "collisions": 0.0},
        "algorithm_metadata": {"algorithm": "simple_policy"},
        "config_hash": "na",
        "git_hash": "na",
        "timestamps": {"start": "2024-01-01T00:00:00+00:00"},
        "termination_reason": "success",
        "status": "success",
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
        "integrity": {"contradictions": []},
    }


# -- _error_signature tests --


class TestErrorSignature:
    """Tests for the _error_signature helper used by the circuit breaker."""

    def test_runtime_error_signature(self) -> None:
        sig = _error_signature(RuntimeError("CUDA out of memory on device 0"))
        assert sig == ("RuntimeError", "CUDA out of memory on device 0")

    def test_value_error_signature(self) -> None:
        sig = _error_signature(ValueError("invalid literal"))
        assert sig == ("ValueError", "invalid literal")

    def test_same_type_and_message_produces_same_signature(self) -> None:
        exc1 = RuntimeError("CUDA out of memory: tried to allocate 1024 MiB")
        exc2 = RuntimeError("CUDA out of memory: tried to allocate 1024 MiB")
        assert _error_signature(exc1) == _error_signature(exc2)

    def test_same_type_different_message_produces_different_signature(self) -> None:
        exc1 = RuntimeError("CUDA out of memory")
        exc2 = RuntimeError("Network timeout")
        assert _error_signature(exc1) != _error_signature(exc2)

    def test_different_type_same_message_produces_different_signature(self) -> None:
        exc1 = RuntimeError("same message")
        exc2 = ValueError("same message")
        assert _error_signature(exc1) != _error_signature(exc2)

    def test_long_message_truncated(self) -> None:
        long_msg = "A" * 500
        sig = _error_signature(RuntimeError(long_msg))
        assert sig[0] == "RuntimeError"
        from robot_sf.benchmark.runner import _CIRCUIT_BREAKER_MSG_PREFIX_LEN

        assert len(sig[1]) == _CIRCUIT_BREAKER_MSG_PREFIX_LEN

    def test_whitespace_normalization(self) -> None:
        msg = "\r\n  leading and trailing  \r\n"
        sig = _error_signature(ValueError(msg))
        assert sig[1] == "leading and trailing"


# -- _run_batch_sequential circuit breaker tests --


class TestSequentialCircuitBreaker:
    """Tests for the circuit breaker in _run_batch_sequential."""

    def test_identical_failures_trip_circuit_breaker(self, tmp_path: Path) -> None:
        """When 10+ consecutive identical failures occur, the arm aborts."""

        def failing_worker(job):
            raise RuntimeError("CUDA out of memory on device 0")

        out_file = tmp_path / "episodes.jsonl"
        jobs = [(_dummy_scenario(), i) for i in range(20)]
        schema = _dummy_schema()

        with patch(f"{_RUNNER_MOD}._run_job_worker", side_effect=failing_worker):
            written, failures, abort_meta = _run_batch_sequential(
                jobs,
                out_path=out_file,
                schema=schema,
                fixed_params=_dummy_fixed_params(),
                progress_cb=None,
                fail_fast=False,
                circuit_breaker_threshold=DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
            )

        assert written == 0
        assert len(failures) == DEFAULT_CIRCUIT_BREAKER_THRESHOLD
        assert abort_meta is not None
        assert abort_meta["status"] == "aborted_systematic_failure"
        assert abort_meta["signature"]["type"] == "RuntimeError"
        assert "CUDA out of memory" in abort_meta["signature"]["message_prefix"]
        assert abort_meta["consecutive_failures"] == DEFAULT_CIRCUIT_BREAKER_THRESHOLD
        assert abort_meta["first_fail_index"] == 1
        assert abort_meta["episodes_completed_before_onset"] == 0
        assert abort_meta["projected_episodes_saved"] == 20 - DEFAULT_CIRCUIT_BREAKER_THRESHOLD

    def test_mixed_errors_do_not_trip_circuit_breaker(self, tmp_path: Path) -> None:
        """Distinct error signatures reset the streak so abort never triggers."""
        call_counter = [0]

        def alternating_worker(job):
            call_counter[0] += 1
            if call_counter[0] % 2 == 0:
                raise RuntimeError("error A")
            raise RuntimeError("error B")

        out_file = tmp_path / "episodes.jsonl"
        num_jobs = 30
        jobs = [(_dummy_scenario(), i) for i in range(num_jobs)]
        schema = _dummy_schema()

        with patch(f"{_RUNNER_MOD}._run_job_worker", side_effect=alternating_worker):
            written, failures, abort_meta = _run_batch_sequential(
                jobs,
                out_path=out_file,
                schema=schema,
                fixed_params=_dummy_fixed_params(),
                progress_cb=None,
                fail_fast=False,
                circuit_breaker_threshold=DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
            )

        assert written == 0
        assert len(failures) == num_jobs
        assert abort_meta is None

    def test_success_resets_counter(self, tmp_path: Path) -> None:
        """A successful job between failures resets the circuit breaker counter."""
        call_counter = [0]

        def intermittent_worker(job):
            call_counter[0] += 1
            if call_counter[0] % 5 == 0:
                return _valid_record(call_counter[0])
            raise RuntimeError("CUDA out of memory on device 0")

        out_file = tmp_path / "episodes.jsonl"
        num_jobs = 60
        jobs = [(_dummy_scenario(), i) for i in range(num_jobs)]
        schema = _dummy_schema()

        with patch(f"{_RUNNER_MOD}._run_job_worker", side_effect=intermittent_worker):
            written, failures, abort_meta = _run_batch_sequential(
                jobs,
                out_path=out_file,
                schema=schema,
                fixed_params=_dummy_fixed_params(),
                progress_cb=None,
                fail_fast=False,
                circuit_breaker_threshold=DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
            )

        # Every 5th job succeeds, so we never reach 10 consecutive failures
        assert abort_meta is None
        assert written > 0
        assert len(failures) > 0

    def test_custom_threshold(self, tmp_path: Path) -> None:
        """The circuit breaker respects a custom threshold."""

        def failing_worker(job):
            raise ValueError("OOM")

        out_file = tmp_path / "episodes.jsonl"
        jobs = [(_dummy_scenario(), i) for i in range(20)]
        schema = _dummy_schema()

        with patch(f"{_RUNNER_MOD}._run_job_worker", side_effect=failing_worker):
            written, failures, abort_meta = _run_batch_sequential(
                jobs,
                out_path=out_file,
                schema=schema,
                fixed_params=_dummy_fixed_params(),
                progress_cb=None,
                fail_fast=False,
                circuit_breaker_threshold=3,
            )

        assert written == 0
        assert len(failures) == 3
        assert abort_meta is not None
        assert abort_meta["consecutive_failures"] == 3
        assert abort_meta["projected_episodes_saved"] == 17

    def test_threshold_one_trips_on_first_failure(self, tmp_path: Path) -> None:
        """A threshold of 1 aborts immediately after the first failure."""

        def failing_worker(job):
            raise RuntimeError("CUDA out of memory on device 0")

        out_file = tmp_path / "episodes.jsonl"
        jobs = [(_dummy_scenario(), i) for i in range(5)]
        schema = _dummy_schema()

        with patch(f"{_RUNNER_MOD}._run_job_worker", side_effect=failing_worker):
            written, failures, abort_meta = _run_batch_sequential(
                jobs,
                out_path=out_file,
                schema=schema,
                fixed_params=_dummy_fixed_params(),
                progress_cb=None,
                fail_fast=False,
                circuit_breaker_threshold=1,
            )

        assert written == 0
        assert len(failures) == 1
        assert abort_meta is not None
        assert abort_meta["consecutive_failures"] == 1
        assert abort_meta["first_fail_index"] == 1
        assert abort_meta["projected_episodes_saved"] == 4

    def test_zero_threshold_disables_circuit_breaker(self, tmp_path: Path) -> None:
        """A threshold of 0 disables the circuit breaker entirely."""

        def failing_worker(job):
            raise RuntimeError("always fails")

        out_file = tmp_path / "episodes.jsonl"
        num_jobs = 15
        jobs = [(_dummy_scenario(), i) for i in range(num_jobs)]
        schema = _dummy_schema()

        with patch(f"{_RUNNER_MOD}._run_job_worker", side_effect=failing_worker):
            written, failures, abort_meta = _run_batch_sequential(
                jobs,
                out_path=out_file,
                schema=schema,
                fixed_params=_dummy_fixed_params(),
                progress_cb=None,
                fail_fast=False,
                circuit_breaker_threshold=0,
            )

        assert written == 0
        assert len(failures) == num_jobs
        assert abort_meta is None

    def test_failures_then_success_no_abort(self, tmp_path: Path) -> None:
        """Failures followed by a success never triggers abort."""
        call_counter = [0]

        def failing_then_success_worker(job):
            call_counter[0] += 1
            if call_counter[0] >= 5:
                return _valid_record(call_counter[0])
            raise RuntimeError("CUDA out of memory on device 0")

        out_file = tmp_path / "episodes.jsonl"
        jobs = [(_dummy_scenario(), i) for i in range(10)]
        schema = _dummy_schema()

        with patch(f"{_RUNNER_MOD}._run_job_worker", side_effect=failing_then_success_worker):
            written, failures, abort_meta = _run_batch_sequential(
                jobs,
                out_path=out_file,
                schema=schema,
                fixed_params=_dummy_fixed_params(),
                progress_cb=None,
                fail_fast=False,
                circuit_breaker_threshold=DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
            )

        # 4 failures, then success resets counter, no abort
        assert abort_meta is None
        assert len(failures) == 4
        assert written == 6  # seeds 4-9 succeed (call_counter >= 5)

    def test_abort_metadata_includes_bookkeeping(self, tmp_path: Path) -> None:
        """Abort metadata contains all required bookkeeping fields."""

        def failing_worker(job):
            raise TypeError("Model checkpoint load failed")

        out_file = tmp_path / "episodes.jsonl"
        jobs = [(_dummy_scenario(), i) for i in range(15)]
        schema = _dummy_schema()

        with patch(f"{_RUNNER_MOD}._run_job_worker", side_effect=failing_worker):
            _written, _failures, abort_meta = _run_batch_sequential(
                jobs,
                out_path=out_file,
                schema=schema,
                fixed_params=_dummy_fixed_params(),
                progress_cb=None,
                fail_fast=False,
                circuit_breaker_threshold=5,
            )

        assert abort_meta is not None
        assert "status" in abort_meta
        assert "signature" in abort_meta
        assert "consecutive_failures" in abort_meta
        assert "first_fail_index" in abort_meta
        assert "episodes_completed_before_onset" in abort_meta
        assert "projected_episodes_saved" in abort_meta
        assert abort_meta["consecutive_failures"] == 5
        assert abort_meta["projected_episodes_saved"] == 10


def test_map_serial_circuit_breaker_aborts_identical_failures(tmp_path: Path) -> None:
    """Map-based serial arms use the same fail-closed breaker contract."""
    calls = 0

    def failing_map_job(_job: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("CUDA out of memory on device 0")

    def bridge(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            adapter_requested_seen=False,
            adapter_native_steps=0,
            adapter_adapted_steps=0,
            runtime_algorithm_contract={},
        )

    result = _serial_execute_map_jobs(
        jobs=[({"name": "map"}, seed) for seed in range(8)],
        fixed_params={},
        out_path=tmp_path / "episodes.jsonl",
        schema={},
        run_map_job=failing_map_job,
        write_validated_to_handle=lambda *_args: None,
        apply_worker_metadata_bridge=bridge,
        scenario_id=lambda scenario: str(scenario["name"]),
        feasibility_totals={},
        circuit_breaker_threshold=3,
    )

    _wrote, _records, failures, _seen, _native, _adapted, _contract, abort = result
    assert calls == 3
    assert len(failures) == 3
    assert abort is not None
    assert abort["status"] == "aborted_systematic_failure"
    assert abort["projected_episodes_saved"] == 5


def test_circuit_breaker_threshold_validation_is_fail_closed() -> None:
    """None selects the default, zero disables, and negative values are invalid."""
    assert normalize_circuit_breaker_threshold(None) == DEFAULT_CIRCUIT_BREAKER_THRESHOLD
    assert normalize_circuit_breaker_threshold(0) == 0
    with pytest.raises(ValueError, match="non-negative"):
        normalize_circuit_breaker_threshold(-1)
