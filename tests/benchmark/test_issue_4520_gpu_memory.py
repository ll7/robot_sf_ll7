"""Tests for memory teardown behavior between consecutive serial arms (issue #4520)."""

from __future__ import annotations

import gc
from pathlib import Path  # noqa: TC003
from typing import Any
from unittest.mock import Mock

import pytest

from robot_sf.benchmark.map_runner_batch_runner import _serial_execute_map_jobs


def test_serial_execute_map_jobs_gpu_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure serial multi-arm execution runs gc.collect and torch.cuda.empty_cache."""
    import torch

    empty_cache_called = 0

    def mock_empty_cache() -> None:
        nonlocal empty_cache_called
        empty_cache_called += 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)

    gc_collect_called = 0
    original_gc_collect = gc.collect

    def mock_gc_collect(*args: Any, **kwargs: Any) -> int:
        nonlocal gc_collect_called
        gc_collect_called += 1
        return original_gc_collect(*args, **kwargs)

    monkeypatch.setattr(gc, "collect", mock_gc_collect)

    # Setup mock arguments
    jobs = [({"name": "scenario_1"}, 42), ({"name": "scenario_2"}, 43)]
    fixed_params: dict[str, Any] = {}
    out_path = tmp_path / "episodes.jsonl"
    schema: dict[str, Any] = {}

    run_map_job = Mock(return_value={"scenario_id": "dummy"})
    write_validated_to_handle = Mock()

    class DummyBridgeUpdate:
        adapter_requested_seen = False
        adapter_native_steps = 0
        adapter_adapted_steps = 0
        runtime_algorithm_contract = None

    apply_worker_metadata_bridge = Mock(return_value=DummyBridgeUpdate())
    scenario_id = Mock(return_value="scenario-id")
    feasibility_totals: dict[str, Any] = {}

    # Execute
    wrote, _, failures, _, _, _, _, _ = _serial_execute_map_jobs(
        jobs=jobs,
        fixed_params=fixed_params,
        out_path=out_path,
        schema=schema,
        run_map_job=run_map_job,
        write_validated_to_handle=write_validated_to_handle,
        apply_worker_metadata_bridge=apply_worker_metadata_bridge,
        scenario_id=scenario_id,
        feasibility_totals=feasibility_totals,
    )

    # Asserts
    assert wrote == 2
    assert len(failures) == 0
    assert empty_cache_called == 2
    assert gc_collect_called >= 2
