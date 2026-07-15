"""Tests for execution-context capture and thread-env pinning (issue #5816)."""

from __future__ import annotations

import importlib
import os

import robot_sf.benchmark.utils as bench_utils


def test_capture_execution_context_shape() -> None:
    ctx = bench_utils.capture_execution_context()
    assert set(ctx.keys()) == {"hostname", "cpu_model", "thread_env"}
    assert isinstance(ctx["hostname"], str) and ctx["hostname"]
    assert isinstance(ctx["cpu_model"], str) and ctx["cpu_model"]
    assert set(ctx["thread_env"].keys()) == {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    }


def test_pin_thread_env_sets_defaults(monkeypatch) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    importlib.reload(bench_utils)
    pinned = bench_utils.pin_thread_env_for_determinism()
    assert pinned == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ[var] == "1"
    importlib.reload(bench_utils)


def test_pin_thread_env_preserves_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    importlib.reload(bench_utils)
    pinned = bench_utils.pin_thread_env_for_determinism()
    assert pinned["OMP_NUM_THREADS"] == "8"
    assert pinned["OPENBLAS_NUM_THREADS"] == "1"
    assert pinned["MKL_NUM_THREADS"] == "1"
    importlib.reload(bench_utils)


def test_pin_thread_env_is_idempotent(monkeypatch) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    importlib.reload(bench_utils)
    first = bench_utils.pin_thread_env_for_determinism()
    second = bench_utils.pin_thread_env_for_determinism()
    assert first == second
    importlib.reload(bench_utils)
