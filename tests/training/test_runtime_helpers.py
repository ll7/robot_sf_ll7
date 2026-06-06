"""Unit tests for reusable training runtime helpers."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from robot_sf.training.runtime_helpers import append_jsonl_record, resolve_ray_runtime_env

_JSONL_APPEND_RECORDS = int(os.environ.get("ROBOT_SF_JSONL_APPEND_RECORDS", "250"))
_JSONL_APPEND_SOFT_RECORDS_PER_SEC = float(
    os.environ.get("ROBOT_SF_JSONL_APPEND_SOFT_RECORDS_PER_SEC", "500.0")
)
_ENFORCE_PERF = os.environ.get("ROBOT_SF_PERF_ENFORCE", "0") == "1"


def test_append_jsonl_record_appends_multiple_lines(tmp_path: Path) -> None:
    """JSONL helper should append records without truncating previous lines."""
    out_path = tmp_path / "result.json"
    append_jsonl_record(out_path, {"iteration": 1, "reward_mean": 1.0})
    append_jsonl_record(out_path, {"iteration": 2, "reward_mean": 2.0})

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["iteration"] == 1
    assert json.loads(lines[1])["iteration"] == 2


def test_append_jsonl_record_preserves_input_order_by_default(tmp_path: Path) -> None:
    """JSONL helper should avoid key sorting unless explicitly requested."""
    out_path = tmp_path / "result.jsonl"

    append_jsonl_record(out_path, {"z": 1, "a": 2})

    assert out_path.read_text(encoding="utf-8") == '{"z":1,"a":2}\n'


def test_append_jsonl_record_can_sort_keys_for_legacy_determinism(tmp_path: Path) -> None:
    """Callers that need sorted keys should be able to request them explicitly."""
    out_path = tmp_path / "result.jsonl"

    append_jsonl_record(out_path, {"z": 1, "a": 2}, sort_keys=True)

    assert out_path.read_text(encoding="utf-8") == '{"a":2,"z":1}\n'


def test_append_jsonl_record_handles_numpy_and_path_payloads(tmp_path: Path) -> None:
    """Runtime logging should serialize common training payload value types."""
    out_path = tmp_path / "result.jsonl"

    append_jsonl_record(
        out_path,
        {
            "checkpoint": tmp_path / "checkpoint.pt",
            "iteration": np.int64(3),
            "reward": np.float64(1.25),
            "window": np.array([1, 2, 3], dtype=np.int64),
        },
    )

    record = json.loads(out_path.read_text(encoding="utf-8"))
    assert record == {
        "checkpoint": str(tmp_path / "checkpoint.pt"),
        "iteration": 3,
        "reward": 1.25,
        "window": [1, 2, 3],
    }


def test_append_jsonl_record_sanitizes_nested_payloads_iteratively(tmp_path: Path) -> None:
    """Nested runtime logging values should be explicit JSON-compatible structures."""
    out_path = tmp_path / "result.jsonl"

    append_jsonl_record(
        out_path,
        {
            "nested": [
                {"path": tmp_path / "model.pt"},
                {"values": (np.float64(1.5), np.array([2, 3], dtype=np.int64))},
            ],
        },
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == {
        "nested": [
            {"path": str(tmp_path / "model.pt")},
            {"values": [1.5, [2, 3]]},
        ],
    }


def test_append_jsonl_record_rejects_non_string_object_keys(tmp_path: Path) -> None:
    """JSONL records should fail fast before writing objects with non-string keys."""
    out_path = tmp_path / "result.jsonl"

    with pytest.raises(TypeError, match="JSON object keys must be strings"):
        append_jsonl_record(out_path, {1: "bad"})  # type: ignore[dict-item]

    assert not out_path.exists()


def test_append_jsonl_record_writes_non_finite_numbers_as_null(tmp_path: Path) -> None:
    """orjson-compatible JSONL should represent non-finite values as JSON null."""
    out_path = tmp_path / "result.jsonl"

    append_jsonl_record(
        out_path,
        {
            "nan": math.nan,
            "inf": math.inf,
            "np_nan": np.float64(np.nan),
        },
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == {
        "nan": None,
        "inf": None,
        "np_nan": None,
    }


def test_append_jsonl_record_synthetic_batch_throughput(tmp_path: Path) -> None:
    """Guard JSONL append throughput with a hardware-tolerant synthetic batch."""

    out_path = tmp_path / "append_perf.jsonl"
    payload = {
        "iteration": 0,
        "metrics": {
            "reward_mean": np.float64(1.25),
            "loss": np.float64(0.125),
            "window": np.arange(16, dtype=np.int64),
        },
        "checkpoint": tmp_path / "model.pt",
    }

    started = time.perf_counter()
    for iteration in range(_JSONL_APPEND_RECORDS):
        append_jsonl_record(out_path, {**payload, "iteration": iteration})
    elapsed_sec = time.perf_counter() - started

    lines = out_path.read_text(encoding="utf-8").splitlines()
    records_per_sec = _JSONL_APPEND_RECORDS / elapsed_sec if elapsed_sec > 0 else 0.0
    message = (
        "append_jsonl_record synthetic throughput below soft threshold: "
        f"{records_per_sec:.1f} records/sec < {_JSONL_APPEND_SOFT_RECORDS_PER_SEC:.1f}; "
        f"records={_JSONL_APPEND_RECORDS}, elapsed_sec={elapsed_sec:.4f}, "
        f"bytes={out_path.stat().st_size}"
    )

    assert len(lines) == _JSONL_APPEND_RECORDS
    assert json.loads(lines[-1])["iteration"] == _JSONL_APPEND_RECORDS - 1
    if records_per_sec < _JSONL_APPEND_SOFT_RECORDS_PER_SEC:
        if _ENFORCE_PERF:
            pytest.fail(message)
        pytest.skip(f"{message}; set ROBOT_SF_PERF_ENFORCE=1 to enforce on this machine")


def test_resolve_ray_runtime_env_sets_defaults() -> None:
    """Runtime env helper should fill deterministic defaults."""
    runtime_env = resolve_ray_runtime_env(
        runtime_env_base={},
        working_dir=Path.cwd(),
        py_executable="/tmp/python",
        expected_virtual_env=None,
    )
    assert runtime_env["working_dir"] == str(Path.cwd().resolve())
    assert runtime_env["py_executable"] == "/tmp/python"
    excludes = runtime_env["excludes"]
    assert isinstance(excludes, list)
    assert ".git" in excludes
    assert runtime_env["env_vars"] == {}


def test_resolve_ray_runtime_env_rejects_non_mapping_env_vars() -> None:
    """Non-mapping env_vars should fail fast before ray.init()."""
    with pytest.raises(RuntimeError, match="ray.runtime_env.env_vars"):
        resolve_ray_runtime_env(
            runtime_env_base={"env_vars": "oops"},
            working_dir=Path.cwd(),
            py_executable="/tmp/python",
            expected_virtual_env=None,
        )


def test_resolve_ray_runtime_env_rejects_custom_py_executable_outside_venv(tmp_path: Path) -> None:
    """Custom worker interpreter should remain inside the active virtualenv."""
    venv_path = tmp_path / ".venv"
    venv_path.mkdir()
    with pytest.raises(RuntimeError, match="Configured worker interpreter"):
        resolve_ray_runtime_env(
            runtime_env_base={"py_executable": "/usr/bin/python3"},
            working_dir=Path.cwd(),
            py_executable="/tmp/python",
            expected_virtual_env=str(venv_path),
        )
