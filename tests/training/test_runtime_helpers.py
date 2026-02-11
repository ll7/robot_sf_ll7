"""Unit tests for reusable training runtime helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.training.runtime_helpers import append_jsonl_record, resolve_ray_runtime_env


def test_append_jsonl_record_appends_multiple_lines(tmp_path: Path) -> None:
    """JSONL helper should append records without truncating previous lines."""
    out_path = tmp_path / "result.json"
    append_jsonl_record(out_path, {"iteration": 1, "reward_mean": 1.0})
    append_jsonl_record(out_path, {"iteration": 2, "reward_mean": 2.0})

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["iteration"] == 1
    assert json.loads(lines[1])["iteration"] == 2


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
    assert ".git" in runtime_env["excludes"]
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
