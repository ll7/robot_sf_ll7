"""Reusable runtime helpers for training launchers.

These helpers keep script entrypoints thin while preserving deterministic runtime
behavior for Ray and durable JSONL progress logging.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

DEFAULT_RAY_RUNTIME_ENV_EXCLUDES: tuple[str, ...] = (
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "output",
    "results",
    "dist",
    "robot_sf.egg-info",
    "*.mp4",
    "*.gif",
)


def append_jsonl_record(path: Path, payload: dict[str, object]) -> None:
    """Append one JSON object as a single line in a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")


def _resolve_executable_path(raw_command: str) -> Path:
    """Resolve the executable token from a potentially compound command string.

    Returns:
        Absolute executable path extracted from the command string.
    """
    parts = shlex.split(raw_command)
    executable = parts[0] if parts else raw_command
    return Path(executable).resolve()


def resolve_ray_runtime_env(
    *,
    runtime_env_base: dict[str, object],
    working_dir: Path,
    py_executable: str,
    expected_virtual_env: str | None,
) -> dict[str, object]:
    """Build a validated Ray runtime environment payload with stable defaults.

    Returns:
        Runtime env mapping ready to pass to ``ray.init(runtime_env=...)``.
    """
    runtime_env = dict(runtime_env_base)
    has_custom_py_executable = "py_executable" in runtime_env
    runtime_env.setdefault("working_dir", str(working_dir.resolve()))
    runtime_env.setdefault("excludes", list(DEFAULT_RAY_RUNTIME_ENV_EXCLUDES))
    runtime_env.setdefault("py_executable", py_executable)

    env_vars = runtime_env.get("env_vars")
    if env_vars is None:
        runtime_env["env_vars"] = {}
    elif not isinstance(env_vars, dict):
        raise RuntimeError("ray.runtime_env.env_vars must be a mapping of string key/value pairs.")

    py_executable_resolved = str(runtime_env.get("py_executable", "")).strip()
    if not py_executable_resolved:
        raise RuntimeError(
            "ray.runtime_env.py_executable resolved to an empty value; set a valid interpreter path."
        )

    if expected_virtual_env and has_custom_py_executable:
        expected_path = Path(expected_virtual_env).resolve()
        executable_path = _resolve_executable_path(py_executable_resolved)
        if not executable_path.is_relative_to(expected_path):
            raise RuntimeError(
                "Configured worker interpreter does not belong to the active VIRTUAL_ENV. "
                f"VIRTUAL_ENV={expected_path}, py_executable={executable_path}. "
                "Use the project venv interpreter to avoid worker/driver mismatch."
            )
    return runtime_env


__all__ = [
    "DEFAULT_RAY_RUNTIME_ENV_EXCLUDES",
    "append_jsonl_record",
    "resolve_ray_runtime_env",
]
