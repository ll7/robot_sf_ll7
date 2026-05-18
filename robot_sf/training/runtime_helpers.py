"""Reusable runtime helpers for training launchers.

These helpers keep script entrypoints thin while preserving deterministic runtime
behavior for Ray and durable JSONL progress logging.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np
import orjson

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

_JsonContainer = dict[str, object] | list[object]


def _json_compatible(value: object) -> object:
    """Convert a runtime logging payload to JSON-compatible values.

    Returns:
        JSON-compatible value with string-only dictionary keys.
    """
    root, root_is_container = _json_shell(value)
    if not root_is_container:
        return root

    if not isinstance(root, dict | list):
        raise TypeError("JSON root container must be an object or array")

    stack: list[tuple[object, _JsonContainer]] = [(_json_container_source(value), root)]
    while stack:
        source, target = stack.pop()
        if isinstance(source, dict):
            _extend_json_object(source, target, stack)
        elif isinstance(source, list | tuple):
            _extend_json_array(source, target, stack)
    return root


def _extend_json_object(
    source: dict[object, object],
    target: _JsonContainer,
    stack: list[tuple[object, _JsonContainer]],
) -> None:
    """Append converted object children to the traversal stack."""
    if not isinstance(target, dict):
        raise TypeError("JSON object target must be a dictionary")
    for key, child in source.items():
        if not isinstance(key, str):
            raise TypeError("JSON object keys must be strings")
        converted, is_container = _json_shell(child)
        target[key] = converted
        if is_container:
            stack.append((_json_container_source(child), _require_json_container(converted)))


def _extend_json_array(
    source: list[object] | tuple[object, ...],
    target: _JsonContainer,
    stack: list[tuple[object, _JsonContainer]],
) -> None:
    """Append converted array children to the traversal stack."""
    if not isinstance(target, list):
        raise TypeError("JSON array target must be a list")
    for child in source:
        converted, is_container = _json_shell(child)
        target.append(converted)
        if is_container:
            stack.append((_json_container_source(child), _require_json_container(converted)))


def _require_json_container(value: object) -> _JsonContainer:
    """Return a JSON container or raise an internal conversion error."""
    if isinstance(value, dict | list):
        return value
    raise TypeError("JSON container conversion produced a scalar")


def _json_container_source(
    value: object,
) -> dict[object, object] | list[object] | tuple[object, ...]:
    """Return the source container that corresponds to a converted shell."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict | list | tuple):
        return value
    raise TypeError("JSON container source must be an object or array")


def _json_shell(value: object) -> tuple[object, bool]:
    """Return a converted scalar or empty container shell."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, Path):
        return str(value), False
    if isinstance(value, dict):
        return {}, True
    if isinstance(value, list | tuple):
        return [], True
    return value, False


def append_jsonl_record(path: Path, payload: dict[str, object], *, sort_keys: bool = False) -> None:
    """Append one JSON object as a single line in a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    options = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
    if sort_keys:
        options |= orjson.OPT_SORT_KEYS
    encoded = orjson.dumps(_json_compatible(payload), option=options)
    with path.open("ab") as handle:
        handle.write(encoded)


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
