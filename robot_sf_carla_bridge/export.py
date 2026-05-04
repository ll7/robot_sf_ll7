"""T0 neutral replay export helpers for future CARLA oracle replay."""

from __future__ import annotations

import functools
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema

EXPORT_SCHEMA_VERSION = "carla-replay-export.v1"
_SCHEMA_RESOURCE = "schemas/carla_replay_export.v1.json"


@functools.lru_cache(maxsize=1)
def load_export_schema() -> dict[str, Any]:
    """Load the versioned T0 neutral export JSON schema.

    Returns:
        Parsed JSON schema dictionary.
    """

    schema_path = files("robot_sf_carla_bridge").joinpath(_SCHEMA_RESOURCE)
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_export_payload(payload: dict[str, Any]) -> None:
    """Validate one T0 neutral export payload.

    Raises:
        jsonschema.ValidationError: if ``payload`` does not satisfy the export schema.
    """

    jsonschema.validate(instance=payload, schema=load_export_schema())


def _coerce_payload(value: Any) -> Any:
    """Iteratively coerce non-JSON-native values (Path, numpy types) for serialization."""

    try:
        import numpy as np  # local import keeps numpy optional for the bridge package
    except ImportError:  # pragma: no cover - numpy is a hard dep elsewhere
        np = None  # type: ignore[assignment]

    def _coerce_scalar(item: Any) -> Any:
        if isinstance(item, Path):
            return item.as_posix()
        if np is not None:
            if isinstance(item, np.ndarray):
                return item.tolist()
            if isinstance(item, np.generic):
                return item.item()
        return item

    if isinstance(value, dict):
        root: dict[str, Any] = {}
        stack: list[tuple[Any, Any]] = [(root, value)]
        while stack:
            target, source = stack.pop()
            for key, child in source.items():
                str_key = key if isinstance(key, str) else str(key)
                if isinstance(child, dict):
                    nested: dict[str, Any] = {}
                    target[str_key] = nested
                    stack.append((nested, child))
                elif isinstance(child, (list, tuple)):
                    target[str_key] = [
                        _coerce_payload(item)
                        if isinstance(item, (dict, list, tuple))
                        else _coerce_scalar(item)
                        for item in child
                    ]
                else:
                    target[str_key] = _coerce_scalar(child)
        return root
    if isinstance(value, (list, tuple)):
        return [
            _coerce_payload(item) if isinstance(item, (dict, list, tuple)) else _coerce_scalar(item)
            for item in value
        ]
    return _coerce_scalar(value)


def write_export_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Validate and write a T0 export payload as stable UTF-8 JSON.

    Returns:
        The output path that was written.
    """

    validate_export_payload(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coerced = _coerce_payload(payload)
    path.write_text(json.dumps(coerced, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
