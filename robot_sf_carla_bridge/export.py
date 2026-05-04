"""T0 neutral replay export helpers for future CARLA oracle replay."""

from __future__ import annotations

import functools
import json
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

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


def _json_safe(value: Any) -> Any:
    """Recursively convert common Python objects to JSON-safe values.

    Returns:
        JSON-safe value composed only of native JSON-compatible containers and scalars.
    """

    if isinstance(value, Path):
        return value.as_posix()

    # Resolve numpy-like objects lazily so this module stays importable even if
    # callers provide array/scalar wrappers without importing numpy here.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())

    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())

    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in value.items()}

    if isinstance(value, list | tuple):
        return [_json_safe(nested) for nested in value]

    return value


def write_export_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Validate and write a T0 export payload as stable UTF-8 JSON.

    Returns:
        The output path that was written.
    """

    normalized_payload = cast("dict[str, Any]", _json_safe(payload))
    validate_export_payload(normalized_payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(normalized_payload, indent=2, sort_keys=True)
    path.write_text(serialized + "\n", encoding="utf-8")
    return path
