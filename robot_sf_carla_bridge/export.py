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


class _PayloadEncoder(json.JSONEncoder):
    """JSON encoder that converts Path and numpy values to native types."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return o.as_posix()
        # numpy is a hard dep of robot_sf, but the bridge package itself is
        # importable without numpy; resolve the conversion lazily so the
        # export module stays usable in numpy-less environments.
        tolist = getattr(o, "tolist", None)
        if callable(tolist):
            return tolist()
        item = getattr(o, "item", None)
        if callable(item):
            return item()
        return super().default(o)


def write_export_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Validate and write a T0 export payload as stable UTF-8 JSON.

    Returns:
        The output path that was written.
    """

    validate_export_payload(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, cls=_PayloadEncoder)
    path.write_text(serialized + "\n", encoding="utf-8")
    return path
