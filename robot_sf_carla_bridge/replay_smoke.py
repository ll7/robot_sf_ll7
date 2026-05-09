"""T1 oracle replay smoke setup for one CARLA T0 export payload."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import jsonschema

from robot_sf_carla_bridge.availability import require_carla
from robot_sf_carla_bridge.export import load_export_manifest_payloads, validate_export_payload
from robot_sf_carla_bridge.schema_catalog import (
    list_carla_bridge_schema_catalog,
    load_schema_catalog_schema,
)

T1_ORACLE_REPLAY_SMOKE_SCHEMA_VERSION = "carla-t1-oracle-replay-smoke.v1"
_CATALOG_T0_EXPORT_PAYLOAD_NAME = "t0_export_payload"


def _catalog_schema_versions(catalog: dict[str, Any]) -> dict[str, str]:
    """Return schema versions keyed by catalog entry name."""

    versions: dict[str, str] = {}
    for entry in catalog.get("schemas", []):
        if isinstance(entry, dict):
            versions[str(entry.get("name"))] = str(entry.get("schema_version"))
    return versions


def validate_t1_replay_catalog_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a T0 payload against the current CARLA bridge schema catalog.

    Returns:
        JSON-safe validation metadata for the selected payload.

    Raises:
        ValueError: if the catalog does not expose the T0 export payload schema contract or the
            payload version does not match it.
        jsonschema.ValidationError: if the catalog or payload does not satisfy its JSON Schema.
    """

    catalog = list_carla_bridge_schema_catalog()
    jsonschema.validate(instance=catalog, schema=load_schema_catalog_schema())
    versions = _catalog_schema_versions(catalog)
    expected_payload_version = versions.get(_CATALOG_T0_EXPORT_PAYLOAD_NAME)
    if expected_payload_version is None:
        raise ValueError("CARLA schema catalog missing t0_export_payload contract")

    validate_export_payload(payload)
    payload_version = payload.get("schema_version")
    if payload_version != expected_payload_version:
        raise ValueError(
            "T0 export payload schema_version "
            f"{payload_version!r} does not match catalog version {expected_payload_version!r}"
        )

    return {
        "schema_version": catalog["schema_version"],
        "t0_export_payload_schema_version": expected_payload_version,
    }


def select_t0_export_payload(
    manifest_path: str | Path,
    *,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Select one validated T0 export payload from a manifest.

    Returns:
        Record with ``scenario_id``, resolved ``path``, ``payload_index``, and ``payload``.

    Raises:
        ValueError: if the manifest has no payloads or the requested scenario is absent.
    """

    records = load_export_manifest_payloads(manifest_path)
    if not records:
        raise ValueError("T0 export manifest contains no payloads to replay")

    selected_index = 0
    if scenario_id is not None:
        selected_index = next(
            (
                index
                for index, record in enumerate(records)
                if record.get("scenario_id") == scenario_id
            ),
            -1,
        )
        if selected_index < 0:
            available = ", ".join(str(record.get("scenario_id")) for record in records)
            raise ValueError(
                f"T0 export manifest does not contain scenario_id {scenario_id!r}; "
                f"available: {available}"
            )

    selected = dict(records[selected_index])
    payload = selected.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("selected T0 export payload must be a JSON object")
    selected["payload"] = cast("dict[str, Any]", payload)
    selected["payload_index"] = selected_index
    return selected


def build_t1_oracle_replay_smoke_setup(
    manifest_path: str | Path,
    *,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Build a setup-only T1 oracle replay smoke summary for one T0 export.

    The function intentionally stops before live CARLA replay or metric comparison. It proves that
    one schema-valid T0 payload can reach a CARLA-facing setup boundary when the optional CARLA
    Python API is available.

    Returns:
        JSON-safe setup summary.
    """

    manifest = Path(manifest_path)
    record = select_t0_export_payload(manifest, scenario_id=scenario_id)
    payload = cast("dict[str, Any]", record["payload"])
    catalog_metadata = validate_t1_replay_catalog_payload(payload)
    carla_module = require_carla()

    scenario = cast("dict[str, Any]", payload["scenario"])
    robot = cast("dict[str, Any]", payload["robot"])
    static_geometry = cast("dict[str, Any]", payload["static_geometry"])
    simulation = cast("dict[str, Any]", payload["simulation"])
    pedestrians = cast("list[dict[str, Any]]", payload["pedestrians"])

    return {
        "schema_version": T1_ORACLE_REPLAY_SMOKE_SCHEMA_VERSION,
        "status": "oracle-replay",
        "mode": "oracle-replay",
        "stage": "setup-only",
        "manifest": manifest.as_posix(),
        "selected_payload": {
            "scenario_id": str(record["scenario_id"]),
            "path": Path(record["path"]).as_posix(),
            "payload_index": int(record["payload_index"]),
        },
        "catalog": catalog_metadata,
        "scenario": {
            "id": scenario["id"],
            "map_id": scenario["map_id"],
            "source_config": scenario["source_config"],
            "certificate_status": scenario["certificate"]["status"],
        },
        "robot": {
            "start": robot["start"],
            "goal": robot["goal"],
            "footprint": robot["footprint"],
        },
        "pedestrian_count": len(pedestrians),
        "static_obstacle_count": len(static_geometry.get("obstacles", [])),
        "simulation": {
            "dt_s": simulation["dt_s"],
            "horizon_s": simulation["horizon_s"],
            "termination": simulation["termination"],
        },
        "carla": {
            "dependency": "carla",
            "module": getattr(carla_module, "__name__", "carla"),
            "available": True,
        },
        "boundary": {
            "full_metrics_parity": False,
            "multi_map_replay": False,
            "long_running_benchmark": False,
            "note": "setup-only smoke; no CARLA benchmark readiness or parity claim",
        },
    }
