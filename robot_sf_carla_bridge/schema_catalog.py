"""Schema catalog metadata for import-safe CARLA bridge automation."""

from __future__ import annotations

from typing import Any

from robot_sf_carla_bridge.availability import AVAILABILITY_SCHEMA_VERSION
from robot_sf_carla_bridge.export import (
    BATCH_VALIDATION_SUMMARY_SCHEMA_VERSION,
    EXPORT_MANIFEST_SCHEMA_VERSION,
    EXPORT_SCHEMA_VERSION,
)

SCHEMA_CATALOG_VERSION = "carla-bridge-schema-catalog.v1"


def list_carla_bridge_schema_catalog() -> dict[str, Any]:
    """Return JSON-safe metadata for CARLA bridge schema contracts.

    Returns:
        Deterministic schema catalog metadata.
    """

    return {
        "schema_version": SCHEMA_CATALOG_VERSION,
        "schemas": [
            {
                "name": "availability",
                "loader": "load_availability_schema",
                "schema_version": AVAILABILITY_SCHEMA_VERSION,
            },
            {
                "name": "t0_export_payload",
                "loader": "load_export_schema",
                "schema_version": EXPORT_SCHEMA_VERSION,
            },
            {
                "name": "t0_export_manifest",
                "loader": "load_export_manifest_schema",
                "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
            },
            {
                "name": "t0_batch_validation_summary",
                "loader": "load_batch_validation_summary_schema",
                "schema_version": BATCH_VALIDATION_SUMMARY_SCHEMA_VERSION,
            },
        ],
    }
