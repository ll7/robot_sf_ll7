"""Manifest validation utilities for visual artifacts (T042).

Provides optional JSON Schema validation for the three visual manifests:
 - plot_artifacts.json
 - video_artifacts.json
 - performance_visuals.json

Design:
 - Imports jsonschema lazily; if missing, returns without error.
 - Collects concise validation errors instead of raising for invalid input.
 - Accepts base directory containing manifests and path to contracts dir.
 - Exposed single entry: validate_visual_manifests(base_dir: Path, contracts_dir: Path) -> list[str]
   Returns validation errors; an empty list means no validation errors were found.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST_FILES = {
    "plot_artifacts.json": "plot_artifacts.schema.json",
    "video_artifacts.json": "video_artifacts.schema.json",
    "performance_visuals.json": "performance_visuals.schema.json",
}


def validate_visual_manifests(base_dir: Path, contracts_dir: Path) -> list[str]:
    """Validate visual manifests against JSON Schemas if jsonschema available.

    Parameters
    ----------
    base_dir : Path
        Directory containing the generated manifests.
    contracts_dir : Path
        Directory containing schema files.

    Returns
    -------
    list[str]
        Descriptive validation errors. Empty when every present manifest is valid,
        no manifests are present, or jsonschema is not installed.
    """
    try:
        jsonschema = importlib.import_module("jsonschema")  # type: ignore
    except ImportError:
        logger.debug("jsonschema not installed; skipping visuals manifest validation")
        return []

    errors: list[str] = []
    for manifest_name, schema_name in MANIFEST_FILES.items():
        manifest_path = base_dir / manifest_name
        if not manifest_path.exists():
            continue
        schema_path = contracts_dir / schema_name
        if not schema_path.exists():  # defensive; should exist
            logger.warning("Schema file missing for %s", manifest_name)
            continue
        try:
            data = _load_json(manifest_path)
            schema = _load_json(schema_path)
            jsonschema.validate(data, schema)  # type: ignore[arg-type]
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            location = "/".join(str(part) for part in exc.path) or "<root>"
            errors.append(f"Validation failed for {manifest_name} at {location}: {exc.message}")
        except (jsonschema.SchemaError, OSError, ValueError, TypeError) as exc:
            errors.append(f"Error validating {manifest_name}: {exc.__class__.__name__}: {exc}")
    return errors


__all__ = ["validate_visual_manifests"]
