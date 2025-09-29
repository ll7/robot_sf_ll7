"""Manifest validation utilities for visual artifacts (T042).

Provides optional JSON Schema validation for the three visual manifests:
 - plot_artifacts.json
 - video_artifacts.json
 - performance_visuals.json

Design:
 - Imports jsonschema lazily; if missing, returns without error.
 - Raises ValueError with concise context when validation fails (caller may catch).
 - Accepts base directory containing manifests and path to contracts dir.
 - Exposed single entry: validate_visual_manifests(base_dir: Path, contracts_dir: Path) -> list[str]
   Returns list of validated manifest filenames (subset of the three if present).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST_FILES = {
    "plot_artifacts.json": "plot_artifacts.schema.json",
    "video_artifacts.json": "video_artifacts.schema.json",
    "performance_visuals.json": "performance_visuals.schema.json",
}


def _load_json(path: Path):  # type: ignore[no-untyped-def]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        Filenames successfully validated. Empty if jsonschema not installed.
    """
    try:
        import jsonschema  # type: ignore
    except ImportError:
        logger.debug("jsonschema not installed; skipping visuals manifest validation")
        return []

    validated: list[str] = []
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
            validated.append(manifest_name)
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            raise ValueError(
                f"Validation failed for {manifest_name}: {exc.message} at path {'/'.join(str(p) for p in exc.path)}",
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"Error validating {manifest_name}: {exc.__class__.__name__}: {exc}",
            ) from exc
    return validated


__all__ = ["validate_visual_manifests"]
