"""Validate provenance required for a public GeoJSON map import."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

_SCHEMA_VERSION = "robot_sf.geojson_import_provenance.v1"


def validate_import_provenance(
    manifest_path: str | Path, source_path: str | Path
) -> dict[str, Any]:
    """Validate a public-source provenance manifest against its raw GeoJSON checksum.

    Returns:
        Parsed, validated provenance manifest.

    Raises:
        ValueError: If the manifest is incomplete, its checksum does not match, or it
            promotes the import to benchmark evidence without a separate review.
    """
    manifest = Path(manifest_path)
    source = Path(source_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"GeoJSON provenance manifest not found: {manifest}")
    if not source.is_file():
        raise FileNotFoundError(f"GeoJSON source file not found: {source}")
    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid GeoJSON provenance YAML {manifest}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("GeoJSON provenance manifest must contain a mapping")
    if data.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
    _require_mapping_fields(
        data.get("source"), "source", ("url", "accessed_on", "license", "citation")
    )
    raw_input = data.get("raw_input")
    _require_mapping_fields(raw_input, "raw_input", ("sha256",))
    if raw_input["sha256"] != _sha256(source):
        raise ValueError("raw_input.sha256 does not match the supplied GeoJSON")
    classification = data.get("classification")
    if classification != "exploratory_only":
        raise ValueError(
            "classification must be 'exploratory_only'; map import alone is not benchmark evidence"
        )
    return data


def _require_mapping_fields(value: Any, name: str, fields: tuple[str, ...]) -> None:
    """Require non-empty fields in one manifest mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    missing = [field for field in fields if not str(value.get(field, "")).strip()]
    if missing:
        raise ValueError(f"{name} missing required field(s): {', '.join(missing)}")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one local file.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
