"""Unit tests for the real ``validate_visual_manifests`` implementation.

Covers ``robot_sf/benchmark/full_classic/validation.py`` across its branches:
 - valid manifests (all three present, and a partial subset)
 - empty directories (no manifests present)
 - missing schema files (manifest present, schema absent)
 - invalid manifests (JSON Schema validation failure -> ValueError)
 - malformed JSON (manifest load failure -> ValueError)
 - missing ``jsonschema`` module (ImportError -> empty list, no error)

Schemas and manifests are written inline into ``tmp_path`` so the tests are
self-contained and deterministic, independent of the spec contracts directory.
"""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.full_classic import validation as validation_mod
from robot_sf.benchmark.full_classic.validation import MANIFEST_FILES, validate_visual_manifests

if TYPE_CHECKING:
    from pathlib import Path

# Minimal self-contained JSON Schemas, one per manifest file. Each requires a
# distinct top-level key so validity / failure can be controlled precisely.
_SCHEMAS: dict[str, dict] = {
    "plot_artifacts.schema.json": {
        "type": "object",
        "required": ["plots"],
        "properties": {"plots": {"type": "array"}},
    },
    "video_artifacts.schema.json": {
        "type": "object",
        "required": ["videos"],
        "properties": {"videos": {"type": "array"}},
    },
    "performance_visuals.schema.json": {
        "type": "object",
        "required": ["metrics"],
        "properties": {"metrics": {"type": "object"}},
    },
}

# Valid manifest payloads, one per schema above.
_VALID_MANIFESTS: dict[str, dict] = {
    "plot_artifacts.json": {"plots": []},
    "video_artifacts.json": {"videos": []},
    "performance_visuals.json": {"metrics": {}},
}


@pytest.fixture
def contracts_dir(tmp_path: Path) -> Path:
    """Temp directory pre-populated with the three manifest schemas."""
    cdir = tmp_path / "contracts"
    cdir.mkdir()
    for schema_name, schema in _SCHEMAS.items():
        (cdir / schema_name).write_text(json.dumps(schema), encoding="utf-8")
    return cdir


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """Temp base directory that will hold manifest files."""
    bdir = tmp_path / "manifests"
    bdir.mkdir()
    return bdir


def _write_manifest(base_dir: Path, name: str, payload: dict) -> None:
    """Write a manifest payload as JSON into the base directory."""
    (base_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_valid_all_manifests_returns_validated(base_dir: Path, contracts_dir: Path) -> None:
    """All three valid manifests validate, returned in MANIFEST_FILES order."""
    for name, payload in _VALID_MANIFESTS.items():
        _write_manifest(base_dir, name, payload)
    result = validate_visual_manifests(base_dir, contracts_dir)
    assert result == list(MANIFEST_FILES.keys())


def test_valid_partial_manifests_return_only_present(base_dir: Path, contracts_dir: Path) -> None:
    """Only present manifests are validated; absent ones are skipped silently."""
    _write_manifest(base_dir, "plot_artifacts.json", {"plots": []})
    result = validate_visual_manifests(base_dir, contracts_dir)
    assert result == ["plot_artifacts.json"]


def test_empty_directory_returns_empty(base_dir: Path, contracts_dir: Path) -> None:
    """A base directory with no manifests yields an empty result and no error."""
    assert validate_visual_manifests(base_dir, contracts_dir) == []


def test_missing_schema_file_skips_manifest(base_dir: Path, contracts_dir: Path) -> None:
    """A manifest whose schema file is missing is skipped, not validated."""
    _write_manifest(base_dir, "plot_artifacts.json", {"plots": []})
    (contracts_dir / "plot_artifacts.schema.json").unlink()
    result = validate_visual_manifests(base_dir, contracts_dir)
    assert result == []


def test_invalid_manifest_raises_value_error(base_dir: Path, contracts_dir: Path) -> None:
    """A manifest failing schema validation raises ValueError with context."""
    # Missing the required "plots" key -> jsonschema.ValidationError.
    _write_manifest(base_dir, "plot_artifacts.json", {"not_plots": True})
    with pytest.raises(ValueError, match="Validation failed for plot_artifacts.json"):
        validate_visual_manifests(base_dir, contracts_dir)


def test_malformed_json_raises_value_error(base_dir: Path, contracts_dir: Path) -> None:
    """A manifest with malformed JSON raises ValueError with a load-error context."""
    (base_dir / "plot_artifacts.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Error validating plot_artifacts.json"):
        validate_visual_manifests(base_dir, contracts_dir)


def test_missing_jsonschema_module_returns_empty(
    base_dir: Path, contracts_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When jsonschema cannot be imported, validation is skipped, returning []."""
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "jsonschema":
            raise ImportError("simulated: No module named 'jsonschema'")
        return real_import_module(name, package)

    monkeypatch.setattr(validation_mod.importlib, "import_module", fake_import_module)
    # Even with a valid manifest present, nothing validates without jsonschema.
    _write_manifest(base_dir, "plot_artifacts.json", {"plots": []})
    assert validate_visual_manifests(base_dir, contracts_dir) == []
