"""Focused runtime tests for benchmark schema helper entities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark import schema_loader
from robot_sf.benchmark.schema_reference import SchemaReference
from robot_sf.benchmark.schemas.episode_schema import EpisodeSchema

ROOT = Path(__file__).resolve().parents[2]
EPISODE_SCHEMA_PATH = ROOT / "robot_sf" / "benchmark" / "schemas" / "episode.schema.v1.json"


def _schema_dict(*, version: str = "v1", **overrides: object) -> dict[str, object]:
    schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://example.test/episode.schema.{version}.json",
        "title": f"RobotSF Benchmark Episode ({version})",
        "type": "object",
        "properties": {
            "episode_id": {"type": "string"},
            "scenario_id": {"type": "string"},
            "seed": {"type": "integer"},
            "metrics": {"type": "object"},
            "version": {"const": version},
        },
        "required": ["episode_id", "scenario_id", "seed", "metrics"],
    }
    schema.update(overrides)
    return schema


def _write_schema(path: Path, schema: dict[str, object] | None = None) -> Path:
    path.write_text(json.dumps(schema or _schema_dict()), encoding="utf-8")
    return path


def test_schema_loader_rejects_invalid_schema_name() -> None:
    """Schema names must follow the canonical dotted version filename."""
    with pytest.raises(ValueError, match="Invalid schema name format"):
        schema_loader._parse_schema_name("episode.schema.v01.json.bak")


def test_load_schema_rejects_non_object_when_integrity_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integrity validation rejects schema loaders that return non-object data."""

    class FakeSchemaReference:
        def __init__(self, schema_path: str, version: str) -> None:
            self.schema_path = schema_path
            self.version = version

        def load_schema(self) -> object:
            return type("LoadedSchema", (), {"schema_data": []})()

    monkeypatch.setattr(schema_loader, "SchemaReference", FakeSchemaReference)

    with pytest.raises(ValueError, match="not a valid JSON object"):
        schema_loader.load_schema("episode.schema.v1.json")


def test_load_schema_rejects_missing_json_schema_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """Integrity validation requires the JSON Schema dialect marker."""

    class FakeSchemaReference:
        def __init__(self, schema_path: str, version: str) -> None:
            self.schema_path = schema_path
            self.version = version

        def load_schema(self) -> object:
            return type("LoadedSchema", (), {"schema_data": {"title": "missing schema"}})()

    monkeypatch.setattr(schema_loader, "SchemaReference", FakeSchemaReference)

    with pytest.raises(ValueError, match=r"missing \$schema field"):
        schema_loader.load_schema("episode.schema.v1.json")


def test_validate_episode_data_loads_default_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader validates with the default schema reference when none is supplied."""

    calls: list[str] = []

    class FakeDefaultReference:
        def load_schema(self) -> None:
            calls.append("load")

        def validate_episode_data(self, episode_data: dict[str, object]) -> None:
            calls.append(f"validate:{episode_data['episode_id']}")

    monkeypatch.setattr(schema_loader, "DEFAULT_EPISODE_SCHEMA_REF", FakeDefaultReference())

    schema_loader.validate_episode_data({"episode_id": "episode-1"})

    assert calls == ["load", "validate:episode-1"]


def test_schema_reference_cache_and_loaded_properties() -> None:
    """Schema references cache schemas while preserving loaded-reference state."""
    SchemaReference.clear_cache()
    ref = SchemaReference("benchmark/schemas/episode.schema.v1.json", "v1")

    first = ref.load_schema()
    second = SchemaReference("benchmark/schemas/episode.schema.v1.json", "v1").load_schema()

    assert first is second
    assert ref.is_loaded is True
    assert ref.loaded_schema is first
    assert "status=loaded" in str(ref)
    assert repr(ref) == (
        "SchemaReference(schema_path='benchmark/schemas/episode.schema.v1.json', version='v1')"
    )
    assert ref == SchemaReference("benchmark/schemas/episode.schema.v1.json", "v1")
    assert ref != object()
    assert hash(ref) == hash(SchemaReference("benchmark/schemas/episode.schema.v1.json", "v1"))


def test_schema_reference_rejects_missing_schema_path_and_version() -> None:
    """Schema references fail fast on empty constructor arguments."""
    with pytest.raises(ValueError, match="schema_path cannot be empty"):
        SchemaReference("", "v1")

    with pytest.raises(ValueError, match="version cannot be empty"):
        SchemaReference("benchmark/schemas/episode.schema.v1.json", "")


def test_schema_reference_reports_missing_file() -> None:
    """Missing schema files report the resolved package-root path."""
    ref = SchemaReference("benchmark/schemas/missing.schema.v1.json", "v1")

    with pytest.raises(FileNotFoundError, match="resolved from package root"):
        ref.load_schema()


def test_schema_reference_reports_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema references reject files whose declared version differs from the expected one."""
    package_root = tmp_path / "robot_sf"
    schema_path = package_root / "benchmark" / "schemas" / "episode.schema.v2.json"
    schema_path.parent.mkdir(parents=True)
    _write_schema(schema_path, _schema_dict(properties={"version": {"const": "v2"}}))
    monkeypatch.setattr("robot_sf.__file__", str(package_root / "__init__.py"))

    ref = SchemaReference("benchmark/schemas/episode.schema.v2.json", "v1")

    with pytest.raises(ValueError, match="Schema version mismatch"):
        ref.load_schema()


def test_schema_reference_requires_loaded_schema_for_accessors() -> None:
    """Validation and property access require an explicitly loaded schema."""
    ref = SchemaReference("benchmark/schemas/episode.schema.v1.json", "v1")

    with pytest.raises(RuntimeError, match="Schema not loaded"):
        ref.validate_episode_data({})

    with pytest.raises(RuntimeError, match="Schema not loaded"):
        ref.get_schema_property("metrics")


def test_schema_reference_property_lookup_after_load() -> None:
    """Loaded schema references expose property definitions and reject missing ones."""
    ref = SchemaReference("benchmark/schemas/episode.schema.v1.json", "v1")
    ref.load_schema()

    assert ref.get_schema_property("metrics")["type"] == "object"
    with pytest.raises(KeyError, match="not defined"):
        ref.get_schema_property("missing")


def test_episode_schema_rejects_missing_file_and_invalid_json(tmp_path: Path) -> None:
    """EpisodeSchema reports missing and malformed schema files clearly."""
    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        EpisodeSchema(tmp_path / "missing.schema.v1.json")

    invalid_json = tmp_path / "bad.schema.v1.json"
    invalid_json.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        EpisodeSchema(invalid_json)


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ([], "Schema must be a JSON object"),
        (_schema_dict(**{"$schema": "https://json-schema.org/draft-07/schema"}), "draft 2020-12"),
        (_schema_dict(type="array"), "root type"),
        (_schema_dict(required=["episode_id"]), "must require properties"),
    ],
)
def test_episode_schema_rejects_invalid_structure(
    tmp_path: Path, schema: dict[str, object] | list[object], message: str
) -> None:
    """EpisodeSchema validates the minimum supported JSON Schema structure."""
    schema_path = tmp_path / "episode.schema.v1.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        EpisodeSchema(schema_path)


def test_episode_schema_extracts_version_from_enum_title_and_unknown(tmp_path: Path) -> None:
    """EpisodeSchema extracts versions from const, enum, title/id fallback, or unknown."""
    enum_path = _write_schema(
        tmp_path / "enum.schema.v1.json",
        _schema_dict(properties={"version": {"enum": ["v3"]}}),
    )
    assert EpisodeSchema(enum_path).version == "v3"

    title_path = _write_schema(
        tmp_path / "title.schema.v1.json",
        _schema_dict(
            title="Legacy Episode v4",
            **{"$id": "https://example.test/no-version.json", "properties": {}},
        ),
    )
    assert EpisodeSchema(title_path).version == "v4"

    unknown_path = _write_schema(
        tmp_path / "unknown.schema.v1.json",
        _schema_dict(
            title="Legacy Episode",
            **{"$id": "https://example.test/no-version.json", "properties": {}},
        ),
    )
    assert EpisodeSchema(unknown_path).version == "unknown"


def test_episode_schema_accessors_validation_and_identity(tmp_path: Path) -> None:
    """EpisodeSchema exposes accessors, validation failures, identity, and compatibility checks."""
    schema = EpisodeSchema(EPISODE_SCHEMA_PATH)
    assert schema.schema_id.endswith("episode.schema.v1.json")
    assert {"episode_id", "version", "scenario_id", "seed"} <= set(schema.required_properties)
    assert schema.get_property_schema("metrics")["type"] == "object"
    assert str(schema).startswith("EpisodeSchema(version=v1")
    assert repr(schema).startswith("EpisodeSchema(schema_path=")

    with pytest.raises(KeyError, match="not defined"):
        schema.get_property_schema("missing")

    with pytest.raises(ValueError, match="validation failed"):
        schema.validate_episode_data({"episode_id": "incomplete"})

    same_content = EpisodeSchema(EPISODE_SCHEMA_PATH)
    assert schema == same_content
    assert schema != object()
    assert hash(schema) == hash(same_content)

    future_path = _write_schema(tmp_path / "episode.schema.v2.json", _schema_dict(version="v2"))
    future_schema = EpisodeSchema(future_path)
    assert future_schema.is_backward_compatible_with(schema) is True

    invalid_version_path = _write_schema(
        tmp_path / "episode.schema.unknown.json",
        _schema_dict(version="unknown"),
    )
    invalid_version = EpisodeSchema(invalid_version_path)
    assert invalid_version.is_backward_compatible_with(schema) is False
