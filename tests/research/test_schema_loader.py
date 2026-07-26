"""Contract tests for :mod:`robot_sf.research.schema_loader`.

These tests lock the public schema-loading and validation contracts exposed by
``robot_sf/research/schema_loader.py``: schema path resolution (canonical and
fallback search locations), JSON loading, ``jsonschema`` validation error
translation, and file validation.

All filesystem effects stay inside ``tmp_path``. The loader resolves schema
locations relative to its own ``__file__``, so the ``isolated_schema_dirs``
fixture redirects that root/path boundary into a temporary directory tree that
mirrors the real layout. The tests therefore never depend on which schemas are
shipped and never edit shipped schemas.

Message assertions intentionally rely only on the loader's own message prefixes
plus actionable tokens drawn from our own inputs, never on full third-party
``jsonschema`` wording.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import jsonschema
import pytest

from robot_sf.research import schema_loader
from robot_sf.research.exceptions import ValidationError
from robot_sf.research.schema_loader import (
    get_schema_path,
    load_schema,
    validate_data,
    validate_file,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def isolated_schema_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Redirect schema_loader's schema-root/path boundary into ``tmp_path``.

    ``schema_loader`` derives its search roots from its own ``__file__``:
    ``<parent>.parent / "benchmark" / "schemas"`` (canonical) and
    ``<parent>.parent.parent / "specs" / "270-imitation-report" / "contracts"``
    (fallback). Pointing ``__file__`` at ``tmp_path/robot_sf/research/schema_loader.py``
    reproduces that exact layout under ``tmp_path`` so every filesystem effect is
    isolated and tests remain independent of shipped schemas.
    """
    fake_module = tmp_path / "robot_sf" / "research" / "schema_loader.py"
    fake_module.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(schema_loader, "__file__", str(fake_module))

    canonical_dir = tmp_path / "robot_sf" / "benchmark" / "schemas"
    fallback_dir = tmp_path / "specs" / "270-imitation-report" / "contracts"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        root=tmp_path,
        canonical_dir=canonical_dir,
        fallback_dir=fallback_dir,
    )


def _write_json(path: Path, payload: object) -> None:
    """Serialize ``payload`` to ``path`` as JSON for the test scaffolding."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_get_schema_path_resolves_canonical_location(
    isolated_schema_dirs: SimpleNamespace,
) -> None:
    """A schema present in the canonical dir resolves there, never the fallback."""
    name = "report_metadata.schema.v1.json"
    schema_file = isolated_schema_dirs.canonical_dir / name
    _write_json(schema_file, {"type": "object"})

    resolved = get_schema_path(name)

    assert resolved == schema_file
    assert resolved.exists()
    assert resolved.is_relative_to(isolated_schema_dirs.canonical_dir)
    assert not resolved.is_relative_to(isolated_schema_dirs.fallback_dir)


def test_get_schema_path_falls_back_to_specs_dir(
    isolated_schema_dirs: SimpleNamespace,
) -> None:
    """A schema absent from canonical but present in specs uses the fallback path."""
    name = "hypothesis_result.schema.json"
    schema_file = isolated_schema_dirs.fallback_dir / name
    _write_json(schema_file, {"type": "object"})

    resolved = get_schema_path(name)

    assert resolved == schema_file
    assert resolved.exists()
    assert resolved.is_relative_to(isolated_schema_dirs.fallback_dir)


def test_get_schema_path_raises_for_missing_schema(
    isolated_schema_dirs: SimpleNamespace,
) -> None:
    """A schema present in neither search location raises a ValidationError."""
    missing_name = "definitely_not_a_shipped_schema.v999.json"

    with pytest.raises(ValidationError) as exc_info:
        get_schema_path(missing_name)

    assert missing_name in str(exc_info.value)


def test_load_schema_parses_valid_json(
    isolated_schema_dirs: SimpleNamespace,
) -> None:
    """A valid JSON schema file is parsed into the expected mapping."""
    name = "valid.schema.json"
    payload = {"type": "object", "required": ["experiment_name"]}
    _write_json(isolated_schema_dirs.canonical_dir / name, payload)

    loaded = load_schema(name)

    assert loaded == payload


def test_load_schema_raises_on_invalid_json(
    isolated_schema_dirs: SimpleNamespace,
) -> None:
    """Malformed JSON in a schema file is translated to a ValidationError."""
    name = "broken.schema.json"
    (isolated_schema_dirs.canonical_dir / name).write_text("{ not valid json", encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        load_schema(name)

    assert name in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_load_schema_raises_for_missing_file(
    isolated_schema_dirs: SimpleNamespace,
) -> None:
    """A missing schema filename surfaces as a ValidationError through load_schema."""
    with pytest.raises(ValidationError) as exc_info:
        load_schema("absent.schema.json")

    assert "absent.schema.json" in str(exc_info.value)


def test_validate_data_accepts_conformant_instance() -> None:
    """An instance matching the schema validates without raising."""
    schema = {"type": "object", "required": ["experiment_name"], "additionalProperties": True}

    validate_data({"experiment_name": "demo"}, schema)


def test_validate_data_translates_instance_validation_error() -> None:
    """An instance violating the schema raises an actionable ValidationError."""
    required_field = "experiment_name"
    schema = {"type": "object", "required": [required_field]}

    with pytest.raises(ValidationError) as exc_info:
        validate_data({}, schema)

    message = str(exc_info.value)
    assert message.startswith("Schema validation failed:")
    # The offending field is drawn from our own schema, not third-party wording.
    assert required_field in message
    assert isinstance(exc_info.value.__cause__, jsonschema.ValidationError)


def test_validate_data_translates_schema_error() -> None:
    """A malformed schema raises an actionable ValidationError, not a raw SchemaError."""
    bad_type = "notAValidType"

    with pytest.raises(ValidationError) as exc_info:
        validate_data({"anything": 1}, {"type": bad_type})

    message = str(exc_info.value)
    assert message.startswith("Invalid schema:")
    # The offending token is drawn from our own schema, not third-party wording.
    assert bad_type in message
    assert isinstance(exc_info.value.__cause__, jsonschema.SchemaError)


def test_validate_file_raises_for_missing_input(tmp_path: Path) -> None:
    """A non-existent input file raises a ValidationError naming the path."""
    missing = tmp_path / "absent.json"

    with pytest.raises(ValidationError) as exc_info:
        validate_file(missing, "ignored.schema.json")

    assert str(missing) in str(exc_info.value)


def test_validate_file_raises_on_invalid_input_json(tmp_path: Path) -> None:
    """Malformed JSON in the input file is translated to a ValidationError."""
    bad_input = tmp_path / "bad.json"
    bad_input.write_text("{ broken", encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        validate_file(bad_input, "any.schema.json")

    assert str(bad_input) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_validate_file_passes_for_conformant_file(
    tmp_path: Path, isolated_schema_dirs: SimpleNamespace
) -> None:
    """A valid JSON file matching the schema validates end-to-end without raising."""
    schema_name = "report_metadata.schema.v1.json"
    schema_payload = {
        "type": "object",
        "required": ["experiment_name"],
        "additionalProperties": True,
    }
    _write_json(isolated_schema_dirs.canonical_dir / schema_name, schema_payload)

    data_file = tmp_path / "metadata.json"
    _write_json(data_file, {"experiment_name": "demo"})

    validate_file(data_file, schema_name)


def test_validate_file_translates_instance_error_for_nonconformant_data(
    tmp_path: Path, isolated_schema_dirs: SimpleNamespace
) -> None:
    """A file whose data violates the schema surfaces an actionable ValidationError."""
    schema_name = "strict.schema.json"
    schema_payload = {"type": "object", "required": ["experiment_name"]}
    _write_json(isolated_schema_dirs.canonical_dir / schema_name, schema_payload)

    data_file = tmp_path / "metadata.json"
    _write_json(data_file, {"wrong_field": 1})

    with pytest.raises(ValidationError) as exc_info:
        validate_file(data_file, schema_name)

    message = str(exc_info.value)
    assert message.startswith("Schema validation failed:")
    assert "experiment_name" in message
