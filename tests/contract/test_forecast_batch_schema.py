"""Contract tests for the ForecastBatch.v1 JSON schema and typed validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.forecast_batch import (
    ActorForecast,
    CoordinateFrame,
    ForecastBatch,
    ForecastBatchProvenance,
    validate_forecast_batch,
)
from robot_sf.benchmark.schemas.forecast_batch_schema import ForecastBatchSchema

try:
    import jsonschema  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("jsonschema dependency required for contract tests") from e


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "forecast_batch.schema.v1.json"
)

CLI_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "validation" / "validate_forecast_batch.py"
)


def _load_schema() -> dict:
    """Load the canonical ForecastBatch schema from the repository."""
    text = SCHEMA_PATH.read_text()
    return json.loads(text)


def _valid_batch_dict() -> dict[str, object]:
    """Return a JSON-compatible valid ForecastBatch fixture."""
    return ForecastBatch(
        provenance=ForecastBatchProvenance(
            predictor_id="cv-baseline-v1",
            predictor_family="constant_velocity",
            observation_tier="deployable_observation",
            frame=CoordinateFrame(name="world", units="m", axes=("x", "y")),
            dt_s=0.5,
            horizons_s=[0.5, 1.0],
            scenario_id="crosswalk_001",
            seed=7,
            timestamp="2026-06-15T12:00:00Z",
            fallback_status="native",
            degraded_status="none",
            actor_ids=["ped_1", "ped_2"],
            actor_mask=[True, True],
            actor_mask_metadata={
                "semantics": "true means forecast payload is available for actor_id",
                "missing_actor_reasons": {},
            },
            feature_schema={"name": "socnav_observation_v1", "features": ["position_m"]},
        ),
        forecasts=[
            ActorForecast(actor_id="ped_1", deterministic=[[1.0, 0.0], [1.5, 0.0]]),
            ActorForecast(actor_id="ped_2", deterministic=[[0.0, 1.0], [0.0, 1.5]]),
        ],
    ).to_dict()


def test_forecast_batch_schema_entity_loads_and_reports_version() -> None:
    """The ForecastBatchSchema entity should load the canonical JSON schema."""
    schema = ForecastBatchSchema(SCHEMA_PATH)

    assert schema.version == "ForecastBatch.v1"
    assert schema.title == "RobotSF ForecastBatch (v1)"
    assert schema.schema_id.endswith("forecast_batch.schema.v1.json")
    assert schema.schema_data["type"] == "object"
    assert "schema_version" in schema.required_properties
    assert "provenance" in schema.required_properties
    assert "forecasts" in schema.required_properties
    assert schema.get_property_schema("schema_version")["const"] == "ForecastBatch.v1"
    assert "ForecastBatch.v1" in str(schema)
    assert schema == ForecastBatchSchema(SCHEMA_PATH)
    assert schema != object()
    assert hash(schema) == hash(ForecastBatchSchema(SCHEMA_PATH))


def test_forecast_batch_schema_entity_reports_missing_property() -> None:
    """Property lookup should fail closed for unknown schema keys."""
    schema = ForecastBatchSchema(SCHEMA_PATH)

    with pytest.raises(KeyError, match="not defined"):
        schema.get_property_schema("not_a_property")


def test_forecast_batch_schema_entity_rejects_missing_file(tmp_path: Path) -> None:
    """A missing schema file should raise a useful error."""
    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        ForecastBatchSchema(tmp_path / "missing.schema.json")


def test_forecast_batch_schema_entity_rejects_malformed_json(tmp_path: Path) -> None:
    """Malformed schema JSON should fail before validation."""
    schema_path = tmp_path / "bad.schema.json"
    schema_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        ForecastBatchSchema(schema_path)


@pytest.mark.parametrize(
    ("schema_update", "message"),
    [
        (lambda schema: schema.pop("$id"), "missing required field"),
        (
            lambda schema: schema.__setitem__("$schema", "https://json-schema.org/draft-07/schema"),
            "draft 2020-12",
        ),
        (lambda schema: schema.__setitem__("type", "array"), "root type"),
        (
            lambda schema: schema.__setitem__("required", ["schema_version"]),
            "must require properties",
        ),
    ],
)
def test_forecast_batch_schema_entity_rejects_invalid_schema_structure(
    tmp_path: Path,
    schema_update,
    message: str,
) -> None:
    """Schema loader should reject structures that cannot describe ForecastBatch.v1."""
    schema_data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_update(schema_data)
    schema_path = tmp_path / "invalid.schema.json"
    schema_path.write_text(json.dumps(schema_data), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ForecastBatchSchema(schema_path)


@pytest.mark.parametrize(
    ("version_property", "expected_version"),
    [
        ({"enum": ["ForecastBatch.v1"]}, "ForecastBatch.v1"),
        ({}, "v1"),
    ],
)
def test_forecast_batch_schema_entity_extracts_version_fallbacks(
    tmp_path: Path,
    version_property: dict[str, object],
    expected_version: str,
) -> None:
    """Version extraction should handle enum and title/id fallback schemas."""
    schema_data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_data["properties"]["schema_version"] = version_property
    schema_path = tmp_path / "fallback.schema.json"
    schema_path.write_text(json.dumps(schema_data), encoding="utf-8")

    assert ForecastBatchSchema(schema_path).version == expected_version


def test_forecast_batch_schema_entity_extracts_unknown_version(tmp_path: Path) -> None:
    """A schema without version clues should report unknown rather than guessing."""
    schema_data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_data["title"] = "ForecastBatch schema"
    schema_data["$id"] = "https://example.com/robot-sf/forecast_batch.schema.json"
    schema_data["properties"]["schema_version"] = {}
    schema_path = tmp_path / "unknown.schema.json"
    schema_path.write_text(json.dumps(schema_data), encoding="utf-8")

    assert ForecastBatchSchema(schema_path).version == "unknown"


def test_forecast_batch_schema_entity_backward_compatibility_is_conservative() -> None:
    """Matching ForecastBatch versions are compatible; unknown versions are not."""
    schema = ForecastBatchSchema(SCHEMA_PATH)

    assert schema.is_backward_compatible_with(ForecastBatchSchema(SCHEMA_PATH)) is True
    assert schema.is_backward_compatible_with(object()) is False


def test_forecast_batch_schema_valid_minimal_passes() -> None:
    """A minimal valid ForecastBatch should pass JSON Schema validation."""
    schema = _load_schema()
    batch = _valid_batch_dict()
    jsonschema.validate(instance=batch, schema=schema)


def test_forecast_batch_schema_rejects_missing_frame() -> None:
    """A missing top-level frame should fail JSON Schema validation."""
    schema = _load_schema()
    batch = _valid_batch_dict()
    provenance = dict(batch["provenance"])  # type: ignore[arg-type]
    provenance.pop("frame")
    batch["provenance"] = provenance

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=batch, schema=schema)


def test_forecast_batch_schema_rejects_wrong_horizon_shape() -> None:
    """A deterministic trajectory with the wrong horizon count should fail the contract."""
    batch = _valid_batch_dict()
    batch["forecasts"] = [{"actor_id": "ped_1", "deterministic": [[1.0, 0.0]]}]
    batch["provenance"]["actor_mask"] = [True, False]
    batch["provenance"]["actor_mask_metadata"]["missing_actor_reasons"] = {"ped_2": "not visible"}

    with pytest.raises(ValueError, match="horizons_s"):
        validate_forecast_batch(batch)


def test_forecast_batch_schema_rejects_missing_timestamp() -> None:
    """A missing provenance timestamp should fail the typed validator."""
    batch = _valid_batch_dict()
    provenance = dict(batch["provenance"])  # type: ignore[arg-type]
    provenance.pop("timestamp")
    batch["provenance"] = provenance

    with pytest.raises(ValueError, match="timestamp"):
        validate_forecast_batch(batch)


def test_forecast_batch_schema_rejects_invalid_timestamp() -> None:
    """A non-ISO timestamp should fail the typed validator."""
    batch = _valid_batch_dict()
    batch["provenance"]["timestamp"] = "2026-06-15"

    with pytest.raises(ValueError, match="timestamp"):
        validate_forecast_batch(batch)


def test_forecast_batch_schema_entity_rejects_non_datetime_timestamp() -> None:
    """The structural schema validator should enforce JSON Schema date-time format."""
    schema = ForecastBatchSchema(SCHEMA_PATH)
    batch = _valid_batch_dict()
    batch["provenance"]["timestamp"] = "not-a-date-time"

    with pytest.raises(ValueError, match="timestamp"):
        schema.validate_forecast_batch_data(batch)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-06-15T99:99:99Z",
        "2026-06-15T12:00:00",
    ],
)
def test_forecast_batch_schema_rejects_malformed_or_naive_timestamp(timestamp: str) -> None:
    """Forecast timestamps should be parseable and timezone-qualified."""
    batch = _valid_batch_dict()
    batch["provenance"]["timestamp"] = timestamp

    with pytest.raises(ValueError, match="timestamp"):
        validate_forecast_batch(batch)


def test_forecast_batch_schema_rejects_stale_provenance() -> None:
    """Empty required provenance strings should fail the typed validator."""
    batch = _valid_batch_dict()
    batch["provenance"]["predictor_id"] = ""

    with pytest.raises(ValueError, match="predictor_id"):
        validate_forecast_batch(batch)


def test_forecast_batch_schema_tracks_fallback_and_degraded_status() -> None:
    """Fallback and degraded status should survive validation and be discoverable."""
    batch = _valid_batch_dict()
    batch["provenance"]["fallback_status"] = "fallback"
    batch["provenance"]["degraded_status"] = "degraded_observation"

    loaded = validate_forecast_batch(batch)

    assert loaded.provenance.fallback_status == "fallback"
    assert loaded.provenance.degraded_status == "degraded_observation"


def test_forecast_batch_schema_probabilistic_example_passes() -> None:
    """A probabilistic forecast with samples, modes, Gaussian, and reachable sets passes."""
    batch = _valid_batch_dict()
    batch["provenance"]["actor_mask"] = [True, False]
    batch["provenance"]["actor_mask_metadata"]["missing_actor_reasons"] = {"ped_2": "occluded"}
    batch["forecasts"] = [
        {
            "actor_id": "ped_1",
            "samples": [
                [[1.0, 0.0], [1.5, 0.0]],
                [[1.0, 0.1], [1.4, 0.2]],
            ],
            "mode_probabilities": [0.6, 0.4],
            "gaussian": [
                {"mean": [1.0, 0.0], "cov": [[0.1, 0.0], [0.0, 0.1]]},
                {"mean": [1.5, 0.0], "cov": [[0.2, 0.0], [0.0, 0.2]]},
            ],
            "reachable_set": [
                {
                    "center": [1.0, 0.0],
                    "radius_m": 0.5,
                    "set_type": "conformal_tube",
                },
                {
                    "center": [1.5, 0.0],
                    "semi_axes_m": [0.6, 0.4],
                    "set_type": "confidence_ellipse",
                },
            ],
        }
    ]

    loaded = validate_forecast_batch(batch)

    assert loaded.forecasts[0].gaussian is not None
    assert len(loaded.forecasts[0].gaussian) == 2
    assert loaded.forecasts[0].reachable_set is not None
    assert loaded.forecasts[0].reachable_set[1]["set_type"] == "confidence_ellipse"


def test_forecast_batch_schema_gaussian_must_align_with_horizons() -> None:
    """Gaussian parameter lists must have one entry per horizon."""
    batch = _valid_batch_dict()
    batch["forecasts"] = [
        {
            "actor_id": "ped_1",
            "deterministic": [[1.0, 0.0], [1.5, 0.0]],
            "gaussian": [{"mean": [1.0, 0.0], "cov": [[0.1, 0.0], [0.0, 0.1]]}],
        }
    ]

    with pytest.raises(ValueError, match="gaussian"):
        validate_forecast_batch(batch)


def test_forecast_batch_schema_rejects_empty_actor_forecast_payload() -> None:
    """A forecast item with only actor_id is incomplete and should fail closed."""
    schema = _load_schema()
    batch = _valid_batch_dict()
    batch["forecasts"] = [{"actor_id": "ped_1"}, {"actor_id": "ped_2"}]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=batch, schema=schema)
    with pytest.raises(ValueError, match="prediction representation"):
        validate_forecast_batch(batch)


def test_forecast_batch_schema_rejects_mode_probabilities_without_samples() -> None:
    """Mode probabilities are only meaningful when aligned with sample trajectories."""
    batch = _valid_batch_dict()
    batch["forecasts"] = [
        {
            "actor_id": "ped_1",
            "deterministic": [[1.0, 0.0], [1.5, 0.0]],
            "mode_probabilities": [1.0],
        },
        {
            "actor_id": "ped_2",
            "deterministic": [[0.0, 1.0], [0.0, 1.5]],
        },
    ]

    with pytest.raises(ValueError, match="sampled trajectories"):
        validate_forecast_batch(batch)


@pytest.mark.parametrize(
    "forecast",
    [
        {"actor_id": "ped_1", "deterministic": []},
        {"actor_id": "ped_1", "samples": []},
        {"actor_id": "ped_1", "gaussian": []},
        {"actor_id": "ped_1", "reachable_set": []},
        {"actor_id": "ped_1", "occupancy_summary": {}},
        {
            "actor_id": "ped_1",
            "samples": [[[1.0, 0.0], [1.5, 0.0]]],
            "mode_probabilities": [1.2],
        },
    ],
)
def test_forecast_batch_schema_rejects_structurally_empty_payloads(
    forecast: dict[str, object],
) -> None:
    """A present forecast representation must contain useful payload content."""
    schema = _load_schema()
    batch = _valid_batch_dict()
    batch["forecasts"] = [forecast, {"actor_id": "ped_2", "occupancy_summary": {"kind": "grid"}}]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=batch, schema=schema)


def test_forecast_batch_schema_reachable_set_must_align_with_horizons() -> None:
    """Reachable or conformal sets must have one entry per horizon."""
    batch = _valid_batch_dict()
    batch["forecasts"] = [
        {
            "actor_id": "ped_1",
            "reachable_set": [
                {"center": [1.0, 0.0], "radius_m": 0.5, "set_type": "conformal_tube"}
            ],
        },
        {
            "actor_id": "ped_2",
            "deterministic": [[0.0, 1.0], [0.0, 1.5]],
        },
    ]

    with pytest.raises(ValueError, match="reachable_set"):
        validate_forecast_batch(batch)


def test_forecast_batch_schema_rejects_unknown_reachable_set_type() -> None:
    """The JSON schema should reject unsupported reachable-set types."""
    schema = _load_schema()
    batch = _valid_batch_dict()
    batch["forecasts"] = [
        {
            "actor_id": "ped_1",
            "deterministic": [[1.0, 0.0], [1.5, 0.0]],
            "reachable_set": [{"center": [1.0, 0.0], "set_type": "unknown_kind"}],
        }
    ]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=batch, schema=schema)


@pytest.mark.parametrize(
    "reachable_set",
    [
        [{"center": [1.0, 0.0], "set_type": "conformal_tube"}],
        [{"center": [1.0, 0.0], "set_type": "confidence_ellipse"}],
    ],
)
def test_forecast_batch_schema_requires_reachable_set_shape_fields(
    reachable_set: list[dict[str, object]],
) -> None:
    """Reachable-set variants should declare their required geometry parameters."""
    schema = _load_schema()
    batch = _valid_batch_dict()
    batch["forecasts"] = [
        {
            "actor_id": "ped_1",
            "deterministic": [[1.0, 0.0], [1.5, 0.0]],
            "reachable_set": reachable_set,
        }
    ]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=batch, schema=schema)


def test_forecast_batch_schema_entity_validates_batch_data() -> None:
    """ForecastBatchSchema.validate_forecast_batch_data should fail closed."""
    schema = ForecastBatchSchema(SCHEMA_PATH)
    batch = _valid_batch_dict()
    schema.validate_forecast_batch_data(batch)

    batch["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="schema_version"):
        schema.validate_forecast_batch_data(batch)


def test_forecast_batch_cli_validates_valid_batch(tmp_path: Path) -> None:
    """The CLI validator should accept a valid ForecastBatch artifact."""
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(_valid_batch_dict()), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--input", str(batch_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)

    assert summary["status"] == "valid"
    assert summary["schema_version"] == "ForecastBatch.v1"


def test_forecast_batch_cli_rejects_invalid_batch(tmp_path: Path) -> None:
    """The CLI validator should reject an artifact with missing timestamp."""
    batch_path = tmp_path / "batch.json"
    batch = _valid_batch_dict()
    batch["provenance"].pop("timestamp")
    batch_path.write_text(json.dumps(batch), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--input", str(batch_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "timestamp" in result.stdout
