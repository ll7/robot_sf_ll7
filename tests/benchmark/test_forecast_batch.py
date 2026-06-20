"""Tests for the ForecastBatch.v1 artifact contract."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from robot_sf.benchmark.forecast_batch import (
    ActorForecast,
    CoordinateFrame,
    ForecastBatch,
    ForecastBatchProvenance,
    ObservationQuality,
    load_forecast_batch,
    save_forecast_batch,
    validate_forecast_batch,
)


def _provenance(**overrides: object) -> ForecastBatchProvenance:
    """Build valid baseline provenance for tests."""
    data: dict[str, object] = {
        "predictor_id": "cv-baseline-v1",
        "predictor_family": "constant_velocity",
        "observation_tier": "deployable_observation",
        "frame": CoordinateFrame(
            name="world",
            units="m",
            axes=("x", "y"),
            origin="map origin",
        ),
        "dt_s": 0.5,
        "horizons_s": [0.5, 1.0],
        "scenario_id": "crosswalk_001",
        "seed": 7,
        "timestamp": "2026-06-15T12:00:00Z",
        "fallback_status": "native",
        "degraded_status": "none",
        "actor_ids": ["ped_1", "ped_2"],
        "actor_mask": [True, True],
        "actor_mask_metadata": {
            "semantics": "true means forecast payload is available for actor_id",
            "missing_actor_reasons": {},
        },
        "feature_schema": {
            "name": "socnav_observation_v1",
            "features": ["position_m", "velocity_mps"],
        },
        "oracle_state": False,
    }
    data.update(overrides)
    return ForecastBatchProvenance(**data)


def _batch_dict() -> dict[str, object]:
    """Return a JSON-compatible valid ForecastBatch fixture."""
    return ForecastBatch(
        provenance=_provenance(),
        forecasts=[
            ActorForecast(actor_id="ped_1", deterministic=[[1.0, 0.0], [1.5, 0.0]]),
            ActorForecast(actor_id="ped_2", deterministic=[[0.0, 1.0], [0.0, 1.5]]),
        ],
        metadata={"artifact_role": "schema_smoke"},
    ).to_dict()


def _observation_quality_dict(**overrides: object) -> dict[str, object]:
    """Build valid observation-quality metadata for ForecastBatch tests."""
    data: dict[str, object] = {
        "visibility": ["simulator_declared_visibility"],
        "occlusion": ["none_unless_declared"],
        "latency_s": 0.0,
        "dropout_probability": 0.0,
        "range_limit_m": None,
        "angular_noise_std_rad": 0.0,
        "false_negative_rate": 0.0,
        "false_positive_rate": 0.0,
        "notes": "Diagnostic benchmark metadata only; not hardware calibration.",
    }
    data.update(overrides)
    return data


def test_forecast_batch_deterministic_round_trip(tmp_path) -> None:
    """Deterministic forecasts should serialize and reload with provenance intact."""
    batch = ForecastBatch.from_dict(_batch_dict())
    path = tmp_path / "forecast_batch.json"

    save_forecast_batch(batch, path)
    loaded = load_forecast_batch(path)

    assert loaded.schema_version == "ForecastBatch.v1"
    assert loaded.provenance.observation_tier == "deployable_observation"
    assert loaded.provenance.frame.units == "m"
    assert loaded.forecasts[0].deterministic is not None
    assert loaded.forecasts[0].deterministic.shape == (2, 2)


def test_forecast_batch_observation_quality_round_trip() -> None:
    """Forecast provenance can carry observation-quality assumptions."""
    batch = ForecastBatch.from_dict(_batch_dict())
    batch.provenance.observation_quality = ObservationQuality.from_dict(
        _observation_quality_dict(false_negative_rate=0.25)
    )

    loaded = ForecastBatch.from_dict(batch.to_dict())

    assert loaded.provenance.observation_quality is not None
    assert loaded.provenance.observation_quality.false_negative_rate == 0.25
    assert loaded.provenance.observation_quality.notes.startswith("Diagnostic")


@pytest.mark.parametrize("field", ["visibility", "occlusion"])
def test_forecast_batch_observation_quality_rejects_scalar_string_lists(field: str) -> None:
    """Observation-quality list fields should not accept scalar strings."""
    data = _batch_dict()
    provenance = copy.deepcopy(data["provenance"])
    assert isinstance(provenance, dict)
    provenance["observation_quality"] = _observation_quality_dict(
        **{field: "simulator_declared_visibility"}
    )
    data["provenance"] = provenance

    with pytest.raises(ValueError, match=field):
        validate_forecast_batch(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latency_s", math.nan),
        ("latency_s", math.inf),
        ("range_limit_m", math.inf),
        ("angular_noise_std_rad", -math.inf),
    ],
)
def test_forecast_batch_observation_quality_rejects_non_finite_numbers(
    field: str, value: float
) -> None:
    """Observation-quality numeric metadata should be finite on typed paths."""
    data = _batch_dict()
    provenance = copy.deepcopy(data["provenance"])
    assert isinstance(provenance, dict)
    provenance["observation_quality"] = _observation_quality_dict(**{field: value})
    data["provenance"] = provenance

    with pytest.raises(ValueError, match=field):
        validate_forecast_batch(data)


def test_forecast_batch_sampled_modes_round_trip() -> None:
    """Sampled trajectories and mode probabilities are optional but validated when present."""
    batch = ForecastBatch(
        provenance=_provenance(
            actor_mask=[True, False],
            actor_mask_metadata={
                "semantics": "true means forecast payload is available for actor_id",
                "missing_actor_reasons": {"ped_2": "not visible"},
            },
        ),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                samples=[
                    [[1.0, 0.0], [1.5, 0.0]],
                    [[1.0, 0.1], [1.4, 0.2]],
                ],
                mode_probabilities=[0.6, 0.4],
                uncertainty_metadata={"sample_source": "scenario_branches"},
            ),
        ],
    )

    loaded = ForecastBatch.from_dict(batch.to_dict())

    assert loaded.forecasts[0].samples is not None
    assert loaded.forecasts[0].samples.shape == (2, 2, 2)
    assert loaded.forecasts[0].mode_probabilities == [0.6, 0.4]


def test_forecast_batch_actor_classes_round_trip() -> None:
    """Actor classes should survive the ForecastBatch artifact boundary."""
    batch = ForecastBatch(
        provenance=_provenance(actor_classes={"ped_1": "pedestrian", "ped_2": "bicycle"}),
        forecasts=[
            ActorForecast(actor_id="ped_1", deterministic=[[1.0, 0.0], [1.5, 0.0]]),
            ActorForecast(actor_id="ped_2", deterministic=[[0.0, 1.0], [0.0, 1.5]]),
        ],
    )

    loaded = ForecastBatch.from_dict(batch.to_dict())

    assert loaded.provenance.actor_classes == {"ped_1": "pedestrian", "ped_2": "bicycle"}


def test_forecast_batch_preserves_positional_oracle_state_argument_order() -> None:
    """Adding actor_classes should not shift the positional oracle_state argument."""
    provenance = ForecastBatchProvenance(
        "cv-baseline-v1",
        "constant_velocity",
        "deployable_observation",
        CoordinateFrame(name="world", units="m", axes=("x", "y"), origin="map origin"),
        0.5,
        [0.5, 1.0],
        "crosswalk_001",
        7,
        "native",
        "none",
        ["ped_1", "ped_2"],
        [True, True],
        {
            "semantics": "true means forecast payload is available for actor_id",
            "missing_actor_reasons": {},
        },
        {"name": "socnav_observation_v1", "features": ["position_m", "velocity_mps"]},
        "2026-06-15T12:00:00Z",
        True,
    )

    assert provenance.oracle_state is True
    assert provenance.actor_classes == {}


def test_forecast_batch_actor_class_list_aligns_with_actor_ids() -> None:
    """Aligned actor-class lists should normalize to an actor-id keyed mapping."""
    provenance = _provenance(actor_classes=["pedestrian", "bicycle"])

    assert provenance.actor_classes == {"ped_1": "pedestrian", "ped_2": "bicycle"}


def test_forecast_batch_actor_classes_reject_unknown_actor_ids() -> None:
    """Actor-class labels must only reference declared actors."""
    with pytest.raises(ValueError, match="actor_classes"):
        _provenance(actor_classes={"bike_1": "bicycle"})


def test_forecast_batch_actor_classes_reject_misaligned_lists() -> None:
    """List-style actor classes must align exactly with actor_ids."""
    with pytest.raises(ValueError, match="actor_classes"):
        _provenance(actor_classes=["pedestrian"])


def test_forecast_batch_serializes_numpy_scalar_metadata(tmp_path) -> None:
    """JSON output should normalize numpy scalar metadata values."""
    batch = ForecastBatch(
        provenance=_provenance(feature_schema={"name": "socnav_v1", "input_dim": np.int64(4)}),
        forecasts=[
            ActorForecast(
                actor_id="ped_1",
                deterministic=[[1.0, 0.0], [1.5, 0.0]],
                occupancy_summary={"peak_probability": np.float32(0.25)},
            ),
            ActorForecast(actor_id="ped_2", deterministic=[[0.0, 1.0], [0.0, 1.5]]),
        ],
        metadata={"generator_version": np.int64(2)},
    )
    path = tmp_path / "forecast_batch.json"

    save_forecast_batch(batch, path)
    loaded = load_forecast_batch(path)

    assert loaded.metadata["generator_version"] == 2
    assert loaded.forecasts[0].occupancy_summary["peak_probability"] == 0.25


def test_forecast_batch_accepts_numpy_horizons_and_tuple_actor_masks() -> None:
    """Scientific-Python container types should normalize at the artifact boundary."""
    batch = ForecastBatch(
        provenance=_provenance(
            horizons_s=np.array([0.5, 1.0], dtype=np.float32),
            actor_ids=("ped_1", "ped_2"),
            actor_mask=(True, False),
            actor_mask_metadata={
                "semantics": "true means forecast payload is available for actor_id",
                "missing_actor_reasons": {"ped_2": "not visible"},
            },
        ),
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=[[1.0, 0.0], [1.5, 0.0]])],
    )

    assert batch.provenance.horizons_s == [0.5, 1.0]
    assert batch.provenance.actor_ids == ["ped_1", "ped_2"]


def test_forecast_batch_allows_missing_actor_with_explicit_mask_metadata() -> None:
    """Missing actor payloads are allowed only when the actor mask documents semantics."""
    provenance = _provenance(
        actor_mask=[True, False],
        actor_mask_metadata={
            "semantics": "true means forecast payload is available for actor_id",
            "missing_actor_reasons": {"ped_2": "not visible at forecast time"},
        },
    )

    batch = ForecastBatch(
        provenance=provenance,
        forecasts=[ActorForecast(actor_id="ped_1", deterministic=[[1.0, 0.0], [1.5, 0.0]])],
    )

    assert batch.provenance.actor_mask == [True, False]
    assert batch.provenance.actor_mask_metadata["missing_actor_reasons"]["ped_2"]


@pytest.mark.parametrize(
    ("actor_mask", "forecast_ids"),
    [
        ([True, True], ["ped_1"]),
        ([True, False], ["ped_1", "ped_2"]),
    ],
)
def test_forecast_batch_requires_payloads_to_match_actor_mask(
    actor_mask: list[bool],
    forecast_ids: list[str],
) -> None:
    """Forecast payloads must exactly match actors marked present by actor_mask."""
    batch = _batch_dict()
    provenance = copy.deepcopy(batch["provenance"])
    assert isinstance(provenance, dict)
    provenance["actor_mask"] = actor_mask
    provenance["actor_mask_metadata"] = {
        "semantics": "true means forecast payload is available for actor_id",
        "missing_actor_reasons": {"ped_2": "not visible"},
    }
    batch["provenance"] = provenance
    batch["forecasts"] = [
        {"actor_id": actor_id, "deterministic": [[1.0, 0.0], [1.5, 0.0]]}
        for actor_id in forecast_ids
    ]

    with pytest.raises(ValueError, match="actor_mask"):
        validate_forecast_batch(batch)


def test_forecast_batch_rejects_duplicate_forecast_actor_ids() -> None:
    """Duplicate forecast payloads for one actor are ambiguous and rejected."""
    data = _batch_dict()
    data["forecasts"] = [
        {"actor_id": "ped_1", "deterministic": [[1.0, 0.0], [1.5, 0.0]]},
        {"actor_id": "ped_1", "deterministic": [[1.0, 0.0], [1.5, 0.0]]},
    ]

    with pytest.raises(ValueError, match="duplicate actor_ids"):
        validate_forecast_batch(data)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("observation_tier", "", "observation_tier"),
        ("dt_s", 0.0, "dt_s"),
        ("dt_s", True, "bool"),
        ("horizons_s", [], "horizons_s"),
        ("horizons_s", [1.0, 0.5], "horizons_s"),
        ("horizons_s", [True, 1.0], "bool"),
        ("seed", "7", "seed"),
        ("frame", {"name": "", "units": "m", "axes": ["x", "y"]}, "frame.name"),
        ("actor_mask", [1, True], "actor_mask"),
        ("actor_mask_metadata", {}, "actor_mask_metadata"),
        ("observation_quality", {"visibility": ["sim"]}, "observation_quality"),
    ],
)
def test_forecast_batch_rejects_missing_required_provenance(
    field: str,
    value: object,
    match: str,
) -> None:
    """Required provenance fields fail closed."""
    data = _batch_dict()
    provenance = copy.deepcopy(data["provenance"])
    assert isinstance(provenance, dict)
    provenance[field] = value
    data["provenance"] = provenance

    with pytest.raises(ValueError, match=match):
        validate_forecast_batch(data)


def test_forecast_batch_rejects_undeclared_oracle_fields() -> None:
    """Oracle-looking fields must be explicitly marked as oracle_state."""
    data = _batch_dict()
    provenance = copy.deepcopy(data["provenance"])
    assert isinstance(provenance, dict)
    provenance["feature_schema"] = {"name": "deployable_oracle_features"}
    provenance["oracle_state"] = False
    data["provenance"] = provenance

    with pytest.raises(ValueError, match="oracle_state"):
        validate_forecast_batch(data)

    provenance["oracle_state"] = True
    loaded = validate_forecast_batch(data)
    assert loaded.provenance.oracle_state is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metadata", {"oracle_hint": "diagnostic"}),
        (
            "actor_mask_metadata",
            {
                "semantics": "true means forecast payload is available for actor_id",
                "missing_actor_reasons": {"ped_2": "oracle_occlusion_state"},
            },
        ),
        ("uncertainty_metadata", {"source": "oracle_rollout"}),
    ],
)
def test_forecast_batch_rejects_undeclared_oracle_metadata(
    field: str,
    value: dict[str, object],
) -> None:
    """Oracle-looking metadata is rejected outside explicit oracle-state artifacts."""
    data = _batch_dict()
    if field == "metadata":
        data["metadata"] = value
    elif field == "actor_mask_metadata":
        provenance = copy.deepcopy(data["provenance"])
        assert isinstance(provenance, dict)
        provenance["actor_mask_metadata"] = value
        data["provenance"] = provenance
    else:
        forecasts = copy.deepcopy(data["forecasts"])
        assert isinstance(forecasts, list)
        forecasts[0][field] = value
        data["forecasts"] = forecasts

    with pytest.raises(ValueError, match="oracle_state"):
        validate_forecast_batch(data)


def test_forecast_batch_rejects_horizon_shape_mismatch() -> None:
    """Forecast payload steps must align with horizons_s."""
    data = _batch_dict()
    forecasts = copy.deepcopy(data["forecasts"])
    assert isinstance(forecasts, list)
    forecasts[0]["deterministic"] = [[1.0, 0.0]]
    data["forecasts"] = forecasts

    with pytest.raises(ValueError, match="horizons_s"):
        validate_forecast_batch(data)
