"""Tests for forecast observation-tier adapters."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.forecast_observation_adapters import (
    ForecastObservationAdapter,
    OracleFullStateForecastAdapter,
    TrackedAgentsForecastAdapter,
    build_constant_velocity_forecast_batch,
)


def _feature_schema() -> dict[str, object]:
    return {
        "name": "forecast_observation_adapter_fixture_v1",
        "features": ["position_m", "velocity_mps"],
    }


def _trace() -> dict[str, object]:
    return {
        "scenario_id": "adapter_crosswalk",
        "seed": 11,
        "frames": [
            {
                "time_s": 0.0,
                "pedestrians": [
                    {"id": 1, "position": [0.0, 0.0], "velocity": [1.0, 0.0]},
                    {"id": 2, "position": [2.0, 0.0], "velocity": [0.0, 1.0]},
                    {"id": 3, "position": [4.0, 0.0], "velocity": [0.0, 0.5]},
                ],
                "tracked_agents": [
                    {"id": 1, "position": [0.1, 0.0], "velocity": [0.9, 0.0]},
                    {"id": 2, "position": [2.0, 0.2], "velocity": [0.0, 0.8]},
                    {
                        "id": 3,
                        "position": [4.0, 0.0],
                        "velocity": [0.0, 0.0],
                        "visible": False,
                        "missing_reason": "occluded by parked vehicle",
                    },
                ],
            },
            {
                "time_s": 0.1,
                "pedestrians": [
                    {"id": 1, "position": [0.1, 0.0], "velocity": [1.0, 0.0]},
                    {"id": 2, "position": [2.0, 0.1], "velocity": [0.0, 1.0]},
                    {"id": 3, "position": [4.0, 0.05], "velocity": [0.0, 0.5]},
                ],
                "tracked_agents": [
                    {"id": 1, "position": [0.2, 0.0], "velocity": [0.9, 0.0]},
                    {"id": 2, "position": [2.0, 0.3], "velocity": [0.0, 0.8]},
                ],
            },
        ],
    }


def test_same_trace_adapts_to_oracle_and_tracked_tiers() -> None:
    """The same frame should expose separate oracle and deployable tiers."""
    oracle = OracleFullStateForecastAdapter().adapt_trace(
        _trace(),
        feature_schema={"name": "oracle_state_fixture_v1"},
        horizons_s=[0.5, 1.0],
    )
    tracked = TrackedAgentsForecastAdapter().adapt_trace(
        _trace(),
        feature_schema=_feature_schema(),
        horizons_s=[0.5, 1.0],
        expected_actor_ids=["1", "2", "3"],
    )

    assert oracle.provenance.observation_tier == "oracle_full_state"
    assert oracle.provenance.oracle_state is True
    assert tracked.provenance.observation_tier == "tracked_agents"
    assert tracked.provenance.oracle_state is False
    assert tracked.provenance.actor_mask == [True, True, False]
    assert len(oracle.actors) == 3
    assert len(tracked.actors) == 2


def test_one_pedestrian_happy_path_records_single_actor_denominator() -> None:
    """A single visible actor should remain a valid one-actor forecast denominator."""
    trace = {
        "scenario_id": "single_actor",
        "seed": 3,
        "frames": [
            {
                "time_s": 0.0,
                "tracked_agents": [
                    {"id": 7, "position": [1.0, 2.0], "velocity": [0.5, 0.0]},
                ],
            },
        ],
    }

    observed = TrackedAgentsForecastAdapter().adapt_trace(
        trace,
        feature_schema=_feature_schema(),
    )

    assert observed.provenance.actor_ids == ["7"]
    assert observed.provenance.actor_mask == [True]
    assert observed.provenance.degraded_status == "none"
    assert observed.actors[0].state.id == 7


def test_constant_velocity_forecast_batch_smoke_for_two_tiers() -> None:
    """Adapted observations should produce valid ForecastBatch artifacts."""
    trace = _trace()
    oracle_batch = build_constant_velocity_forecast_batch(
        OracleFullStateForecastAdapter().adapt_trace(
            trace,
            feature_schema={"name": "oracle_state_fixture_v1"},
            horizons_s=[0.5, 1.0],
        ),
    )
    tracked_batch = build_constant_velocity_forecast_batch(
        TrackedAgentsForecastAdapter().adapt_trace(
            trace,
            feature_schema=_feature_schema(),
            horizons_s=[0.5, 1.0],
            expected_actor_ids=["1", "2", "3"],
        ),
    )

    assert oracle_batch.provenance.observation_tier == "oracle_full_state"
    assert tracked_batch.provenance.observation_tier == "tracked_agents"
    assert [forecast.actor_id for forecast in oracle_batch.forecasts] == ["1", "2", "3"]
    assert [forecast.actor_id for forecast in tracked_batch.forecasts] == ["1", "2"]


def test_empty_scene_fails_closed_before_forecast_artifact() -> None:
    """ForecastBatch provenance requires an explicit non-empty actor denominator."""
    trace = {"scenario_id": "empty", "frames": [{"pedestrians": [], "tracked_agents": []}]}

    with pytest.raises(ValueError, match="actor_ids must be non-empty"):
        OracleFullStateForecastAdapter().adapt_trace(trace, feature_schema=_feature_schema())


def test_missing_observation_tier_fails_closed() -> None:
    """A missing declared source key must not silently fall back to oracle state."""
    trace = {"scenario_id": "missing", "frames": [{"pedestrians": []}]}

    with pytest.raises(ValueError, match="missing tracked_agents"):
        TrackedAgentsForecastAdapter().adapt_trace(
            trace,
            feature_schema=_feature_schema(),
            expected_actor_ids=["1"],
        )


def test_missing_feature_schema_fails_closed() -> None:
    """Adapters must require feature metadata before provenance exists."""
    with pytest.raises(ValueError, match="feature_schema is required"):
        OracleFullStateForecastAdapter().adapt_trace(_trace(), feature_schema={})


def test_dense_masked_agents_record_degraded_observation_metadata() -> None:
    """Masked actors remain in the denominator with a reviewable reason."""
    observed = TrackedAgentsForecastAdapter().adapt_trace(
        _trace(),
        feature_schema=_feature_schema(),
        expected_actor_ids=["1", "2", "3", "4"],
    )

    assert observed.provenance.degraded_status == "degraded_observation"
    assert observed.provenance.actor_mask == [True, True, False, False]
    assert observed.provenance.actor_mask_metadata["missing_actor_reasons"] == {
        "3": "occluded by parked vehicle",
        "4": "not present in tracked-agent observation",
    }


def test_non_oracle_adapter_provenance_rejects_oracle_looking_feature_schema() -> None:
    """Deployable-tier provenance must not smuggle oracle metadata through schemas."""
    with pytest.raises(ValueError, match="oracle fields require explicit oracle_state=True"):
        TrackedAgentsForecastAdapter().adapt_trace(
            _trace(),
            feature_schema={"name": "oracle_state_fixture_v1"},
        )


def test_adapter_constructor_requires_declared_observation_tier() -> None:
    """Missing observation-tier declarations should fail before adaptation."""
    with pytest.raises(ValueError, match="observation_tier is required"):
        ForecastObservationAdapter(
            observation_tier="",
            source_key="tracked_agents",
            oracle_state=False,
            missing_reason="missing",
        )
