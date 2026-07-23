"""Contract tests for surfacing the foresight-model-load fallback (issue #6190).

``PredictionPlannerAdapter`` silently degrades to constant-velocity prediction
when the predictive model cannot be loaded with ``allow_fallback=True``. These
tests pin the structured-metadata contract and the end-to-end degradation ->
exclusion chain so the harness can disposition a degraded run before the
determinism assertion fires.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robot_sf.benchmark.algorithm_metadata import (
    PREDICTIVE_FORESIGHT_MODEL_FALLBACK_STATUS,
    enrich_algorithm_metadata,
)
from robot_sf.benchmark.exact_repeat_campaign import _record_is_degraded
from robot_sf.planner.predictive_foresight import PredictiveForesightConfig
from robot_sf.planner.socnav import PredictionPlannerAdapter, SocNavPlannerConfig

_UNKNOWN_MODEL_ID = "nonexistent_predictive_model_for_fallback_test"


def _two_pedestrian_state() -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((2, 4), dtype=np.float32)
    state[:, :2] = [[1.0, 0.0], [0.0, 1.0]]
    mask = np.ones(2, dtype=np.float32)
    return state, mask


def test_foresight_adapter_records_degraded_metadata_on_model_load_failure():
    """A model-load failure with allow_fallback=True records all six structured fields.

    Issue #6190 structured-metadata contract: requested model id + checkpoint
    provenance, model load status, effective prediction mode, fallback_used,
    fallback reason, and evidence_eligible (derived by the metadata layer).
    """
    config = SocNavPlannerConfig(predictive_model_id=_UNKNOWN_MODEL_ID)
    adapter = PredictionPlannerAdapter(config, allow_fallback=True)
    state, mask = _two_pedestrian_state()

    future = adapter._predict_trajectories(state, mask)

    # The constant-velocity fallback actually executed.
    assert future.shape == (2, adapter.config.predictive_horizon_steps, 2)
    assert adapter.foresight_degraded() is True

    provenance = adapter.foresight_diagnostics()["foresight_prediction"]
    assert provenance["load_status"] == "failed"
    assert provenance["effective_prediction_mode"] == "constant_velocity"
    assert provenance["fallback_used"] is True
    assert provenance["fallback_reason"] == "predictive_model_load_failed"
    assert provenance["load_error"] is not None
    # Structured provenance carries the requested asset identifiers.
    assert provenance["requested_model_id"] == _UNKNOWN_MODEL_ID
    assert provenance["requested_checkpoint_path"] == _UNKNOWN_MODEL_ID


def test_enrich_algorithm_metadata_marks_foresight_fallback_degraded_and_ineligible():
    """The metadata layer derives the degraded status and evidence_eligible=False."""
    config = SocNavPlannerConfig(predictive_model_id=_UNKNOWN_MODEL_ID)
    adapter = PredictionPlannerAdapter(config, allow_fallback=True)
    state, mask = _two_pedestrian_state()
    adapter._predict_trajectories(state, mask)

    metadata = enrich_algorithm_metadata(
        algo="prediction_planner", metadata=dict(adapter.foresight_diagnostics())
    )

    assert metadata["status"] == PREDICTIVE_FORESIGHT_MODEL_FALLBACK_STATUS
    block = metadata["foresight_prediction"]
    assert block["evidence_eligible"] is False
    assert block["fallback_used"] is True


def test_foresight_fallback_record_is_dispositioned_as_degraded():
    """End-to-end: a foresight-fallback record is degraded so the classifier acts.

    metadata -> enriched status -> ``_record_is_degraded`` returns True, which is
    what ``_classify_repeat_failure`` uses to disposition the run before the
    determinism assertion is reached.
    """
    config = SocNavPlannerConfig(predictive_model_id=_UNKNOWN_MODEL_ID)
    adapter = PredictionPlannerAdapter(config, allow_fallback=True)
    state, mask = _two_pedestrian_state()
    adapter._predict_trajectories(state, mask)

    metadata = enrich_algorithm_metadata(
        algo="prediction_planner", metadata=dict(adapter.foresight_diagnostics())
    )
    record = {"algorithm_metadata": metadata, "outcome": {}}

    assert _record_is_degraded(record) is True


def test_foresight_adapter_fail_closed_when_fallback_disabled():
    """Scientific profile: allow_fallback=False fails closed on model-load failure.

    A scientific benchmark profile must never quietly contribute degraded numbers.
    """
    config = SocNavPlannerConfig(predictive_model_id=_UNKNOWN_MODEL_ID)
    adapter = PredictionPlannerAdapter(config, allow_fallback=False)
    state, mask = _two_pedestrian_state()
    with pytest.raises(Exception):
        adapter._predict_trajectories(state, mask)


def test_loaded_foresight_run_is_evidence_eligible_and_not_degraded():
    """A successfully loaded foresight run stays healthy and evidence-eligible."""
    metadata = enrich_algorithm_metadata(
        algo="prediction_planner",
        metadata={
            "foresight_prediction": {
                "requested_model_id": "real_model",
                "load_status": "loaded",
                "effective_prediction_mode": "predictive_foresight",
                "fallback_used": False,
                "fallback_reason": None,
                "load_error": None,
            }
        },
    )
    assert metadata["status"] == "ok"
    assert metadata["foresight_prediction"]["evidence_eligible"] is True
    assert _record_is_degraded({"algorithm_metadata": metadata, "outcome": {}}) is False


def test_non_predictive_planner_metadata_is_untouched():
    """A planner without foresight provenance is unaffected by the new logic."""
    metadata = enrich_algorithm_metadata(algo="goal", metadata={})
    assert metadata["status"] == "ok"
    assert "foresight_prediction" not in metadata


def test_predictive_foresight_encoder_surfaces_adapter_degradation():
    """``PredictiveForesightEncoder`` (allow_fallback=True) surfaces the fallback."""
    config = PredictiveForesightConfig(model_id=_UNKNOWN_MODEL_ID)
    # Import lazily so the encoder is constructed against the test model id.
    from robot_sf.planner.predictive_foresight import PredictiveForesightEncoder

    encoder = PredictiveForesightEncoder(config)
    state, mask = _two_pedestrian_state()
    encoder._adapter._predict_trajectories(state, mask)

    assert encoder.foresight_degraded() is True
    diag = encoder.foresight_diagnostics()["foresight_prediction"]
    assert diag["load_status"] == "failed"
    assert diag["fallback_used"] is True


def test_evidence_eligible_derivation_covers_constant_velocity_without_explicit_flag() -> None:
    """A failed load with the constant-velocity mode is ineligible even when the
    explicit ``fallback_used`` flag is absent (defensive derivation)."""
    metadata: dict[str, Any] = enrich_algorithm_metadata(
        algo="prediction_planner",
        metadata={
            "foresight_prediction": {
                "load_status": "failed",
                "effective_prediction_mode": "constant_velocity",
            }
        },
    )
    assert metadata["status"] == PREDICTIVE_FORESIGHT_MODEL_FALLBACK_STATUS
    assert metadata["foresight_prediction"]["evidence_eligible"] is False
