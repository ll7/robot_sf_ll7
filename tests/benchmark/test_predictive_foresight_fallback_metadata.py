"""Contract tests for surfacing the foresight-model-load fallback (issue #6190).

``PredictionPlannerAdapter`` silently degrades to constant-velocity prediction
when the predictive model cannot be loaded with ``allow_fallback=True``. These
tests pin the structured-metadata contract and the end-to-end degradation ->
exclusion chain so the harness can disposition a degraded run before the
determinism assertion fires.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig
from robot_sf.benchmark import runner as runner_module
from robot_sf.benchmark.aggregate import compute_aggregates, compute_aggregates_with_ci
from robot_sf.benchmark.algorithm_metadata import (
    PREDICTIVE_FORESIGHT_MODEL_FALLBACK_STATUS,
    enrich_algorithm_metadata,
)
from robot_sf.benchmark.exact_repeat_campaign import _record_is_degraded
from robot_sf.models import get_registry_entry
from robot_sf.planner import socnav as socnav_module
from robot_sf.planner.predictive_foresight import PredictiveForesightConfig
from robot_sf.planner.socnav import PredictionPlannerAdapter, SocNavPlannerConfig

_UNKNOWN_MODEL_ID = "nonexistent_predictive_model_for_fallback_test"


class _ForesightDemandingPPOModel:
    """Minimal PPO model whose Dict contract forces predictive feature construction."""

    def __init__(self) -> None:
        self.observation_space = SimpleNamespace(
            spaces={
                "predictive_min_clearance": SimpleNamespace(
                    shape=(1,),
                    dtype=np.float32,
                ),
            },
        )

    def predict(self, _observation: Any, *, deterministic: bool) -> tuple[np.ndarray, None]:
        """Return a stable velocity action after the foresight feature is materialized."""
        assert deterministic is True
        return np.array([0.0, 0.0], dtype=np.float32), None


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


def test_foresight_failed_load_retains_digest_for_readable_checkpoint(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readable but invalid checkpoint retains its digest after load failure."""
    checkpoint = tmp_path / "invalid-predictive-checkpoint.pt"
    checkpoint.write_bytes(b"not a valid predictive checkpoint")

    def fail_deserialization(*_args: Any, **_kwargs: Any) -> None:
        """Simulate checkpoint deserialization/schema validation failure."""
        raise ValueError("invalid predictive checkpoint payload")

    monkeypatch.setattr("robot_sf.planner.socnav.load_predictive_checkpoint", fail_deserialization)
    adapter = PredictionPlannerAdapter(
        SocNavPlannerConfig(predictive_checkpoint_path=str(checkpoint)),
        allow_fallback=True,
    )
    state, mask = _two_pedestrian_state()

    adapter._predict_trajectories(state, mask)

    provenance = adapter.foresight_diagnostics()["foresight_prediction"]
    assert provenance["load_status"] == "failed"
    assert (
        provenance["observed_checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )


def test_foresight_failed_load_retains_registry_expected_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing registered model retains its expected digest without local bytes."""
    model_id = "predictive_proxy_selected_v2_full"
    expected_digest = get_registry_entry(model_id)["github_release"]["sha256"]

    def missing_model_path(_model_id: str) -> None:
        """Prevent model hydration so the fallback path is exercised locally."""
        raise FileNotFoundError("test model cache miss")

    monkeypatch.setattr(socnav_module, "resolve_model_path", missing_model_path)
    adapter = PredictionPlannerAdapter(
        SocNavPlannerConfig(predictive_model_id=model_id), allow_fallback=True
    )
    state, mask = _two_pedestrian_state()

    adapter._predict_trajectories(state, mask)

    provenance = adapter.foresight_diagnostics()["foresight_prediction"]
    assert provenance["load_status"] == "failed"
    assert provenance["requested_model_id"] == model_id
    assert provenance["requested_checkpoint_sha256"] == expected_digest
    assert provenance["observed_checkpoint_sha256"] is None


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="requires fork isolation")
def test_run_episode_relays_ppo_child_foresight_fallback_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real PPO child fallback reaches episode metadata and degraded classification."""

    def load_test_ppo_model(self: PPOPlanner) -> None:
        """Install a Dict-policy model so normal PPO stepping invokes foresight."""
        self._model = _ForesightDemandingPPOModel()
        self._status = "ok"
        self._fallback_reason = None

    planner = PPOPlanner(
        PPOPlannerConfig(
            model_path="unused-test-model.zip",
            obs_mode="dict",
            predictive_foresight_enabled=True,
            predictive_foresight_model_id=_UNKNOWN_MODEL_ID,
        ),
        defer_model_loading=True,
    )
    monkeypatch.setattr(PPOPlanner, "_load_model", load_test_ppo_model)
    monkeypatch.setattr(
        runner_module,
        "_load_baseline_planner",
        lambda _algo, _config_path, _seed: (planner, runner_module.Observation, {}),
    )

    record = runner_module.run_episode(
        {
            "id": "ppo-child-foresight-fallback",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
        },
        seed=17,
        horizon=1,
        dt=0.1,
        record_forces=False,
        algo="ppo",
    )

    metadata = record["algorithm_metadata"]
    provenance = metadata["foresight_prediction"]
    assert metadata["status"] == PREDICTIVE_FORESIGHT_MODEL_FALLBACK_STATUS
    assert provenance["load_status"] == "failed"
    assert provenance["effective_prediction_mode"] == "constant_velocity"
    assert provenance["fallback_used"] is True
    assert provenance["evidence_eligible"] is False
    assert _record_is_degraded(record) is True


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


def test_aggregate_analyzer_excludes_foresight_fallback_from_evidence() -> None:
    """A real fallback is classifier-detected then excluded by the aggregate analyzer."""
    adapter = PredictionPlannerAdapter(
        SocNavPlannerConfig(predictive_model_id=_UNKNOWN_MODEL_ID), allow_fallback=True
    )
    state, mask = _two_pedestrian_state()
    adapter._predict_trajectories(state, mask)
    fallback_metadata = enrich_algorithm_metadata(
        algo="prediction_planner", metadata=dict(adapter.foresight_diagnostics())
    )
    fallback_record = {
        "episode_id": "fallback",
        "scenario_id": "scenario",
        "algo": "prediction_planner",
        "metrics": {"success": 0.0},
        "algorithm_metadata": fallback_metadata,
    }
    assert _record_is_degraded(fallback_record) is True

    records = [
        {
            "episode_id": "eligible",
            "scenario_id": "scenario",
            "algo": "prediction_planner",
            "metrics": {"success": 1.0},
            "algorithm_metadata": {
                "foresight_prediction": {"evidence_eligible": True},
            },
        },
        fallback_record,
    ]

    summary = compute_aggregates(records, group_by="algo")

    assert summary["prediction_planner"]["success"]["mean"] == 1.0
    assert summary["_meta"]["evidence_eligibility"] == {
        "input_record_count": 2,
        "eligible_record_count": 1,
        "excluded_record_count": 1,
        "policy": (
            "Rows with algorithm_metadata.foresight_prediction.evidence_eligible=false "
            "are excluded from benchmark evidence aggregation."
        ),
    }


def test_ci_analyzer_preserves_foresight_exclusion_audit_counts() -> None:
    """Bootstrap aggregation reports the original and excluded record counts."""
    records = [
        {
            "episode_id": "eligible",
            "scenario_id": "scenario",
            "algo": "prediction_planner",
            "metrics": {"success": 1.0},
            "algorithm_metadata": {
                "foresight_prediction": {"evidence_eligible": True},
            },
        },
        {
            "episode_id": "fallback",
            "scenario_id": "scenario",
            "algo": "prediction_planner",
            "metrics": {"success": 0.0},
            "algorithm_metadata": {
                "foresight_prediction": {"evidence_eligible": False},
            },
        },
    ]

    summary = compute_aggregates_with_ci(
        records,
        group_by="algo",
        bootstrap_samples=10,
        bootstrap_seed=1,
    )

    assert summary["prediction_planner"]["success"]["mean"] == 1.0
    assert summary["_meta"]["evidence_eligibility"]["input_record_count"] == 2
    assert summary["_meta"]["evidence_eligibility"]["eligible_record_count"] == 1
    assert summary["_meta"]["evidence_eligibility"]["excluded_record_count"] == 1


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
