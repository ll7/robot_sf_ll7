"""Focused coverage for the extracted Prediction planner-family module."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from robot_sf.planner import socnav
from robot_sf.planner import socnav_prediction as prediction
from tests import test_socnav_planner_adapter as prediction_contracts

_LAZY_NAMES = ("PredictionPlannerAdapter", "SocNavBenchSamplingAdapter", "make_prediction_policy")

# The canonical readiness optional lane collects ``tests/planner`` but not the top-level
# adapter suite. Reuse its prediction-specific characterization cases here so moving the
# implementation into this optional module does not silently reduce changed-line proof.
_PATCHED_PREDICTION_CONTRACTS: tuple[Callable[[pytest.MonkeyPatch], None], ...] = (
    prediction_contracts.test_prediction_adapter_adaptive_lattice_expands_near_field,
    prediction_contracts.test_prediction_adapter_candidate_set_computes_min_pred_dist_once,
    prediction_contracts.test_prediction_adapter_fallback_when_model_missing,
    prediction_contracts.test_prediction_adapter_mcts_mode_is_deterministic,
    prediction_contracts.test_prediction_adapter_probabilistic_risk_mode_is_deterministic,
    prediction_contracts.test_prediction_adapter_progress_escape_injects_motion_in_clear_space,
    prediction_contracts.test_prediction_adapter_progress_escape_keeps_lower_cost_rollout,
    prediction_contracts.test_prediction_adapter_progress_escape_respects_clearance_gate,
    prediction_contracts.test_prediction_adapter_progress_risk_penalty_reduces_speed,
    prediction_contracts.test_prediction_adapter_requires_model_when_fallback_disabled,
    prediction_contracts.test_prediction_adapter_reverse_candidates_appear_in_near_field,
    prediction_contracts.test_prediction_adapter_sequence_search_is_deterministic,
    prediction_contracts.test_prediction_adapter_sequence_search_keeps_progress_escape,
    prediction_contracts.test_prediction_adapter_speed_clearance_gain_reduces_speed,
    prediction_contracts.test_prediction_adapter_ttc_penalty_reduces_speed,
    prediction_contracts.test_prediction_planner_caching_rollout_in_score_action,
)

_UNPATCHED_PREDICTION_CONTRACTS: tuple[Callable[[], None], ...] = (
    prediction_contracts.test_prediction_adapter_baseline_partial_miss_uses_constant_velocity_fallback,
    prediction_contracts.test_prediction_adapter_consumes_configured_forecast_variant,
    prediction_contracts.test_prediction_adapter_cvar_objective_penalizes_worse_tail,
    prediction_contracts.test_prediction_adapter_invalid_forecast_variant_fails_closed,
    prediction_contracts.test_prediction_adapter_reconfigures_forecast_variant_runtime_state,
    prediction_contracts.test_prediction_rollout_robot_boundary_steps_match_scalar_reference,
    prediction_contracts.test_prediction_rollout_robot_vectorized_parity,
)


@pytest.mark.parametrize("cached", [False, True], ids=["rollout", "cached-distances"])
def test_surface_clearance_costs_use_body_gaps_in_both_distance_paths(cached: bool) -> None:
    """Collision, near-miss, clearance, and TTC costs use surface gaps exactly once."""
    adapter = prediction.PredictionPlannerAdapter(
        prediction.SocNavPlannerConfig(
            predictive_clearance_model="surface_v2",
            predictive_robot_radius=1.0,
            predictive_pedestrian_radius=0.4,
            predictive_rollout_dt=0.2,
            predictive_safe_distance=0.3,
            predictive_near_distance=0.8,
            predictive_ttc_distance=0.5,
            predictive_speed_clearance_gain=0.0,
        )
    )
    # A stationary robot sees one pedestrian at 1.2 m then 1.6 m. Its body
    # gaps are -0.2 m and 0.2 m; a padded row at the origin must be ignored.
    future = np.asarray([[[1.2, 0.0], [1.6, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    kwargs = {
        "future_peds": future,
        "mask": np.asarray([1.0, 0.0]),
        "v": 0.0,
        "w": 0.0,
        "steps": 2,
        "valid_dists": np.asarray([[1.2, 1.6]]) if cached else None,
    }

    # Shortfalls: collision 0.5 + 0.1; near miss 1.0 + 0.6; TTC 0.7 and 0.3.
    assert adapter._collision_cost(**kwargs) == pytest.approx((0.6, 1.6))
    assert adapter._min_clearance(**kwargs) == pytest.approx(-0.2)
    assert adapter._ttc_penalty(**kwargs) == pytest.approx(0.7 / 0.200001 + 0.3 / 0.400001)


def test_facade_wildcard_import_includes_lazy_public_exports() -> None:
    """Lazy public symbols remain visible through facade introspection and wildcard import."""
    for name in _LAZY_NAMES:
        assert name in dir(socnav)
        assert name in socnav.__all__
    assert socnav.PredictionPlannerAdapter is prediction.PredictionPlannerAdapter
    assert socnav.SocNavBenchSamplingAdapter is prediction.SocNavBenchSamplingAdapter
    assert socnav.make_prediction_policy is prediction.make_prediction_policy


def test_lazy_resolution_caches_into_facade_globals(monkeypatch) -> None:
    """Resolving a lazy symbol imports the family module and caches it in the facade."""
    for name in _LAZY_NAMES:
        monkeypatch.delattr(socnav, name, raising=False)
    assert "PredictionPlannerAdapter" not in vars(socnav)

    resolved = socnav.PredictionPlannerAdapter
    assert resolved is prediction.PredictionPlannerAdapter
    # ``__getattr__`` caches the resolved value so subsequent lookups skip the import.
    assert vars(socnav)["PredictionPlannerAdapter"] is prediction.PredictionPlannerAdapter
    assert socnav.PredictionPlannerAdapter is resolved


def test_prediction_adapter_importable_and_instantiable() -> None:
    """The predictive adapter can be imported and instantiated from the extracted module."""
    adapter = prediction.PredictionPlannerAdapter(allow_fallback=True)
    assert isinstance(adapter, prediction.SamplingPlannerAdapter)
    assert adapter.config is not None
    assert adapter.get_forecast_variant_execution_mode() == "native"


def test_bench_sampling_adapter_importable_and_instantiable() -> None:
    """The upstream-delegating bench adapter remains constructible in fallback mode."""
    adapter = prediction.SocNavBenchSamplingAdapter(allow_fallback=True)
    assert isinstance(adapter, prediction.SamplingPlannerAdapter)


def test_invalid_forecast_variant_reports_blocked_when_fallback_is_allowed() -> None:
    """An unsupported forecast remains observable instead of silently appearing native."""
    config = prediction.SocNavPlannerConfig(forecast_variant="unsupported")
    adapter = prediction.PredictionPlannerAdapter(config, allow_fallback=True)

    assert adapter.get_forecast_variant_execution_mode() == "blocked"


def test_factory_produces_policy_with_correct_adapter_type() -> None:
    """Factory function wraps the correct adapter inside the policy."""
    policy = prediction.make_prediction_policy(allow_fallback=True)
    assert isinstance(policy, prediction.SocNavPlannerPolicy)
    assert isinstance(policy.adapter, prediction.PredictionPlannerAdapter)


def test_adapter_reads_model_dependencies_from_live_facade(tmp_path, monkeypatch) -> None:
    """Facade dependency patches remain effective after prediction-family extraction."""
    checkpoint = tmp_path / "predictive-checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    loaded: dict[str, Any] = {}

    class FakeModel:
        """Minimal predictive model returned by the patched checkpoint loader."""

        def to(self, device: str) -> None:
            loaded["device"] = device

        def eval(self) -> None:
            loaded["evaluated"] = True

    model = FakeModel()

    def fake_resolve_model_path(model_id: str):
        loaded["model_id"] = model_id
        return checkpoint

    def fake_load_predictive_checkpoint(path, **kwargs):
        loaded["path"] = path
        loaded["loader_kwargs"] = kwargs
        return model, {"feature_schema": None}

    monkeypatch.setattr(socnav, "resolve_model_path", fake_resolve_model_path)
    monkeypatch.setattr(socnav, "load_predictive_checkpoint", fake_load_predictive_checkpoint)
    monkeypatch.setattr(socnav, "torch", object())

    adapter = prediction.PredictionPlannerAdapter()

    assert adapter._build_model() is model
    assert loaded["model_id"] == adapter.config.predictive_model_id
    assert loaded["path"] == checkpoint
    assert loaded["device"] == "cpu"
    assert loaded["evaluated"] is True


@pytest.mark.parametrize(
    "contract",
    _PATCHED_PREDICTION_CONTRACTS,
    ids=lambda contract: contract.__name__,
)
def test_extracted_module_runs_patched_prediction_contract(
    contract: Callable[[pytest.MonkeyPatch], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing monkeypatch-based prediction behavior remains covered in the optional lane."""
    contract(monkeypatch)


@pytest.mark.parametrize(
    "contract",
    _UNPATCHED_PREDICTION_CONTRACTS,
    ids=lambda contract: contract.__name__,
)
def test_extracted_module_runs_unpatched_prediction_contract(
    contract: Callable[[], None],
) -> None:
    """Existing deterministic prediction behavior remains covered in the optional lane."""
    contract()
