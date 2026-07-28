"""Focused coverage for the extracted Prediction planner-family module."""

from robot_sf.planner import socnav
from robot_sf.planner import socnav_prediction as prediction

_LAZY_NAMES = ("PredictionPlannerAdapter", "SocNavBenchSamplingAdapter", "make_prediction_policy")


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


def test_factory_produces_policy_with_correct_adapter_type() -> None:
    """Factory function wraps the correct adapter inside the policy."""
    policy = prediction.make_prediction_policy(allow_fallback=True)
    assert isinstance(policy, prediction.SocNavPlannerPolicy)
    assert isinstance(policy.adapter, prediction.PredictionPlannerAdapter)
