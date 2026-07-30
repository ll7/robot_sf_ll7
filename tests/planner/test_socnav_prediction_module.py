"""Focused coverage for the extracted Prediction planner-family module."""

from typing import Any

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
