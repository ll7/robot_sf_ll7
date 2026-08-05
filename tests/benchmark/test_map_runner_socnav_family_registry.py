"""Characterization of the data-driven SocNav-family registry (issue #6467).

The historical ``map_runner._build_socnav_family_adapter`` dispatched over a
hand-written 25-branch chain. Issue #6467 replaced that chain with a declarative
spec registry in ``robot_sf.benchmark.map_runner_policies.socnav_family``. These
tests pin the registry contract so future planners are added as one spec entry
without touching ``map_runner``.

The keys whose construction requires an external checkout, torch, ``skfmm``, or a
full candidate config are characterized through monkeypatched resolvers/config
builders instead of real construction, mirroring the existing
``test_map_runner_utils`` wiring tests.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from robot_sf.benchmark.map_runner import _build_socnav_family_adapter
from robot_sf.benchmark.map_runner_policies.socnav_family import (
    _ORCA_VARIANTS,
    SOCNAV_FAMILY_SPECS,
    build_socnav_family_adapter,
)
from robot_sf.planner.socnav_base import SocNavPlannerConfig

HISTORICAL_SOCNAV_KEYS = frozenset(
    {
        "socnav_sampling",
        "sampling",
        "social_force",
        "sf",
        "orca",
        "socnav_orca_nonholonomic",
        "socnav_orca_dd",
        "socnav_orca_relaxed",
        "socnav_hrvo",
        "hrvo",
        "social_navigation_pyenvs_orca",
        "social_nav_pyenvs_orca",
        "social_navigation_pyenvs_socialforce",
        "social_nav_pyenvs_socialforce",
        "social_navigation_pyenvs_sfm_helbing",
        "social_nav_pyenvs_sfm_helbing",
        "social_navigation_pyenvs_hsfm_new_guo",
        "social_nav_pyenvs_hsfm_new_guo",
        "crowdnav_height",
        "sonic_crowdnav",
        "sonic_gst",
        "gensafenav_ours_gst",
        "gensafe_ours_gst",
        "ours_gst",
        "gensafenav_gst_predictor_rand",
        "gensafe_gst_predictor_rand",
        "gst_predictor_rand",
        "sacadrl",
        "sa_cadrl",
        "prediction_planner",
        "hybrid_portfolio",
        "planner_selector_v2_diagnostic",
        "hybrid_orca_sampler",
        "socnav_bench",
        "nmpc_social",
        "nmpc",
        "dwa",
        "rvo",
    }
)

#: Keys whose adapter constructs without external checkpoints/torch/skfmm.
CONSTRUCTION_CAPABLE_KEYS = [
    ("socnav_sampling", "SamplingPlannerAdapter"),
    ("sampling", "SamplingPlannerAdapter"),
    ("social_force", "SocialForcePlannerAdapter"),
    ("sf", "SocialForcePlannerAdapter"),
    ("orca", "ORCAPlannerAdapter"),
    ("socnav_orca_nonholonomic", "ORCAPlannerAdapter"),
    ("socnav_orca_dd", "ORCAPlannerAdapter"),
    ("socnav_orca_relaxed", "ORCAPlannerAdapter"),
    ("socnav_hrvo", "HRVOPlannerAdapter"),
    ("hrvo", "HRVOPlannerAdapter"),
    ("sacadrl", "SACADRLPlannerAdapter"),
    ("sa_cadrl", "SACADRLPlannerAdapter"),
    ("prediction_planner", "PredictionPlannerAdapter"),
    ("hybrid_portfolio", "HybridPortfolioAdapter"),
    ("hybrid_orca_sampler", "HybridORCASamplerAdapter"),
    ("nmpc_social", "NMPCSocialPlannerAdapter"),
    ("nmpc", "NMPCSocialPlannerAdapter"),
    ("dwa", "DWAPlannerAdapter"),
    ("rvo", "SamplingPlannerAdapter"),
]

ORCA_VARIANT_OVERRIDES = [
    (
        "socnav_orca_nonholonomic",
        {
            "orca_heading_slowdown": 0.8,
            "orca_commit_distance": 1.8,
            "orca_commit_lateral_gain": 0.6,
        },
    ),
    (
        "socnav_orca_dd",
        {
            "orca_time_horizon": 3.0,
            "orca_neighbor_dist": 8.0,
            "orca_max_neighbors": 6,
            "orca_stall_speed_threshold": 0.1,
        },
    ),
    (
        "socnav_orca_relaxed",
        {
            "orca_time_horizon": 8.0,
            "orca_obstacle_range": 8.0,
            "orca_obstacle_threshold": 0.6,
            "orca_head_on_bias": 0.4,
            "orca_symmetry_bias": 0.15,
        },
    ),
]

SOCNAV_CONFIG_FIELDS = [field.name for field in fields(SocNavPlannerConfig)]


class _RecordingAdapter:
    """Minimal adapter test double that records its config."""

    def __init__(self, config: Any) -> None:
        self.config = config


class TestRegistryCoverage:
    """The registry must cover exactly the historical dispatch key surface."""

    def test_registry_key_set_matches_historical_dispatch(self) -> None:
        registered = {algo for spec in SOCNAV_FAMILY_SPECS for algo in spec.algorithms}
        assert registered == HISTORICAL_SOCNAV_KEYS

    def test_every_key_resolves_to_exactly_one_spec(self) -> None:
        seen: set[str] = set()
        for spec in SOCNAV_FAMILY_SPECS:
            assert spec.algorithms
            overlap = seen & set(spec.algorithms)
            assert not overlap, f"Duplicate registry keys: {sorted(overlap)}"
            seen.update(spec.algorithms)

    def test_orca_variants_declare_historical_override_values(self) -> None:
        assert _ORCA_VARIANTS == dict(ORCA_VARIANT_OVERRIDES)


class TestUnknownKeyError:
    """Unknown keys must fail exactly like the historical dispatch."""

    def test_registry_entry_point_raises_original_label(self) -> None:
        with pytest.raises(ValueError, match="Unknown map-based algorithm 'totally_bogus'"):
            build_socnav_family_adapter(
                algo_key="totally_bogus",
                algo="totally_bogus",
                algo_config={},
                meta={},
                socnav_cfg=SocNavPlannerConfig(),
            )

    def test_map_runner_wrapper_raises_original_label(self) -> None:
        with pytest.raises(ValueError, match="Unknown map-based algorithm 'totally_bogus'"):
            _build_socnav_family_adapter("totally_bogus", "totally_bogus", {}, meta={})


class TestAdapterConstruction:
    """Dep-free keys must construct the same adapter types as the historical chain."""

    @pytest.mark.parametrize(("algo_key", "expected_type"), CONSTRUCTION_CAPABLE_KEYS)
    def test_key_constructs_expected_adapter(
        self,
        algo_key: str,
        expected_type: str,
    ) -> None:
        adapter = _build_socnav_family_adapter(algo_key, algo_key, {}, meta={})
        assert type(adapter).__name__ == expected_type

    def test_registry_and_wrapper_agree(self) -> None:
        via_registry = build_socnav_family_adapter(
            algo_key="nmpc",
            algo="nmpc",
            algo_config={},
            meta={},
            socnav_cfg=SocNavPlannerConfig(),
        )
        via_wrapper = _build_socnav_family_adapter("nmpc", "nmpc", {}, meta={})
        assert type(via_registry).__name__ == type(via_wrapper).__name__


class TestOrcaVariants:
    """ORCA variants must apply per-variant overrides on a deepcopy of the shared config."""

    @pytest.mark.parametrize(("variant", "expected_overrides"), ORCA_VARIANT_OVERRIDES)
    def test_variant_applies_field_overrides(
        self,
        variant: str,
        expected_overrides: dict[str, Any],
    ) -> None:
        adapter = _build_socnav_family_adapter(variant, variant, {}, meta={})
        for name, value in expected_overrides.items():
            assert getattr(adapter.config, name) == value

    @pytest.mark.parametrize(("variant", "expected_overrides"), ORCA_VARIANT_OVERRIDES)
    def test_variant_preserves_unmodified_fields(
        self,
        variant: str,
        expected_overrides: dict[str, Any],
    ) -> None:
        adapter = _build_socnav_family_adapter(variant, variant, {}, meta={})
        base = SocNavPlannerConfig()
        for name in SOCNAV_CONFIG_FIELDS:
            if name not in expected_overrides:
                assert getattr(adapter.config, name) == getattr(base, name)

    def test_base_orca_keeps_shared_config_defaults(self) -> None:
        adapter = _build_socnav_family_adapter("orca", "orca", {}, meta={})
        base = SocNavPlannerConfig()
        for name in SOCNAV_CONFIG_FIELDS:
            assert getattr(adapter.config, name) == getattr(base, name)

    def test_base_orca_reuses_shared_config_object(self) -> None:
        shared = SocNavPlannerConfig()
        adapter = build_socnav_family_adapter(
            algo_key="orca",
            algo="orca",
            algo_config={},
            meta={},
            socnav_cfg=shared,
        )
        assert adapter.config is shared

    def test_variant_deepcopies_shared_config(self) -> None:
        shared = SocNavPlannerConfig()
        adapter = build_socnav_family_adapter(
            algo_key="socnav_orca_dd",
            algo="socnav_orca_dd",
            algo_config={},
            meta={},
            socnav_cfg=shared,
        )
        assert adapter.config is not shared
        assert adapter.config.orca_time_horizon == 3.0
        assert adapter.config.orca_neighbor_dist == 8.0


class TestMetadataMutations:
    """The few meta-mutating specs must keep their historical side effects."""

    def test_prediction_planner_stamps_mode_metadata(self) -> None:
        meta: dict[str, Any] = {"algorithm": "prediction_planner"}
        _build_socnav_family_adapter("prediction_planner", "prediction_planner", {}, meta=meta)
        assert meta["prediction_mode"] == "deterministic"
        assert meta["predictive_uncertainty_mode"] == "deterministic"
        assert meta["predictive_risk_objective"] == "mean"
        assert meta["predictive_risk_sample_count"] == 1
        assert meta["predictive_search_mode"] == "lattice"

    def test_rvo_marks_placeholder_status(self) -> None:
        meta: dict[str, Any] = {"algorithm": "rvo"}
        _build_socnav_family_adapter("rvo", "rvo", {}, meta=meta)
        assert meta["status"] == "placeholder"
        assert meta["fallback_reason"] == "unimplemented"


class TestResolverRouting:
    """External-checkpoint keys must route through the lazy map_runner resolvers."""

    def test_sonic_key_passes_plain_algo_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def _fake_resolver() -> tuple[Any, Any]:
            def _builder(payload: dict[str, Any]) -> str:
                captured["payload"] = payload
                return "dummy-config"

            return _RecordingAdapter, _builder

        monkeypatch.setattr("robot_sf.benchmark.map_runner._sonic_crowdnav_symbols", _fake_resolver)
        adapter = _build_socnav_family_adapter(
            "sonic_crowdnav", "sonic_crowdnav", {"device": "cpu"}, meta={}
        )
        assert adapter.config == "dummy-config"
        assert captured["payload"] == {"device": "cpu"}

    def test_gensafe_ours_key_injects_default_checkpoint_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def _fake_resolver() -> tuple[Any, Any]:
            def _builder(payload: dict[str, Any]) -> str:
                captured["payload"] = payload
                return "dummy-config"

            return _RecordingAdapter, _builder

        monkeypatch.setattr("robot_sf.benchmark.map_runner._sonic_crowdnav_symbols", _fake_resolver)
        adapter = _build_socnav_family_adapter(
            "gensafenav_ours_gst",
            "gensafenav_ours_gst",
            {"max_linear_speed": 0.8},
            meta={},
        )
        assert adapter.config == "dummy-config"
        payload = captured["payload"]
        assert payload["max_linear_speed"] == 0.8
        assert payload["repo_root"] == "output/repos/GenSafeNav"
        assert payload["model_name"] == "Ours_GST"
        assert payload["checkpoint_name"] == "05207.pt"

    def test_gensafe_predictor_rand_key_injects_default_checkpoint_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def _fake_resolver() -> tuple[Any, Any]:
            def _builder(payload: dict[str, Any]) -> str:
                captured["payload"] = payload
                return "dummy-config"

            return _RecordingAdapter, _builder

        monkeypatch.setattr("robot_sf.benchmark.map_runner._sonic_crowdnav_symbols", _fake_resolver)
        _build_socnav_family_adapter(
            "gst_predictor_rand", "gst_predictor_rand", {"model_name": "kept"}, meta={}
        )
        payload = captured["payload"]
        assert payload["model_name"] == "kept"
        assert payload["repo_root"] == "output/repos/GenSafeNav"
        assert payload["checkpoint_name"] == "05207.pt"

    def test_crowdnav_height_key_uses_crowdnav_resolver(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def _fake_resolver() -> tuple[Any, Any]:
            def _builder(payload: dict[str, Any]) -> str:
                captured["payload"] = payload
                return "dummy-config"

            return _RecordingAdapter, _builder

        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner._crowdnav_height_symbols", _fake_resolver
        )
        adapter = _build_socnav_family_adapter(
            "crowdnav_height", "crowdnav_height", {"device": "cpu"}, meta={}
        )
        assert adapter.config == "dummy-config"
        assert captured["payload"] == {"device": "cpu"}


class TestConfigBuilderRouting:
    """Keys with dedicated config builders must route the right builder and kwargs."""

    def test_pyenvs_orca_routes_config_builder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def _fake_builder(cfg: dict[str, Any]) -> str:
            captured["cfg"] = cfg
            return "dummy-config"

        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner.SocialNavigationPyEnvsORCAAdapter", _RecordingAdapter
        )
        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner.build_social_navigation_pyenvs_orca_config",
            _fake_builder,
        )
        adapter = _build_socnav_family_adapter(
            "social_navigation_pyenvs_orca",
            "social_navigation_pyenvs_orca",
            {"max_speed": 1.0},
            meta={},
        )
        assert adapter.config == "dummy-config"
        assert captured["cfg"] == {"max_speed": 1.0}

    @pytest.mark.parametrize(
        ("algo_key", "expected_policy_name"),
        [
            ("social_navigation_pyenvs_socialforce", "socialforce"),
            ("social_navigation_pyenvs_sfm_helbing", "sfm_helbing"),
        ],
    )
    def test_pyenvs_force_model_routes_default_policy_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        algo_key: str,
        expected_policy_name: str,
    ) -> None:
        captured: dict[str, Any] = {}

        def _fake_builder(cfg: dict[str, Any], *, default_policy_name: str) -> str:
            captured["cfg"] = cfg
            captured["default_policy_name"] = default_policy_name
            return "dummy-config"

        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner.SocialNavigationPyEnvsForceModelAdapter",
            _RecordingAdapter,
        )
        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner.build_social_navigation_pyenvs_force_model_config",
            _fake_builder,
        )
        adapter = _build_socnav_family_adapter(algo_key, algo_key, {"max_speed": 1.0}, meta={})
        assert adapter.config == "dummy-config"
        assert captured["cfg"] == {"max_speed": 1.0}
        assert captured["default_policy_name"] == expected_policy_name

    def test_pyenvs_hsfm_routes_default_policy_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def _fake_builder(cfg: dict[str, Any], *, default_policy_name: str) -> str:
            captured["default_policy_name"] = default_policy_name
            return "dummy-config"

        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner.SocialNavigationPyEnvsHSFMAdapter", _RecordingAdapter
        )
        monkeypatch.setattr(
            "robot_sf.benchmark.map_runner.build_social_navigation_pyenvs_hsfm_config",
            _fake_builder,
        )
        _build_socnav_family_adapter(
            "social_navigation_pyenvs_hsfm_new_guo",
            "social_navigation_pyenvs_hsfm_new_guo",
            {},
            meta={},
        )
        assert captured["default_policy_name"] == "hsfm_new_guo"
