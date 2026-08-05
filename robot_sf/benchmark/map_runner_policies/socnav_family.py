"""Data-driven registry for SocNav-family planner adapters (issue #6467).

Replaces the hand-written 25-branch dispatch chain that previously lived in
``robot_sf.benchmark.map_runner._build_socnav_family_adapter`` with a declarative
spec table. Each registered algorithm key maps to a :class:`SocnavFamilySpec` that
declares the adapter class, optional config builder, config overrides, fallback
behavior, and metadata mutations. Adding a new SocNav-family planner is a one-line
registry entry instead of another branch in ``map_runner``.

Adapter classes, config builders, and lazy symbol resolvers are resolved through
``robot_sf.benchmark.map_runner`` module attributes at build time so tests that
monkeypatch those names keep the exact construction behavior of the historical
inline dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from robot_sf.benchmark.map_runner_policy_resolution import _prediction_planner_metadata_overrides

AdapterBuilder: TypeAlias = Callable[[dict[str, Any], Any, dict[str, Any]], Any]  # noqa: UP040
MetadataMutator: TypeAlias = Callable[[dict[str, Any], dict[str, Any]], None]  # noqa: UP040


@dataclass(frozen=True)
class SocnavFamilySpec:
    """Declarative recipe for building one SocNav-family planner adapter.

    A spec is either *uniform* (``builder`` is ``None`` and the generic config/
    adapter construction path applies) or *bespoke* (``builder`` is set and
    constructs the adapter directly, used for structurally different planners such
    as the hybrid portfolios or the lazy torch-backed checkpoint wrappers).
    """

    algorithms: frozenset[str]
    adapter_attr: str | None = None
    config_builder_attr: str | None = None
    config_builder_kwargs: dict[str, Any] = field(default_factory=dict)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    allow_fallback_default: bool = False
    pass_allow_fallback: bool = False
    builder: AdapterBuilder | None = None
    metadata_mutator: MetadataMutator | None = None


#: Tuned ORCA variants expressed purely as per-variant field overrides applied on
#: top of the shared filtered ``SocNavPlannerConfig`` (issue #6467 suggested shape).
_ORCA_VARIANTS: dict[str, dict[str, Any]] = {
    "socnav_orca_nonholonomic": {
        "orca_heading_slowdown": 0.8,
        "orca_commit_distance": 1.8,
        "orca_commit_lateral_gain": 0.6,
    },
    "socnav_orca_dd": {
        "orca_time_horizon": 3.0,
        "orca_neighbor_dist": 8.0,
        "orca_max_neighbors": 6,
        "orca_stall_speed_threshold": 0.1,
    },
    "socnav_orca_relaxed": {
        "orca_time_horizon": 8.0,
        "orca_obstacle_range": 8.0,
        "orca_obstacle_threshold": 0.6,
        "orca_head_on_bias": 0.4,
        "orca_symmetry_bias": 0.15,
    },
}


def _map_runner_attr(name: str) -> Any:
    """Resolve a module-level symbol on ``map_runner`` at call time.

    Resolution is deferred so tests can monkeypatch adapter classes and config
    builders on the ``map_runner`` module and keep construction behavior identical
    to the historical inline dispatch.

    Returns:
        Any: The resolved symbol.
    """

    from robot_sf.benchmark import map_runner  # noqa: PLC0415

    symbol = getattr(map_runner, name, None)
    if symbol is None:
        raise RuntimeError(f"SocNav-family registry cannot resolve map_runner attribute '{name}'.")
    return symbol


def _orca_spec(name: str, overrides: dict[str, Any]) -> SocnavFamilySpec:
    """Build the ORCA-family spec entry for one variant (base key uses no overrides).

    Returns:
        SocnavFamilySpec: Registry spec routing ``name`` through the ORCA adapter.
    """
    return SocnavFamilySpec(
        algorithms=frozenset({name}),
        adapter_attr="ORCAPlannerAdapter",
        config_overrides=overrides,
        allow_fallback_default=False,
        pass_allow_fallback=True,
    )


def _prediction_planner_metadata_mutator(
    meta: dict[str, Any],
    algo_config: dict[str, Any],
) -> None:
    """Stamp predictive-planner mode labels onto metadata."""
    meta.update(_prediction_planner_metadata_overrides(algo_config))


def _rvo_placeholder_metadata_mutator(
    meta: dict[str, Any],
    algo_config: dict[str, Any],
) -> None:
    """Mark the RVO key as an unimplemented placeholder."""
    del algo_config
    meta.update({"status": "placeholder", "fallback_reason": "unimplemented"})


def _build_crowdnav_height_adapter(
    algo_config: dict[str, Any],
    socnav_cfg: Any,
    meta: dict[str, Any],
) -> Any:
    """Build the CrowdNav HEIGHT upstream-checkpoint adapter (lazy torch import).

    Returns:
        Any: The constructed CrowdNav HEIGHT adapter.
    """
    del socnav_cfg, meta
    adapter_cls, config_builder = _map_runner_attr("_crowdnav_height_symbols")()
    return adapter_cls(config=config_builder(algo_config))


def _build_sonic_crowdnav_adapter(
    algo_config: dict[str, Any],
    socnav_cfg: Any,
    meta: dict[str, Any],
) -> Any:
    """Build the SoNIC upstream-checkpoint adapter (lazy torch import).

    Returns:
        Any: The constructed SoNIC adapter.
    """
    del socnav_cfg, meta
    adapter_cls, config_builder = _map_runner_attr("_sonic_crowdnav_symbols")()
    return adapter_cls(config=config_builder(algo_config))


def _build_gensafenav_ours_gst_adapter(
    algo_config: dict[str, Any],
    socnav_cfg: Any,
    meta: dict[str, Any],
) -> Any:
    """Build the GenSafeNav Ours_GST adapter with its default checkpoint payload.

    Returns:
        Any: The constructed GenSafeNav Ours_GST adapter.
    """
    del socnav_cfg, meta
    adapter_cls, config_builder = _map_runner_attr("_sonic_crowdnav_symbols")()
    payload = {
        **algo_config,
        "repo_root": algo_config.get("repo_root", "output/repos/GenSafeNav"),
        "model_name": algo_config.get("model_name", "Ours_GST"),
        "checkpoint_name": algo_config.get("checkpoint_name", "05207.pt"),
    }
    return adapter_cls(config=config_builder(payload))


def _build_gensafenav_gst_predictor_rand_adapter(
    algo_config: dict[str, Any],
    socnav_cfg: Any,
    meta: dict[str, Any],
) -> Any:
    """Build the GenSafeNav GST_predictor_rand adapter with its default checkpoint payload.

    Returns:
        Any: The constructed GenSafeNav GST_predictor_rand adapter.
    """
    del socnav_cfg, meta
    adapter_cls, config_builder = _map_runner_attr("_sonic_crowdnav_symbols")()
    payload = {
        **algo_config,
        "repo_root": algo_config.get("repo_root", "output/repos/GenSafeNav"),
        "model_name": algo_config.get("model_name", "GST_predictor_rand"),
        "checkpoint_name": algo_config.get("checkpoint_name", "05207.pt"),
    }
    return adapter_cls(config=config_builder(payload))


def _build_hybrid_portfolio_adapter(
    algo_config: dict[str, Any],
    socnav_cfg: Any,
    meta: dict[str, Any],
) -> Any:
    """Build the hybrid portfolio (risk-DWA + MPPI + ORCA + prediction) adapter.

    Returns:
        Any: The constructed hybrid portfolio adapter.
    """
    del socnav_cfg, meta
    allow_fallback = bool(algo_config.get("allow_fallback", True))
    hybrid_cfg = _map_runner_attr("build_hybrid_portfolio_build_config")(algo_config)
    return _map_runner_attr("HybridPortfolioAdapter")(
        hybrid_config=hybrid_cfg.hybrid,
        risk_dwa=_map_runner_attr("RiskDWAPlannerAdapter")(config=hybrid_cfg.risk_dwa),
        mppi=_map_runner_attr("MPPISocialPlannerAdapter")(config=hybrid_cfg.mppi),
        orca=_map_runner_attr("ORCAPlannerAdapter")(
            config=hybrid_cfg.socnav, allow_fallback=allow_fallback
        ),
        prediction=_map_runner_attr("PredictionPlannerAdapter")(
            config=hybrid_cfg.socnav, allow_fallback=allow_fallback
        ),
    )


def _build_hybrid_orca_sampler_adapter(
    algo_config: dict[str, Any],
    socnav_cfg: Any,
    meta: dict[str, Any],
) -> Any:
    """Build the ORCA-with-sampler guard adapter.

    Returns:
        Any: The constructed hybrid ORCA sampler adapter.
    """
    del socnav_cfg, meta
    allow_fallback = bool(algo_config.get("allow_fallback", True))
    hybrid_cfg = _map_runner_attr("build_hybrid_orca_sampler_build_config")(algo_config)
    return _map_runner_attr("HybridORCASamplerAdapter")(
        config=hybrid_cfg.guard,
        orca_adapter=_map_runner_attr("ORCAPlannerAdapter")(
            config=hybrid_cfg.socnav, allow_fallback=allow_fallback
        ),
        sampler_adapter=_map_runner_attr("MPPISocialPlannerAdapter")(config=hybrid_cfg.mppi),
    )


def _build_planner_selector_v2_diagnostic_adapter(
    algo_config: dict[str, Any],
    socnav_cfg: Any,
    meta: dict[str, Any],
) -> Any:
    """Build the diagnostic planner-selector-v2 adapter and stamp its claim boundary.

    Returns:
        Any: The constructed planner-selector-v2 diagnostic adapter.
    """
    del socnav_cfg
    adapter = _map_runner_attr("_build_planner_selector_v2_adapter")(algo_config)
    meta["selector_boundary"] = {
        "diagnostic_only": True,
        "benchmark_strength": False,
        "learned_policy_used": False,
        "claim_boundary": "diagnostic_only_not_benchmark_success",
    }
    return adapter


SOCNAV_FAMILY_SPECS: tuple[SocnavFamilySpec, ...] = (
    # Classical in-repo adapters sharing the filtered SocNavPlannerConfig.
    SocnavFamilySpec(
        algorithms=frozenset({"socnav_sampling", "sampling"}),
        adapter_attr="SamplingPlannerAdapter",
    ),
    SocnavFamilySpec(
        algorithms=frozenset({"social_force", "sf"}),
        adapter_attr="SocialForcePlannerAdapter",
    ),
    SocnavFamilySpec(
        algorithms=frozenset({"hrvo", "socnav_hrvo"}),
        adapter_attr="HRVOPlannerAdapter",
    ),
    # ORCA family: base key plus the tuned variants (per-variant config overrides).
    _orca_spec("orca", {}),
    *(_orca_spec(name, overrides) for name, overrides in _ORCA_VARIANTS.items()),
    # Upstream SocNavBench wrapper.
    SocnavFamilySpec(
        algorithms=frozenset({"socnav_bench"}),
        adapter_attr="SocNavBenchSamplingAdapter",
        allow_fallback_default=False,
        pass_allow_fallback=True,
    ),
    # SACADRL and the predictive planner.
    SocnavFamilySpec(
        algorithms=frozenset({"sacadrl", "sa_cadrl"}),
        adapter_attr="SACADRLPlannerAdapter",
        allow_fallback_default=False,
        pass_allow_fallback=True,
    ),
    SocnavFamilySpec(
        algorithms=frozenset({"prediction_planner"}),
        adapter_attr="PredictionPlannerAdapter",
        allow_fallback_default=False,
        pass_allow_fallback=True,
        metadata_mutator=_prediction_planner_metadata_mutator,
    ),
    # NMPC and DWA build their own dedicated configs from the algorithm payload.
    SocnavFamilySpec(
        algorithms=frozenset({"nmpc_social", "nmpc"}),
        adapter_attr="NMPCSocialPlannerAdapter",
        config_builder_attr="build_nmpc_social_config",
    ),
    SocnavFamilySpec(
        algorithms=frozenset({"dwa"}),
        adapter_attr="DWAPlannerAdapter",
        config_builder_attr="build_dwa_config",
    ),
    # RVO placeholder (kept as a registered key for error-surface compatibility).
    SocnavFamilySpec(
        algorithms=frozenset({"rvo"}),
        adapter_attr="SamplingPlannerAdapter",
        metadata_mutator=_rvo_placeholder_metadata_mutator,
    ),
    # External Social-Navigation-PyEnvs wrappers.
    SocnavFamilySpec(
        algorithms=frozenset({"social_navigation_pyenvs_orca", "social_nav_pyenvs_orca"}),
        adapter_attr="SocialNavigationPyEnvsORCAAdapter",
        config_builder_attr="build_social_navigation_pyenvs_orca_config",
    ),
    SocnavFamilySpec(
        algorithms=frozenset(
            {"social_navigation_pyenvs_socialforce", "social_nav_pyenvs_socialforce"}
        ),
        adapter_attr="SocialNavigationPyEnvsForceModelAdapter",
        config_builder_attr="build_social_navigation_pyenvs_force_model_config",
        config_builder_kwargs={"default_policy_name": "socialforce"},
    ),
    SocnavFamilySpec(
        algorithms=frozenset(
            {"social_navigation_pyenvs_sfm_helbing", "social_nav_pyenvs_sfm_helbing"}
        ),
        adapter_attr="SocialNavigationPyEnvsForceModelAdapter",
        config_builder_attr="build_social_navigation_pyenvs_force_model_config",
        config_builder_kwargs={"default_policy_name": "sfm_helbing"},
    ),
    SocnavFamilySpec(
        algorithms=frozenset(
            {"social_navigation_pyenvs_hsfm_new_guo", "social_nav_pyenvs_hsfm_new_guo"}
        ),
        adapter_attr="SocialNavigationPyEnvsHSFMAdapter",
        config_builder_attr="build_social_navigation_pyenvs_hsfm_config",
        config_builder_kwargs={"default_policy_name": "hsfm_new_guo"},
    ),
    # Torch-backed checkpoint wrappers resolved lazily through map_runner helpers.
    SocnavFamilySpec(
        algorithms=frozenset({"crowdnav_height"}),
        builder=_build_crowdnav_height_adapter,
    ),
    SocnavFamilySpec(
        algorithms=frozenset({"sonic_crowdnav", "sonic_gst"}),
        builder=_build_sonic_crowdnav_adapter,
    ),
    SocnavFamilySpec(
        algorithms=frozenset({"gensafenav_ours_gst", "gensafe_ours_gst", "ours_gst"}),
        builder=_build_gensafenav_ours_gst_adapter,
    ),
    SocnavFamilySpec(
        algorithms=frozenset(
            {"gensafenav_gst_predictor_rand", "gensafe_gst_predictor_rand", "gst_predictor_rand"}
        ),
        builder=_build_gensafenav_gst_predictor_rand_adapter,
    ),
    # Composite and diagnostic planners with bespoke construction.
    SocnavFamilySpec(
        algorithms=frozenset({"hybrid_portfolio"}),
        builder=_build_hybrid_portfolio_adapter,
    ),
    SocnavFamilySpec(
        algorithms=frozenset({"hybrid_orca_sampler"}),
        builder=_build_hybrid_orca_sampler_adapter,
    ),
    SocnavFamilySpec(
        algorithms=frozenset({"planner_selector_v2_diagnostic"}),
        builder=_build_planner_selector_v2_diagnostic_adapter,
    ),
)

_SOCNAV_FAMILY_LOOKUP: dict[str, SocnavFamilySpec] = {}
for _spec in SOCNAV_FAMILY_SPECS:
    for _algo in _spec.algorithms:
        _existing = _SOCNAV_FAMILY_LOOKUP.get(_algo)
        if _existing is not None:
            raise RuntimeError(f"Duplicate SocNav-family registry key '{_algo}'.")
        _SOCNAV_FAMILY_LOOKUP[_algo] = _spec


def _build_uniform_adapter(
    spec: SocnavFamilySpec,
    algo_config: dict[str, Any],
    socnav_cfg: Any,
) -> Any:
    """Build a spec entry through the generic config/adapter construction path.

    Returns:
        Any: The constructed planner adapter.
    """
    if spec.adapter_attr is None:
        raise RuntimeError(
            f"SocNav-family registry entry {sorted(spec.algorithms)} is missing an adapter."
        )
    adapter_cls = _map_runner_attr(spec.adapter_attr)
    if spec.config_builder_attr is not None:
        config_builder = _map_runner_attr(spec.config_builder_attr)
        config = config_builder(algo_config, **spec.config_builder_kwargs)
    else:
        config = socnav_cfg
    if spec.config_overrides:
        config = deepcopy(config)
        for override_name, override_value in spec.config_overrides.items():
            setattr(config, override_name, override_value)
    kwargs: dict[str, Any] = {"config": config}
    if spec.pass_allow_fallback:
        kwargs["allow_fallback"] = bool(
            algo_config.get("allow_fallback", spec.allow_fallback_default)
        )
    return adapter_cls(**kwargs)


def build_socnav_family_adapter(
    *,
    algo_key: str,
    algo: str,
    algo_config: dict[str, Any],
    meta: dict[str, Any],
    socnav_cfg: Any,
) -> Any:
    """Build a SocNav-family planner adapter from the data-driven registry.

    Args:
        algo_key: Lowercased algorithm key used for registry lookup.
        algo: Original algorithm label (used only for the unknown-algorithm error).
        algo_config: Algorithm configuration payload.
        meta: Algorithm metadata dict; mutated in place by a few specs.
        socnav_cfg: Shared filtered ``SocNavPlannerConfig`` for the base planners.

    Returns:
        Any: The constructed planner adapter.

    Raises:
        ValueError: If ``algo_key`` is not a registered SocNav-family key.
    """
    spec = _SOCNAV_FAMILY_LOOKUP.get(algo_key)
    if spec is None:
        raise ValueError(f"Unknown map-based algorithm '{algo}'.")
    if spec.builder is not None:
        adapter = spec.builder(algo_config, socnav_cfg, meta)
    else:
        adapter = _build_uniform_adapter(spec, algo_config, socnav_cfg)
    if spec.metadata_mutator is not None:
        spec.metadata_mutator(meta, algo_config)
    return adapter


__all__ = [
    "SOCNAV_FAMILY_SPECS",
    "SocnavFamilySpec",
    "build_socnav_family_adapter",
]
