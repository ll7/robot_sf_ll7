"""Map-based benchmark runner using Gym environments and scenario YAMLs."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TextIO

import numpy as np
import yaml
from loguru import logger

from robot_sf.baselines.dr_mpc import DRMPCPlanner, build_dr_mpc_config
from robot_sf.baselines.drl_vo import DrlVoPlanner
from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig
from robot_sf.baselines.sac import SACPlanner
from robot_sf.baselines.sicnav import SICNavPlanner, build_sicnav_config
from robot_sf.benchmark import planner_command_contract as planner_commands
from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    infer_execution_mode_from_counts,
    resolve_observation_mode,
)
from robot_sf.benchmark.algorithm_readiness import (
    BenchmarkProfile,
    require_algorithm_allowed,
)
from robot_sf.benchmark.fallback_policy import availability_payload
from robot_sf.benchmark.latency_stress import (
    LatencyStressProfile,
    load_latency_stress_profile,
    not_available_latency_metrics,
)
from robot_sf.benchmark.local_model_artifacts import validate_no_local_model_artifacts
from robot_sf.benchmark.metrics import (
    EpisodeData,
    compute_all_metrics,
    post_process_metrics,
)
from robot_sf.benchmark.observation_noise import (
    apply_observation_noise,
    make_observation_noise_rng,
    merge_observation_noise_stats,
    new_observation_noise_stats,
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.obstacle_sampling import sample_obstacle_points
from robot_sf.benchmark.path_utils import compute_shortest_path_length
from robot_sf.benchmark.scenario_schema import validate_scenario_list
from robot_sf.benchmark.schema_validator import load_schema, validate_episode
from robot_sf.benchmark.synthetic_actuation import (
    SyntheticActuationController,
    SyntheticActuationProfile,
    not_available_saturation_metrics,
    validate_actuation_profile_claim_boundary,
    validate_synthetic_actuation_profile,
)
from robot_sf.benchmark.termination_reason import (
    build_outcome_payload,
    collision_event,
    outcome_contradictions,
    resolve_termination_reason,
    route_complete_success,
    status_from_termination_reason,
)
from robot_sf.benchmark.thresholds import ensure_metric_parameters
from robot_sf.benchmark.utils import (
    _config_hash,
    _git_hash_fallback,
    attach_track_metadata,
    index_existing,
    normalize_track_field,
    validate_episode_success_integrity,
)
from robot_sf.common.math_utils import wrap_angle_pi as _normalize_heading
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig
from robot_sf.planner.adaptive_proxemic_selector import (
    AdaptiveProxemicSelectorAdapter,
    build_adaptive_proxemic_selector_config,
)
from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter
from robot_sf.planner.crowdnav_height import (
    CrowdNavHeightAdapter,
    build_crowdnav_height_config,
)
from robot_sf.planner.gap_prediction import (
    GapAwarePredictionAdapter,
    build_gap_prediction_config,
)
from robot_sf.planner.grid_route import GridRoutePlannerAdapter, build_grid_route_config
from robot_sf.planner.guarded_ppo import (
    GuardedPPOAdapter,
    build_guarded_ppo_config,
    build_guarded_ppo_fallback,
    build_guarded_ppo_prior,
)
from robot_sf.planner.hybrid_orca_sampler import (
    HybridORCASamplerAdapter,
    build_hybrid_orca_sampler_build_config,
)
from robot_sf.planner.hybrid_portfolio import (
    HybridPortfolioAdapter,
    build_hybrid_portfolio_build_config,
)
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)
from robot_sf.planner.kinematics_model import (
    KinematicsModel,
    resolve_benchmark_kinematics_model,
)
from robot_sf.planner.learned_risk_surface import (
    RiskSurfacePlannerAdapter,
    build_local_risk_surface_spec,
)
from robot_sf.planner.lidar_occupancy import (
    LidarOccupancyPlannerAdapter,
    build_lidar_occupancy_config,
)
from robot_sf.planner.lidar_occupancy_grid import build_lidar_grid_route_adapter
from robot_sf.planner.lidar_tracked_agents import build_lidar_tracked_social_force_adapter
from robot_sf.planner.mppi_social import (
    MPPISocialPlannerAdapter,
    build_mppi_social_config,
)
from robot_sf.planner.nmpc_social import (
    NMPCSocialPlannerAdapter,
    build_nmpc_social_config,
)
from robot_sf.planner.planner_selector_v2_diagnostic import (
    PlannerSelectorV2DiagnosticAdapter,
    build_planner_selector_v2_diagnostic_config,
)
from robot_sf.planner.policy_stack_v1 import (
    PolicyStackV1Adapter,
    build_policy_stack_v1_build_config,
)
from robot_sf.planner.predictive_mppi import (
    PredictiveMPPIAdapter,
    build_predictive_mppi_config,
)
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, build_risk_dwa_config
from robot_sf.planner.safety_barrier import (
    SafetyBarrierPlannerAdapter,
    build_safety_barrier_config,
)
from robot_sf.planner.safety_shield import (
    ShieldDecision,
    new_shield_stats,
    shield_contract_metadata,
    shield_metrics_from_stats,
    update_shield_stats,
)
from robot_sf.planner.social_navigation_pyenvs_force_model import (
    SocialNavigationPyEnvsForceModelAdapter,
    build_social_navigation_pyenvs_force_model_config,
)
from robot_sf.planner.social_navigation_pyenvs_hsfm import (
    SocialNavigationPyEnvsHSFMAdapter,
    build_social_navigation_pyenvs_hsfm_config,
)
from robot_sf.planner.social_navigation_pyenvs_orca import (
    SocialNavigationPyEnvsORCAAdapter,
    build_social_navigation_pyenvs_orca_config,
)
from robot_sf.planner.socnav import (
    HRVOPlannerAdapter,
    ORCAPlannerAdapter,
    PredictionPlannerAdapter,
    SACADRLPlannerAdapter,
    SamplingPlannerAdapter,
    SocialForcePlannerAdapter,
    SocNavBenchSamplingAdapter,
    SocNavPlannerConfig,
    TrivialReferencePlannerAdapter,
)
from robot_sf.planner.sonic_crowdnav import (
    SonicCrowdNavAdapter,
    build_sonic_crowdnav_config,
)
from robot_sf.planner.stream_gap import StreamGapPlannerAdapter, build_stream_gap_config
from robot_sf.planner.teb_commitment import (
    TEBCommitmentPlannerAdapter,
    build_teb_commitment_config,
)
from robot_sf.planner.topology_guided_local_policy import (
    TopologyGuidedHybridRulePlannerAdapter,
    build_topology_guided_local_policy_config,
)
from robot_sf.robot.action_adapters import holonomic_to_diff_drive_action
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from robot_sf.gym_env.unified_config import RobotSimulationConfig


_SOCNAV_ALGO_KEYS = {
    "socnav_sampling",
    "sampling",
    "orca",
    "hrvo",
    "sacadrl",
    "prediction_planner",
    "sa_cadrl",
    "socnav_bench",
    "hybrid_portfolio",
    "hybrid_orca_sampler",
    "social_navigation_pyenvs_orca",
    "social_nav_pyenvs_orca",
    "social_navigation_pyenvs_socialforce",
    "social_nav_pyenvs_socialforce",
    "social_navigation_pyenvs_sfm_helbing",
    "social_nav_pyenvs_sfm_helbing",
    "social_navigation_pyenvs_hsfm_new_guo",
    "social_nav_pyenvs_hsfm_new_guo",
    "socnav_orca_nonholonomic",
    "socnav_orca_dd",
    "socnav_orca_relaxed",
    "socnav_hrvo",
    "crowdnav_height",
    "sonic_crowdnav",
    "sonic_gst",
    "gensafenav_ours_gst",
    "gensafe_ours_gst",
    "ours_gst",
    "gensafenav_ours_gst_guarded",
    "ours_gst_guarded",
    "gensafenav_gst_predictor_rand",
    "gensafe_gst_predictor_rand",
    "gst_predictor_rand",
    "gensafenav_gst_predictor_rand_guarded",
    "gst_predictor_rand_guarded",
    "sicnav",
    "dr_mpc",
    "trivial_reference",
    "reference_adapter",
}
_PPO_PAPER_REQUIRED_PROVENANCE = (
    "training_config",
    "training_commit",
    "dataset_version",
    "checkpoint_id",
    "normalization_id",
    "deterministic_seed_set",
)
_DEFAULT_KINEMATICS = "differential_drive"
_STRICT_LEARNED_POLICY_PROFILES = {"baseline-safe", "paper-baseline"}
_PPO_ALLOWED_OBS_MODES = {"vector", "dict", "native_dict", "multi_input"}
_PPO_ALLOWED_ACTION_SPACES = {"velocity", "unicycle"}
_PPO_WARN_ROBOT_KINEMATICS = {"holonomic", "omni", "omnidirectional"}

_default_robot_command_space = planner_commands.default_robot_command_space
_init_feasibility_metadata = planner_commands.init_feasibility_metadata
_planner_kinematics_compatibility = planner_commands.planner_kinematics_compatibility
_project_with_feasibility = planner_commands.project_with_feasibility
_validate_planner_contract = planner_commands.validate_planner_contract


def _load_synthetic_actuation_profile(payload: Any) -> SyntheticActuationProfile | None:
    """Normalize optional synthetic-actuation payloads into the typed profile contract.

    Returns:
        A validated profile, or ``None`` when the payload is absent.
    """
    if payload is None:
        return None
    if isinstance(payload, SyntheticActuationProfile):
        validate_synthetic_actuation_profile(payload)
        return payload
    if not isinstance(payload, dict):
        raise TypeError("synthetic_actuation_profile must be a mapping when provided")
    validate_actuation_profile_claim_boundary(payload)
    claim_scope = str(payload.get("claim_scope", "synthetic-only")).strip() or "synthetic-only"
    if claim_scope != "synthetic-only":
        raise ValueError("synthetic_actuation_profile.claim_scope must be 'synthetic-only'")
    profile = SyntheticActuationProfile(
        name=str(payload.get("name", "")),
        profile_version=str(payload.get("profile_version", "v0")),
        claim_scope=claim_scope,
        claim_boundary=str(payload.get("claim_boundary", "")),
        max_linear_accel_m_s2=float(payload.get("max_linear_accel_m_s2")),
        max_linear_decel_m_s2=float(payload.get("max_linear_decel_m_s2")),
        max_yaw_rate_rad_s=float(payload.get("max_yaw_rate_rad_s")),
        max_angular_accel_rad_s2=float(payload.get("max_angular_accel_rad_s2")),
        latency_mode=str(payload.get("latency_mode", "")),
        update_mode=str(payload.get("update_mode", "")),
    )
    validate_synthetic_actuation_profile(profile)
    return profile


def _load_latency_stress_profile(payload: Any) -> LatencyStressProfile | None:
    """Normalize optional latency-stress payloads into the typed profile contract.

    Returns:
        A validated profile, or ``None`` when the payload is absent.
    """
    return load_latency_stress_profile(payload)


def _holonomic_world_velocity_command(vx: float, vy: float) -> dict[str, float | str]:
    """Build an explicit world-frame holonomic velocity command payload.

    Returns:
        dict[str, float | str]: Structured command payload for holonomic env actions.
    """
    return {
        "command_kind": "holonomic_vxy_world",
        "vx": float(vx),
        "vy": float(vy),
    }


def _apply_direct_world_velocity_metadata(
    meta: dict[str, Any],
    *,
    adapter_boundary: str | None = None,
) -> None:
    """Mark planner metadata as direct world-velocity execution for holonomic benchmarks."""
    planner_meta = meta.get("planner_kinematics")
    if isinstance(planner_meta, dict):
        planner_meta["planner_command_space"] = "holonomic_vxy_world"
        planner_meta["benchmark_command_space"] = "holonomic_vxy_world"
        planner_meta["projection_policy"] = "world_velocity_passthrough"
        planner_meta["execution_detail"] = "direct_holonomic_world_velocity"
    upstream_reference = meta.get("upstream_reference")
    if isinstance(upstream_reference, dict) and adapter_boundary is not None:
        upstream_reference["adapter_boundary"] = adapter_boundary


def _build_adapter_policy(
    *,
    algo_key: str,
    algo_config: dict[str, Any],
    meta: dict[str, Any],
    adapter: Any,
    adapter_name: str,
    robot_kinematics: str | None,
    normalized_robot_command_mode: str | None,
    limitations: str | None = None,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Construct a projected adapter policy with standard metadata wiring.

    Returns:
        tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
            Projected policy callable and populated benchmark metadata.
    """
    meta.update(
        {
            "status": "ok",
            "config": algo_config,
            "config_hash": _config_hash(algo_config),
        }
    )
    meta = enrich_algorithm_metadata(
        algo=algo_key,
        metadata=meta,
        execution_mode="adapter",
        adapter_name=adapter_name,
        robot_kinematics=robot_kinematics,
    )
    _init_feasibility_metadata(meta)
    planner_meta = meta.get("planner_kinematics")
    if isinstance(planner_meta, dict):
        planner_meta["planner_command_space"] = _default_robot_command_space(
            robot_kinematics,
            algo_config,
            robot_command_mode=normalized_robot_command_mode,
        )
        if limitations is not None:
            planner_meta["limitations"] = limitations
    adapter_kinematics_model = resolve_benchmark_kinematics_model(
        robot_kinematics=robot_kinematics,
        command_limits=algo_config,
    )

    def _policy(obs: dict[str, Any]) -> tuple[float, float]:
        """Run an adapter-backed planner and project command feasibility.

        Returns:
            tuple[float, float]: Projected linear and angular command.
        """
        linear, angular = adapter.plan(obs)
        return _project_with_feasibility(
            model=adapter_kinematics_model,
            command=(float(linear), float(angular)),
            meta=meta,
        )

    _attach_planner_reset(_policy, adapter)
    _policy._planner_adapter = adapter
    if hasattr(adapter, "bind_env"):
        _policy._planner_bind_env = adapter.bind_env
    if hasattr(adapter, "close"):
        _policy._planner_close = adapter.close
    if hasattr(adapter, "diagnostics"):

        def _planner_stats() -> dict[str, Any]:
            """Expose adapter diagnostics for episode metadata.

            Returns:
                dict[str, Any]: Adapter diagnostic payload.
            """
            return adapter.diagnostics()

        _policy._planner_stats = _planner_stats

    return _policy, meta


def _attach_planner_reset(policy: Callable[..., Any], adapter: Any) -> None:
    """Attach a planner reset hook that tolerates adapters without seed support."""
    reset = getattr(adapter, "reset", None)
    if not callable(reset):
        return

    def _planner_reset(seed: int | None = None) -> None:
        """Reset an adapter, using seed-aware reset when supported."""
        if seed is None:
            reset()
            return
        try:
            reset(seed=seed)
        except TypeError:
            reset()

    policy._planner_reset = _planner_reset


def _finalize_feasibility_metadata(meta: dict[str, Any]) -> None:
    """Finalize per-episode feasibility rates/means and strip internal accumulators."""
    feasibility = meta.get("kinematics_feasibility")
    if not isinstance(feasibility, dict):
        return
    total = int(feasibility.get("commands_evaluated", 0))
    infeasible = int(feasibility.get("infeasible_native_count", 0))
    projected = int(feasibility.get("projected_count", 0))
    sum_linear = float(feasibility.pop("_sum_abs_delta_linear", 0.0))
    sum_angular = float(feasibility.pop("_sum_abs_delta_angular", 0.0))
    max_linear = float(feasibility.pop("_max_abs_delta_linear", 0.0))
    max_angular = float(feasibility.pop("_max_abs_delta_angular", 0.0))
    if total > 0:
        feasibility["projection_rate"] = float(projected / total)
        feasibility["infeasible_rate"] = float(infeasible / total)
        feasibility["mean_abs_delta_linear"] = float(sum_linear / total)
        feasibility["mean_abs_delta_angular"] = float(sum_angular / total)
    else:
        feasibility["projection_rate"] = 0.0
        feasibility["infeasible_rate"] = 0.0
        feasibility["mean_abs_delta_linear"] = 0.0
        feasibility["mean_abs_delta_angular"] = 0.0
    feasibility["max_abs_delta_linear"] = max_linear
    feasibility["max_abs_delta_angular"] = max_angular


class _ExternalMPCAdapter:
    """Bridge an external MPC wrapper into the map-runner adapter contract.

    The external wrapper is expected to expose ``step(obs) -> dict`` while the
    benchmark runner expects ``plan(obs) -> (linear, angular)``. The adapter also
    normalizes the upstream observation into the structured Robot SF payload used
    by the other external planner bridges.
    """

    def __init__(
        self,
        planner: Any,
        *,
        algo_config: dict[str, Any],
        robot_kinematics: str | None,
        planner_name: str,
    ) -> None:
        self._planner = planner
        self._algo_config = algo_config
        self._robot_kinematics = robot_kinematics
        self._planner_name = planner_name

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the wrapped planner if it exposes the standard hook."""
        if hasattr(self._planner, "reset"):
            if seed is None:
                self._planner.reset()
            else:
                try:
                    self._planner.reset(seed=seed)
                except TypeError:
                    self._planner.reset()

    def close(self) -> None:
        """Release wrapped planner resources if available."""
        if hasattr(self._planner, "close"):
            self._planner.close()

    def plan(self, obs: dict[str, Any]) -> tuple[float, float]:
        """Produce a map-runner command from an external MPC planner step.

        Returns:
            tuple[float, float]: Projected ``(linear, angular)`` command.
        """
        action = self._planner.step(_obs_to_external_mpc_format(obs))
        if not isinstance(action, dict):
            raise TypeError(
                f"{self._planner_name} returned unsupported action payload: {type(action)}"
            )
        linear, angular, _conversion_mode = _ppo_action_to_unicycle(
            action,
            obs,
            self._algo_config,
            robot_kinematics=self._robot_kinematics,
            project_command=False,
        )
        return linear, angular


def _parse_algo_config(algo_config_path: str | None) -> dict[str, Any]:
    """Load an optional planner YAML config.

    Returns:
        dict[str, Any]: Parsed config mapping, or an empty mapping when omitted.
    """
    if not algo_config_path:
        return {}
    path = Path(algo_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Algorithm config file not found: {algo_config_path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("Algorithm config must be a mapping (YAML dict).")
    validate_no_local_model_artifacts(data, config_path=path)
    return data


def _deep_merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge nested planner config overrides without mutating either input.

    Returns:
        A new mapping containing ``base`` with ``overrides`` applied recursively.
    """
    merged = deepcopy(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_config(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_config_path(anchor: Path | None, raw_path: Any) -> Path | None:
    """Resolve candidate-manifest config paths from manifest-local or repo-root form.

    Returns:
        Resolved path, or ``None`` when the raw path is empty or not a string.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    if anchor is not None:
        anchored = (anchor / path).resolve()
        if anchored.exists():
            return anchored
    return path.resolve()


def _scenario_id(scenario: dict[str, Any]) -> str:
    """Resolve a scenario identifier from common manifest fields.

    Returns:
        str: Scenario id string, or ``"unknown"``.
    """
    return str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )


def _scenario_family(scenario: dict[str, Any]) -> str:
    """Classify a scenario into the report family used by benchmark summaries.

    Returns:
        str: Scenario family label.
    """
    scenario_id = _scenario_id(scenario)
    if scenario_id.startswith("francis2023_"):
        return "francis2023"
    if scenario_id.startswith("classic_"):
        return "classic"
    return str(scenario.get("family") or scenario.get("metadata", {}).get("family") or "nominal")


def _is_policy_search_candidate_manifest(config: dict[str, Any]) -> bool:
    """Return whether a config has policy-search candidate manifest fields."""
    return any(
        key in config
        for key in (
            "base_config_path",
            "params",
            "family_overrides",
            "scenario_overrides",
            "scenario_algo_overrides",
        )
    )


def _load_base_candidate_config(
    manifest: dict[str, Any],
    *,
    config_anchor: Path | None,
) -> dict[str, Any]:
    """Load and merge a policy-search candidate's base config and params.

    Returns:
        dict[str, Any]: Effective candidate planner config.
    """
    base_cfg: dict[str, Any] = {}
    base_path = _resolve_config_path(config_anchor, manifest.get("base_config_path"))
    if base_path is not None:
        base_cfg = _parse_algo_config(str(base_path))
    params = manifest.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError("Policy-search candidate params must be a mapping.")
    return _deep_merge_config(base_cfg, params)


def _scenario_algo_override_runtime(
    override: dict[str, Any],
    *,
    default_algo: str,
    scenario_key: str,
    config_anchor: Path | None,
) -> tuple[str, dict[str, Any]]:
    """Resolve one scenario-level algorithm override.

    Returns:
        Effective algorithm key and flattened runtime config for the scenario.
    """
    algo = str(override.get("algo", default_algo)).strip().lower()
    if not algo:
        raise ValueError(f"Scenario algo override is missing algo: {scenario_key}")
    base_cfg: dict[str, Any] = {}
    base_path = _resolve_config_path(config_anchor, override.get("base_config_path"))
    if base_path is not None:
        base_cfg = _parse_algo_config(str(base_path))
    params = override.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError("Policy-search scenario_algo_overrides params must be a mapping.")
    return algo, _deep_merge_config(base_cfg, params)


def _resolve_policy_search_candidate_runtime(
    *,
    default_algo: str,
    algo_config_path: str | None,
    scenario: dict[str, Any],
    algo_config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve a policy-search candidate manifest to the runtime algo/config for a scenario.

    Returns:
        Effective algorithm key and flattened runtime config for the scenario.
    """
    manifest = (
        dict(algo_config) if algo_config is not None else _parse_algo_config(algo_config_path)
    )
    if not _is_policy_search_candidate_manifest(manifest):
        return default_algo, manifest

    config_anchor = Path(algo_config_path).resolve().parent if algo_config_path else None
    scenario_key = _scenario_id(scenario)
    algo_overrides = manifest.get("scenario_algo_overrides")
    if isinstance(algo_overrides, dict):
        override = algo_overrides.get(scenario_key)
        if isinstance(override, dict):
            return _scenario_algo_override_runtime(
                override,
                default_algo=default_algo,
                scenario_key=scenario_key,
                config_anchor=config_anchor,
            )

    effective = _load_base_candidate_config(manifest, config_anchor=config_anchor)
    family_overrides = manifest.get("family_overrides")
    if isinstance(family_overrides, dict):
        family_cfg = family_overrides.get(_scenario_family(scenario), {})
        if isinstance(family_cfg, dict):
            effective = _deep_merge_config(effective, family_cfg)
    scenario_overrides = manifest.get("scenario_overrides")
    if isinstance(scenario_overrides, dict):
        scenario_cfg = scenario_overrides.get(scenario_key, {})
        if isinstance(scenario_cfg, dict):
            effective = _deep_merge_config(effective, scenario_cfg)
    return default_algo, effective


def _apply_planner_selector_v2_context(
    algo: str,
    policy_cfg: dict[str, Any],
    *,
    scenario: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Attach selector-v2 scenario context wherever effective runtime config is materialized.

    Returns:
        Runtime config with selector context for planner-selector v2, otherwise the original config.
    """
    if str(algo).strip().lower() != "planner_selector_v2_diagnostic":
        return policy_cfg
    return _deep_merge_config(
        policy_cfg,
        {
            "selector_context": {
                "scenario_id": _scenario_id(scenario),
                "scenario_family": _scenario_family(scenario),
                "seed": int(seed),
            }
        },
    )


def _build_planner_selector_v2_child_adapter(
    *,
    candidate_name: str,
    candidate_config_path: str,
    scenario: dict[str, Any],
) -> Any:
    """Build one existing local candidate adapter for planner-selector v2.

    Returns:
        Adapter instance for a supported local child candidate.
    """
    path = Path(candidate_config_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    manifest = _parse_algo_config(str(path))
    child_default_algo = str(manifest.get("algo", "")).strip().lower()
    if not child_default_algo:
        raise ValueError(f"Selector child candidate is missing algo: {candidate_name}")
    child_algo, child_config = _resolve_policy_search_candidate_runtime(
        default_algo=child_default_algo,
        algo_config_path=str(path),
        algo_config=manifest,
        scenario=scenario,
    )
    if child_algo == "hybrid_rule_local_planner":
        return HybridRuleLocalPlannerAdapter(
            config=build_hybrid_rule_local_planner_config(child_config)
        )
    if child_algo == "orca":
        return ORCAPlannerAdapter(
            config=_build_socnav_config(child_config),
            allow_fallback=bool(child_config.get("allow_fallback", False)),
        )
    raise ValueError(
        "planner_selector_v2_diagnostic only supports existing local hybrid-rule/ORCA "
        f"child candidates; {candidate_name!r} resolved to {child_algo!r}"
    )


def _build_planner_selector_v2_adapter(
    algo_config: dict[str, Any],
) -> PlannerSelectorV2DiagnosticAdapter:
    """Build the diagnostic selector and all configured local child candidates.

    Returns:
        Configured planner-selector v2 adapter.
    """
    build = build_planner_selector_v2_diagnostic_config(algo_config)
    scenario_stub = {
        "name": build.selector.scenario_id,
        "family": build.selector.scenario_family,
    }
    adapters = {
        name: _build_planner_selector_v2_child_adapter(
            candidate_name=name,
            candidate_config_path=path,
            scenario=scenario_stub,
        )
        for name, path in sorted(build.candidate_config_paths.items())
    }
    return PlannerSelectorV2DiagnosticAdapter(
        config=build.selector,
        candidate_adapters=adapters,
    )


def _prediction_planner_metadata_overrides(
    algo_config: dict[str, Any],
) -> dict[str, Any]:
    """Expose predictive-planner search and uncertainty modes as first-class metadata.

    Returns:
        dict[str, Any]: Explicit mode labels for audit-friendly benchmark metadata.
    """
    uncertainty_mode = str(algo_config.get("predictive_uncertainty_mode", "deterministic")).strip()
    search_mode = "mcts_lite"
    if not bool(algo_config.get("predictive_mcts_enabled", False)):
        search_mode = (
            "sequence_beam"
            if bool(algo_config.get("predictive_sequence_search_enabled", False))
            else "lattice"
        )
    sample_count = int(algo_config.get("predictive_risk_sample_count", 1))
    return {
        "prediction_mode": "probabilistic"
        if uncertainty_mode.lower() != "deterministic" or sample_count > 1
        else "deterministic",
        "predictive_uncertainty_mode": uncertainty_mode,
        "predictive_risk_objective": str(
            algo_config.get("predictive_risk_objective", "mean")
        ).strip(),
        "predictive_risk_sample_count": sample_count,
        "predictive_search_mode": search_mode,
    }


def _is_socnav_algorithm(algo: str) -> bool:
    """Return whether an algorithm key routes through the SocNav planner family."""
    return algo.strip().lower() in _SOCNAV_ALGO_KEYS


def _ppo_paper_gate_status(config: dict[str, Any]) -> tuple[bool, str | None]:
    """Return whether PPO config satisfies paper-grade provenance/quality gates."""
    profile = str(config.get("profile", "experimental")).strip().lower()
    if profile not in {"paper", "paper-baseline"}:
        return False, None

    provenance = config.get("provenance")
    if not isinstance(provenance, dict):
        return False, "missing 'provenance' mapping"
    missing = [k for k in _PPO_PAPER_REQUIRED_PROVENANCE if not provenance.get(k)]
    if missing:
        return False, f"missing provenance keys: {', '.join(missing)}"

    gate = config.get("quality_gate")
    if not isinstance(gate, dict):
        return False, "missing 'quality_gate' mapping"
    min_success = gate.get("min_success_rate")
    measured_success = gate.get("measured_success_rate")
    try:
        min_success_f = float(min_success)
        measured_success_f = float(measured_success)
    except (TypeError, ValueError):
        return (
            False,
            "quality gate requires numeric min_success_rate and measured_success_rate",
        )
    if not math.isfinite(min_success_f) or not math.isfinite(measured_success_f):
        return False, "quality gate success-rate values must be finite"
    if measured_success_f < min_success_f:
        return (
            False,
            f"quality gate failed: measured_success_rate={measured_success_f:.3f} "
            f"< min_success_rate={min_success_f:.3f}",
        )
    return True, None


def _evaluate_learned_policy_contract(
    *,
    algo: str,
    algo_config: dict[str, Any],
    benchmark_profile: str,
    robot_kinematics: str | None = None,
) -> dict[str, Any]:
    """Evaluate learned-policy compatibility against a benchmark contract schema.

    Returns:
        Contract evaluation payload with schema, observed fields, and status.
    """
    algo_key = algo.strip().lower()
    if algo_key != "ppo":
        return {"status": "not_applicable"}

    obs_mode = str(algo_config.get("obs_mode", "vector")).strip().lower()
    action_space = str(algo_config.get("action_space", "velocity")).strip().lower()
    kinematics = str(robot_kinematics or _DEFAULT_KINEMATICS).strip().lower()

    critical_mismatches: list[str] = []
    warnings: list[str] = []
    if obs_mode == "image":
        critical_mismatches.append(
            "obs_mode=image is incompatible with map-runner preflight contract "
            "(expected vector/dict-style inputs).",
        )
    elif obs_mode not in _PPO_ALLOWED_OBS_MODES:
        critical_mismatches.append(
            f"Unsupported obs_mode='{obs_mode}'. Allowed: {sorted(_PPO_ALLOWED_OBS_MODES)}.",
        )

    if action_space not in _PPO_ALLOWED_ACTION_SPACES:
        critical_mismatches.append(
            f"Unsupported action_space='{action_space}'. "
            f"Allowed: {sorted(_PPO_ALLOWED_ACTION_SPACES)}.",
        )

    if kinematics in _PPO_WARN_ROBOT_KINEMATICS:
        warnings.append(
            f"robot_kinematics='{kinematics}' may require stronger calibration for PPO "
            "adapter conversion.",
        )

    strict_profile = benchmark_profile.strip().lower() in _STRICT_LEARNED_POLICY_PROFILES
    status = "pass"
    if critical_mismatches:
        status = "fail" if strict_profile else "warn"
    elif warnings:
        status = "warn"

    return {
        "status": status,
        "schema": {
            "algorithm": "ppo",
            "observation_modes": sorted(_PPO_ALLOWED_OBS_MODES),
            "action_spaces": sorted(_PPO_ALLOWED_ACTION_SPACES),
            "strict_profiles": sorted(_STRICT_LEARNED_POLICY_PROFILES),
        },
        "observed": {
            "obs_mode": obs_mode,
            "action_space": action_space,
            "robot_kinematics": kinematics,
            "benchmark_profile": benchmark_profile,
        },
        "critical_mismatches": critical_mismatches,
        "warnings": warnings,
    }


def _preflight_policy(  # noqa: C901, PLR0915
    *,
    algo: str,
    algo_config: dict[str, Any],
    benchmark_profile: str,
    missing_prereq_policy: str,
    robot_kinematics: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preflight planner initialization and apply SocNav prereq policy.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Effective policy config and
        preflight status payload.
    """
    policy = missing_prereq_policy.strip().lower()
    if policy not in {"fail-fast", "skip-with-warning", "fallback"}:
        raise ValueError(
            "Unsupported socnav prereq policy "
            f"'{missing_prereq_policy}'. Expected fail-fast|skip-with-warning|fallback.",
        )

    def _build_and_close(cfg: dict[str, Any]) -> dict[str, Any]:
        """Instantiate then close one planner, returning any emitted benchmark metadata.

        Returns:
            dict[str, Any]: Planner metadata emitted during construction-time preflight.
        """
        effective_kinematics = robot_kinematics
        if effective_kinematics is None:
            effective_kinematics = str(
                cfg.get("robot_kinematics", cfg.get("kinematics", _DEFAULT_KINEMATICS))
            )
        robot_cfg = cfg.get("robot_config") if isinstance(cfg.get("robot_config"), dict) else {}
        effective_command_mode = None
        if isinstance(robot_cfg, dict):
            raw_mode = robot_cfg.get("command_mode")
            if raw_mode is not None:
                normalized_mode = str(raw_mode).strip().lower()
                effective_command_mode = normalized_mode or None
        try:
            policy_fn, _meta = _build_policy(
                algo,
                cfg,
                robot_kinematics=effective_kinematics,
                robot_command_mode=effective_command_mode,
            )
        except TypeError as exc:
            # Backward compatibility for tests or monkeypatches with legacy _build_policy signatures.
            message = str(exc)
            if "unexpected keyword argument 'robot_kinematics'" not in message and (
                "unexpected keyword argument 'robot_command_mode'" not in message
            ):
                raise
            try:
                policy_fn, _meta = _build_policy(
                    algo,
                    cfg,
                    robot_kinematics=effective_kinematics,
                )
            except TypeError as legacy_exc:
                if "unexpected keyword argument 'robot_kinematics'" not in str(legacy_exc):
                    raise
                policy_fn, _meta = _build_policy(algo, cfg)
        planner_close = getattr(policy_fn, "_planner_close", None)
        if callable(planner_close):
            planner_close()
        return _meta if isinstance(_meta, dict) else {}

    def _preflight_metadata_issue(meta: dict[str, Any]) -> str | None:
        """Surface benchmark-blocking fallback metadata before any expensive episode work.

        Returns:
            str | None: Fail-closed reason when the planner initialized in fallback mode.
        """
        status = str(meta.get("status", "")).strip().lower()
        if status != "fallback":
            return None
        fallback_reason = str(meta.get("fallback_reason", "")).strip()
        algorithm_label = str(meta.get("algorithm") or algo).strip() or algo
        if fallback_reason:
            return (
                f"Planner '{algorithm_label}' initialized in fallback mode during benchmark "
                f"preflight ({fallback_reason})."
            )
        return (
            f"Planner '{algorithm_label}' initialized in fallback mode during benchmark preflight."
        )

    learned_contract = _evaluate_learned_policy_contract(
        algo=algo,
        algo_config=algo_config,
        benchmark_profile=benchmark_profile,
        robot_kinematics=robot_kinematics,
    )
    contract_status = str(learned_contract.get("status", "not_applicable"))
    if contract_status == "fail":
        mismatches = learned_contract.get("critical_mismatches", [])
        detail = ", ".join(mismatches) if isinstance(mismatches, list) else "unknown mismatch"
        raise ValueError(
            f"Learned-policy compatibility contract failed for '{algo}': {detail}",
        )
    if contract_status == "warn":
        contract_warnings = learned_contract.get("warnings")
        contract_mismatches = learned_contract.get("critical_mismatches")
        details = []
        if isinstance(contract_mismatches, list):
            details.extend(contract_mismatches)
        if isinstance(contract_warnings, list):
            details.extend(contract_warnings)
        logger.warning(
            "Learned-policy contract warning for '{}' (profile='{}'): {}",
            algo,
            benchmark_profile,
            "; ".join(details) if details else "contract warning",
        )

    try:
        planner_meta = _build_and_close(algo_config)
        metadata_issue = _preflight_metadata_issue(planner_meta)
        if metadata_issue is not None:
            logger.warning("{}", metadata_issue)
            return dict(algo_config), {
                "status": "skipped",
                "error": metadata_issue,
                "planner_metadata_status": str(planner_meta.get("status", "")),
                "planner_metadata_fallback_reason": planner_meta.get("fallback_reason"),
                "learned_policy_contract": learned_contract,
            }
        return dict(algo_config), {
            "status": "ok",
            "learned_policy_contract": learned_contract,
        }
    except Exception as exc:
        if not _is_socnav_algorithm(algo):
            raise
        message = (
            f"SocNav preflight failed for algorithm '{algo}': {exc}. "
            "Check missing dependencies/models or choose a different prereq policy."
        )
        if policy == "skip-with-warning":
            logger.warning("{}", message)
            return dict(algo_config), {
                "status": "skipped",
                "error": str(exc),
                "policy": policy,
                "learned_policy_contract": learned_contract,
            }
        if policy == "fallback":
            fallback_cfg = dict(algo_config)
            fallback_cfg["allow_fallback"] = True
            try:
                _build_and_close(fallback_cfg)
                logger.warning(
                    "SocNav preflight failed for '{}'; continuing with allow_fallback=True.",
                    algo,
                )
                return fallback_cfg, {
                    "status": "fallback",
                    "error": str(exc),
                    "policy": policy,
                    "learned_policy_contract": learned_contract,
                }
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"{message} Fallback attempt also failed: {fallback_exc}",
                ) from fallback_exc
        raise RuntimeError(message) from exc


def _build_socnav_config(cfg: dict[str, Any]) -> SocNavPlannerConfig:
    """Build a SocNav planner config from a loose mapping.

    Returns:
        SocNavPlannerConfig: Filtered planner configuration.
    """
    if not isinstance(cfg, dict):
        return SocNavPlannerConfig()
    allowed = {f.name for f in fields(SocNavPlannerConfig)}
    filtered = {key: value for key, value in cfg.items() if key in allowed}
    return SocNavPlannerConfig(**filtered)


def _goal_policy(obs: dict[str, Any], *, max_speed: float = 1.0) -> tuple[float, float]:
    """Compute a simple goal-directed unicycle command from benchmark observations.

    Returns:
        tuple[float, float]: Linear and angular command.
    """
    robot = obs.get("robot")
    goal = obs.get("goal")

    # Prefer the structured benchmark observation, but keep compatibility with
    # the flattened map-runner keys used by the env.
    robot_pos_source = robot.get("position") if isinstance(robot, dict) else None
    if robot_pos_source is None:
        robot_pos_source = obs.get("robot_position", [0.0, 0.0])

    heading_source = robot.get("heading") if isinstance(robot, dict) else None
    if heading_source is None:
        heading_source = obs.get("robot_heading", [0.0])

    goal_pos_source = goal.get("current") if isinstance(goal, dict) else None
    if goal_pos_source is None:
        goal_pos_source = obs.get("goal_current", [0.0, 0.0])

    robot_pos = np.asarray(robot_pos_source, dtype=float)
    heading = float(np.asarray(heading_source, dtype=float)[0])
    goal_pos = np.asarray(goal_pos_source, dtype=float)
    vec = goal_pos - robot_pos
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return 0.0, 0.0
    desired_heading = float(np.arctan2(vec[1], vec[0]))
    heading_error = _normalize_heading(desired_heading - heading)
    angular = float(np.clip(heading_error, -1.0, 1.0))
    linear = float(np.clip(dist, 0.0, max_speed * max(0.0, 1.0 - abs(heading_error) / np.pi)))
    return linear, angular


def _normalize_xy_rows(values: Any) -> np.ndarray:
    """Normalize scalar/list/ndarray payloads to an ``(N, 2)`` float array.

    Returns:
        np.ndarray: ``(N, 2)`` array, or ``(0, 2)`` when input is empty/malformed.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            return np.zeros((0, 2), dtype=float)
        return arr.reshape(-1, 2)
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return arr
        if arr.shape[1] > 2:
            return arr[:, :2]
        return np.pad(arr, ((0, 0), (0, 2 - arr.shape[1])), constant_values=0.0)
    return np.zeros((0, 2), dtype=float)


def _extract_ppo_pedestrians(
    pedestrians: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract count-aware pedestrian positions, velocities, and shared radius.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Pedestrian positions, velocities, and radius.
    """
    ped_pos = _normalize_xy_rows(pedestrians.get("positions", []))
    ped_count_arr = np.asarray(pedestrians.get("count", [ped_pos.shape[0]]), dtype=float).reshape(
        -1
    )
    ped_count = int(ped_count_arr[0]) if ped_count_arr.size else int(ped_pos.shape[0])
    ped_count = max(0, min(ped_count, int(ped_pos.shape[0])))
    ped_pos = ped_pos[:ped_count]

    ped_vel = _normalize_xy_rows(pedestrians.get("velocities", []))
    if ped_vel.shape[0] < ped_count:
        ped_vel = np.pad(
            ped_vel,
            ((0, ped_count - ped_vel.shape[0]), (0, 0)),
            constant_values=0.0,
        )
    ped_vel = ped_vel[:ped_count]

    ped_radius_raw = np.asarray(pedestrians.get("radius", [0.35]), dtype=float).reshape(-1)
    ped_radius = float(ped_radius_raw[0]) if ped_radius_raw.size else 0.35
    return ped_pos, ped_vel, ped_radius


def _extract_ppo_dt(obs: dict[str, Any]) -> float:
    """Resolve PPO dt from structured sim metadata first, then fallback fields.

    Returns:
        float: Timestep for PPO planner observations.
    """
    sim_info = obs.get("sim")
    if isinstance(sim_info, dict) and "timestep" in sim_info:
        dt_source = sim_info.get("timestep")
    else:
        dt_source = obs.get("dt", 0.1)
    dt_raw = np.asarray(0.1 if dt_source is None else dt_source, dtype=float).reshape(-1)
    return float(dt_raw[0]) if dt_raw.size else 0.1


def _obs_to_ppo_format(obs: dict[str, Any]) -> dict[str, Any]:
    """Convert map-runner observations into the PPO baseline observation contract.

    Returns:
        Mapping compatible with ``robot_sf.baselines.ppo.PPOPlanner.step``.
    """
    robot = obs.get("robot", {}) if isinstance(obs.get("robot"), dict) else {}
    goal = obs.get("goal", {}) if isinstance(obs.get("goal"), dict) else {}
    pedestrians = obs.get("pedestrians", {}) if isinstance(obs.get("pedestrians"), dict) else {}

    robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=float).reshape(-1)
    robot_vel = np.asarray(robot.get("velocity", [0.0, 0.0]), dtype=float).reshape(-1)
    if robot_vel.size < 2:
        speed = float(np.asarray(robot.get("speed", [0.0]), dtype=float).reshape(-1)[0])
        heading = float(np.asarray(robot.get("heading", [0.0]), dtype=float).reshape(-1)[0])
        robot_vel = np.array([speed * np.cos(heading), speed * np.sin(heading)], dtype=float)
    robot_goal = np.asarray(goal.get("current", [0.0, 0.0]), dtype=float).reshape(-1)
    robot_heading = float(np.asarray(robot.get("heading", [0.0]), dtype=float).reshape(-1)[0])
    robot_radius = float(np.asarray(robot.get("radius", [0.3]), dtype=float).reshape(-1)[0])

    ped_pos, ped_vel, ped_radius = _extract_ppo_pedestrians(pedestrians)

    agents = []
    for idx in range(ped_pos.shape[0]):
        vel = ped_vel[idx] if idx < ped_vel.shape[0] else np.zeros(2, dtype=float)
        agents.append(
            {
                "position": [float(ped_pos[idx, 0]), float(ped_pos[idx, 1])],
                "velocity": [float(vel[0]), float(vel[1])],
                "radius": ped_radius,
            }
        )

    dt = _extract_ppo_dt(obs)
    return {
        "dt": dt,
        "robot": {
            "position": [float(robot_pos[0]), float(robot_pos[1])]
            if robot_pos.size >= 2
            else [0.0, 0.0],
            "velocity": [float(robot_vel[0]), float(robot_vel[1])]
            if robot_vel.size >= 2
            else [0.0, 0.0],
            "goal": [float(robot_goal[0]), float(robot_goal[1])]
            if robot_goal.size >= 2
            else [0.0, 0.0],
            "heading": robot_heading,
            "radius": robot_radius,
        },
        "agents": agents,
        "obstacles": [],
    }


def _obs_to_external_mpc_format(obs: dict[str, Any]) -> dict[str, Any]:
    """Convert map-runner observations into the external MPC wrapper contract.

    Returns:
        dict[str, Any]: Structured observation with obstacles preserved when present.

    The external MPC wrappers use the same robot/human fields as the PPO bridge,
    but they may also reason about obstacle payloads when the upstream contract
    exposes them, so preserve the raw obstacle list when available.
    """
    payload = _obs_to_ppo_format(obs)
    obstacles = obs.get("obstacles")
    if isinstance(obstacles, list):
        payload["obstacles"] = obstacles
    return payload


class _GoalFallbackAdapter:
    """Simple goal-policy fallback adapter for guarded wrapper experiments."""

    def __init__(self, *, max_speed: float) -> None:
        self._max_speed = float(max_speed)

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return the configured goal fallback command.

        Returns:
            tuple[float, float]: Linear and angular command.
        """
        return _goal_policy(observation, max_speed=self._max_speed)


def _choose_shield_command(
    shield: Any,
    observation: dict[str, Any],
    proposed_command: tuple[float, float],
) -> ShieldDecision:
    """Return a structured shield decision, adapting legacy guard implementations."""
    decision_fn = getattr(shield, "choose_command_decision", None)
    if callable(decision_fn):
        decision = decision_fn(observation, proposed_command)
        if isinstance(decision, ShieldDecision):
            return decision
        if hasattr(decision, "as_command_result"):
            command, label = decision.as_command_result()
            return ShieldDecision(
                proposed_action=(float(proposed_command[0]), float(proposed_command[1])),
                filtered_action=(float(command[0]), float(command[1])),
                decision_label=str(label),
                intervention_reason="legacy_structured_guard_decision",
            )

    command, label = shield.choose_command(observation, proposed_command)
    return ShieldDecision(
        proposed_action=(float(proposed_command[0]), float(proposed_command[1])),
        filtered_action=(float(command[0]), float(command[1])),
        decision_label=str(label),
        intervention_reason="legacy_guard_decision",
    )


def _ppo_action_to_unicycle(
    action: dict[str, Any],
    obs: dict[str, Any],
    cfg: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    kinematics_model: KinematicsModel | None = None,
    project_command: bool = True,
) -> tuple[float, float, str]:
    """Convert PPO action dict into the unicycle command used by map environments.

    Returns:
        Tuple of ``(linear_velocity, angular_velocity, conversion_mode)`` where
        conversion_mode is ``"native"`` or ``"adapter"``.
    """
    model = kinematics_model or resolve_benchmark_kinematics_model(
        robot_kinematics=robot_kinematics,
        command_limits=cfg,
    )
    if "v" in action and "omega" in action:
        if project_command:
            v, omega = model.project((float(action["v"]), float(action["omega"])))
        else:
            v, omega = float(action["v"]), float(action["omega"])
        return v, omega, "native"

    if "vx" not in action or "vy" not in action:
        raise ValueError(f"Unsupported PPO action payload: {action}")

    vx = float(action["vx"])
    vy = float(action["vy"])
    speed = float(np.hypot(vx, vy))
    if speed < 1e-9:
        if project_command:
            v, omega = model.project((0.0, 0.0))
        else:
            v, omega = 0.0, 0.0
        return v, omega, "adapter"

    robot = obs.get("robot", {}) if isinstance(obs.get("robot"), dict) else {}
    heading = float(np.asarray(robot.get("heading", [0.0]), dtype=float).reshape(-1)[0])
    desired_heading = float(np.arctan2(vy, vx))
    heading_error = _normalize_heading(desired_heading - heading)
    omega_max = float(cfg.get("omega_max", cfg.get("max_angular_speed", 1.0)))
    omega_kp = float(cfg.get("omega_kp", cfg.get("heading_error_gain", 1.0)))
    angular_velocity = float(np.clip(omega_kp * heading_error, -omega_max, omega_max))

    if project_command:
        v, omega = model.project((float(speed), angular_velocity))
    else:
        v, omega = float(speed), angular_velocity
    return v, omega, "adapter"


def _update_adapter_impact_metrics(
    meta: dict[str, Any],
    conversion_mode: str,
    *,
    count_native: bool | None = None,
) -> None:
    """Update native-vs-adapted step counters when adapter-impact probing is enabled."""
    impact = meta.get("adapter_impact")
    if not isinstance(impact, dict) or not bool(impact.get("requested", False)):
        return
    if count_native is None:
        count_native = conversion_mode == "native"
    if count_native:
        impact["native_steps"] = int(impact.get("native_steps", 0)) + 1
    else:
        impact["adapted_steps"] = int(impact.get("adapted_steps", 0)) + 1
    impact["status"] = "collecting"


def _build_policy(  # noqa: C901, PLR0912, PLR0915
    algo: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build an action policy and algorithm metadata for map-based benchmarking.

    Args:
        algo: Algorithm key to instantiate.
        algo_config: Algorithm configuration payload.
        robot_kinematics: Runtime robot kinematics label for metadata enrichment.
        robot_command_mode: Runtime robot command mode (for holonomic metadata labels).
        adapter_impact_eval: Whether to collect native-vs-adapter step counters.

    Returns:
        tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
        Policy callable and enriched metadata dictionary. For PPO, adapter-impact
        counters are mutated in-place in the returned metadata during episode rollout.
    """
    algo_key = algo.lower().strip()
    meta: dict[str, Any] = {"algorithm": algo_key}
    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    if algo_key in {"goal", "simple", "goal_policy", "simple_policy"}:
        goal_kinematics_model = resolve_benchmark_kinematics_model(
            robot_kinematics=robot_kinematics,
            command_limits=algo_config,
        )

        meta.update(
            {
                "status": "ok",
                "config": algo_config,
                "config_hash": _config_hash(algo_config),
            }
        )
        meta = enrich_algorithm_metadata(
            algo=algo_key,
            metadata=meta,
            execution_mode="native",
            robot_kinematics=robot_kinematics,
        )
        _init_feasibility_metadata(meta)
        planner_meta = meta.get("planner_kinematics")
        if isinstance(planner_meta, dict):
            planner_meta["planner_command_space"] = _default_robot_command_space(
                robot_kinematics,
                algo_config,
                robot_command_mode=normalized_robot_command_mode,
            )

        def _policy(obs: dict[str, Any]) -> tuple[float, float]:
            """Run the built-in goal policy with feasibility projection.

            Returns:
                tuple[float, float]: Projected linear and angular command.
            """
            linear, angular = _goal_policy(obs, max_speed=float(algo_config.get("max_speed", 1.0)))
            return _project_with_feasibility(
                model=goal_kinematics_model,
                command=(linear, angular),
                meta=meta,
            )

        return _policy, meta

    if algo_key == "risk_dwa":
        adapter = RiskDWAPlannerAdapter(config=build_risk_dwa_config(algo_config))
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="RiskDWAPlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
        )

    if algo_key in {"risk_surface_dwa", "risk_surface_dwa_v0"}:
        risk_surface_cfg = (
            algo_config.get("risk_surface")
            if isinstance(algo_config.get("risk_surface"), dict)
            else {}
        )
        risk_dwa_cfg = (
            algo_config.get("risk_dwa") if isinstance(algo_config.get("risk_dwa"), dict) else {}
        )
        adapter = RiskSurfacePlannerAdapter(
            spec=build_local_risk_surface_spec(risk_surface_cfg),
            planner=RiskDWAPlannerAdapter(config=build_risk_dwa_config(risk_dwa_cfg)),
        )
        meta["risk_surface_planner"] = {
            "status": "enabled",
            "producer": "deterministic_pedestrian_risk_surface",
            "wrapped_planner": "risk_dwa",
            "benchmark_strength": False,
            "claim_boundary": "exploratory_smoke_only",
        }
        return _build_adapter_policy(
            algo_key="risk_surface_dwa",
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="RiskSurfacePlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations="deterministic_risk_surface_fixture_not_benchmark_evidence",
        )

    if algo_key in {"lidar_social_force", "lidar_tracked_social_force"}:
        adapter = build_lidar_tracked_social_force_adapter(algo_config)
        return _build_adapter_policy(
            algo_key="lidar_social_force",
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="LidarTrackedSocialForceAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations="lidar_endpoint_tracked_social_force_testing_only",
        )

    if algo_key in {"trivial_reference", "reference_adapter"}:
        adapter = TrivialReferencePlannerAdapter(config=_build_socnav_config(algo_config))
        meta["algorithm"] = "trivial_reference"
        return _build_adapter_policy(
            algo_key="trivial_reference",
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="TrivialReferencePlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations=(
                "Diagnostic adapter template only; do not use as benchmark planner evidence."
            ),
        )

    if algo_key == "policy_stack_v1":
        stack_cfg = build_policy_stack_v1_build_config(algo_config)
        adapter = PolicyStackV1Adapter(
            config=stack_cfg.policy_stack,
            risk_dwa=RiskDWAPlannerAdapter(config=stack_cfg.risk_dwa),
        )
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="PolicyStackV1Adapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
        )

    if algo_key in {
        "hybrid_rule_local_planner",
        "hybrid_rule_v0_minimal",
        "actuation_aware_hybrid_rule_v0",
    }:
        adapter = HybridRuleLocalPlannerAdapter(
            config=build_hybrid_rule_local_planner_config(algo_config)
        )
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="HybridRuleLocalPlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
        )

    if algo_key in {"adaptive_proxemic_selector_v0", "adaptive_proxemic_selector_v1"}:
        selector_algo_config = dict(algo_config)
        selector_algo_config.setdefault(
            "selector_version",
            "v1" if algo_key == "adaptive_proxemic_selector_v1" else "v0",
        )
        selector_config = build_adaptive_proxemic_selector_config(selector_algo_config)
        adapter = AdaptiveProxemicSelectorAdapter(config=selector_config)
        meta["adaptive_proxemic_selector"] = {
            "status": "enabled",
            "selector_version": selector_config.selector_version,
            "diagnostic_only": bool(selector_config.diagnostic_only),
            "claim_boundary": selector_config.claim_boundary,
            "profile_sources": [
                selector_config.profiles[name].source_candidate
                for name in ("conservative", "neutral", "open")
            ],
        }
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=selector_algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="AdaptiveProxemicSelectorAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations=(
                "diagnostic-only selector over fixed proxemic profiles; "
                "not benchmark or comfort evidence"
            ),
        )

    if algo_key == "safety_barrier":
        adapter: Any = SafetyBarrierPlannerAdapter(config=build_safety_barrier_config(algo_config))
        adapter_name = "SafetyBarrierPlannerAdapter"
        limitations = "static_obstacle_first_testing_only"
        if algo_config.get("lidar_occupancy_adapter"):
            adapter = LidarOccupancyPlannerAdapter(
                planner=adapter,
                config=build_lidar_occupancy_config(algo_config),
            )
            adapter_name = "LidarOccupancySafetyBarrierAdapter"
            limitations = "lidar_derived_ego_occupancy_testing_only"
            meta["lidar_occupancy_adapter"] = {
                "status": "enabled",
                "source": "lidar_rays",
                "output": "ego_occupancy_grid",
                "planner": "safety_barrier",
            }
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name=adapter_name,
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations=limitations,
        )

    if algo_key == "grid_route":
        adapter = GridRoutePlannerAdapter(config=build_grid_route_config(algo_config))
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="GridRoutePlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations="static_obstacle_first_testing_only",
        )

    if algo_key == "topology_guided_hybrid_rule_v0":
        adapter = TopologyGuidedHybridRulePlannerAdapter(
            config=build_topology_guided_local_policy_config(algo_config)
        )
        meta["topology_guided_hybrid_rule"] = {
            "diagnostic_only": True,
            "claim_boundary": "diagnostic_only",
            "hypothesis_source": "masked_occupancy_grid_routes",
            "wrapped_planner": "HybridRuleLocalPlannerAdapter",
        }
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="TopologyGuidedHybridRulePlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations="diagnostic_only_topology_hypothesis_selector",
        )

    if algo_key in {"lidar_grid_route", "lidar_occupancy_grid_route"}:
        adapter = build_lidar_grid_route_adapter(algo_config)
        return _build_adapter_policy(
            algo_key="lidar_grid_route",
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="LidarOccupancyGridRouteAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations="lidar_ego_occupancy_grid_route_testing_only",
        )

    if algo_key == "stream_gap":
        adapter = StreamGapPlannerAdapter(config=build_stream_gap_config(algo_config))
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="StreamGapPlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
        )

    if algo_key == "gap_prediction":
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        adapter = GapAwarePredictionAdapter(
            config=build_gap_prediction_config(algo_config),
            allow_fallback=allow_fallback,
        )
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="GapAwarePredictionAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
        )

    if algo_key == "mppi_social":
        adapter = MPPISocialPlannerAdapter(config=build_mppi_social_config(algo_config))
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="MPPISocialPlannerAdapter",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
        )

    if algo_key == "predictive_mppi":
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        adapter = PredictiveMPPIAdapter(
            config=build_predictive_mppi_config(algo_config),
            allow_fallback=allow_fallback,
        )
        meta.update(
            {
                "status": "ok",
                "config": algo_config,
                "config_hash": _config_hash(algo_config),
            }
        )
        meta = enrich_algorithm_metadata(
            algo=algo_key,
            metadata=meta,
            execution_mode="adapter",
            adapter_name="PredictiveMPPIAdapter",
            robot_kinematics=robot_kinematics,
        )
        _init_feasibility_metadata(meta)
        planner_meta = meta.get("planner_kinematics")
        if isinstance(planner_meta, dict):
            planner_meta["planner_command_space"] = _default_robot_command_space(
                robot_kinematics,
                algo_config,
                robot_command_mode=normalized_robot_command_mode,
            )
        adapter_kinematics_model = resolve_benchmark_kinematics_model(
            robot_kinematics=robot_kinematics,
            command_limits=algo_config,
        )

        def _policy(obs: dict[str, Any]) -> tuple[float, float]:
            """Run a SocNav adapter and project command feasibility.

            Returns:
                tuple[float, float]: Projected linear and angular command.
            """
            linear, angular = adapter.plan(obs)
            return _project_with_feasibility(
                model=adapter_kinematics_model,
                command=(float(linear), float(angular)),
                meta=meta,
            )

        _attach_planner_reset(_policy, adapter)
        return _policy, meta

    if algo_key == "sicnav":
        planner = SICNavPlanner(build_sicnav_config(algo_config), seed=None)
        planner_meta = planner.get_metadata()
        if planner_meta.get("status") != "ok":
            raise RuntimeError(
                "SICNav dependency is missing or unresolved. "
                "Point `repo_root` at a checked-out upstream repo or install the package."
            )
        adapter = _ExternalMPCAdapter(
            planner,
            algo_config=algo_config,
            robot_kinematics=robot_kinematics,
            planner_name="SICNavPlanner",
        )
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="SICNavPlanner",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations="external_mpc_dependency_sensitive",
        )

    if algo_key == "dr_mpc":
        planner = DRMPCPlanner(build_dr_mpc_config(algo_config), seed=None)
        planner_meta = planner.get_metadata()
        if planner_meta.get("status") != "ok":
            raise RuntimeError(
                "DR-MPC dependency is missing or unresolved. "
                "Point `repo_root` at a checked-out upstream repo or install the package."
            )
        adapter = _ExternalMPCAdapter(
            planner,
            algo_config=algo_config,
            robot_kinematics=robot_kinematics,
            planner_name="DRMPCPlanner",
        )
        return _build_adapter_policy(
            algo_key=algo_key,
            algo_config=algo_config,
            meta=meta,
            adapter=adapter,
            adapter_name="DRMPCPlanner",
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations="external_mpc_dependency_sensitive",
        )

    socnav_cfg = _build_socnav_config(algo_config)

    if algo_key in {"socnav_sampling", "sampling"}:
        # Keep `socnav_sampling` as the native in-repo sampling adapter baseline.
        # `socnav_bench` is the upstream SocNavBench wrapper.
        adapter = SamplingPlannerAdapter(config=socnav_cfg)
    elif algo_key in {"social_force", "sf"}:
        adapter = SocialForcePlannerAdapter(config=socnav_cfg)
    elif algo_key in {"ppo"}:
        paper_ready, paper_reason = _ppo_paper_gate_status(algo_config)
        if (
            str(algo_config.get("profile", "experimental")).strip().lower()
            in {
                "paper",
                "paper-baseline",
            }
            and not paper_ready
        ):
            raise ValueError(
                "PPO paper profile requested but gate failed: "
                f"{paper_reason}. Provide provenance + quality_gate in algo config.",
            )
        ppo_planner = PPOPlanner(algo_config, seed=None)
        planner_cfg = getattr(ppo_planner, "config", None)
        if isinstance(planner_cfg, dict):
            ppo_obs_mode = str(planner_cfg.get("obs_mode", "vector")).strip().lower()
        else:
            ppo_obs_mode = str(getattr(planner_cfg, "obs_mode", "vector")).strip().lower()
        if hasattr(ppo_planner, "get_metadata"):
            planner_meta = ppo_planner.get_metadata()
            if isinstance(planner_meta, dict):
                meta.update(planner_meta)
        meta = enrich_algorithm_metadata(
            algo=algo_key,
            metadata=meta,
            execution_mode="mixed",
            adapter_name="ppo_action_to_unicycle",
            robot_kinematics=robot_kinematics,
            adapter_impact_requested=adapter_impact_eval,
        )
        _init_feasibility_metadata(meta)
        planner_meta = meta.get("planner_kinematics")
        if isinstance(planner_meta, dict):
            planner_meta["planner_command_space"] = _default_robot_command_space(
                robot_kinematics,
                algo_config,
                robot_command_mode=normalized_robot_command_mode,
            )
        ppo_kinematics_model = resolve_benchmark_kinematics_model(
            robot_kinematics=robot_kinematics,
            command_limits=algo_config,
        )

        def _policy(obs: dict[str, Any]) -> tuple[float, float]:
            """Run PPO planner inference and convert output into benchmark command space.

            Returns:
                tuple[float, float]: Linear and angular command after conversion/projection.
            """
            if ppo_obs_mode in {"dict", "native_dict", "multi_input"}:
                ppo_obs = obs
            else:
                ppo_obs = _obs_to_ppo_format(obs)
            action = ppo_planner.step(ppo_obs)
            if not isinstance(action, dict):
                raise TypeError(f"PPO planner returned non-dict action: {type(action)}")
            linear, angular, conversion_mode = _ppo_action_to_unicycle(
                action,
                obs,
                algo_config,
                robot_kinematics=robot_kinematics,
                kinematics_model=ppo_kinematics_model,
                project_command=False,
            )
            linear, angular = _project_with_feasibility(
                model=ppo_kinematics_model,
                command=(float(linear), float(angular)),
                meta=meta,
            )
            _update_adapter_impact_metrics(meta, conversion_mode)
            return linear, angular

        _policy._planner_close = ppo_planner.close
        ppo_bind_env = getattr(ppo_planner, "bind_env", None)
        if callable(ppo_bind_env):
            _policy._planner_bind_env = ppo_bind_env
        if "status" not in meta:
            meta["status"] = "ok"
        meta.setdefault("algorithm", "ppo")
        meta.setdefault("config", algo_config)
        meta["profile"] = str(algo_config.get("profile", "experimental")).strip().lower()
        provenance = algo_config.get("provenance")
        if isinstance(provenance, dict):
            meta["provenance"] = provenance
        quality_gate = algo_config.get("quality_gate")
        if isinstance(quality_gate, dict):
            meta["quality_gate"] = quality_gate
        meta["paper_ready"] = bool(paper_ready)
        if paper_reason:
            meta["paper_gate_reason"] = paper_reason
        meta["config_hash"] = _config_hash(meta.get("config", algo_config))
        return _policy, meta
    elif algo_key in {"sac"}:
        sac_planner = SACPlanner(algo_config, seed=None)
        planner_cfg = getattr(sac_planner, "config", None)
        if isinstance(planner_cfg, dict):
            sac_obs_mode = str(planner_cfg.get("obs_mode", "dict")).strip().lower()
        else:
            sac_obs_mode = str(getattr(planner_cfg, "obs_mode", "dict")).strip().lower()
        if hasattr(sac_planner, "get_metadata"):
            planner_meta = sac_planner.get_metadata()
            if isinstance(planner_meta, dict):
                meta.update(planner_meta)
        meta = enrich_algorithm_metadata(
            algo=algo_key,
            metadata=meta,
            execution_mode="mixed",
            adapter_name="ppo_action_to_unicycle",
            robot_kinematics=robot_kinematics,
            adapter_impact_requested=adapter_impact_eval,
        )
        _init_feasibility_metadata(meta)
        planner_meta = meta.get("planner_kinematics")
        if isinstance(planner_meta, dict):
            planner_meta["planner_command_space"] = _default_robot_command_space(
                robot_kinematics,
                algo_config,
                robot_command_mode=normalized_robot_command_mode,
            )
        sac_kinematics_model = resolve_benchmark_kinematics_model(
            robot_kinematics=robot_kinematics,
            command_limits=algo_config,
        )

        sac_action_semantics = str(algo_config.get("action_semantics", "delta")).strip().lower()
        sac_action_space = str(algo_config.get("action_space", "unicycle")).strip().lower()
        # Native bypass is only valid for delta-unicycle outputs from the SAC model itself.
        # Fallback steps emit absolute goal-directed commands that must go through the
        # command→env-action conversion path.  We decide per step by checking the planner
        # status *after* each sac_planner.step() call.
        _sac_can_be_native = sac_action_semantics == "delta" and sac_action_space == "unicycle"

        def _policy(obs: dict[str, Any]) -> tuple[float, float]:
            """Run SAC planner inference and handle native-vs-fallback action semantics.

            Returns:
                tuple[float, float]: Linear and angular command for the environment.
            """
            if sac_obs_mode in {"dict", "native_dict", "multi_input"}:
                sac_obs = obs
            else:
                sac_obs = _obs_to_ppo_format(obs)
            action = sac_planner.step(sac_obs)
            if not isinstance(action, dict):
                raise TypeError(f"SAC planner returned non-dict action: {type(action)}")
            # Track per step whether this output is a native env action (model ran) or an
            # absolute goal-directed fallback command (planner in fallback mode).
            _policy._last_step_native = (
                _sac_can_be_native and getattr(sac_planner, "_status", "fallback") == "ok"
            )
            linear, angular, conversion_mode = _ppo_action_to_unicycle(
                action,
                obs,
                algo_config,
                robot_kinematics=robot_kinematics,
                kinematics_model=sac_kinematics_model,
                project_command=False,
            )
            linear, angular = _project_with_feasibility(
                model=sac_kinematics_model,
                command=(float(linear), float(angular)),
                meta=meta,
            )
            _update_adapter_impact_metrics(meta, conversion_mode)
            return linear, angular

        _policy._planner_close = sac_planner.close
        _policy._planner_native_env_action = _sac_can_be_native
        if "status" not in meta:
            meta["status"] = "ok"
        meta.setdefault("algorithm", "sac")
        meta.setdefault("config", algo_config)
        meta["profile"] = str(algo_config.get("profile", "experimental")).strip().lower()
        meta["config_hash"] = _config_hash(meta.get("config", algo_config))
        return _policy, meta
    elif algo_key == "drl_vo":
        drl_planner = DrlVoPlanner(algo_config, seed=None)
        if hasattr(drl_planner, "get_metadata"):
            planner_meta = drl_planner.get_metadata()
            if isinstance(planner_meta, dict):
                meta.update(planner_meta)

        meta = enrich_algorithm_metadata(
            algo=algo_key,
            metadata=meta,
            execution_mode="mixed",
            adapter_name="drl_vo_action_to_unicycle",
            robot_kinematics=robot_kinematics,
            adapter_impact_requested=adapter_impact_eval,
        )
        _init_feasibility_metadata(meta)
        planner_meta = meta.get("planner_kinematics")
        if isinstance(planner_meta, dict):
            planner_meta["planner_command_space"] = _default_robot_command_space(
                robot_kinematics,
                algo_config,
                robot_command_mode=normalized_robot_command_mode,
            )

        drl_kinematics_model = resolve_benchmark_kinematics_model(
            robot_kinematics=robot_kinematics,
            command_limits=algo_config,
        )

        def _policy(obs: dict[str, Any]) -> tuple[float, float]:
            """Run DRL-VO inference through the PPO-format observation adapter.

            Returns:
                tuple[float, float]: Linear and angular command after conversion/projection.
            """
            drl_obs = _obs_to_ppo_format(obs)
            action = drl_planner.step(drl_obs)
            if not isinstance(action, dict):
                raise TypeError(f"DRL-VO planner returned non-dict action: {type(action)}")
            linear, angular, conversion_mode = _ppo_action_to_unicycle(
                action,
                obs,
                algo_config,
                robot_kinematics=robot_kinematics,
                kinematics_model=drl_kinematics_model,
                project_command=False,
            )
            linear, angular = _project_with_feasibility(
                model=drl_kinematics_model,
                command=(float(linear), float(angular)),
                meta=meta,
            )
            _update_adapter_impact_metrics(meta, conversion_mode)
            return linear, angular

        _policy._planner_close = drl_planner.close
        _attach_planner_reset(_policy, drl_planner)
        if "status" not in meta:
            meta["status"] = "ok"
        meta.setdefault("algorithm", "drl_vo")
        meta.setdefault("config", algo_config)
        meta["config_hash"] = _config_hash(meta.get("config", algo_config))
        return _policy, meta
    elif algo_key in {"guarded_ppo"}:
        ppo_allowed = {field.name for field in fields(PPOPlannerConfig)}
        ppo_config = {key: value for key, value in algo_config.items() if key in ppo_allowed}
        ppo_planner = PPOPlanner(ppo_config, seed=None)
        guard_adapter = GuardedPPOAdapter(
            config=build_guarded_ppo_config(algo_config),
            fallback_adapter=build_guarded_ppo_fallback(algo_config),
            prior_adapter=build_guarded_ppo_prior(algo_config),
        )
        planner_cfg = getattr(ppo_planner, "config", None)
        if isinstance(planner_cfg, dict):
            ppo_obs_mode = str(planner_cfg.get("obs_mode", "vector")).strip().lower()
        else:
            ppo_obs_mode = str(getattr(planner_cfg, "obs_mode", "vector")).strip().lower()
        if hasattr(ppo_planner, "get_metadata"):
            planner_meta = ppo_planner.get_metadata()
            if isinstance(planner_meta, dict):
                meta.update(planner_meta)
        meta = enrich_algorithm_metadata(
            algo=algo_key,
            metadata=meta,
            execution_mode="mixed",
            adapter_name="guarded_ppo_action_to_unicycle",
            robot_kinematics=robot_kinematics,
            adapter_impact_requested=adapter_impact_eval,
        )
        _init_feasibility_metadata(meta)
        meta["guard_stats"] = {
            "ppo_clear": 0,
            "ppo_safe": 0,
            "fallback_safe": 0,
            "prior_blend_safe": 0,
            "prior_residual_safe": 0,
            "prior_safe": 0,
            "stop_safe": 0,
            "fallback_best_effort": 0,
            "stop_best_effort": 0,
            "goal_reached": 0,
        }
        meta["safety_shield_contract"] = shield_contract_metadata(
            shield_name="GuardedPPOAdapter",
            prediction_source="short_horizon_rollout",
            fallback_policy=type(guard_adapter.fallback_adapter).__name__,
        )
        meta["shield_stats"] = new_shield_stats()
        planner_meta = meta.get("planner_kinematics")
        if isinstance(planner_meta, dict):
            planner_meta["planner_command_space"] = _default_robot_command_space(
                robot_kinematics,
                algo_config,
                robot_command_mode=normalized_robot_command_mode,
            )
        ppo_kinematics_model = resolve_benchmark_kinematics_model(
            robot_kinematics=robot_kinematics,
            command_limits=algo_config,
        )

        def _policy(obs: dict[str, Any]) -> tuple[float, float]:
            """Run Guarded PPO and apply the short-horizon safety gate.

            Returns:
                tuple[float, float]: Selected and projected command.
            """
            if ppo_obs_mode in {"dict", "native_dict", "multi_input"}:
                ppo_obs = obs
            else:
                ppo_obs = _obs_to_ppo_format(obs)
            action = ppo_planner.step(ppo_obs)
            if not isinstance(action, dict):
                raise TypeError(f"Guarded PPO planner returned non-dict action: {type(action)}")
            linear, angular, conversion_mode = _ppo_action_to_unicycle(
                action,
                obs,
                algo_config,
                robot_kinematics=robot_kinematics,
                kinematics_model=ppo_kinematics_model,
                project_command=False,
            )
            shield_decision = _choose_shield_command(
                guard_adapter,
                obs,
                (float(linear), float(angular)),
            )
            chosen, decision = shield_decision.as_command_result()
            linear, angular = _project_with_feasibility(
                model=ppo_kinematics_model,
                command=(float(chosen[0]), float(chosen[1])),
                meta=meta,
            )
            guard_stats = meta.get("guard_stats")
            if isinstance(guard_stats, dict):
                guard_stats[decision] = int(guard_stats.get(decision, 0)) + 1
            shield_stats = meta.get("shield_stats")
            if isinstance(shield_stats, dict):
                update_shield_stats(shield_stats, shield_decision)
            _update_adapter_impact_metrics(
                meta,
                conversion_mode,
                count_native=conversion_mode == "native" and decision in {"ppo_clear", "ppo_safe"},
            )
            return linear, angular

        _attach_planner_reset(_policy, guard_adapter)

        def _close_guarded_ppo() -> None:
            """Close PPO and guard adapter resources when present."""
            ppo_planner.close()
            guard_close = getattr(guard_adapter, "close", None)
            if callable(guard_close):
                guard_close()

        _policy._planner_close = _close_guarded_ppo
        ppo_bind_env = getattr(ppo_planner, "bind_env", None)
        guard_bind_env = getattr(guard_adapter, "bind_env", None)
        bind_hooks = [hook for hook in (ppo_bind_env, guard_bind_env) if callable(hook)]
        if bind_hooks:

            def _bind_guarded_ppo_env(env: Any) -> None:
                """Bind PPO runtime observation space and guard map context."""
                for hook in bind_hooks:
                    hook(env)

            _policy._planner_bind_env = _bind_guarded_ppo_env
        meta.setdefault("algorithm", "guarded_ppo")
        meta.setdefault("status", "ok")
        meta.setdefault("config", algo_config)
        meta["config_hash"] = _config_hash(meta.get("config", algo_config))
        return _policy, meta
    elif algo_key in {
        "gensafenav_ours_gst_guarded",
        "ours_gst_guarded",
        "gensafenav_gst_predictor_rand_guarded",
        "gst_predictor_rand_guarded",
    }:
        holonomic_vx_vy_mode = (
            str(robot_kinematics or "").strip().lower() in {"holonomic", "omni", "omnidirectional"}
            and normalized_robot_command_mode == "vx_vy"
        )
        if holonomic_vx_vy_mode:
            raise ValueError(
                "Guarded SoNIC / GenSafeNav wrappers do not support holonomic vx_vy benchmark "
                "action space yet. The upstream checkpoint emits ActionXY world velocities, but "
                "the current short-horizon guard and goal fallback only evaluate unicycle_vw "
                "commands, so this path fails closed instead of collapsing ActionXY through a "
                "lossy (v, w) round-trip."
            )

        guarded_root = {
            "gensafenav_ours_gst_guarded",
            "ours_gst_guarded",
        }
        model_name = (
            str(algo_config.get("model_name", "Ours_GST"))
            if algo_key in guarded_root
            else str(algo_config.get("model_name", "GST_predictor_rand"))
        )
        sonic_adapter = SonicCrowdNavAdapter(
            config=build_sonic_crowdnav_config(
                {
                    **algo_config,
                    "repo_root": algo_config.get("repo_root", "output/repos/GenSafeNav"),
                    "model_name": model_name,
                    "checkpoint_name": algo_config.get("checkpoint_name", "05207.pt"),
                }
            )
        )
        guard_adapter = GuardedPPOAdapter(
            config=build_guarded_ppo_config(algo_config),
            fallback_adapter=_GoalFallbackAdapter(
                max_speed=float(algo_config.get("guard_fallback_max_speed", 1.0)),
            ),
        )
        meta.update(
            {
                "status": "ok",
                "config": algo_config,
                "config_hash": _config_hash(algo_config),
            }
        )
        meta = enrich_algorithm_metadata(
            algo=algo_key,
            metadata=meta,
            execution_mode="mixed",
            adapter_name="sonic_guarded_goal_fallback",
            robot_kinematics=robot_kinematics,
            adapter_impact_requested=adapter_impact_eval,
        )
        _init_feasibility_metadata(meta)
        meta["guard_stats"] = {
            "ppo_clear": 0,
            "ppo_safe": 0,
            "fallback_safe": 0,
            "stop_safe": 0,
            "fallback_best_effort": 0,
            "stop_best_effort": 0,
            "goal_reached": 0,
        }
        meta["safety_shield_contract"] = shield_contract_metadata(
            shield_name="GuardedPPOAdapter",
            prediction_source="short_horizon_rollout",
            fallback_policy="goal",
        )
        meta["shield_stats"] = new_shield_stats()
        planner_meta = meta.get("planner_kinematics")
        if isinstance(planner_meta, dict):
            planner_meta["planner_command_space"] = _default_robot_command_space(
                robot_kinematics,
                algo_config,
                robot_command_mode=normalized_robot_command_mode,
            )
            planner_meta["guard_strategy"] = "short_horizon_safety_gate"
            planner_meta["fallback_policy"] = "goal"

        guarded_kinematics_model = resolve_benchmark_kinematics_model(
            robot_kinematics=robot_kinematics,
            command_limits=algo_config,
        )

        def _policy(obs: dict[str, Any]) -> tuple[float, float]:
            """Run SONIC behind the short-horizon safety guard.

            Returns:
                tuple[float, float]: Guard-selected and projected command.
            """
            sonic_command = sonic_adapter.plan(obs)
            shield_decision = _choose_shield_command(
                guard_adapter,
                obs,
                (float(sonic_command[0]), float(sonic_command[1])),
            )
            chosen, decision = shield_decision.as_command_result()
            linear, angular = _project_with_feasibility(
                model=guarded_kinematics_model,
                command=(float(chosen[0]), float(chosen[1])),
                meta=meta,
            )
            guard_stats = meta.get("guard_stats")
            if isinstance(guard_stats, dict):
                guard_stats[decision] = int(guard_stats.get(decision, 0)) + 1
            shield_stats = meta.get("shield_stats")
            if isinstance(shield_stats, dict):
                update_shield_stats(shield_stats, shield_decision)
            _update_adapter_impact_metrics(
                meta,
                "adapter",
                count_native=False,
            )
            return linear, angular

        _attach_planner_reset(_policy, sonic_adapter)
        if hasattr(sonic_adapter, "close"):
            _policy._planner_close = sonic_adapter.close
        return _policy, meta
    elif algo_key in {"orca"}:
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        adapter = ORCAPlannerAdapter(config=socnav_cfg, allow_fallback=allow_fallback)
    elif algo_key in {"socnav_orca_nonholonomic"}:
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        config = deepcopy(socnav_cfg)
        config.orca_heading_slowdown = 0.8
        config.orca_commit_distance = 1.8
        config.orca_commit_lateral_gain = 0.6
        adapter = ORCAPlannerAdapter(config=config, allow_fallback=allow_fallback)
    elif algo_key in {"socnav_orca_dd"}:
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        config = deepcopy(socnav_cfg)
        config.orca_time_horizon = 3.0
        config.orca_neighbor_dist = 8.0
        config.orca_max_neighbors = 6
        config.orca_stall_speed_threshold = 0.1
        adapter = ORCAPlannerAdapter(config=config, allow_fallback=allow_fallback)
    elif algo_key in {"socnav_orca_relaxed"}:
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        config = deepcopy(socnav_cfg)
        config.orca_time_horizon = 8.0
        config.orca_obstacle_range = 8.0
        config.orca_obstacle_threshold = 0.6
        config.orca_head_on_bias = 0.4
        config.orca_symmetry_bias = 0.15
        adapter = ORCAPlannerAdapter(config=config, allow_fallback=allow_fallback)
    elif algo_key in {"socnav_hrvo"}:
        adapter = HRVOPlannerAdapter(config=socnav_cfg)
    elif algo_key in {"hrvo"}:
        adapter = HRVOPlannerAdapter(config=socnav_cfg)
    elif algo_key in {"social_navigation_pyenvs_orca", "social_nav_pyenvs_orca"}:
        adapter = SocialNavigationPyEnvsORCAAdapter(
            config=build_social_navigation_pyenvs_orca_config(algo_config)
        )
    elif algo_key in {
        "social_navigation_pyenvs_socialforce",
        "social_nav_pyenvs_socialforce",
    }:
        adapter = SocialNavigationPyEnvsForceModelAdapter(
            config=build_social_navigation_pyenvs_force_model_config(
                algo_config,
                default_policy_name="socialforce",
            )
        )
    elif algo_key in {
        "social_navigation_pyenvs_sfm_helbing",
        "social_nav_pyenvs_sfm_helbing",
    }:
        adapter = SocialNavigationPyEnvsForceModelAdapter(
            config=build_social_navigation_pyenvs_force_model_config(
                algo_config,
                default_policy_name="sfm_helbing",
            )
        )
    elif algo_key in {
        "social_navigation_pyenvs_hsfm_new_guo",
        "social_nav_pyenvs_hsfm_new_guo",
    }:
        adapter = SocialNavigationPyEnvsHSFMAdapter(
            config=build_social_navigation_pyenvs_hsfm_config(
                algo_config,
                default_policy_name="hsfm_new_guo",
            )
        )
    elif algo_key in {"crowdnav_height"}:
        adapter = CrowdNavHeightAdapter(config=build_crowdnav_height_config(algo_config))
    elif algo_key in {"sonic_crowdnav", "sonic_gst"}:
        adapter = SonicCrowdNavAdapter(config=build_sonic_crowdnav_config(algo_config))
    elif algo_key in {"gensafenav_ours_gst", "gensafe_ours_gst", "ours_gst"}:
        adapter = SonicCrowdNavAdapter(
            config=build_sonic_crowdnav_config(
                {
                    **algo_config,
                    "repo_root": algo_config.get("repo_root", "output/repos/GenSafeNav"),
                    "model_name": algo_config.get("model_name", "Ours_GST"),
                    "checkpoint_name": algo_config.get("checkpoint_name", "05207.pt"),
                }
            )
        )
    elif algo_key in {
        "gensafenav_gst_predictor_rand",
        "gensafe_gst_predictor_rand",
        "gst_predictor_rand",
    }:
        adapter = SonicCrowdNavAdapter(
            config=build_sonic_crowdnav_config(
                {
                    **algo_config,
                    "repo_root": algo_config.get("repo_root", "output/repos/GenSafeNav"),
                    "model_name": algo_config.get("model_name", "GST_predictor_rand"),
                    "checkpoint_name": algo_config.get("checkpoint_name", "05207.pt"),
                }
            )
        )
    elif algo_key in {"sacadrl", "sa_cadrl"}:
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        adapter = SACADRLPlannerAdapter(config=socnav_cfg, allow_fallback=allow_fallback)
    elif algo_key == "prediction_planner":
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        adapter = PredictionPlannerAdapter(config=socnav_cfg, allow_fallback=allow_fallback)
        meta.update(_prediction_planner_metadata_overrides(algo_config))
    elif algo_key == "hybrid_portfolio":
        allow_fallback = bool(algo_config.get("allow_fallback", True))
        hybrid_cfg = build_hybrid_portfolio_build_config(algo_config)
        adapter = HybridPortfolioAdapter(
            hybrid_config=hybrid_cfg.hybrid,
            risk_dwa=RiskDWAPlannerAdapter(config=hybrid_cfg.risk_dwa),
            mppi=MPPISocialPlannerAdapter(config=hybrid_cfg.mppi),
            orca=ORCAPlannerAdapter(config=hybrid_cfg.socnav, allow_fallback=allow_fallback),
            prediction=PredictionPlannerAdapter(
                config=hybrid_cfg.socnav, allow_fallback=allow_fallback
            ),
        )
    elif algo_key == "planner_selector_v2_diagnostic":
        adapter = _build_planner_selector_v2_adapter(algo_config)
        meta["selector_boundary"] = {
            "diagnostic_only": True,
            "benchmark_strength": False,
            "learned_policy_used": False,
            "claim_boundary": "diagnostic_only_not_benchmark_success",
        }
    elif algo_key == "hybrid_orca_sampler":
        allow_fallback = bool(algo_config.get("allow_fallback", True))
        hybrid_cfg = build_hybrid_orca_sampler_build_config(algo_config)
        adapter = HybridORCASamplerAdapter(
            config=hybrid_cfg.guard,
            orca_adapter=ORCAPlannerAdapter(
                config=hybrid_cfg.socnav,
                allow_fallback=allow_fallback,
            ),
            sampler_adapter=MPPISocialPlannerAdapter(config=hybrid_cfg.mppi),
        )
    elif algo_key in {"socnav_bench"}:
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        adapter = SocNavBenchSamplingAdapter(config=socnav_cfg, allow_fallback=allow_fallback)
    elif algo_key == "teb":
        adapter = TEBCommitmentPlannerAdapter(config=build_teb_commitment_config(algo_config))
    elif algo_key in {"nmpc_social", "nmpc"}:
        adapter = NMPCSocialPlannerAdapter(config=build_nmpc_social_config(algo_config))
    elif algo_key in {"rvo", "dwa"}:
        adapter = SamplingPlannerAdapter(config=socnav_cfg)
        meta.update({"status": "placeholder", "fallback_reason": "unimplemented"})
    else:
        raise ValueError(f"Unknown map-based algorithm '{algo}'.")

    if "status" not in meta:
        meta["status"] = "ok"
    meta["config"] = algo_config
    provenance = algo_config.get("provenance")
    if isinstance(provenance, dict):
        meta["provenance"] = provenance
    meta["config_hash"] = _config_hash(algo_config)
    meta = enrich_algorithm_metadata(
        algo=algo_key,
        metadata=meta,
        execution_mode="adapter",
        robot_kinematics=robot_kinematics,
    )
    _init_feasibility_metadata(meta)
    planner_meta = meta.get("planner_kinematics")
    if isinstance(planner_meta, dict):
        planner_meta["planner_command_space"] = _default_robot_command_space(
            robot_kinematics,
            algo_config,
            robot_command_mode=normalized_robot_command_mode,
        )
    adapter_kinematics_model = resolve_benchmark_kinematics_model(
        robot_kinematics=robot_kinematics,
        command_limits=algo_config,
    )
    planner_bind_env = None
    if algo_key in {"hrvo", "socnav_hrvo"} and hasattr(adapter, "bind_static_obstacle_points"):

        def _bind_env(env: Any) -> None:
            """Bind sampled static obstacle points from the environment to HRVO."""
            simulator = getattr(env, "simulator", None)
            if simulator is None or not hasattr(simulator, "iter_obstacle_segments"):
                return
            spacing = max(
                float(getattr(adapter.config, "orca_obstacle_margin", 0.12)) * 2.0,
                0.25,
            )
            points = sample_obstacle_points(
                simulator.iter_obstacle_segments(),
                spacing=spacing,
            )
            adapter.bind_static_obstacle_points(points, spacing=spacing)

        planner_bind_env = _bind_env
    elif hasattr(adapter, "bind_env"):
        planner_bind_env = adapter.bind_env
    holonomic_world_velocity_mode = (
        algo_key
        in {
            "orca",
            "hrvo",
            "socnav_hrvo",
            "social_force",
            "sf",
            "social_navigation_pyenvs_orca",
            "social_nav_pyenvs_orca",
            "social_navigation_pyenvs_socialforce",
            "social_nav_pyenvs_socialforce",
            "social_navigation_pyenvs_sfm_helbing",
            "social_nav_pyenvs_sfm_helbing",
            "sonic_crowdnav",
            "sonic_gst",
            "gensafenav_ours_gst",
            "gensafe_ours_gst",
            "ours_gst",
            "gensafenav_gst_predictor_rand",
            "gensafe_gst_predictor_rand",
            "gst_predictor_rand",
        }
        and str(robot_kinematics or "").strip().lower() in {"holonomic", "omni", "omnidirectional"}
        and normalized_robot_command_mode == "vx_vy"
    )
    if holonomic_world_velocity_mode:
        if algo_key == "orca":
            adapter_boundary = (
                "Use upstream Python-RVO2 to solve reciprocal-avoidance velocity in world "
                "coordinates, then forward that world-frame velocity directly into the "
                "holonomic vx_vy benchmark action space."
            )
        elif algo_key in {"hrvo", "socnav_hrvo"}:
            adapter_boundary = (
                "Run the local HRVO geometry solver in world velocity space, then forward the "
                "selected world-frame velocity directly into the holonomic vx_vy benchmark "
                "action space."
            )
        elif algo_key == "social_force":
            adapter_boundary = (
                "Compute the local social-force translational command as a world-frame velocity "
                "vector, then forward that world-frame velocity directly into the holonomic "
                "vx_vy benchmark action space."
            )
        elif algo_key in {"social_navigation_pyenvs_orca", "social_nav_pyenvs_orca"}:
            adapter_boundary = (
                "Map Robot SF SocNav observations into the upstream Social-Navigation-PyEnvs "
                "JointState contract, run upstream ORCA predict(), and forward the resulting "
                "ActionXY world velocity directly into the holonomic vx_vy benchmark action space."
            )
        elif algo_key in {"sonic_crowdnav", "sonic_gst"}:
            adapter_boundary = (
                "Run the upstream SoNIC checkpoint through the model-only Robot SF wrapper and "
                "forward the resulting ActionXY world velocity directly into the holonomic vx_vy "
                "benchmark action space."
            )
        elif algo_key in {"gensafenav_ours_gst", "gensafe_ours_gst", "ours_gst"}:
            adapter_boundary = (
                "Run the upstream GenSafeNav Ours_GST checkpoint through the model-only Robot SF "
                "wrapper and forward the resulting ActionXY world velocity directly into the "
                "holonomic vx_vy benchmark action space."
            )
        elif algo_key in {
            "gensafenav_gst_predictor_rand",
            "gensafe_gst_predictor_rand",
            "gst_predictor_rand",
        }:
            adapter_boundary = (
                "Run the upstream GenSafeNav GST_predictor_rand checkpoint through the model-only "
                "Robot SF wrapper and forward the resulting ActionXY world velocity directly into "
                "the holonomic vx_vy benchmark action space."
            )
        else:
            adapter_boundary = (
                "Map Robot SF SocNav observations into the upstream Social-Navigation-PyEnvs "
                "JointState contract, run the upstream force-model policy predict(), and forward "
                "the resulting ActionXY world velocity directly into the holonomic vx_vy "
                "benchmark action space."
            )
        _apply_direct_world_velocity_metadata(meta, adapter_boundary=adapter_boundary)

        def _policy(obs: dict[str, Any]) -> dict[str, float | str]:
            """Run holonomic upstream ORCA and return world-velocity action payload.

            Returns:
                dict[str, float | str]: Holonomic world-velocity command.
            """
            velocity_world = np.asarray(adapter.plan_velocity_world(obs), dtype=float).reshape(-1)
            if velocity_world.size < 2:
                raise ValueError(
                    "Holonomic ORCA path expected a world-frame velocity with two components."
                )
            return _holonomic_world_velocity_command(
                float(velocity_world[0]),
                float(velocity_world[1]),
            )

        _attach_planner_reset(_policy, adapter)
        _policy._planner_adapter = adapter
        if planner_bind_env is not None:
            _policy._planner_bind_env = planner_bind_env
        return _policy, meta

    def _policy(obs: dict[str, Any]) -> tuple[float, float]:
        """Run a generic SocNav adapter and project command feasibility.

        Returns:
            tuple[float, float]: Projected linear and angular command.
        """
        linear, angular = adapter.plan(obs)
        return _project_with_feasibility(
            model=adapter_kinematics_model,
            command=(float(linear), float(angular)),
            meta=meta,
        )

    _attach_planner_reset(_policy, adapter)
    _policy._planner_adapter = adapter
    if planner_bind_env is not None:
        _policy._planner_bind_env = planner_bind_env
    if hasattr(adapter, "diagnostics"):

        def _planner_stats() -> dict[str, Any]:
            """Expose generic adapter diagnostics for episode metadata.

            Returns:
                dict[str, Any]: Adapter diagnostic payload.
            """
            return adapter.diagnostics()

        _policy._planner_stats = _planner_stats
    return _policy, meta


def _resolve_seed_list(path: Path) -> dict[str, list[int]]:
    """Load named benchmark seed lists from YAML.

    Returns:
        dict[str, list[int]]: Seed lists keyed by suite name.
    """
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [int(s) for s in v] for k, v in data.items() if isinstance(v, list)}


def _suite_key(scenario_path: Path) -> str:
    """Infer the seed-suite key from a scenario config filename.

    Returns:
        str: Suite key used for seed-list lookup.
    """
    stem = scenario_path.stem.lower()
    if "classic" in stem:
        return "classic_interactions"
    if "francis" in stem:
        return "francis2023"
    return "default"


def _select_seeds(
    scenario: dict[str, Any],
    *,
    suite_seeds: dict[str, list[int]],
    suite_key: str,
) -> list[int]:
    """Resolve per-scenario seeds with suite and default fallbacks.

    Returns:
        list[int]: Seeds to run for the scenario.
    """
    seeds = scenario.get("seeds")
    if isinstance(seeds, list) and seeds:
        return [int(s) for s in seeds]
    if suite_seeds.get(suite_key):
        return list(suite_seeds[suite_key])
    if suite_seeds.get("default"):
        return list(suite_seeds["default"])
    return [0]


def _scenario_identity_payload(  # noqa: PLR0913
    scenario: dict[str, Any],
    *,
    algo: str,
    algo_config: dict[str, Any],
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
    observation_mode: str | None = None,
    observation_level: str | None = None,
    benchmark_track: str | None = None,
    track_schema_version: str | None = None,
    observation_noise: dict[str, Any] | None = None,
    synthetic_actuation_profile: dict[str, Any] | None = None,
    latency_stress_profile: dict[str, Any] | None = None,
    record_simulation_step_trace: bool = False,
) -> dict[str, Any]:
    """Build the canonical scenario payload used for episode identity.

    Resume safety relies on using the same identity dimensions at write-time and
    skip-time. For map runs this includes algorithm and run-shaping options.

    Returns:
        dict[str, Any]: Identity payload consumed by ``compute_episode_id``.
    """
    payload = {key: value for key, value in scenario.items() if key not in {"seed", "seeds"}}
    scenario_id = (
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )
    payload.setdefault("id", scenario_id)
    payload["algo"] = str(algo)
    payload["algo_config_hash"] = _config_hash(algo_config)
    payload["record_forces"] = bool(record_forces)
    if observation_mode is not None:
        payload["observation_mode"] = str(observation_mode)
    if observation_level is not None:
        payload["observation_level"] = str(observation_level)
    if benchmark_track is not None:
        payload["benchmark_track"] = str(benchmark_track)
    if track_schema_version is not None:
        payload["track_schema_version"] = str(track_schema_version)
    noise_spec = normalize_observation_noise_spec(observation_noise)
    if bool(noise_spec["enabled"]):
        payload["observation_noise_profile"] = str(noise_spec["profile"])
        payload["observation_noise_hash"] = observation_noise_hash(noise_spec)
    if synthetic_actuation_profile is not None:
        payload["synthetic_actuation_profile"] = dict(synthetic_actuation_profile)
    if latency_stress_profile is not None:
        payload["latency_stress_profile"] = dict(latency_stress_profile)
    payload["record_simulation_step_trace"] = bool(record_simulation_step_trace)
    if horizon is not None and int(horizon) > 0:
        payload["run_horizon"] = int(horizon)
    if dt is not None and float(dt) > 0.0:
        payload["run_dt"] = float(dt)
    return payload


def _compute_map_episode_id(identity_payload: dict[str, Any], seed: int) -> str:
    """Return a map-runner episode id scoped to algorithm + run dimensions.

    The default benchmark ``compute_episode_id`` uses ``<scenario_id>--<seed>``.
    Map-batch resume needs richer scoping for mixed algorithm/config runs.
    """
    scenario_id = (
        identity_payload.get("id")
        or identity_payload.get("name")
        or identity_payload.get("scenario_id")
        or "unknown"
    )
    identity_hash = _config_hash(identity_payload)
    return f"{scenario_id}--{seed}--{identity_hash}"


def _first_float(value: Any, default: float = 0.0) -> float:
    """Return the first finite numeric value from scalar-or-sequence inputs."""

    if isinstance(value, int | float | np.integer | np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
        return _first_float(value[0], default=default)
    return default


def _observation_heading(obs: Any, *, default: float = 0.0) -> float:
    """Extract robot heading from structured or flat observations.

    Returns:
        Heading in radians, or the provided default when unavailable.
    """

    if isinstance(obs, dict):
        robot = obs.get("robot")
        if isinstance(robot, dict) and "heading" in robot:
            return _first_float(robot.get("heading"), default=default)
        if "robot_heading" in obs:
            return _first_float(obs.get("robot_heading"), default=default)
    return default


def _trace_pedestrians(
    positions: np.ndarray,
    previous_positions: np.ndarray | None,
    dt_seconds: float,
    intent_metadata: list[dict[str, Any] | None] | None = None,
    vru_metadata: list[dict[str, Any] | None] | None = None,
    robot_position: np.ndarray | None = None,
    robot_velocity: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Build trace-export pedestrian frames from simulator position buffers.

    Returns:
        Renderer-neutral pedestrian frame entries.
    """

    if positions.size == 0:
        return []
    pedestrians: list[dict[str, Any]] = []
    metadata_count = max(len(intent_metadata or []), len(vru_metadata or []))
    single_offset = (
        max(0, int(np.asarray(positions).shape[0]) - metadata_count) if metadata_count else None
    )
    for ped_idx, ped_pos in enumerate(np.asarray(positions, dtype=float)):
        if (
            previous_positions is not None
            and previous_positions.shape == positions.shape
            and dt_seconds > 0.0
        ):
            velocity = (ped_pos - previous_positions[ped_idx]) / dt_seconds
        else:
            velocity = np.zeros(2, dtype=float)
        frame = {
            "id": int(ped_idx),
            "position": [float(ped_pos[0]), float(ped_pos[1])],
            "velocity": [float(velocity[0]), float(velocity[1])],
        }
        if single_offset is not None and ped_idx >= single_offset:
            single_idx = ped_idx - single_offset
            if 0 <= single_idx < len(intent_metadata or []):
                metadata = (intent_metadata or [])[single_idx]
                if metadata is not None:
                    frame.update(_intent_trace_payload(metadata, velocity))
            if 0 <= single_idx < len(vru_metadata or []):
                metadata = (vru_metadata or [])[single_idx]
                if metadata is not None:
                    frame.update(
                        _vru_trace_payload(
                            metadata,
                            ped_pos=ped_pos,
                            velocity=velocity,
                            robot_position=robot_position,
                            robot_velocity=robot_velocity,
                        )
                    )
        pedestrians.append(frame)
    return pedestrians


def _single_pedestrian_intent_metadata(scenario: dict[str, Any]) -> list[dict[str, Any] | None]:
    """Return authored intent metadata aligned with scenario single pedestrians."""
    scenario_metadata = (
        scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    )
    scenario_intent = (
        scenario_metadata.get("intent_conditioned_behavior")
        if isinstance(scenario_metadata.get("intent_conditioned_behavior"), dict)
        else {}
    )
    scenario_signal_state = (
        scenario_metadata.get("signal_state")
        if isinstance(scenario_metadata.get("signal_state"), dict)
        else None
    )
    single_peds = (
        scenario.get("single_pedestrians")
        if isinstance(scenario.get("single_pedestrians"), list)
        else []
    )
    result: list[dict[str, Any] | None] = []
    has_intent_metadata = False
    for idx, ped in enumerate(single_peds):
        if not isinstance(ped, dict):
            result.append(None)
            continue
        ped_metadata = ped.get("metadata") if isinstance(ped.get("metadata"), dict) else {}
        intent = (
            ped_metadata.get("intent_conditioned_behavior")
            if isinstance(ped_metadata.get("intent_conditioned_behavior"), dict)
            else {}
        )
        if not intent and not scenario_intent:
            result.append(None)
            continue
        wait_at = ped.get("wait_at") if isinstance(ped.get("wait_at"), list) else []
        trajectory = ped.get("trajectory") if isinstance(ped.get("trajectory"), list) else []
        phases = intent.get("intent_phases")
        if not isinstance(phases, list) or not phases:
            phases = ["waiting", "crossing"] if wait_at and trajectory else ["authored_motion"]
        label = str(
            intent.get("intent_label")
            or ("waiting_then_crossing" if wait_at and trajectory else "authored_single_pedestrian")
        )
        wait_intervals = [
            float(rule.get("wait_s"))
            for rule in wait_at
            if isinstance(rule, dict) and rule.get("wait_s") is not None
        ]
        has_intent_metadata = True
        result.append(
            {
                "single_index": idx,
                "pedestrian_id": str(ped.get("id") or f"single_{idx}"),
                "intent_label": label,
                "intent_phases": [str(phase) for phase in phases],
                "intent_source": str(intent.get("intent_source") or "authored_scenario_metadata"),
                "claim_boundary": str(
                    intent.get("claim_boundary")
                    or "Authored scenario metadata only; not data-grounded human intent evidence."
                ),
                "behavior_parameters": {
                    "trajectory_waypoint_count": len(trajectory),
                    "wait_at": wait_at,
                    "wait_interval_s": wait_intervals,
                    "start_delay_s": float(ped.get("start_delay_s", 0.0) or 0.0),
                    "speed_m_s": (
                        float(ped["speed_m_s"]) if ped.get("speed_m_s") is not None else None
                    ),
                    "role": ped.get("role"),
                    "role_target_id": ped.get("role_target_id"),
                },
                **({"signal_state": scenario_signal_state} if scenario_signal_state else {}),
            }
        )
    return result if has_intent_metadata else []


def _single_pedestrian_vru_metadata(scenario: dict[str, Any]) -> list[dict[str, Any] | None]:
    """Return authored fast-VRU metadata aligned with scenario single pedestrians."""
    single_peds = (
        scenario.get("single_pedestrians")
        if isinstance(scenario.get("single_pedestrians"), list)
        else []
    )
    result: list[dict[str, Any] | None] = []
    has_vru_metadata = False
    for idx, ped in enumerate(single_peds):
        if not isinstance(ped, dict):
            result.append(None)
            continue
        ped_metadata = ped.get("metadata") if isinstance(ped.get("metadata"), dict) else {}
        payload_key = "cyclist_like_vru"
        vru = ped_metadata.get("cyclist_like_vru")
        if not isinstance(vru, dict):
            fast_bicycle = ped_metadata.get("fast_bicycle_actor")
            if isinstance(fast_bicycle, dict):
                payload_key = "fast_bicycle_actor"
                vru = fast_bicycle
            else:
                vru = {}
        if not vru:
            result.append(None)
            continue
        speed_m_s = _metadata_float(vru, "speed_m_s", default=ped.get("speed_m_s"))
        acceleration_m_s2 = _metadata_float(vru, "acceleration_m_s2", default=0.0)
        actor_radius_m = _metadata_float(vru, "actor_radius_m", default=0.35)
        robot_radius_m = _metadata_float(vru, "robot_radius_m", default=0.3)
        has_vru_metadata = True
        result.append(
            {
                "single_index": idx,
                "pedestrian_id": str(ped.get("id") or f"single_{idx}"),
                "actor_type": str(
                    vru.get("actor_type")
                    or ("bicycle" if payload_key == "fast_bicycle_actor" else "cyclist_like_vru")
                ),
                "diagnostic_payload_key": payload_key,
                "speed_m_s": speed_m_s,
                "acceleration_m_s2": acceleration_m_s2,
                "actor_radius_m": actor_radius_m,
                "robot_radius_m": robot_radius_m,
                "interaction_role": str(vru.get("interaction_role") or "fast_moving_vru"),
                "diagnostic_metric_subset": [
                    str(item)
                    for item in (
                        vru.get("diagnostic_metric_subset")
                        if isinstance(vru.get("diagnostic_metric_subset"), list)
                        else [
                            "time_to_conflict_zone_s",
                            "clearance_m",
                            "pass_overtake_state",
                        ]
                    )
                ],
                "claim_boundary": str(
                    vru.get("claim_boundary")
                    or "Authored fast-VRU proxy metadata only; not cyclist realism or "
                    "planner-ranking benchmark evidence."
                ),
            }
        )
    return result if has_vru_metadata else []


def _metadata_float(mapping: Mapping[str, Any], key: str, *, default: Any) -> float:
    """Return a finite metadata float or a finite default."""
    value = mapping.get(key, default)
    try:
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    except (TypeError, ValueError):
        pass
    try:
        fallback = float(default) if default is not None else 0.0
    except (TypeError, ValueError):
        fallback = 0.0
    return fallback if math.isfinite(fallback) else 0.0


def _intent_trace_payload(metadata: dict[str, Any], velocity: np.ndarray) -> dict[str, Any]:
    """Build optional per-pedestrian authored-intent trace fields.

    Returns:
        dict[str, Any]: JSON-serializable intent fields for one trace pedestrian.
    """
    phases = (
        metadata.get("intent_phases") if isinstance(metadata.get("intent_phases"), list) else []
    )
    speed = float(np.linalg.norm(velocity))
    if "waiting" in phases and speed <= 1e-6:
        phase = "waiting"
    elif "crossing" in phases:
        phase = "crossing"
    elif phases:
        phase = str(phases[0])
    else:
        phase = "authored_motion"
    payload = {
        "pedestrian_id": metadata["pedestrian_id"],
        "intent_label": metadata["intent_label"],
        "intent_phase": phase,
        "intent_source": metadata["intent_source"],
        "claim_boundary": metadata["claim_boundary"],
        "behavior_parameters": metadata["behavior_parameters"],
    }
    proxy_payload = _signal_state_proxy_wrapper(
        metadata.get("signal_state"),
        phase,
        metadata["intent_label"],
        metadata["intent_source"],
    )
    if proxy_payload is not None:
        payload["signal_state"] = proxy_payload
    return payload


def _signal_state_trace_payload(signal_state: Any, intent_phase: str) -> dict[str, Any] | None:
    """Return proxy signal-state trace metadata for the current authored intent phase."""
    if not isinstance(signal_state, dict):
        return None
    phase_timeline = (
        signal_state.get("phase_timeline")
        if isinstance(signal_state.get("phase_timeline"), list)
        else []
    )
    matching_phase = next(
        (
            phase
            for phase in phase_timeline
            if isinstance(phase, dict) and phase.get("intent_phase") == intent_phase
        ),
        None,
    )
    if matching_phase is None:
        matching_phase = next(
            (phase for phase in phase_timeline if isinstance(phase, dict)),
            {},
        )
    return {
        "schema_version": str(signal_state.get("schema_version") or "signal-state-proxy.v1"),
        "status": str(signal_state.get("status") or "proxy_diagnostic_only"),
        "signal_id": str(signal_state.get("signal_id") or "unknown_signal"),
        "conflict_zone_id": str(signal_state.get("conflict_zone_id") or "unknown_conflict_zone"),
        "phase": str(matching_phase.get("phase") or "unknown"),
        "intent_phase": intent_phase,
        "robot_right_of_way": bool(matching_phase.get("robot_right_of_way", False)),
        "pedestrian_right_of_way": bool(matching_phase.get("pedestrian_right_of_way", False)),
        "legality_state": str(matching_phase.get("legality_state") or "unknown"),
        "planner_observable": bool(signal_state.get("planner_observable", False)),
        "observation_mode": str(signal_state.get("observation_mode") or "trace_metadata_only"),
        "benchmark_evidence": bool(signal_state.get("benchmark_evidence", False)),
        "claim_boundary": str(
            signal_state.get("claim_boundary")
            or "Proxy signal-state metadata only; not benchmark evidence."
        ),
    }


_PROXY_TRACE_FIELDS_PRESENT = [
    "signal_phase",
    "pedestrian_intent",
    "robot_stop_or_yield_expectation",
    "claim_boundary",
    "trace_fields_present",
    "signal_state",
]

_SIGNAL_STATE_OBSERVABLE_SCHEMA = "signal-state-observable.v1"
_SIGNAL_STATE_OBSERVABLE_STATUS = "planner_observable_signal_state"
_SIGNAL_STATE_OBSERVABLE_MODE = "planner_observable"
_SIGNAL_STATE_OBSERVABLE_FIELDS = [
    "signal_id",
    "conflict_zone_id",
    "phase",
    "phase_elapsed_s",
    "phase_remaining_s",
    "robot_right_of_way",
    "pedestrian_right_of_way",
    "legality_state",
]
_SIGNAL_STATE_RECORDED_ONLY_FIELDS = [
    "schema_version",
    "status",
    "signal_id",
    "conflict_zone_id",
    "phase",
    "intent_phase",
    "robot_right_of_way",
    "pedestrian_right_of_way",
    "legality_state",
    "planner_observable",
    "observation_mode",
    "benchmark_evidence",
    "claim_boundary",
]


def _signal_state_promotion_contract(signal_state: Any) -> dict[str, Any]:
    """Classify signal-state metadata for benchmark promotion decisions.

    Returns:
        dict[str, Any]: Contract state plus planner-consumed and recorded-only fields.
    """
    if not isinstance(signal_state, dict):
        return {
            "contract_state": "unavailable",
            "planner_consumed_fields": [],
            "recorded_only_fields": [],
            "promotion_required_fields": list(_SIGNAL_STATE_OBSERVABLE_FIELDS),
            "fail_closed_reason": "signal_state_metadata_absent",
            "benchmark_evidence": False,
        }

    schema_version = str(signal_state.get("schema_version") or "")
    status = str(signal_state.get("status") or "")
    observation_mode = str(signal_state.get("observation_mode") or "")
    planner_observable = bool(signal_state.get("planner_observable", False))
    benchmark_evidence = bool(signal_state.get("benchmark_evidence", False))
    is_observable = (
        schema_version == _SIGNAL_STATE_OBSERVABLE_SCHEMA
        and status == _SIGNAL_STATE_OBSERVABLE_STATUS
        and observation_mode == _SIGNAL_STATE_OBSERVABLE_MODE
        and planner_observable
        and benchmark_evidence
    )
    if is_observable:
        return {
            "contract_state": "planner_observable",
            "planner_consumed_fields": list(_SIGNAL_STATE_OBSERVABLE_FIELDS),
            "recorded_only_fields": [],
            "promotion_required_fields": [],
            "fail_closed_reason": "",
            "benchmark_evidence": True,
        }

    return {
        "contract_state": "proxy_diagnostic",
        "planner_consumed_fields": [],
        "recorded_only_fields": list(_SIGNAL_STATE_RECORDED_ONLY_FIELDS),
        "promotion_required_fields": list(_SIGNAL_STATE_OBSERVABLE_FIELDS),
        "fail_closed_reason": (
            "signal_state_proxy_or_synthetic_not_planner_observable; "
            "do_not_count_as_signalized_benchmark_evidence"
        ),
        "benchmark_evidence": False,
    }


def _signal_state_for_metric_metadata(signal_state: Any) -> dict[str, Any] | None:
    """Return fail-closed signal-state metadata for runtime metric computation.

    Proxy signal metadata may be useful for trace diagnostics, but signal metrics may only enter
    denominators when the explicit planner-observable benchmark contract is met. Observable rows
    still need metric geometry fields; missing fields stay fail-closed in ``signal_metrics.py``.
    """
    if not isinstance(signal_state, dict):
        return None
    contract = _signal_state_promotion_contract(signal_state)
    if contract["contract_state"] != "planner_observable":
        return {
            "contract_state": contract["contract_state"],
            "benchmark_evidence": False,
        }
    metric_state: dict[str, Any] = {
        "contract_state": "planner_observable",
        "benchmark_evidence": True,
    }
    for key in ("timeline", "stop_line", "crosswalk_polygon"):
        if key in signal_state:
            metric_state[key] = signal_state[key]
    return metric_state


def _episode_metadata_for_signal_metrics(scenario: dict[str, Any]) -> dict[str, Any] | None:
    """Build optional episode metadata consumed by signal metrics.

    Returns:
        Optional episode metadata when the scenario carries usable signal-state evidence.
    """
    metadata = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    signal_state = metadata.get("signal_state") if isinstance(metadata, dict) else None
    metric_signal_state = _signal_state_for_metric_metadata(signal_state)
    if metric_signal_state is None:
        return None
    return {"signal_state": metric_signal_state}


def _synth_robot_stop_or_yield_expectation(
    robot_right_of_way: bool,
    pedestrian_right_of_way: bool,
    legality_state: str,
) -> str:
    """Synthesize a robot stop-or-yield expectation from proxy right-of-way fields.

    Returns:
        str: Diagnostic expectation label for trace/summary metadata.
    """
    if legality_state == "pedestrian_wait_required" and robot_right_of_way:
        return "proceed_clear"
    if legality_state == "pedestrian_crossing_allowed" and pedestrian_right_of_way:
        return "yield_to_pedestrian"
    if pedestrian_right_of_way:
        return "yield_to_pedestrian"
    return "proceed_clear"


def _signal_state_proxy_wrapper(
    signal_state: Any,
    intent_phase: str,
    intent_label: str,
    intent_source: str,
) -> dict[str, Any] | None:
    """Wrap signal-state proxy with bounded diagnostic fields for trace/summary export.

    This wrapper adds pedestrian_intent, robot_stop_or_yield_expectation,
    trace_fields_present, and claim_boundary=proxy_diagnostic on top of the
    existing signal-state trace payload. It does not add runtime simulation
    behavior or planner-observable signal-phase semantics.

    Returns:
        dict[str, Any] | None: Proxy diagnostic payload, or None when the
        input signal_state is absent.
    """
    base = _signal_state_trace_payload(signal_state, intent_phase)
    if base is None:
        return None

    pedestrian_intent = {
        "intent_label": intent_label,
        "intent_phase": intent_phase,
        "intent_source": intent_source,
    }
    robot_stop_or_yield_expectation = _synth_robot_stop_or_yield_expectation(
        base["robot_right_of_way"],
        base["pedestrian_right_of_way"],
        base["legality_state"],
    )
    return {
        **base,
        **_signal_state_promotion_contract(signal_state),
        "signal_phase": base["phase"],
        "pedestrian_intent": pedestrian_intent,
        "robot_stop_or_yield_expectation": robot_stop_or_yield_expectation,
        "trace_fields_present": list(_PROXY_TRACE_FIELDS_PRESENT),
        "claim_boundary": "proxy_diagnostic",
    }


def _vru_trace_payload(
    metadata: dict[str, Any],
    *,
    ped_pos: np.ndarray,
    velocity: np.ndarray,
    robot_position: np.ndarray | None,
    robot_velocity: np.ndarray | None,
) -> dict[str, Any]:
    """Build optional per-pedestrian cyclist-like VRU diagnostic trace fields.

    Returns:
        dict[str, Any]: JSON-serializable cyclist-like VRU diagnostic fields.
    """
    speed = float(np.linalg.norm(velocity))
    diagnostics: dict[str, Any] = {
        "speed_m_s": speed,
        "configured_speed_m_s": float(metadata["speed_m_s"]),
        "acceleration_m_s2": float(metadata["acceleration_m_s2"]),
    }
    if robot_position is not None:
        robot_pos = np.asarray(robot_position, dtype=float)
        robot_vel = (
            np.asarray(robot_velocity, dtype=float)
            if robot_velocity is not None
            else np.zeros(2, dtype=float)
        )
        offset = np.asarray(ped_pos, dtype=float) - robot_pos
        distance = float(np.linalg.norm(offset))
        relative_velocity = np.asarray(velocity, dtype=float) - robot_vel
        closing_speed = 0.0
        if distance > 1e-9:
            closing_speed = -float(np.dot(offset, relative_velocity) / distance)
        clearance = distance - float(metadata["actor_radius_m"]) - float(metadata["robot_radius_m"])
        diagnostics.update(
            {
                "distance_to_robot_m": distance,
                "relative_closing_speed_m_s": closing_speed,
                "time_to_conflict_zone_s": (
                    float(distance / closing_speed) if closing_speed > 1e-9 else None
                ),
                "clearance_m": clearance,
                "pass_overtake_state": _pass_overtake_state(closing_speed),
            }
        )
    payload_key = str(metadata.get("diagnostic_payload_key") or "cyclist_like_vru")
    return {
        "pedestrian_id": metadata["pedestrian_id"],
        "actor_type": metadata["actor_type"],
        "interaction_role": metadata["interaction_role"],
        "claim_boundary": metadata["claim_boundary"],
        payload_key: diagnostics,
    }


def _pass_overtake_state(closing_speed_m_s: float) -> str:
    """Classify a one-step proxy pass/overtake state from relative closing speed.

    Returns:
        str: Diagnostic state for the proxy pass/overtake interaction.
    """
    if closing_speed_m_s > 0.1:
        return "approaching_conflict_zone"
    if closing_speed_m_s < -0.1:
        return "separating_after_pass"
    return "parallel_or_static_relative_motion"


def _intent_conditioned_behavior_summary(
    scenario: dict[str, Any],
    intent_metadata: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Return an analysis-only summary for authored intent-conditioned fixtures."""
    if not intent_metadata:
        return None
    summarized_pedestrians = [metadata for metadata in intent_metadata if metadata is not None]
    if not summarized_pedestrians:
        return None
    summary = {
        "schema_version": "intent-conditioned-behavior-summary.v1",
        "scenario_name": _scenario_id(scenario),
        "status": "diagnostic_metadata_only",
        "benchmark_evidence": False,
        "trace_field_source": "algorithm_metadata.simulation_step_trace.steps[].pedestrians[]",
        "claim_boundary": (
            "Authored intent metadata records fixture phases only; it is not data-grounded "
            "human behavior evidence and must not be used as a planner-ranking claim."
        ),
        "pedestrians": summarized_pedestrians,
    }
    signal_state = next(
        (
            metadata.get("signal_state")
            for metadata in summarized_pedestrians
            if isinstance(metadata.get("signal_state"), dict)
        ),
        None,
    )
    if isinstance(signal_state, dict):
        first_intent = next(
            (
                metadata
                for metadata in summarized_pedestrians
                if isinstance(metadata, dict) and metadata.get("intent_label")
            ),
            {},
        )
        phases = first_intent.get("intent_phases")
        first_phase = str(phases[0]) if isinstance(phases, list) and phases else "unknown"
        proxy_payload = _signal_state_proxy_wrapper(
            signal_state,
            first_phase,
            str(first_intent.get("intent_label", "unknown")),
            str(first_intent.get("intent_source", "unknown")),
        )
        summary["signal_state"] = (
            proxy_payload
            if proxy_payload is not None
            else {
                "schema_version": str(
                    signal_state.get("schema_version") or "signal-state-proxy.v1"
                ),
                "status": str(signal_state.get("status") or "proxy_diagnostic_only"),
                "signal_id": str(signal_state.get("signal_id") or "unknown_signal"),
                "conflict_zone_id": str(
                    signal_state.get("conflict_zone_id") or "unknown_conflict_zone"
                ),
                "planner_observable": bool(signal_state.get("planner_observable", False)),
                "observation_mode": str(
                    signal_state.get("observation_mode") or "trace_metadata_only"
                ),
                "benchmark_evidence": bool(signal_state.get("benchmark_evidence", False)),
                "claim_boundary": "proxy_diagnostic",
            }
        )
    return summary


def _cyclist_like_vru_summary(
    scenario: dict[str, Any],
    vru_metadata: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Return an analysis-only summary for authored cyclist-like VRU fixtures."""
    return _vru_diagnostic_summary(
        scenario,
        vru_metadata,
        payload_key="cyclist_like_vru",
        schema_version="cyclist-like-vru-smoke-summary.v1",
        claim_boundary=(
            "Authored cyclist-like VRU proxy metadata records speed, acceleration, "
            "time-to-conflict, clearance, and pass/overtake diagnostics only; it is not "
            "cyclist realism, cyclist behavior, or planner-ranking evidence."
        ),
    )


def _fast_bicycle_actor_summary(
    scenario: dict[str, Any],
    vru_metadata: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Return an analysis-only summary for authored fast-bicycle actor fixtures."""
    return _vru_diagnostic_summary(
        scenario,
        vru_metadata,
        payload_key="fast_bicycle_actor",
        schema_version="fast-bicycle-actor-summary.v1",
        claim_boundary=(
            "Authored fast-bicycle actor proxy metadata records speed, acceleration, "
            "time-to-conflict, clearance, and pass/overtake diagnostics only; it is not "
            "a full bicycle dynamics model, cyclist realism evidence, or planner-ranking evidence."
        ),
    )


def _vru_diagnostic_summary(
    scenario: dict[str, Any],
    vru_metadata: list[dict[str, Any] | None],
    *,
    payload_key: str,
    schema_version: str,
    claim_boundary: str,
) -> dict[str, Any] | None:
    """Return an analysis-only summary for authored fast-VRU fixtures."""
    if not vru_metadata:
        return None
    summarized = [
        metadata
        for metadata in vru_metadata
        if metadata is not None and metadata.get("diagnostic_payload_key") == payload_key
    ]
    if not summarized:
        return None
    return {
        "schema_version": schema_version,
        "scenario_name": _scenario_id(scenario),
        "status": "diagnostic_metadata_only",
        "benchmark_evidence": False,
        "trace_field_source": "algorithm_metadata.simulation_step_trace.steps[].pedestrians[]",
        "claim_boundary": claim_boundary,
        "pedestrians": summarized,
    }


def _command_action_payload(command: Any) -> dict[str, float]:
    """Normalize planner commands to trace-export selected_action fields.

    Returns:
        Linear and angular velocity action fields.
    """

    if isinstance(command, np.ndarray):
        command = command.tolist()
    if isinstance(command, (list, tuple)) and len(command) >= 2:
        return {
            "linear_velocity": _first_float(command[0]),
            "angular_velocity": _first_float(command[1]),
        }
    return {"linear_velocity": 0.0, "angular_velocity": 0.0}


def _validate_behavior_sanity(scenario: dict[str, Any]) -> list[str]:
    """Check behavior metadata has the fields needed by pedestrian definitions.

    Returns:
        list[str]: Non-fatal behavior sanity errors.
    """
    errors: list[str] = []
    meta = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    behavior = str(meta.get("behavior") or "").strip().lower()
    single_peds = (
        scenario.get("single_pedestrians")
        if isinstance(scenario.get("single_pedestrians"), list)
        else []
    )

    if behavior in {"wait", "join", "leave", "follow", "lead", "accompany"}:
        if not single_peds:
            errors.append("behavior requires single_pedestrians entries")
        else:
            for ped in single_peds:
                if not isinstance(ped, dict):
                    continue
                role = str(ped.get("role") or "").strip().lower()
                if behavior == "wait" and not ped.get("wait_at"):
                    errors.append("wait behavior requires wait_at rules")
                    break
                if behavior in {"join", "leave", "follow", "lead", "accompany"} and not role:
                    errors.append("role behavior requires role field")
                    break

    return errors


def _build_env_config(
    scenario: dict[str, Any],
    *,
    scenario_path: Path,
) -> RobotSimulationConfig:
    """Build the benchmark environment config for one scenario.

    Returns:
        RobotSimulationConfig: Config with SocNav structured observations and grid enabled.
    """
    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    config.observation_mode = ObservationMode.SOCNAV_STRUCT
    config.use_occupancy_grid = True
    config.include_grid_in_observation = True
    config.grid_config = GridConfig(
        # Benchmark default upgraded to higher-resolution occupancy grids.
        resolution=0.2,
        width=32.0,
        height=32.0,
        channels=[
            GridChannel.OBSTACLES,
            GridChannel.PEDESTRIANS,
            GridChannel.COMBINED,
        ],
        use_ego_frame=True,
        center_on_robot=True,
    )
    return config


def _apply_active_observation_mode_to_env_config(
    config: RobotSimulationConfig,
    *,
    active_observation_mode: str,
) -> None:
    """Apply planner observation-mode requirements to the runtime environment config."""
    if active_observation_mode != "sensor_fusion_state":
        return
    config.observation_mode = ObservationMode.DEFAULT_GYM
    config.use_occupancy_grid = False
    config.include_grid_in_observation = False
    config.grid_config = None


_POLICY_ENV_OBSERVATION_OVERRIDE_KEYS = frozenset(
    {
        "predictive_foresight_enabled",
        "predictive_foresight_model_id",
        "predictive_foresight_checkpoint_path",
        "predictive_foresight_device",
        "predictive_foresight_max_agents",
        "predictive_foresight_horizon_steps",
        "predictive_foresight_rollout_dt",
        "predictive_foresight_ego_conditioning",
        "predictive_foresight_near_distance",
        "predictive_foresight_front_corridor_length",
        "predictive_foresight_front_corridor_half_width",
    }
)

_STATIC_DEADLOCK_SUITE_ID = "static_deadlock_recovery"
_STATIC_DEADLOCK_LOW_PROGRESS_WINDOW_STEPS = 10
_STATIC_DEADLOCK_LOW_PROGRESS_THRESHOLD_M = 0.05


def _is_static_deadlock_suite(scenario: Mapping[str, Any]) -> bool:
    """Return whether a scenario row belongs to the static-deadlock mechanism suite."""
    metadata = scenario.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    return str(metadata.get("mechanism_aware_suite_id", "")) == _STATIC_DEADLOCK_SUITE_ID


def _finite_or_none(value: float) -> float | None:
    """Return a JSON-friendly float when finite."""
    value = float(value)
    return value if math.isfinite(value) else None


def _mechanism_row_status(termination_reason: str) -> str:
    """Classify an episode row for mechanism-suite reportability accounting.

    Returns:
        ``"completed"`` for valid terminal rows, or ``"failed"`` for error rows.
    """
    return "failed" if str(termination_reason).strip().lower() == "error" else "completed"


def _recenter_activation_count(planner_decision_trace: list[dict[str, Any]]) -> int:
    """Count planner-decision steps whose static-recenter term was active.

    Returns:
        Number of recorded decision steps with a positive static-recenter term.
    """
    count = 0
    for step in planner_decision_trace:
        value = step.get("static_recenter")
        if isinstance(value, (int, float, np.integer, np.floating)) and float(value) > 0.0:
            count += 1
    return count


def _static_deadlock_trace_fields(
    scenario: Mapping[str, Any],
    *,
    robot_pos_arr: np.ndarray,
    goal_vec: np.ndarray,
    initial_goal_distance: float,
    termination_reason: str,
    outcome: Mapping[str, bool],
    planner_decision_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build static-deadlock mechanism trace fields for suite-tagged episode rows.

    Returns:
        Top-level episode row fields required by the static-deadlock suite, or an empty payload for
        unrelated scenarios.
    """
    if not _is_static_deadlock_suite(scenario):
        return {}

    if robot_pos_arr.size:
        distances = np.linalg.norm(robot_pos_arr - goal_vec, axis=1)
        distance_series = [float(initial_goal_distance), *[float(value) for value in distances]]
    else:
        distance_series = [float(initial_goal_distance)]

    sample_count = max(0, len(distance_series) - 1)
    window_steps = min(_STATIC_DEADLOCK_LOW_PROGRESS_WINDOW_STEPS, sample_count)
    window_start_idx = max(0, len(distance_series) - 1 - window_steps)
    window_start_distance = distance_series[window_start_idx]
    final_distance = distance_series[-1]
    window_progress_delta = window_start_distance - final_distance
    total_progress_delta = float(initial_goal_distance) - final_distance
    route_complete = bool(outcome.get("route_complete", False))
    collision_event = bool(outcome.get("collision_event", False))
    timeout_event = bool(outcome.get("timeout_event", False))
    low_progress_active = (
        sample_count > 0
        and not route_complete
        and window_progress_delta <= _STATIC_DEADLOCK_LOW_PROGRESS_THRESHOLD_M
    )
    local_minimum = bool(low_progress_active and timeout_event and not collision_event)

    return {
        "low_progress_window": {
            "schema_version": "static-deadlock-low-progress-window.v1",
            "window_steps": int(window_steps),
            "sample_count": int(sample_count),
            "start_distance_to_goal_m": _finite_or_none(window_start_distance),
            "end_distance_to_goal_m": _finite_or_none(final_distance),
            "progress_delta_m": _finite_or_none(window_progress_delta),
            "threshold_m": float(_STATIC_DEADLOCK_LOW_PROGRESS_THRESHOLD_M),
            "active": bool(low_progress_active),
        },
        "recenter_activation_count": int(_recenter_activation_count(planner_decision_trace)),
        "distance_to_goal_delta": {
            "schema_version": "static-deadlock-distance-to-goal-delta.v1",
            "initial_distance_to_goal_m": _finite_or_none(initial_goal_distance),
            "final_distance_to_goal_m": _finite_or_none(final_distance),
            "delta_m": _finite_or_none(total_progress_delta),
            "interpretation": "positive values indicate progress toward the goal",
        },
        "local_minimum_indicator": {
            "schema_version": "static-deadlock-local-minimum-indicator.v1",
            "is_local_minimum": local_minimum,
            "status": "detected" if local_minimum else "not_detected",
            "reason": (
                "timeout with low progress and no collision"
                if local_minimum
                else "route completed, collision occurred, or low-progress timeout was not observed"
            ),
        },
        "row_status": _mechanism_row_status(termination_reason),
    }


def _apply_policy_env_observation_overrides(
    config: RobotSimulationConfig,
    policy_cfg: Mapping[str, Any],
) -> None:
    """Apply candidate observation-contract env overrides before env construction."""

    raw_overrides = policy_cfg.get("env_overrides")
    overrides = raw_overrides if isinstance(raw_overrides, Mapping) else {}
    for key in _POLICY_ENV_OBSERVATION_OVERRIDE_KEYS:
        if key in overrides:
            setattr(config, key, overrides[key])


def _validate_sensor_fusion_adapter_config(
    *,
    algo: str,
    active_observation_mode: str,
    algo_config: dict[str, Any],
) -> None:
    """Fail closed when a planner requests sensor-fusion input without an adapter."""
    if active_observation_mode != "sensor_fusion_state":
        return
    algo_key = str(algo).strip().lower()
    if algo_key == "safety_barrier" and not algo_config.get("lidar_occupancy_adapter"):
        raise ValueError(
            "safety_barrier with sensor_fusion_state/lidar_2d requires "
            "algo_config['lidar_occupancy_adapter']."
        )


def _robot_kinematics_label(config: RobotSimulationConfig) -> str:
    """Derive the runtime robot kinematics label from simulation config.

    Returns:
        Canonical kinematics label used in benchmark metadata.
    """
    robot_cfg = getattr(config, "robot_config", None)
    if robot_cfg is None:
        return _DEFAULT_KINEMATICS
    cls_name = robot_cfg.__class__.__name__.lower()
    if "bicycle" in cls_name:
        return "bicycle_drive"
    if "differential" in cls_name:
        return "differential_drive"
    if "holonomic" in cls_name or "omni" in cls_name:
        return "holonomic"
    return cls_name or _DEFAULT_KINEMATICS


def _robot_max_speed(config: RobotSimulationConfig) -> float | None:
    """Extract a positive robot max-speed setting from simulation config if available.

    Returns:
        Configured positive max speed, or ``None`` when not available.
    """
    robot_cfg = getattr(config, "robot_config", None)
    if robot_cfg is None:
        return None
    for attr in ("max_linear_speed", "max_velocity", "max_speed"):
        value = getattr(robot_cfg, attr, None)
        if isinstance(value, (int, float)) and float(value) > 0:
            return float(value)
    return None


def _scenario_robot_kinematics_label(scenario: dict[str, Any]) -> str:
    """Derive the scenario-declared robot kinematics label from scenario metadata.

    Returns:
        Canonical kinematics label inferred from scenario robot configuration fields.
    """
    robot_cfg = scenario.get("robot_config")
    if not isinstance(robot_cfg, dict):
        return _DEFAULT_KINEMATICS
    raw = str(robot_cfg.get("type") or robot_cfg.get("model") or "").strip().lower()
    if "bicycle" in raw:
        return "bicycle_drive"
    if "holonomic" in raw or "omni" in raw:
        return "holonomic"
    if "differential" in raw or raw == "":
        return _DEFAULT_KINEMATICS
    return raw


def _vel_and_acc(positions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite-difference velocity and acceleration arrays.

    Returns:
        tuple[np.ndarray, np.ndarray]: Velocity and acceleration with input shape.
    """
    if positions.shape[0] < 2:
        return np.zeros_like(positions), np.zeros_like(positions)
    vel = np.gradient(positions, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def _stack_ped_positions(traj: list[np.ndarray], *, fill_value: float = np.nan) -> np.ndarray:
    """Stack variable-count pedestrian position arrays into one padded tensor.

    Returns:
        np.ndarray: Array shaped ``(time, max_pedestrians, 2)``.
    """
    if not traj:
        return np.zeros((0, 0, 2), dtype=float)
    first_shape = traj[0].shape
    if all(arr.shape == first_shape for arr in traj):
        return np.stack(traj).astype(float, copy=False)
    max_k = max(p.shape[0] for p in traj)
    stacked = np.full((len(traj), max_k, 2), fill_value, dtype=float)
    for i, arr in enumerate(traj):
        if arr.size == 0:
            continue
        stacked[i, : arr.shape[0]] = arr
    return stacked


def _command_xy_payload(command: tuple[float, float] | dict[str, Any]) -> np.ndarray:
    """Extract a world-frame XY payload from tuple or structured commands.

    Returns:
        np.ndarray: Two-element world-frame XY payload.
    """
    if isinstance(command, dict):
        return np.array(
            [float(command.get("vx", 0.0)), float(command.get("vy", 0.0))],
            dtype=float,
        )
    return np.array([float(command[0]), float(command[1])], dtype=float)


def _policy_command_to_env_action(  # noqa: C901
    *,
    env: Any,
    config: RobotSimulationConfig,
    command: tuple[float, float] | dict[str, Any],
) -> np.ndarray:
    """Convert a policy command into the robot's native environment action space.

    Returns:
        np.ndarray: Action vector compatible with ``env.step``.
    """
    simulator = getattr(env, "simulator", None)
    sim_robots = getattr(simulator, "robots", None)
    if not isinstance(sim_robots, list) or not sim_robots:
        return _command_xy_payload(command)
    robot = sim_robots[0]
    robot_cfg = getattr(config, "robot_config", None)
    if robot_cfg is None:
        return _command_xy_payload(command)

    if isinstance(command, dict):
        command_kind = str(command.get("command_kind", "")).strip().lower()
        if command_kind != "holonomic_vxy_world":
            raise ValueError(f"Unsupported structured policy command: {command}")
        velocity_world = np.array(
            [float(command.get("vx", 0.0)), float(command.get("vy", 0.0))],
            dtype=float,
        )
        max_linear_speed = float(
            getattr(robot_cfg, "max_linear_speed", getattr(robot_cfg, "max_speed", 0.0)) or 0.0
        )
        max_angular_speed = float(getattr(robot_cfg, "max_angular_speed", 0.0) or 0.0)

    cls_name = robot_cfg.__class__.__name__.lower()
    if isinstance(command, dict):
        if "holonomic" in cls_name:
            mode = str(getattr(robot_cfg, "command_mode", "vx_vy")).strip().lower()
            if mode == "vx_vy":
                return velocity_world
            command_vw = holonomic_to_diff_drive_action(
                velocity_world,
                robot.pose,
                max_linear_speed=max_linear_speed,
                max_angular_speed=max_angular_speed,
            )
            return np.asarray(command_vw, dtype=float)

        command_vw = holonomic_to_diff_drive_action(
            velocity_world,
            robot.pose,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
        )
        if "bicycle" in cls_name:
            adapter = PlannerActionAdapter(
                robot=robot,
                action_space=env.action_space,
                time_step=float(config.sim_config.time_per_step_in_secs),
            )
            return np.asarray(
                adapter.from_velocity_command(tuple(command_vw.tolist())), dtype=float
            )
        current_linear, current_angular = robot.current_speed
        d_linear = float(command_vw[0]) - float(current_linear)
        d_angular = float(command_vw[1]) - float(current_angular)
        return np.array([d_linear, d_angular], dtype=float)

    if "bicycle" in cls_name:
        adapter = PlannerActionAdapter(
            robot=robot,
            action_space=env.action_space,
            time_step=float(config.sim_config.time_per_step_in_secs),
        )
        return np.asarray(adapter.from_velocity_command(command), dtype=float)

    if "holonomic" in cls_name:
        mode = str(getattr(robot_cfg, "command_mode", "vx_vy")).strip().lower()
        linear, angular = float(command[0]), float(command[1])
        if mode == "vx_vy":
            # Preserve turning intent by projecting at midpoint heading over this step.
            step_dt = float(getattr(config.sim_config, "time_per_step_in_secs", 0.0) or 0.0)
            heading = float(robot.pose[1]) + (angular * max(step_dt, 0.0) * 0.5)
            vx = linear * math.cos(heading)
            vy = linear * math.sin(heading)
            return np.array([vx, vy], dtype=float)
        return np.array([linear, angular], dtype=float)

    current_linear, current_angular = robot.current_speed
    d_linear = float(command[0]) - float(current_linear)
    d_angular = float(command[1]) - float(current_angular)
    return np.array([d_linear, d_angular], dtype=float)


def _normalize_pedestrian_impact_controls(
    *,
    experimental_ped_impact: bool,
    ped_impact_radius_m: float,
    ped_impact_window_steps: int,
) -> tuple[float, int]:
    """Normalize pedestrian-impact controls and fail fast for invalid opt-in values.

    Returns:
        Normalized radius/window pair for downstream metric computation.
    """

    radius = float(ped_impact_radius_m)
    window_value = float(ped_impact_window_steps)
    window_steps = int(window_value)
    if experimental_ped_impact:
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("ped_impact_radius_m must be a finite value > 0.")
        if (
            not math.isfinite(window_value)
            or float(window_steps) != window_value
            or window_steps < 1
        ):
            raise ValueError("ped_impact_window_steps must be an integer >= 1.")
    return radius, window_steps


def _collision_metric_value(metrics: dict[str, Any], key: str) -> float:
    """Return a finite collision metric value, treating missing/non-finite values as zero."""
    value = metrics.get(key)
    if value is None:
        value = 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _floor_collision_metrics_from_flags(
    metrics: dict[str, Any],
    *,
    collision_seen: bool,
    ped_collision_seen: bool,
    obstacle_collision_seen: bool,
    robot_collision_seen: bool,
) -> None:
    """Preserve exact environment collision flags when sampled metrics miss the contact.

    Obstacle metrics are computed from sampled wall points, while the environment detects obstacle
    collisions against exact geometry. When the exact detector reports a collision between sampled
    points, keep the episode usable by flooring the corresponding count to one instead of failing
    the outcome/metric integrity check.
    """

    collision_keys = {
        "ped_collision_count": ped_collision_seen,
        "obstacle_collision_count": obstacle_collision_seen,
        "agent_collision_count": robot_collision_seen,
    }
    for key, typed_collision_seen in collision_keys.items():
        if typed_collision_seen and _collision_metric_value(metrics, key) <= 0.0:
            metrics[key] = 1.0

    typed_collision_count = sum(_collision_metric_value(metrics, key) for key in collision_keys)
    sampled_collision_count = max(
        _collision_metric_value(metrics, "total_collision_count"),
        _collision_metric_value(metrics, "collisions"),
        _collision_metric_value(metrics, "wall_collisions"),
    )
    if typed_collision_count > 0.0:
        aggregate_collision_count = max(sampled_collision_count, typed_collision_count)
        metrics["total_collision_count"] = aggregate_collision_count
        metrics["collisions"] = aggregate_collision_count
        if obstacle_collision_seen and _collision_metric_value(metrics, "wall_collisions") <= 0.0:
            metrics["wall_collisions"] = _collision_metric_value(
                metrics, "obstacle_collision_count"
            )
    elif collision_seen:
        aggregate_collision_count = max(sampled_collision_count, 1.0)
        metrics["total_collision_count"] = aggregate_collision_count
        metrics["collisions"] = aggregate_collision_count


def _run_map_episode(  # noqa: C901,PLR0912,PLR0913,PLR0915
    scenario: dict[str, Any],
    seed: int,
    *,
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
    algo: str,
    scenario_path: Path,
    algo_config: dict[str, Any] | None = None,
    algo_config_path: str | None = None,
    adapter_impact_eval: bool = False,
    experimental_ped_impact: bool = False,
    ped_impact_radius_m: float = 2.0,
    ped_impact_window_steps: int = 5,
    observation_mode: str | None = None,
    observation_level: str | None = None,
    benchmark_track: str | None = None,
    track_schema_version: str | None = None,
    observation_noise: dict[str, Any] | None = None,
    synthetic_actuation_profile: dict[str, Any] | None = None,
    latency_stress_profile: dict[str, Any] | None = None,
    record_planner_decision_trace: bool = False,
    record_simulation_step_trace: bool = False,
) -> dict[str, Any]:
    """Run one scenario/seed episode and return a benchmark JSONL record.

    Returns:
        dict[str, Any]: Episode record with metrics, provenance, and planner metadata.
    """
    ped_impact_radius_m, ped_impact_window_steps = _normalize_pedestrian_impact_controls(
        experimental_ped_impact=experimental_ped_impact,
        ped_impact_radius_m=ped_impact_radius_m,
        ped_impact_window_steps=ped_impact_window_steps,
    )
    ts_start = datetime.now(UTC).isoformat()
    start_time = time.time()
    scenario = _scenario_with_episode_seed_defaults(scenario, seed=seed)
    scenario_id = str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )
    benchmark_track = normalize_track_field(benchmark_track, field_name="benchmark_track")
    track_schema_version = normalize_track_field(
        track_schema_version,
        field_name="track_schema_version",
    )
    noise_spec = normalize_observation_noise_spec(observation_noise)
    noise_rng = make_observation_noise_rng(noise_spec, seed=seed, scenario_id=scenario_id)
    noise_stats = new_observation_noise_stats()
    config = _build_env_config(scenario, scenario_path=scenario_path)
    max_steps = int(scenario.get("simulation_config", {}).get("max_episode_steps", 0) or 0)
    horizon_val = int(horizon) if horizon and horizon > 0 else max_steps
    if horizon_val <= 0:
        horizon_val = 200
    if dt is not None and dt > 0:
        config.sim_config.time_per_step_in_secs = float(dt)

    robot_kinematics = _robot_kinematics_label(config)
    actuation_profile = _load_synthetic_actuation_profile(synthetic_actuation_profile)
    latency_profile = _load_latency_stress_profile(latency_stress_profile)
    if actuation_profile is not None and robot_kinematics != _DEFAULT_KINEMATICS:
        raise ValueError(
            "synthetic_actuation_profile requires differential_drive scenarios; "
            f"got {robot_kinematics!r} for scenario {scenario_id!r}"
        )
    if (
        latency_profile is not None
        and latency_profile.action_delay_steps > 0
        and robot_kinematics != _DEFAULT_KINEMATICS
    ):
        raise ValueError(
            "latency_stress_profile.action_delay_steps requires differential_drive scenarios; "
            f"got {robot_kinematics!r} for scenario {scenario_id!r}"
        )
    robot_command_mode = (
        str(getattr(getattr(config, "robot_config", None), "command_mode", "vx_vy")).strip().lower()
    )
    raw_policy_cfg = (
        dict(algo_config) if algo_config is not None else _parse_algo_config(algo_config_path)
    )
    algo, policy_cfg = _resolve_policy_search_candidate_runtime(
        default_algo=algo,
        algo_config_path=algo_config_path,
        algo_config=raw_policy_cfg,
        scenario=scenario,
    )
    policy_cfg = _apply_planner_selector_v2_context(
        algo,
        policy_cfg,
        scenario=scenario,
        seed=int(seed),
    )
    active_observation_mode = resolve_observation_mode(
        algo,
        observation_mode,
        observation_level=observation_level,
    )
    _apply_active_observation_mode_to_env_config(
        config,
        active_observation_mode=active_observation_mode,
    )
    _apply_policy_env_observation_overrides(config, policy_cfg)
    _validate_sensor_fusion_adapter_config(
        algo=algo,
        active_observation_mode=active_observation_mode,
        algo_config=policy_cfg,
    )
    _validate_planner_contract(
        algo=algo,
        robot_kinematics=robot_kinematics,
        algo_config=policy_cfg,
        observation_mode=active_observation_mode,
        observation_level=observation_level,
    )
    policy_fn, algo_meta = _build_policy(
        algo,
        policy_cfg,
        robot_kinematics=robot_kinematics,
        robot_command_mode=robot_command_mode,
        adapter_impact_eval=adapter_impact_eval,
    )
    algo_meta = enrich_algorithm_metadata(
        algo=algo,
        metadata=algo_meta,
        robot_kinematics=robot_kinematics,
        observation_mode=active_observation_mode,
        observation_level=observation_level,
    )
    active_observation_level = str(algo_meta["observation_level"]["key"])
    attach_track_metadata(
        algo_meta,
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
        observation_level=active_observation_level,
        observation_mode=active_observation_mode,
    )
    planner_close = getattr(policy_fn, "_planner_close", None)
    planner_reset = getattr(policy_fn, "_planner_reset", None)
    planner_bind_env = getattr(policy_fn, "_planner_bind_env", None)
    planner_stats = getattr(policy_fn, "_planner_stats", None)
    planner_native_action = getattr(policy_fn, "_planner_native_env_action", False)

    planner_runtime_snapshot: dict[str, Any] | None = None
    actuation_controller = (
        SyntheticActuationController(
            profile=actuation_profile, dt=config.sim_config.time_per_step_in_secs
        )
        if actuation_profile is not None
        else None
    )
    current_command = (0.0, 0.0)
    actuation_summary: dict[str, Any] = not_available_saturation_metrics()
    synthetic_actuation_trace: list[dict[str, Any]] = []
    planner_decision_trace: list[dict[str, Any]] = []
    simulation_step_trace: list[dict[str, Any]] = []
    single_pedestrian_intent_metadata = _single_pedestrian_intent_metadata(scenario)
    single_pedestrian_vru_metadata = _single_pedestrian_vru_metadata(scenario)

    env = make_robot_env(config=config, seed=int(seed), debug=False)
    obs, _ = env.reset(seed=int(seed))
    if callable(planner_bind_env):
        planner_bind_env(env)
    if callable(planner_reset):
        planner_reset(seed=int(seed))

    robot_positions: list[np.ndarray] = []
    ped_positions: list[np.ndarray] = []
    ped_forces: list[np.ndarray] = []
    reached_goal_step: int | None = None
    termination_reason = "max_steps"
    collision_seen = False
    ped_collision_seen = False
    obstacle_collision_seen = False
    robot_collision_seen = False
    timeout_seen = False

    map_def = None
    goal_vec = np.asarray(env.simulator.goal_pos[0], dtype=float)
    initial_robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
    initial_goal_distance = float(np.linalg.norm(initial_robot_pos - goal_vec))
    previous_trace_robot_pos = np.array(initial_robot_pos, dtype=float, copy=True)
    previous_trace_ped_pos: np.ndarray | None = None
    previous_trace_heading = _observation_heading(obs)
    try:
        for step_idx in range(horizon_val):
            policy_obs, step_noise_stats = apply_observation_noise(obs, noise_spec, noise_rng)
            merge_observation_noise_stats(noise_stats, step_noise_stats)
            policy_command = policy_fn(policy_obs)
            actuation_step = None
            planner_step_decision = None
            if record_planner_decision_trace and callable(planner_stats):
                try:
                    planner_runtime = planner_stats()
                except (RuntimeError, ValueError, TypeError):
                    planner_runtime = None
                if isinstance(planner_runtime, dict) and isinstance(
                    planner_runtime.get("last_decision"), dict
                ):
                    planner_step_decision = dict(planner_runtime["last_decision"])
            # Use per-step flag when available (e.g. SAC with fallback); fall back to the
            # static cached value for planners that set _planner_native_env_action once.
            step_is_native = getattr(policy_fn, "_last_step_native", planner_native_action)
            if actuation_controller is not None and step_is_native:
                raise ValueError(
                    "synthetic_actuation_profile requires absolute differential-drive commands; "
                    "native env actions cannot be wrapped safely"
                )
            if actuation_controller is not None:
                if (
                    not isinstance(policy_command, (tuple, list, np.ndarray))
                    or len(policy_command) < 2
                ):
                    raise TypeError(
                        "synthetic_actuation_profile expects planner commands shaped like "
                        "(linear_velocity, angular_velocity)"
                    )
                actuation_step = actuation_controller.apply(
                    current_command=current_command,
                    requested_command=(float(policy_command[0]), float(policy_command[1])),
                )
                policy_command = actuation_step.applied_command
                current_command = actuation_step.applied_command
            if step_is_native:
                # Policy already outputs native env actions (e.g. delta velocities);
                # skip the absolute→delta conversion done by _policy_command_to_env_action.
                action = np.asarray(policy_command, dtype=np.float32)
            else:
                action = _policy_command_to_env_action(
                    env=env,
                    config=config,
                    command=policy_command,
                )
            obs, _reward, terminated, truncated, info = env.step(action)

            # Snapshot mutable simulator buffers; do not keep view aliases across steps.
            robot_pos = np.array(env.simulator.robot_pos[0], dtype=float, copy=True)
            peds = np.array(env.simulator.ped_pos, dtype=float, copy=True)
            if record_forces:
                forces = getattr(env.simulator, "last_ped_forces", None)
                if forces is None:
                    forces_arr = np.zeros_like(peds, dtype=float)
                else:
                    forces_arr = np.array(forces, dtype=float, copy=True)
                    if forces_arr.shape != peds.shape:
                        forces_arr = np.zeros_like(peds, dtype=float)

            robot_positions.append(robot_pos)
            ped_positions.append(peds)
            if record_forces:
                ped_forces.append(forces_arr)
            if record_simulation_step_trace:
                dt_seconds = float(config.sim_config.time_per_step_in_secs)
                robot_velocity = (
                    (robot_pos - previous_trace_robot_pos) / dt_seconds
                    if dt_seconds > 0.0
                    else np.zeros(2, dtype=float)
                )
                heading = _observation_heading(obs, default=previous_trace_heading)
                planner_payload: dict[str, Any] = {
                    "event": "step",
                    "selected_action": _command_action_payload(policy_command),
                }
                if actuation_step is not None:
                    planner_payload["amv"] = {
                        "requested_linear_m_s": float(actuation_step.requested_command[0]),
                        "requested_angular_rad_s": float(actuation_step.requested_command[1]),
                        "applied_linear_m_s": float(actuation_step.applied_command[0]),
                        "applied_angular_rad_s": float(actuation_step.applied_command[1]),
                        "command_clipped": bool(actuation_step.command_clipped),
                        "yaw_rate_saturated": bool(actuation_step.yaw_rate_saturated),
                    }
                if record_forces and peds.size:
                    planner_payload["ammv"] = {
                        "pedestrian_force_vectors": [
                            [float(force[0]), float(force[1])] for force in forces_arr
                        ]
                    }
                simulation_step_trace.append(
                    {
                        "step": int(step_idx),
                        "time_s": float((step_idx + 1) * dt_seconds),
                        "robot": {
                            "position": [float(robot_pos[0]), float(robot_pos[1])],
                            "heading": float(heading),
                            "velocity": [float(robot_velocity[0]), float(robot_velocity[1])],
                        },
                        "pedestrians": _trace_pedestrians(
                            peds,
                            previous_trace_ped_pos,
                            dt_seconds,
                            single_pedestrian_intent_metadata,
                            single_pedestrian_vru_metadata,
                            robot_pos,
                            robot_velocity,
                        ),
                        "planner": planner_payload,
                    }
                )
                previous_trace_robot_pos = np.array(robot_pos, dtype=float, copy=True)
                previous_trace_ped_pos = np.array(peds, dtype=float, copy=True)
                previous_trace_heading = float(heading)
            if actuation_step is not None:
                distance_to_goal = float(np.linalg.norm(robot_pos - goal_vec))
                route_progress = float(initial_goal_distance - distance_to_goal)
                progress_ratio = (
                    route_progress / initial_goal_distance if initial_goal_distance > 1e-9 else 0.0
                )
                synthetic_actuation_trace.append(
                    {
                        "step": int(step_idx),
                        "requested_linear_m_s": float(actuation_step.requested_command[0]),
                        "requested_angular_rad_s": float(actuation_step.requested_command[1]),
                        "applied_linear_m_s": float(actuation_step.applied_command[0]),
                        "applied_angular_rad_s": float(actuation_step.applied_command[1]),
                        "command_clipped": bool(actuation_step.command_clipped),
                        "yaw_rate_saturated": bool(actuation_step.yaw_rate_saturated),
                        "linear_accel_applied_m_s2": float(
                            actuation_step.linear_accel_applied_m_s2
                        ),
                        "angular_accel_applied_rad_s2": float(
                            actuation_step.angular_accel_applied_rad_s2
                        ),
                        "distance_to_goal_m": distance_to_goal,
                        "route_progress_from_start_m": route_progress,
                        "route_progress_ratio": float(progress_ratio),
                        "robot_x_m": float(robot_pos[0]),
                        "robot_y_m": float(robot_pos[1]),
                    }
                )
            if planner_step_decision is not None:
                selected_terms = planner_step_decision.get("selected_terms")
                selected_terms = selected_terms if isinstance(selected_terms, dict) else {}
                progress_windows_raw = planner_step_decision.get("progress_windows")
                progress_windows = (
                    progress_windows_raw if isinstance(progress_windows_raw, dict) else {}
                )
                selected_command = planner_step_decision.get("selected_command")
                selected_command = selected_command if isinstance(selected_command, list) else []
                distance_to_goal = float(np.linalg.norm(robot_pos - goal_vec))
                planner_decision_trace.append(
                    {
                        "step": int(step_idx),
                        "selected_source": str(
                            planner_step_decision.get("selected_source", "unknown")
                        ),
                        "selected_command": [
                            float(value)
                            for value in selected_command[:2]
                            if isinstance(value, int | float | np.integer | np.floating)
                        ],
                        "selected_score": float(planner_step_decision["selected_score"])
                        if isinstance(
                            planner_step_decision.get("selected_score"),
                            int | float | np.integer | np.floating,
                        )
                        and math.isfinite(float(planner_step_decision["selected_score"]))
                        else None,
                        "static_recenter": float(selected_terms.get("static_recenter", 0.0)),
                        "route_arc_progress": float(selected_terms.get("route_arc_progress", 0.0)),
                        "goal_progress": float(selected_terms.get("goal_progress", 0.0)),
                        "progress_windows": {
                            str(key): float(value)
                            for key, value in progress_windows.items()
                            if isinstance(value, int | float | np.integer | np.floating)
                        },
                        "distance_to_goal_m": distance_to_goal,
                        "route_progress_from_start_m": float(
                            initial_goal_distance - distance_to_goal
                        ),
                        "robot_x_m": float(robot_pos[0]),
                        "robot_y_m": float(robot_pos[1]),
                    }
                )

            meta = info.get("meta", {}) if isinstance(info, dict) else {}
            step_collision = collision_event(info)
            step_route_complete = route_complete_success(info)
            step_success = step_route_complete and not step_collision
            step_timeout = bool(meta.get("is_timesteps_exceeded", False))
            collision_seen = collision_seen or step_collision
            ped_collision_seen = ped_collision_seen or bool(
                meta.get("is_pedestrian_collision", False)
            )
            obstacle_collision_seen = obstacle_collision_seen or bool(
                meta.get("is_obstacle_collision", False)
            )
            robot_collision_seen = robot_collision_seen or bool(
                meta.get("is_robot_collision", False)
            )
            timeout_seen = timeout_seen or step_timeout
            if reached_goal_step is None and step_success:
                reached_goal_step = step_idx
            if step_success:
                termination_reason = resolve_termination_reason(
                    terminated=True,
                    truncated=False,
                    success=True,
                    collision=step_collision,
                )
                break
            if terminated or truncated:
                termination_reason = resolve_termination_reason(
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    success=step_success,
                    collision=step_collision,
                )
                break
        if getattr(env, "simulator", None) is not None:
            map_def = env.simulator.map_def
            goal_vec = np.asarray(env.simulator.goal_pos[0], dtype=float)
    finally:
        if callable(planner_stats):
            try:
                planner_runtime = planner_stats()
            except (RuntimeError, ValueError, TypeError):
                logger.debug("Planner stats hook failed before close", exc_info=True)
                planner_runtime = None
            if isinstance(planner_runtime, dict):
                planner_runtime_snapshot = dict(planner_runtime)
        if callable(planner_close):
            try:
                planner_close()
            except (RuntimeError, ValueError, TypeError):
                logger.debug("Planner close hook failed", exc_info=True)
        env.close()

    robot_pos_arr = np.asarray(robot_positions, dtype=float)
    robot_vel_arr, robot_acc_arr = _vel_and_acc(
        robot_pos_arr, config.sim_config.time_per_step_in_secs
    )
    ped_pos_arr = _stack_ped_positions(ped_positions)
    ped_forces_arr = (
        _stack_ped_positions(ped_forces, fill_value=np.nan)
        if record_forces
        else np.zeros_like(ped_pos_arr, dtype=float)
    )

    obstacles = (
        sample_obstacle_points(map_def.obstacles, map_def.bounds) if map_def is not None else None
    )
    if robot_pos_arr.size:
        shortest_path = compute_shortest_path_length(map_def, robot_pos_arr[0], goal_vec)
    else:
        shortest_path = float("nan")

    if robot_pos_arr.size == 0:
        metrics_raw = {
            "success": 0.0,
            "time_to_goal_norm": float("nan"),
            "collisions": 0.0,
        }
    else:
        robot_config = getattr(config, "robot_config", None)
        ep = EpisodeData(
            robot_pos=robot_pos_arr,
            robot_vel=robot_vel_arr,
            robot_acc=robot_acc_arr,
            peds_pos=ped_pos_arr,
            ped_forces=ped_forces_arr,
            obstacles=obstacles,
            goal=goal_vec,
            dt=float(config.sim_config.time_per_step_in_secs),
            reached_goal_step=reached_goal_step,
            robot_radius=float(getattr(robot_config, "radius", 1.0)),
            ped_radius=float(getattr(config.sim_config, "ped_radius", 0.4)),
            episode_metadata=_episode_metadata_for_signal_metrics(scenario),
        )
        metrics_raw = compute_all_metrics(
            ep,
            horizon=horizon_val,
            shortest_path_len=shortest_path,
            robot_max_speed=_robot_max_speed(config),
            experimental_ped_impact=experimental_ped_impact,
            ped_impact_radius_m=ped_impact_radius_m,
            ped_impact_window_steps=ped_impact_window_steps,
        )
    _floor_collision_metrics_from_flags(
        metrics_raw,
        collision_seen=collision_seen,
        ped_collision_seen=ped_collision_seen,
        obstacle_collision_seen=obstacle_collision_seen,
        robot_collision_seen=robot_collision_seen,
    )
    impact = algo_meta.get("adapter_impact")
    if isinstance(impact, dict) and bool(impact.get("requested", False)):
        native_steps = int(impact.get("native_steps", 0))
        adapted_steps = int(impact.get("adapted_steps", 0))
        total = native_steps + adapted_steps
        if total > 0:
            execution_mode = infer_execution_mode_from_counts(native_steps, adapted_steps)
            impact["status"] = "complete"
            impact["execution_mode"] = execution_mode
            impact["adapter_fraction"] = float(adapted_steps / total)
            algo_meta = enrich_algorithm_metadata(
                algo=algo,
                metadata=algo_meta,
                execution_mode=execution_mode,
                robot_kinematics=robot_kinematics,
                observation_mode=active_observation_mode,
                observation_level=active_observation_level,
            )
            attach_track_metadata(
                algo_meta,
                benchmark_track=benchmark_track,
                track_schema_version=track_schema_version,
                observation_level=active_observation_level,
                observation_mode=active_observation_mode,
            )
        else:
            impact["status"] = "not_applicable"
            impact["adapter_fraction"] = 0.0
    _finalize_feasibility_metadata(algo_meta)
    if isinstance(planner_runtime_snapshot, dict):
        algo_meta["planner_runtime"] = planner_runtime_snapshot
    if record_planner_decision_trace:
        algo_meta["planner_decision_trace"] = {
            "schema_version": "planner-decision-trace.v1",
            "dt": float(config.sim_config.time_per_step_in_secs),
            "initial_goal_distance_m": initial_goal_distance,
            "steps": planner_decision_trace,
        }
    if record_simulation_step_trace:
        algo_meta["simulation_step_trace"] = {
            "schema_version": "simulation-step-trace.v1",
            "dt": float(config.sim_config.time_per_step_in_secs),
            "initial_goal_distance_m": initial_goal_distance,
            "steps": simulation_step_trace,
        }
    intent_summary = _intent_conditioned_behavior_summary(
        scenario,
        single_pedestrian_intent_metadata,
    )
    if intent_summary is not None:
        algo_meta["intent_conditioned_behavior"] = intent_summary
    vru_summary = _cyclist_like_vru_summary(
        scenario,
        single_pedestrian_vru_metadata,
    )
    if vru_summary is not None:
        algo_meta["cyclist_like_vru"] = vru_summary
    fast_bicycle_summary = _fast_bicycle_actor_summary(
        scenario,
        single_pedestrian_vru_metadata,
    )
    if fast_bicycle_summary is not None:
        algo_meta["fast_bicycle_actor"] = fast_bicycle_summary
    if actuation_controller is not None:
        actuation_summary = actuation_controller.summary()
        algo_meta["synthetic_actuation"] = {
            "profile": actuation_profile.to_metadata(),
            "summary": dict(actuation_summary),
            "trace": {
                "schema_version": "synthetic-actuation-step-trace.v1",
                "dt": float(config.sim_config.time_per_step_in_secs),
                "initial_goal_distance_m": initial_goal_distance,
                "steps": synthetic_actuation_trace,
            },
        }
    if latency_profile is not None:
        algo_meta["latency_stress"] = {
            "profile": latency_profile.to_metadata(dt=config.sim_config.time_per_step_in_secs),
            "metrics": not_available_latency_metrics(),
        }
    visibility_settings = getattr(config, "observation_visibility", None)
    if visibility_settings is not None and hasattr(visibility_settings, "to_metadata"):
        algo_meta["observation_visibility"] = visibility_settings.to_metadata()
    shield_stats = algo_meta.get("shield_stats")
    if isinstance(shield_stats, dict):
        metrics_raw.update(shield_metrics_from_stats(shield_stats))
    metrics = post_process_metrics(
        metrics_raw,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
    )
    if actuation_controller is not None:
        for metric_name, metric_value in actuation_summary.items():
            if metric_name in {
                "schema_version",
                "status",
                "step_count",
                "command_clip_steps",
                "yaw_rate_saturation_steps",
            }:
                continue
            metrics[metric_name] = metric_value

    ts_end = datetime.now(UTC).isoformat()
    scenario_params = _scenario_identity_payload(
        scenario,
        algo=algo,
        algo_config=policy_cfg,
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
        observation_mode=active_observation_mode,
        observation_level=active_observation_level,
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
        observation_noise=noise_spec,
        synthetic_actuation_profile=(
            actuation_profile.to_metadata() if actuation_profile is not None else None
        ),
        latency_stress_profile=(
            latency_profile.to_metadata(dt=config.sim_config.time_per_step_in_secs)
            if latency_profile is not None
            else None
        ),
        record_simulation_step_trace=record_simulation_step_trace,
    )
    steps_taken = int(robot_pos_arr.shape[0])
    wall_time = float(max(1e-9, time.time() - start_time))
    timing = {"steps_per_second": float(steps_taken) / wall_time if wall_time > 0 else 0.0}
    route_complete = reached_goal_step is not None
    timeout_event = timeout_seen or termination_reason in {"truncated", "max_steps"}
    outcome = build_outcome_payload(
        route_complete=route_complete,
        collision=collision_seen,
        timeout=timeout_event,
    )
    status = status_from_termination_reason(termination_reason)
    contradictions = outcome_contradictions(
        termination_reason=termination_reason,
        outcome=outcome,
        metrics=metrics,
    )
    if contradictions:
        raise ValueError(
            f"Episode integrity contradictions for scenario '{scenario_id}', seed={seed}: "
            + "; ".join(contradictions)
        )
    static_deadlock_fields = _static_deadlock_trace_fields(
        scenario,
        robot_pos_arr=robot_pos_arr,
        goal_vec=goal_vec,
        initial_goal_distance=initial_goal_distance,
        termination_reason=termination_reason,
        outcome=outcome,
        planner_decision_trace=planner_decision_trace,
    )
    record = {
        "version": "v1",
        "episode_id": _compute_map_episode_id(scenario_params, seed),
        "scenario_id": scenario_id,
        "seed": seed,
        "scenario_params": scenario_params,
        "metrics": metrics,
        "algorithm_metadata": algo_meta,
        "observation_noise": noise_spec,
        "observation_noise_hash": observation_noise_hash(noise_spec),
        "observation_noise_stats": noise_stats,
        "algo": algo,
        "observation_mode": active_observation_mode,
        "observation_level": active_observation_level,
        "config_hash": _config_hash(scenario_params),
        "git_hash": _git_hash_fallback(),
        "timestamps": {"start": ts_start, "end": ts_end},
        "status": status,
        "steps": steps_taken,
        "horizon": horizon_val,
        "wall_time_sec": wall_time,
        "timing": timing,
        "termination_reason": termination_reason,
        "outcome": outcome,
        "integrity": {"contradictions": contradictions},
    }
    record.update(static_deadlock_fields)
    if benchmark_track is not None:
        record["benchmark_track"] = benchmark_track
    if track_schema_version is not None:
        record["track_schema_version"] = track_schema_version
    ensure_metric_parameters(record)
    return record


def _scenario_with_episode_seed_defaults(
    scenario: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Return a scenario copy with seed-derived defaults for stochastic subcomponents.

    Some scenario-level generators use their own NumPy ``default_rng`` instances.  When those
    fields are left unset they bypass the episode seed and make benchmark rows depend on process
    history.  Fill only missing values here so explicit scenario provenance remains unchanged.
    """
    updated = deepcopy(scenario)
    sim_config = updated.setdefault("simulation_config", {})
    if isinstance(sim_config, dict) and sim_config.get("route_spawn_seed") is None:
        sim_config["route_spawn_seed"] = int(seed)
    return updated


def _write_validated(out_path: Path, schema: dict[str, Any], record: dict[str, Any]) -> None:
    """Validate an episode record and append it as JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        _write_validated_to_handle(handle, schema, record)


def _write_validated_to_handle(
    handle: TextIO,
    schema: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Validate one episode record and append it to an open JSONL handle."""
    violations = validate_episode_success_integrity(record)
    if violations:
        raise ValueError("Episode integrity contradictions detected: " + "; ".join(violations))
    validate_episode(record, schema)
    handle.write(json.dumps(record, sort_keys=True) + "\n")


def _run_map_job_worker(
    job: tuple[dict[str, Any], int, dict[str, Any]],
) -> dict[str, Any]:
    """Execute one serialized map-runner job.

    Returns:
        dict[str, Any]: Episode record returned by ``_run_map_episode``.
    """
    scenario, seed, params = job
    return _run_map_episode(
        scenario,
        seed,
        horizon=params.get("horizon"),
        dt=params.get("dt"),
        record_forces=bool(params.get("record_forces", True)),
        snqi_weights=params.get("snqi_weights"),
        snqi_baseline=params.get("snqi_baseline"),
        algo=str(params.get("algo", "goal")),
        algo_config=params.get("algo_config"),
        algo_config_path=params.get("algo_config_path"),
        scenario_path=Path(params.get("scenario_path")),
        adapter_impact_eval=bool(params.get("adapter_impact_eval", False)),
        experimental_ped_impact=bool(params.get("experimental_ped_impact", False)),
        ped_impact_radius_m=float(params.get("ped_impact_radius_m", 2.0)),
        ped_impact_window_steps=int(params.get("ped_impact_window_steps", 5)),
        observation_mode=params.get("observation_mode"),
        observation_level=params.get("observation_level"),
        benchmark_track=params.get("benchmark_track"),
        track_schema_version=params.get("track_schema_version"),
        observation_noise=params.get("observation_noise"),
        synthetic_actuation_profile=params.get("synthetic_actuation_profile"),
        latency_stress_profile=params.get("latency_stress_profile"),
        record_simulation_step_trace=bool(params.get("record_simulation_step_trace", False)),
    )


def _accumulate_batch_metadata(
    rec: dict[str, Any],
    *,
    feasibility_totals: dict[str, float],
) -> tuple[bool, int, int]:
    """Aggregate adapter-impact and feasibility counters from one episode record.

    Returns:
        tuple[bool, int, int]: ``(adapter_requested_seen, native_steps, adapted_steps)`` deltas.
    """
    impact_meta = (rec.get("algorithm_metadata") or {}).get("adapter_impact") or {}
    feasibility_meta = (rec.get("algorithm_metadata") or {}).get("kinematics_feasibility") or {}
    adapter_requested_seen = False
    adapter_native_steps = 0
    adapter_adapted_steps = 0
    if isinstance(impact_meta, dict):
        adapter_requested_seen = bool(impact_meta.get("requested", False))
        adapter_native_steps = int(impact_meta.get("native_steps", 0) or 0)
        adapter_adapted_steps = int(impact_meta.get("adapted_steps", 0) or 0)
    if isinstance(feasibility_meta, dict):
        commands_evaluated = int(feasibility_meta.get("commands_evaluated", 0) or 0)
        feasibility_totals["commands_evaluated"] += commands_evaluated
        feasibility_totals["infeasible_native_count"] += int(
            feasibility_meta.get("infeasible_native_count", 0) or 0
        )
        feasibility_totals["projected_count"] += int(
            feasibility_meta.get("projected_count", 0) or 0
        )
        feasibility_totals["sum_abs_delta_linear"] += (
            float(feasibility_meta.get("mean_abs_delta_linear", 0.0)) * commands_evaluated
        )
        feasibility_totals["sum_abs_delta_angular"] += (
            float(feasibility_meta.get("mean_abs_delta_angular", 0.0)) * commands_evaluated
        )
        feasibility_totals["max_abs_delta_linear"] = max(
            float(feasibility_totals["max_abs_delta_linear"]),
            float(feasibility_meta.get("max_abs_delta_linear", 0.0) or 0.0),
        )
        feasibility_totals["max_abs_delta_angular"] = max(
            float(feasibility_totals["max_abs_delta_angular"]),
            float(feasibility_meta.get("max_abs_delta_angular", 0.0) or 0.0),
        )
    return adapter_requested_seen, adapter_native_steps, adapter_adapted_steps


class _WorkerMetadataBridgeUpdate(NamedTuple):
    """Normalized metadata update emitted by either direct or serialized worker paths."""

    runtime_algorithm_contract: dict[str, Any]
    adapter_requested_seen: bool
    adapter_native_steps: int
    adapter_adapted_steps: int


def _apply_worker_metadata_bridge(
    rec: dict[str, Any],
    *,
    feasibility_totals: dict[str, float],
    runtime_algorithm_contract: dict[str, Any] | None,
) -> _WorkerMetadataBridgeUpdate:
    """Fold one worker episode record into the batch-level metadata contract.

    This is the explicit bridge between per-episode worker payloads and the batch summary.  It is
    used by both direct function-call execution and serialized worker execution so metadata-only
    additions have one testable hop.

    Returns:
        Worker metadata update with merged runtime contract and adapter-impact deltas.
    """
    requested_seen, native_steps, adapted_steps = _accumulate_batch_metadata(
        rec,
        feasibility_totals=feasibility_totals,
    )
    merged_runtime_contract = _merge_runtime_algorithm_contract(
        runtime_algorithm_contract or {},
        rec.get("algorithm_metadata"),
    )
    return _WorkerMetadataBridgeUpdate(
        runtime_algorithm_contract=merged_runtime_contract,
        adapter_requested_seen=requested_seen,
        adapter_native_steps=native_steps,
        adapter_adapted_steps=adapted_steps,
    )


def _merge_runtime_algorithm_contract(  # noqa: C901
    base_contract: dict[str, Any],
    runtime_algorithm_metadata: Any,
) -> dict[str, Any]:
    """Merge runtime-resolved algorithm contract fields into a batch summary contract.

    Returns:
        dict[str, Any]: The merged contract mapping, or the original input on mismatch.
    """
    if not isinstance(base_contract, dict) or not isinstance(runtime_algorithm_metadata, dict):
        return base_contract

    def _merge_mapping(target: dict[str, Any], source: dict[str, Any]) -> None:
        """Merge authoritative runtime contract values into a nested mapping."""
        authoritative_keys = {
            "robot_kinematics",
            "execution_mode",
            "adapter_name",
            "planner_command_space",
            "benchmark_command_space",
            "projection_policy",
            "execution_detail",
            "adapter_boundary",
        }

        def _is_placeholder(value: Any) -> bool:
            """Return whether a contract value should be replaced by runtime data."""
            if value is None:
                return True
            if isinstance(value, str):
                normalized = value.strip().lower()
                return normalized in {"", "unknown", "unspecified", "mixed"}
            return False

        for key, value in source.items():
            current = target.get(key)
            if _is_placeholder(current):
                target[key] = value
                continue
            if isinstance(current, dict) and isinstance(value, dict):
                _merge_mapping(current, value)
                continue
            if _is_placeholder(value) or current == value:
                continue
            if key in authoritative_keys:
                target[key] = value
                continue
            target[key] = "mixed"

    runtime_planner_kinematics = runtime_algorithm_metadata.get("planner_kinematics")
    if isinstance(runtime_planner_kinematics, dict):
        planner_kinematics = base_contract.get("planner_kinematics")
        if not isinstance(planner_kinematics, dict):
            planner_kinematics = {}
            base_contract["planner_kinematics"] = planner_kinematics
        _merge_mapping(planner_kinematics, runtime_planner_kinematics)

    runtime_upstream_reference = runtime_algorithm_metadata.get("upstream_reference")
    if isinstance(runtime_upstream_reference, dict):
        upstream_reference = base_contract.get("upstream_reference")
        if not isinstance(upstream_reference, dict):
            upstream_reference = {}
            base_contract["upstream_reference"] = upstream_reference
        _merge_mapping(upstream_reference, runtime_upstream_reference)

    return base_contract


def run_map_batch(  # noqa: C901,PLR0912,PLR0913,PLR0915
    scenarios_or_path: list[dict[str, Any]] | str | Path,
    out_path: str | Path,
    schema_path: str | Path,
    *,
    horizon: int | None = None,
    dt: float | None = None,
    record_forces: bool = True,
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
    algo: str = "goal",
    algo_config_path: str | None = None,
    benchmark_profile: BenchmarkProfile = "baseline-safe",
    socnav_missing_prereq_policy: str = "fail-fast",
    adapter_impact_eval: bool = False,
    experimental_ped_impact: bool = False,
    ped_impact_radius_m: float = 2.0,
    ped_impact_window_steps: int = 5,
    observation_mode: str | None = None,
    observation_level: str | None = None,
    benchmark_track: str | None = None,
    track_schema_version: str | None = None,
    observation_noise: dict[str, Any] | None = None,
    synthetic_actuation_profile: dict[str, Any] | None = None,
    latency_stress_profile: dict[str, Any] | None = None,
    record_simulation_step_trace: bool = False,
    workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Run map-based scenarios and append episode records.

    Returns:
        Summary payload with counts and failure details.
    """
    ped_impact_radius_m, ped_impact_window_steps = _normalize_pedestrian_impact_controls(
        experimental_ped_impact=experimental_ped_impact,
        ped_impact_radius_m=ped_impact_radius_m,
        ped_impact_window_steps=ped_impact_window_steps,
    )
    scenarios_is_path = isinstance(scenarios_or_path, (str, Path))
    if scenarios_is_path:
        scenario_path = Path(scenarios_or_path)
        scenarios = load_scenarios(scenario_path)
    else:
        scenario_path = Path(".")
        scenarios = list(scenarios_or_path)

    errors = validate_scenario_list([dict(s) for s in scenarios])
    if errors:
        raise ValueError(f"Scenario validation failed: {errors[:3]} (total {len(errors)})")

    suite_seeds = _resolve_seed_list(Path("configs/benchmarks/seed_list_v1.yaml"))
    suite_key = _suite_key(scenario_path)
    noise_spec = normalize_observation_noise_spec(observation_noise)
    noise_hash = observation_noise_hash(noise_spec)
    actuation_profile = _load_synthetic_actuation_profile(synthetic_actuation_profile)
    latency_profile = _load_latency_stress_profile(latency_stress_profile)
    latency_metadata_dt = float(dt) if dt is not None and float(dt) > 0.0 else 0.1
    benchmark_track = normalize_track_field(benchmark_track, field_name="benchmark_track")
    track_schema_version = normalize_track_field(
        track_schema_version,
        field_name="track_schema_version",
    )

    filtered: list[dict[str, Any]] = []
    for scenario in scenarios:
        if (
            scenario.get("supported") is False
            or scenario.get("metadata", {}).get("supported") is False
        ):
            continue
        errors = _validate_behavior_sanity(scenario)
        if errors:
            logger.warning("Skipping scenario '{}': {}", scenario.get("name"), errors)
            continue
        filtered.append(scenario)

    jobs: list[tuple[dict[str, Any], int]] = []
    scenario_kinematics = sorted({_scenario_robot_kinematics_label(sc) for sc in filtered})
    if not scenario_kinematics:
        kinematics_tag = "unknown"
    elif len(scenario_kinematics) == 1:
        kinematics_tag = scenario_kinematics[0]
    else:
        kinematics_tag = "mixed"
    batch_observation_mode = str(observation_mode).strip() if observation_mode is not None else None
    algo_contract = enrich_algorithm_metadata(
        algo=algo,
        metadata={},
        robot_kinematics=kinematics_tag,
        adapter_impact_requested=adapter_impact_eval,
        observation_mode=batch_observation_mode,
        observation_level=observation_level,
    )
    active_observation_mode = str(algo_contract["observation_spec"]["active_mode"])
    active_observation_level = str(algo_contract["observation_level"]["key"])
    attach_track_metadata(
        algo_contract,
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
        observation_level=active_observation_level,
        observation_mode=active_observation_mode,
    )
    planner_meta = algo_contract.get("planner_kinematics")
    if isinstance(planner_meta, dict):
        planner_meta["scenario_kinematics"] = scenario_kinematics
    for scenario in filtered:
        seeds = _select_seeds(scenario, suite_seeds=suite_seeds, suite_key=suite_key)
        for seed in seeds:
            jobs.append((scenario, int(seed)))
    preflight_skipped_jobs = 0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = load_schema(schema_path)
    raw_policy_cfg = _parse_algo_config(algo_config_path)
    _, policy_cfg = _resolve_policy_search_candidate_runtime(
        default_algo=algo,
        algo_config_path=algo_config_path,
        algo_config=raw_policy_cfg,
        scenario={},
    )
    robot_command_mode: str | None = None
    for scenario in filtered:
        robot_cfg = scenario.get("robot_config")
        if not isinstance(robot_cfg, dict):
            continue
        raw_mode = robot_cfg.get("command_mode")
        if raw_mode is None:
            continue
        robot_command_mode = str(raw_mode).strip().lower()
        break
    ppo_paper_ready, _paper_reason = (
        _ppo_paper_gate_status(policy_cfg) if algo.strip().lower() == "ppo" else (False, None)
    )
    readiness = require_algorithm_allowed(
        algo=algo,
        benchmark_profile=benchmark_profile,
        ppo_paper_ready=ppo_paper_ready,
        allow_testing_algorithms=bool(policy_cfg.get("allow_testing_algorithms", False)),
    )
    policy_cfg, preflight = _preflight_policy(
        algo=algo,
        algo_config=policy_cfg,
        benchmark_profile=benchmark_profile,
        missing_prereq_policy=socnav_missing_prereq_policy,
        robot_kinematics=kinematics_tag,
    )
    if algo.strip().lower() == "prediction_planner":
        algo_contract.update(_prediction_planner_metadata_overrides(policy_cfg))
    incompatible_kinematics: dict[str, str] = {}
    incompatible_scenarios: dict[str, str] = {}
    compatible_contract: dict[str, Any] | None = None
    for scenario in filtered:
        scenario_id = _scenario_id(scenario)
        validation_kinematics = _scenario_robot_kinematics_label(scenario)
        try:
            validation_algo, validation_cfg = _resolve_policy_search_candidate_runtime(
                default_algo=algo,
                algo_config_path=algo_config_path,
                algo_config=raw_policy_cfg,
                scenario=scenario,
            )
            validation_observation_mode = resolve_observation_mode(
                validation_algo,
                batch_observation_mode,
                observation_level=observation_level,
            )
            compatible_contract = _validate_planner_contract(
                algo=validation_algo,
                robot_kinematics=validation_kinematics,
                algo_config=validation_cfg,
                observation_mode=validation_observation_mode,
                observation_level=observation_level,
            )
            algo_contract["planner_contract"] = compatible_contract
        except planner_commands.PlannerContractValidationError as exc:
            incompatible_scenarios[scenario_id] = str(exc)
            incompatible_kinematics[validation_kinematics] = str(exc)
    if incompatible_kinematics:
        preflight["incompatible_scenario_kinematics"] = dict(
            sorted(incompatible_kinematics.items())
        )
        preflight["incompatible_scenarios"] = dict(sorted(incompatible_scenarios.items()))
        if compatible_contract is None:
            preflight["status"] = "skipped"
            preflight["compatibility_status"] = "incompatible"
            preflight["compatibility_reason"] = "; ".join(
                f"{scenario_id}: {reason}"
                for scenario_id, reason in sorted(incompatible_scenarios.items())
            )
        else:
            runnable_jobs: list[tuple[dict[str, Any], int]] = []
            for scenario, seed in jobs:
                if _scenario_id(scenario) in incompatible_scenarios:
                    preflight_skipped_jobs += 1
                    continue
                runnable_jobs.append((scenario, seed))
            jobs = runnable_jobs
            preflight["status"] = "partial"
            preflight["compatibility_status"] = "partial"
            preflight["skipped_jobs"] = preflight_skipped_jobs
    compatible, incompatible_reason = _planner_kinematics_compatibility(
        algo=algo,
        robot_kinematics=kinematics_tag,
        algo_config=policy_cfg,
    )
    if not compatible:
        preflight["status"] = "skipped"
        preflight["compatibility_status"] = "incompatible"
        preflight["compatibility_reason"] = incompatible_reason
    if actuation_profile is not None:
        if kinematics_tag != _DEFAULT_KINEMATICS:
            preflight["status"] = "skipped"
            preflight["compatibility_status"] = "incompatible"
            preflight["compatibility_reason"] = (
                "synthetic_actuation_profile requires differential_drive-only scenarios"
            )
        preflight["synthetic_actuation_profile"] = actuation_profile.to_metadata()
    if latency_profile is not None:
        latency_metadata = latency_profile.to_metadata(dt=latency_metadata_dt)
        if latency_profile.action_delay_steps > 0 and kinematics_tag != _DEFAULT_KINEMATICS:
            preflight["status"] = "skipped"
            preflight["compatibility_status"] = "incompatible"
            preflight["compatibility_reason"] = (
                "latency_stress_profile.action_delay_steps requires "
                "differential_drive-only scenarios"
            )
        preflight["latency_stress_profile"] = latency_metadata
        preflight["latency_stress_metrics"] = not_available_latency_metrics()
    preflight["algorithm_metadata_contract"] = algo_contract
    if preflight.get("status") == "skipped":
        summary = {
            "total_jobs": 0,
            "written": 0,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "skipped_jobs": len(jobs),
            "failures": [],
            "out_path": str(out_path),
            "algorithm_readiness": {
                "name": readiness.canonical_name if readiness is not None else algo,
                "tier": readiness.tier if readiness is not None else "unknown",
                "profile": benchmark_profile,
            },
            "algorithm_metadata_contract": algo_contract,
            "preflight": preflight,
            "observation_noise": noise_spec,
            "observation_noise_hash": noise_hash,
            "latency_stress_profile": (
                latency_profile.to_metadata(dt=latency_metadata_dt)
                if latency_profile is not None
                else None
            ),
            "latency_stress_metrics": (
                not_available_latency_metrics() if latency_profile is not None else None
            ),
        }
        if benchmark_track is not None:
            summary["benchmark_track"] = benchmark_track
        if track_schema_version is not None:
            summary["track_schema_version"] = track_schema_version
        summary["benchmark_availability"] = availability_payload(summary)
        return summary

    if resume and out_path.exists():
        existing = index_existing(out_path)
        if existing:
            filtered_jobs: list[tuple[dict[str, Any], int]] = []
            for sc, seed in jobs:
                identity_algo, identity_cfg = _resolve_policy_search_candidate_runtime(
                    default_algo=algo,
                    algo_config_path=algo_config_path,
                    algo_config=raw_policy_cfg,
                    scenario=sc,
                )
                identity_cfg = _apply_planner_selector_v2_context(
                    identity_algo,
                    identity_cfg,
                    scenario=sc,
                    seed=int(seed),
                )
                identity_observation_mode = resolve_observation_mode(
                    identity_algo,
                    batch_observation_mode,
                    observation_level=observation_level,
                )
                identity_contract = enrich_algorithm_metadata(
                    algo=identity_algo,
                    metadata={},
                    observation_mode=identity_observation_mode,
                    observation_level=observation_level,
                )
                identity_observation_level = str(identity_contract["observation_level"]["key"])
                identity_payload = _scenario_identity_payload(
                    sc,
                    algo=identity_algo,
                    algo_config=identity_cfg,
                    horizon=horizon,
                    dt=dt,
                    record_forces=record_forces,
                    observation_mode=identity_observation_mode,
                    observation_level=identity_observation_level,
                    benchmark_track=benchmark_track,
                    track_schema_version=track_schema_version,
                    observation_noise=noise_spec,
                    synthetic_actuation_profile=(
                        actuation_profile.to_metadata() if actuation_profile is not None else None
                    ),
                    latency_stress_profile=(
                        latency_profile.to_metadata(dt=latency_metadata_dt)
                        if latency_profile is not None
                        else None
                    ),
                    record_simulation_step_trace=record_simulation_step_trace,
                )
                if _compute_map_episode_id(identity_payload, seed) not in existing:
                    filtered_jobs.append((sc, seed))
            jobs = filtered_jobs

    fixed_params = {
        "horizon": horizon,
        "dt": dt,
        "record_forces": record_forces,
        "snqi_weights": snqi_weights,
        "snqi_baseline": snqi_baseline,
        "algo": algo,
        "algo_config": raw_policy_cfg,
        "algo_config_path": algo_config_path,
        "scenario_path": str(scenario_path),
        "adapter_impact_eval": bool(adapter_impact_eval),
        "experimental_ped_impact": bool(experimental_ped_impact),
        "ped_impact_radius_m": float(ped_impact_radius_m),
        "ped_impact_window_steps": int(ped_impact_window_steps),
        "observation_noise": noise_spec,
        "observation_mode": batch_observation_mode,
        "observation_level": observation_level,
        "benchmark_track": benchmark_track,
        "track_schema_version": track_schema_version,
        "synthetic_actuation_profile": (
            actuation_profile.to_metadata() if actuation_profile is not None else None
        ),
        "latency_stress_profile": (
            latency_profile.to_metadata(dt=latency_metadata_dt)
            if latency_profile is not None
            else None
        ),
        "latency_stress_metrics": (
            not_available_latency_metrics() if latency_profile is not None else None
        ),
        "record_simulation_step_trace": bool(record_simulation_step_trace),
    }

    total_jobs = len(jobs)
    wrote = 0
    failures: list[dict[str, Any]] = []
    adapter_native_steps = 0
    adapter_adapted_steps = 0
    adapter_samples_seen = False
    runtime_algorithm_contract: dict[str, Any] | None = None
    feasibility_totals = {
        "commands_evaluated": 0,
        "infeasible_native_count": 0,
        "projected_count": 0,
        "sum_abs_delta_linear": 0.0,
        "sum_abs_delta_angular": 0.0,
        "max_abs_delta_linear": 0.0,
        "max_abs_delta_angular": 0.0,
    }
    batch_started = time.perf_counter()
    if workers <= 1:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            for scenario, seed in jobs:
                try:
                    rec = _run_map_job_worker((scenario, seed, fixed_params))
                    bridge_update = _apply_worker_metadata_bridge(
                        rec,
                        feasibility_totals=feasibility_totals,
                        runtime_algorithm_contract=runtime_algorithm_contract,
                    )
                    adapter_samples_seen = (
                        adapter_samples_seen or bridge_update.adapter_requested_seen
                    )
                    adapter_native_steps += bridge_update.adapter_native_steps
                    adapter_adapted_steps += bridge_update.adapter_adapted_steps
                    runtime_algorithm_contract = bridge_update.runtime_algorithm_contract
                    _write_validated_to_handle(handle, schema, rec)
                    wrote += 1
                except Exception as exc:  # pragma: no cover - error path
                    failures.append(
                        {
                            "scenario_id": scenario.get("name", "unknown"),
                            "seed": seed,
                            "error": repr(exc),
                        }
                    )
    else:
        results_by_idx: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            future_to_job: dict[Any, tuple[int, dict[str, Any], int]] = {}
            for idx, (scenario, seed) in enumerate(jobs, start=1):
                fut = ex.submit(_run_map_job_worker, (scenario, seed, fixed_params))
                future_to_job[fut] = (idx, scenario, seed)
            for fut in as_completed(future_to_job):
                idx, scenario, seed = future_to_job[fut]
                try:
                    rec = fut.result()
                    bridge_update = _apply_worker_metadata_bridge(
                        rec,
                        feasibility_totals=feasibility_totals,
                        runtime_algorithm_contract=runtime_algorithm_contract,
                    )
                    adapter_samples_seen = (
                        adapter_samples_seen or bridge_update.adapter_requested_seen
                    )
                    adapter_native_steps += bridge_update.adapter_native_steps
                    adapter_adapted_steps += bridge_update.adapter_adapted_steps
                    runtime_algorithm_contract = bridge_update.runtime_algorithm_contract
                    results_by_idx[idx] = rec
                except Exception as exc:  # pragma: no cover
                    failures.append(
                        {
                            "scenario_id": scenario.get("name", "unknown"),
                            "seed": seed,
                            "error": repr(exc),
                        }
                    )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            for idx in sorted(results_by_idx):
                try:
                    _write_validated_to_handle(handle, schema, results_by_idx[idx])
                    wrote += 1
                except Exception as exc:  # pragma: no cover - write/validate path
                    rec = results_by_idx[idx]
                    failures.append(
                        {
                            "scenario_id": rec.get("scenario_id")
                            or rec.get("scenario", {}).get("name", "unknown"),
                            "seed": rec.get("seed", -1),
                            "error": repr(exc),
                        }
                    )

    impact_contract = algo_contract.get("adapter_impact")
    if (
        isinstance(impact_contract, dict)
        and bool(impact_contract.get("requested", False))
        and adapter_samples_seen
    ):
        impact_contract["native_steps"] = int(adapter_native_steps)
        impact_contract["adapted_steps"] = int(adapter_adapted_steps)
        total_steps = adapter_native_steps + adapter_adapted_steps
        if total_steps > 0:
            execution_mode = infer_execution_mode_from_counts(
                adapter_native_steps, adapter_adapted_steps
            )
            impact_contract["status"] = "complete"
            impact_contract["execution_mode"] = execution_mode
            impact_contract["adapter_fraction"] = float(adapter_adapted_steps / total_steps)
            algo_contract = enrich_algorithm_metadata(
                algo=algo,
                metadata=algo_contract,
                execution_mode=execution_mode,
                robot_kinematics=kinematics_tag,
                observation_mode=active_observation_mode,
                observation_level=active_observation_level,
            )
            attach_track_metadata(
                algo_contract,
                benchmark_track=benchmark_track,
                track_schema_version=track_schema_version,
                observation_level=active_observation_level,
                observation_mode=active_observation_mode,
            )
        else:
            impact_contract["status"] = "not_applicable"
            impact_contract["adapter_fraction"] = 0.0

    algo_contract = _merge_runtime_algorithm_contract(
        algo_contract,
        runtime_algorithm_contract,
    )
    preflight["algorithm_metadata_contract"] = algo_contract
    planner_contract = algo_contract.get("planner_kinematics")
    if isinstance(planner_contract, dict) and planner_contract.get("planner_command_space") in {
        None,
        "unknown",
    }:
        planner_contract["planner_command_space"] = _default_robot_command_space(
            kinematics_tag,
            policy_cfg,
            robot_command_mode=robot_command_mode,
        )
    total_commands = int(feasibility_totals["commands_evaluated"])
    algo_contract["kinematics_feasibility"] = {
        "commands_evaluated": total_commands,
        "infeasible_native_count": int(feasibility_totals["infeasible_native_count"]),
        "projected_count": int(feasibility_totals["projected_count"]),
        "projection_rate": (
            float(feasibility_totals["projected_count"] / total_commands)
            if total_commands > 0
            else 0.0
        ),
        "infeasible_rate": (
            float(feasibility_totals["infeasible_native_count"] / total_commands)
            if total_commands > 0
            else 0.0
        ),
        "mean_abs_delta_linear": (
            float(feasibility_totals["sum_abs_delta_linear"] / total_commands)
            if total_commands > 0
            else 0.0
        ),
        "mean_abs_delta_angular": (
            float(feasibility_totals["sum_abs_delta_angular"] / total_commands)
            if total_commands > 0
            else 0.0
        ),
        "max_abs_delta_linear": float(feasibility_totals["max_abs_delta_linear"]),
        "max_abs_delta_angular": float(feasibility_totals["max_abs_delta_angular"]),
    }

    summary = {
        "total_jobs": total_jobs,
        "workers": int(workers),
        "parallel_execution": bool(workers > 1),
        "batch_runtime_sec": float(max(time.perf_counter() - batch_started, 0.0)),
        "written": wrote,
        "successful_jobs": wrote,
        "failed_jobs": len(failures),
        "skipped_jobs": preflight_skipped_jobs,
        "failures": failures,
        "out_path": str(out_path),
        "algorithm_readiness": {
            "name": readiness.canonical_name if readiness is not None else algo,
            "tier": readiness.tier if readiness is not None else "unknown",
            "profile": benchmark_profile,
        },
        "algorithm_metadata_contract": algo_contract,
        "preflight": preflight,
        "observation_noise": noise_spec,
        "observation_noise_hash": noise_hash,
        "observation_level": active_observation_level,
        "synthetic_actuation_profile": (
            actuation_profile.to_metadata() if actuation_profile is not None else None
        ),
        "latency_stress_profile": (
            latency_profile.to_metadata(dt=latency_metadata_dt)
            if latency_profile is not None
            else None
        ),
        "latency_stress_metrics": (
            not_available_latency_metrics() if latency_profile is not None else None
        ),
    }
    if benchmark_track is not None:
        summary["benchmark_track"] = benchmark_track
    if track_schema_version is not None:
        summary["track_schema_version"] = track_schema_version
    summary["benchmark_availability"] = availability_payload(summary)
    return summary


__all__ = ["run_map_batch"]
