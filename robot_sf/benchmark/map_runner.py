"""Map-based benchmark runner using Gym environments and scenario YAMLs."""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
from loguru import logger

from robot_sf.baselines.dr_mpc import DRMPCPlanner, build_dr_mpc_config
from robot_sf.baselines.drl_vo import DrlVoPlanner
from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig
from robot_sf.baselines.sac import SACPlanner
from robot_sf.baselines.sicnav import SICNavPlanner, build_sicnav_config
from robot_sf.benchmark import map_runner_policy_resolution as _policy_resolution
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
    not_available_latency_metrics,
)
from robot_sf.benchmark.map_runner_actions import DEFAULT_KINEMATICS as _DEFAULT_KINEMATICS
from robot_sf.benchmark.map_runner_actions import (
    command_xy_payload as _command_xy_payload,  # noqa: F401 - compatibility re-export for tests.
)
from robot_sf.benchmark.map_runner_actions import (
    policy_command_to_env_action as _policy_command_to_env_action,
)
from robot_sf.benchmark.map_runner_actions import robot_kinematics_label as _robot_kinematics_label
from robot_sf.benchmark.map_runner_actions import robot_max_speed as _robot_max_speed
from robot_sf.benchmark.map_runner_actions import (
    scenario_robot_kinematics_label as _scenario_robot_kinematics_label,
)
from robot_sf.benchmark.map_runner_actions import stack_ped_positions as _stack_ped_positions
from robot_sf.benchmark.map_runner_actions import vel_and_acc as _vel_and_acc
from robot_sf.benchmark.map_runner_batch_runner import execute_map_jobs as _execute_map_jobs
from robot_sf.benchmark.map_runner_batch_summary import (
    WorkerMetadataBridgeUpdate as _WorkerMetadataBridgeUpdate,  # noqa: F401 - compatibility export.
)
from robot_sf.benchmark.map_runner_batch_summary import (
    accumulate_batch_metadata as _accumulate_batch_metadata,  # noqa: F401 - compatibility export.
)
from robot_sf.benchmark.map_runner_batch_summary import (
    apply_worker_metadata_bridge as _apply_worker_metadata_bridge,
)
from robot_sf.benchmark.map_runner_batch_summary import (
    merge_runtime_algorithm_contract as _merge_runtime_algorithm_contract,
)
from robot_sf.benchmark.map_runner_env import (
    apply_active_observation_mode_to_env_config as _apply_active_observation_mode_to_env_config,
)
from robot_sf.benchmark.map_runner_env import (
    apply_policy_env_observation_overrides as _apply_policy_env_observation_overrides,
)
from robot_sf.benchmark.map_runner_env import build_env_config as _build_env_config
from robot_sf.benchmark.map_runner_env import (
    validate_sensor_fusion_adapter_config as _validate_sensor_fusion_adapter_config,
)
from robot_sf.benchmark.map_runner_identity import compute_map_episode_id as _compute_map_episode_id
from robot_sf.benchmark.map_runner_identity import resolve_seed_list as _resolve_seed_list
from robot_sf.benchmark.map_runner_identity import (
    scenario_identity_payload as _scenario_identity_payload,
)
from robot_sf.benchmark.map_runner_identity import (
    scenario_with_episode_seed_defaults as _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.map_runner_identity import select_seeds as _select_seeds
from robot_sf.benchmark.map_runner_identity import suite_key as _suite_key
from robot_sf.benchmark.map_runner_jsonl import write_validated_to_handle as _write_jsonl_record
from robot_sf.benchmark.map_runner_metrics import (
    floor_collision_metrics_from_flags as _floor_collision_metrics_from_flags,
)
from robot_sf.benchmark.map_runner_metrics import (
    normalize_pedestrian_impact_controls as _normalize_pedestrian_impact_controls,
)
from robot_sf.benchmark.map_runner_policy_metadata import (
    apply_direct_world_velocity_metadata as _apply_direct_world_velocity_metadata,
)
from robot_sf.benchmark.map_runner_policy_metadata import (
    attach_planner_reset as _attach_planner_reset,
)
from robot_sf.benchmark.map_runner_policy_metadata import (
    finalize_feasibility_metadata as _finalize_feasibility_metadata,
)
from robot_sf.benchmark.map_runner_policy_metadata import (
    holonomic_world_velocity_command as _holonomic_world_velocity_command,
)
from robot_sf.benchmark.map_runner_profile_metadata import (
    load_latency_profile as _load_latency_profile,
)
from robot_sf.benchmark.map_runner_profile_metadata import (
    load_synthetic_actuation_profile as _load_synthetic_actuation_profile_impl,
)
from robot_sf.benchmark.map_runner_provenance import (
    map_result_provenance as _map_result_provenance,
)
from robot_sf.benchmark.map_runner_static_deadlock import (
    static_deadlock_trace_fields as _static_deadlock_trace_fields,
)
from robot_sf.benchmark.map_runner_trace import (
    _command_action_payload,
    _cyclist_like_vru_summary,
    _episode_metadata_for_signal_metrics,
    _fast_bicycle_actor_summary,
    _intent_conditioned_behavior_summary,
    _observation_heading,
    _scenario_id,
    _signal_state_for_metric_metadata,  # noqa: F401 - compatibility re-export for tests.
    _signal_state_promotion_contract,  # noqa: F401 - compatibility re-export for tests.
    _signal_state_proxy_wrapper,  # noqa: F401 - compatibility re-export for tests.
    _single_pedestrian_intent_metadata,
    _single_pedestrian_vru_metadata,
    _trace_pedestrians,
)
from robot_sf.benchmark.map_runner_worker import execute_map_job as _execute_map_job
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
from robot_sf.benchmark.schema_validator import load_schema
from robot_sf.benchmark.synthetic_actuation import (
    SyntheticActuationController,
    not_available_saturation_metrics,
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
)
from robot_sf.common.math_utils import wrap_angle_pi as _normalize_heading
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.planner.adaptive_proxemic_selector import (
    AdaptiveProxemicSelectorAdapter,
    build_adaptive_proxemic_selector_config,
)
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
from robot_sf.training.scenario_loader import load_scenarios

if TYPE_CHECKING:
    from collections.abc import Callable


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
_STRICT_LEARNED_POLICY_PROFILES = {"baseline-safe", "paper-baseline"}
_PPO_ALLOWED_OBS_MODES = {"vector", "dict", "native_dict", "multi_input"}
_PPO_ALLOWED_ACTION_SPACES = {"velocity", "unicycle"}
_PPO_WARN_ROBOT_KINEMATICS = {"holonomic", "omni", "omnidirectional"}

_default_robot_command_space = planner_commands.default_robot_command_space
_init_feasibility_metadata = planner_commands.init_feasibility_metadata
_planner_kinematics_compatibility = planner_commands.planner_kinematics_compatibility
_project_with_feasibility = planner_commands.project_with_feasibility
_validate_planner_contract = planner_commands.validate_planner_contract


_load_synthetic_actuation_profile = _load_synthetic_actuation_profile_impl
_load_latency_stress_profile = _load_latency_profile


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


_parse_algo_config = _policy_resolution._parse_algo_config
_deep_merge_config = _policy_resolution._deep_merge_config
_resolve_config_path = _policy_resolution._resolve_config_path
_scenario_family = _policy_resolution._scenario_family
_is_policy_search_candidate_manifest = _policy_resolution._is_policy_search_candidate_manifest
_load_base_candidate_config = _policy_resolution._load_base_candidate_config
_scenario_algo_override_runtime = _policy_resolution._scenario_algo_override_runtime
_resolve_policy_search_candidate_runtime = (
    _policy_resolution._resolve_policy_search_candidate_runtime
)
_apply_planner_selector_v2_context = _policy_resolution._apply_planner_selector_v2_context
_build_planner_selector_v2_child_adapter = (
    _policy_resolution._build_planner_selector_v2_child_adapter
)
_build_planner_selector_v2_adapter = _policy_resolution._build_planner_selector_v2_adapter
_prediction_planner_metadata_overrides = _policy_resolution._prediction_planner_metadata_overrides


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
        meta["residual_clipping_stats"] = {
            "schema_version": "orca-residual-clipping-stats.v1",
            "decision_count": 0,
            "clipped_count": 0,
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
            residual_stats = meta.get("residual_clipping_stats")
            if isinstance(residual_stats, dict):
                decision_metadata = shield_decision.to_metadata()
                fallback_state = decision_metadata.get("fallback_controller_state")
                action_adaptation = (
                    fallback_state.get("action_adaptation")
                    if isinstance(fallback_state, dict)
                    else None
                )
                if isinstance(action_adaptation, dict):
                    residual_stats["decision_count"] = (
                        int(residual_stats.get("decision_count", 0)) + 1
                    )
                    if bool(action_adaptation.get("residual_clipped", False)):
                        residual_stats["clipped_count"] = (
                            int(residual_stats.get("clipped_count", 0)) + 1
                        )
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
    _write_jsonl_record(handle, schema, record)


def _run_map_job_worker(
    job: tuple[dict[str, Any], int, dict[str, Any]],
) -> dict[str, Any]:
    """Execute one serialized map-runner job.

    Returns:
        dict[str, Any]: Episode record returned by ``_run_map_episode``.
    """
    return _execute_map_job(job, run_map_episode=_run_map_episode)


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
        summary["provenance"] = _map_result_provenance(
            schema_path=schema_path,
            scenario_path=scenario_path,
            scenarios=filtered,
            algo=algo,
            algo_config_path=algo_config_path,
            benchmark_profile=benchmark_profile,
            suite_key=suite_key,
            total_jobs=len(jobs),
            written=0,
            artifact_pointer_status="not_available",
        )
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
    batch_execution = _execute_map_jobs(
        jobs=jobs,
        fixed_params=fixed_params,
        out_path=out_path,
        schema=schema,
        workers=workers,
        run_map_job=_run_map_job_worker,
        write_validated_to_handle=_write_validated_to_handle,
        apply_worker_metadata_bridge=_apply_worker_metadata_bridge,
        scenario_id=_scenario_id,
        executor_cls=ProcessPoolExecutor,
        as_completed_fn=as_completed,
    )
    wrote = batch_execution.wrote
    failures = batch_execution.failures
    adapter_native_steps = batch_execution.adapter_native_steps
    adapter_adapted_steps = batch_execution.adapter_adapted_steps
    adapter_samples_seen = batch_execution.adapter_samples_seen
    runtime_algorithm_contract = batch_execution.runtime_algorithm_contract
    feasibility_totals = batch_execution.feasibility_totals

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
        "batch_runtime_sec": batch_execution.batch_runtime_sec,
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
    artifact_pointer_status = "local_jsonl_present" if out_path.exists() else "not_available"
    summary["provenance"] = _map_result_provenance(
        schema_path=schema_path,
        scenario_path=scenario_path,
        scenarios=filtered,
        algo=algo,
        algo_config_path=algo_config_path,
        benchmark_profile=benchmark_profile,
        suite_key=suite_key,
        total_jobs=total_jobs,
        written=wrote,
        artifact_pointer_status=artifact_pointer_status,
    )
    summary["benchmark_availability"] = availability_payload(summary)
    return summary


__all__ = ["run_map_batch"]
