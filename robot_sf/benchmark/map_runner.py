"""Map-based benchmark runner using Gym environments and scenario YAMLs."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
from loguru import logger

from robot_sf.baselines.dr_mpc import DRMPCPlanner, build_dr_mpc_config
from robot_sf.baselines.drl_vo import DrlVoPlanner
from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig
from robot_sf.baselines.sac import SACPlanner
from robot_sf.baselines.sicnav import SICNavPlanner, build_sicnav_config
from robot_sf.benchmark import map_runner_episode as _map_runner_episode_module
from robot_sf.benchmark import map_runner_policy_resolution as _policy_resolution
from robot_sf.benchmark import planner_command_contract as planner_commands
from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    resolve_learned_checkpoint_observation_contract,
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
    policy_command_to_env_action as _policy_command_to_env_action,  # noqa: F401 - compatibility re-export.
)
from robot_sf.benchmark.map_runner_actions import (
    robot_kinematics_label as _robot_kinematics_label,  # noqa: F401 - compatibility re-export.
)
from robot_sf.benchmark.map_runner_actions import (
    robot_max_speed as _robot_max_speed,  # noqa: F401 - compatibility re-export.
)
from robot_sf.benchmark.map_runner_actions import (
    scenario_robot_kinematics_label as _scenario_robot_kinematics_label,
)
from robot_sf.benchmark.map_runner_actions import (
    stack_ped_positions as _stack_ped_positions,  # noqa: F401 - compatibility re-export.
)
from robot_sf.benchmark.map_runner_actions import (
    vel_and_acc as _vel_and_acc,  # noqa: F401 - compatibility re-export.
)
from robot_sf.benchmark.map_runner_batch_plan import (
    build_seed_jobs as _build_seed_jobs,
)
from robot_sf.benchmark.map_runner_batch_plan import (
    build_worker_fixed_params as _build_worker_fixed_params,
)
from robot_sf.benchmark.map_runner_batch_plan import (
    resolve_batch_kinematics_tag as _resolve_batch_kinematics_tag,
)
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
    build_completed_batch_summary as _build_completed_batch_summary,
)
from robot_sf.benchmark.map_runner_env import (
    apply_active_observation_mode_to_env_config as _apply_active_observation_mode_to_env_config,  # noqa: F401 - compatibility re-export.
)
from robot_sf.benchmark.map_runner_env import (
    apply_policy_env_observation_overrides as _apply_policy_env_observation_overrides,  # noqa: F401 - compatibility re-export.
)
from robot_sf.benchmark.map_runner_env import build_env_config as _build_env_config
from robot_sf.benchmark.map_runner_env import (
    representative_metric_affecting_config as _representative_metric_affecting_config,
)
from robot_sf.benchmark.map_runner_env import (
    validate_sensor_fusion_adapter_config as _validate_sensor_fusion_adapter_config,  # noqa: F401 - compatibility re-export.
)
from robot_sf.benchmark.map_runner_episode import run_map_episode as _execute_map_episode
from robot_sf.benchmark.map_runner_identity import (
    _compute_map_episode_id,
    _resolve_seed_list,
    _scenario_identity_payload,
    _scenario_with_episode_seed_defaults,  # noqa: F401 - compatibility re-export.
    _select_seeds,  # noqa: F401 - compatibility re-export for tests.
    _suite_key,
)
from robot_sf.benchmark.map_runner_jsonl import write_validated_to_handle as _write_jsonl_record
from robot_sf.benchmark.map_runner_metrics import (
    floor_collision_metrics_from_flags as _floor_collision_metrics_from_flags,  # noqa: F401 - compatibility re-export.
)
from robot_sf.benchmark.map_runner_metrics import (
    normalize_pedestrian_impact_controls as _normalize_pedestrian_impact_controls,
)
from robot_sf.benchmark.map_runner_metrics import summarize_collision_metrics
from robot_sf.benchmark.map_runner_observations import (
    extract_ppo_dt as _extract_ppo_dt,  # noqa: F401 - compatibility re-export for tests.
)
from robot_sf.benchmark.map_runner_observations import (
    extract_ppo_pedestrians as _extract_ppo_pedestrians,  # noqa: F401 - compatibility re-export.
)
from robot_sf.benchmark.map_runner_observations import (
    normalize_xy_rows as _normalize_xy_rows,  # noqa: F401 - compatibility re-export for tests.
)
from robot_sf.benchmark.map_runner_observations import (
    obs_to_external_mpc_format as _obs_to_external_mpc_format,
)
from robot_sf.benchmark.map_runner_observations import obs_to_ppo_format as _obs_to_ppo_format
from robot_sf.benchmark.map_runner_policies import adapters as _adapter_policy_builders
from robot_sf.benchmark.map_runner_policies import adaptive_proxemic as _adaptive_proxemic_builder
from robot_sf.benchmark.map_runner_policies import diffusion_policy as _diffusion_policy_builder
from robot_sf.benchmark.map_runner_policies import distributional_rl as _distributional_rl_builder
from robot_sf.benchmark.map_runner_policies import gap_reference as _gap_reference_builder
from robot_sf.benchmark.map_runner_policies import goal as _goal_policy_builder
from robot_sf.benchmark.map_runner_policies import group_avoidance as _group_avoidance_builder
from robot_sf.benchmark.map_runner_policies import hybrid_global_rl as _hybrid_global_rl_builder
from robot_sf.benchmark.map_runner_policies import registry as _policy_builder_registry
from robot_sf.benchmark.map_runner_policies import rule_and_grid as _rule_and_grid_builder
from robot_sf.benchmark.map_runner_policies import safety_barrier as _safety_barrier_builder
from robot_sf.benchmark.map_runner_policy_actions import (
    ppo_action_to_unicycle as _ppo_action_to_unicycle_impl,
)
from robot_sf.benchmark.map_runner_policy_actions import (
    update_adapter_impact_metrics as _update_adapter_impact_metrics,
)
from robot_sf.benchmark.map_runner_policy_common import (
    build_adapter_policy as _build_adapter_policy,
)
from robot_sf.benchmark.map_runner_policy_metadata import (
    apply_direct_world_velocity_metadata as _apply_direct_world_velocity_metadata,
)
from robot_sf.benchmark.map_runner_policy_metadata import (
    attach_planner_reset as _attach_planner_reset,
)
from robot_sf.benchmark.map_runner_policy_metadata import (
    finalize_feasibility_metadata as _finalize_feasibility_metadata,  # noqa: F401 - compatibility re-export.
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
from robot_sf.benchmark.map_runner_trace import (
    _command_action_payload,  # noqa: F401 - compatibility re-export.
    _cyclist_like_vru_summary,  # noqa: F401 - compatibility re-export.
    _episode_metadata_for_signal_metrics,  # noqa: F401 - compatibility re-export.
    _fast_bicycle_actor_summary,  # noqa: F401 - compatibility re-export.
    _intent_conditioned_behavior_summary,  # noqa: F401 - compatibility re-export.
    _observation_heading,  # noqa: F401 - compatibility re-export.
    _scenario_id,
    _signal_state_for_metric_metadata,  # noqa: F401 - compatibility re-export for tests.
    _signal_state_promotion_contract,  # noqa: F401 - compatibility re-export for tests.
    _signal_state_proxy_wrapper,  # noqa: F401 - compatibility re-export for tests.
    _single_pedestrian_intent_metadata,  # noqa: F401 - compatibility re-export.
    _single_pedestrian_vru_metadata,  # noqa: F401 - compatibility re-export.
    _trace_pedestrians,  # noqa: F401 - compatibility re-export.
)
from robot_sf.benchmark.map_runner_worker import execute_map_job as _execute_map_job
from robot_sf.benchmark.metrics import compute_all_metrics, post_process_metrics
from robot_sf.benchmark.observation_noise import (
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.obstacle_sampling import sample_obstacle_points
from robot_sf.benchmark.path_utils import compute_shortest_path_length
from robot_sf.benchmark.policy_builders import build_registered_adapter_policy_spec
from robot_sf.benchmark.result_provenance import (
    build_result_provenance_manifest as _build_result_provenance_manifest,
)
from robot_sf.benchmark.result_provenance import (
    manifest_path_for_result_jsonl as _provenance_manifest_path,
)
from robot_sf.benchmark.result_provenance import (
    write_result_provenance_manifest as _write_result_provenance_manifest,
)
from robot_sf.benchmark.scenario_schema import validate_scenario_list
from robot_sf.benchmark.schema_validator import load_schema
from robot_sf.benchmark.tracking_precision_contract import (
    normalize_tracking_precision_spec,
    tracking_precision_hash,
)
from robot_sf.benchmark.utils import (
    _config_hash,
    attach_track_metadata,
    index_existing,
    normalize_track_field,
)
from robot_sf.common.math_utils import wrap_angle_pi as _normalize_heading
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.planner.gap_prediction import GapAwarePredictionAdapter  # noqa: F401
from robot_sf.planner.grid_route import (  # noqa: F401
    GridRoutePlannerAdapter,
    build_grid_route_config,
)
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
from robot_sf.planner.kinematics_model import resolve_benchmark_kinematics_model
from robot_sf.planner.lidar_occupancy import (  # noqa: F401
    LidarOccupancyPlannerAdapter,
    build_lidar_occupancy_config,
)
from robot_sf.planner.mppi_social import (
    MPPISocialPlannerAdapter,
    build_mppi_social_config,
)
from robot_sf.planner.nmpc_social import (
    NMPCSocialPlannerAdapter,
    build_nmpc_social_config,
)
from robot_sf.planner.predictive_mppi import (
    PredictiveMPPIAdapter,
    build_predictive_mppi_config,
)
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter
from robot_sf.planner.safety_barrier import (  # noqa: F401
    SafetyBarrierPlannerAdapter,
    build_safety_barrier_config,
)
from robot_sf.planner.safety_shield import (
    ShieldDecision,
    new_shield_stats,
    shield_contract_metadata,
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
)
from robot_sf.planner.stream_gap import StreamGapPlannerAdapter  # noqa: F401
from robot_sf.training.scenario_loader import load_scenarios

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class _CrowdNavHeightConfigFallback:
    """Lightweight config used when tests monkeypatch the optional adapter."""

    repo_root: Path = Path("output/repos/CrowdNav_HEIGHT")
    model_dir: Path = Path("output/external_checkpoints/crowdnav_height_extracted/HEIGHT/HEIGHT")
    checkpoint_name: str = "237800.pt"
    device: str = "cpu"
    max_linear_speed: float = 0.5
    max_angular_speed: float = 1.0


@dataclass(frozen=True)
class _SonicCrowdNavConfigFallback:
    """Lightweight config used when tests monkeypatch the optional adapter."""

    repo_root: Path = Path("output/repos/SoNIC-Social-Nav")
    model_name: str = "SoNIC_GST"
    checkpoint_name: str = "05207.pt"
    device: str = "cpu"
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0


CrowdNavHeightAdapter: Any | None = None
build_crowdnav_height_config: Any | None = None
SonicCrowdNavAdapter: Any | None = None
build_sonic_crowdnav_config: Any | None = None


def _fallback_crowdnav_height_config(data: dict[str, Any] | None) -> _CrowdNavHeightConfigFallback:
    """Build config without importing torch-backed CrowdNav module.

    Returns:
        _CrowdNavHeightConfigFallback: Minimal adapter config.
    """

    payload = data or {}
    max_linear_speed = float(payload.get("max_linear_speed", 0.5))
    max_angular_speed = float(payload.get("max_angular_speed", 1.0))
    if max_linear_speed < 0.0 or max_angular_speed < 0.0:
        raise ValueError("max_linear_speed and max_angular_speed must be non-negative")
    return _CrowdNavHeightConfigFallback(
        repo_root=Path(str(payload.get("repo_root", _CrowdNavHeightConfigFallback.repo_root))),
        model_dir=Path(str(payload.get("model_dir", _CrowdNavHeightConfigFallback.model_dir))),
        checkpoint_name=str(
            payload.get("checkpoint_name", _CrowdNavHeightConfigFallback.checkpoint_name)
        ),
        device=str(payload.get("device", "cpu")).strip() or "cpu",
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )


def _crowdnav_height_symbols() -> tuple[Any, Any]:
    """Return CrowdNav HEIGHT adapter/config builder, importing torch-backed code lazily."""

    global CrowdNavHeightAdapter, build_crowdnav_height_config
    if CrowdNavHeightAdapter is None:
        from robot_sf.planner.crowdnav_height import (  # noqa: PLC0415
            CrowdNavHeightAdapter as _CrowdNavHeightAdapter,
        )
        from robot_sf.planner.crowdnav_height import (  # noqa: PLC0415
            build_crowdnav_height_config as _build_crowdnav_height_config,
        )

        CrowdNavHeightAdapter = _CrowdNavHeightAdapter
        build_crowdnav_height_config = _build_crowdnav_height_config
    builder = build_crowdnav_height_config or _fallback_crowdnav_height_config
    return CrowdNavHeightAdapter, builder


def _fallback_sonic_crowdnav_config(data: dict[str, Any] | None) -> _SonicCrowdNavConfigFallback:
    """Build SoNIC config without importing the torch-backed module.

    Returns:
        _SonicCrowdNavConfigFallback: Minimal adapter config.
    """

    payload = data or {}
    max_linear_speed = float(payload.get("max_linear_speed", 1.0))
    max_angular_speed = float(payload.get("max_angular_speed", 1.0))
    if max_linear_speed < 0.0 or max_angular_speed < 0.0:
        raise ValueError("max_linear_speed and max_angular_speed must be non-negative")
    return _SonicCrowdNavConfigFallback(
        repo_root=Path(str(payload.get("repo_root", _SonicCrowdNavConfigFallback.repo_root))),
        model_name=str(payload.get("model_name", _SonicCrowdNavConfigFallback.model_name)).strip()
        or _SonicCrowdNavConfigFallback.model_name,
        checkpoint_name=str(
            payload.get("checkpoint_name", _SonicCrowdNavConfigFallback.checkpoint_name)
        ).strip()
        or _SonicCrowdNavConfigFallback.checkpoint_name,
        device=str(payload.get("device", "cpu")).strip() or "cpu",
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )


def _sonic_crowdnav_symbols() -> tuple[Any, Any]:
    """Return SoNIC adapter/config builder, importing torch-backed code lazily.

    Returns:
        tuple[Any, Any]: Adapter class and config-builder callable.
    """

    global SonicCrowdNavAdapter, build_sonic_crowdnav_config
    if SonicCrowdNavAdapter is None:
        from robot_sf.planner.sonic_crowdnav import (  # noqa: PLC0415
            SonicCrowdNavAdapter as _SonicCrowdNavAdapter,
        )
        from robot_sf.planner.sonic_crowdnav import (  # noqa: PLC0415
            build_sonic_crowdnav_config as _build_sonic_crowdnav_config,
        )

        SonicCrowdNavAdapter = _SonicCrowdNavAdapter
        build_sonic_crowdnav_config = _build_sonic_crowdnav_config
    builder = build_sonic_crowdnav_config or _fallback_sonic_crowdnav_config
    return SonicCrowdNavAdapter, builder


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
_apply_scenario_uncertainty_envelope_config = (
    _policy_resolution._apply_scenario_uncertainty_envelope_config
)
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
    kinematics_model: Any | None = None,
    project_command: bool = True,
) -> tuple[float, float, str]:
    """Compatibility wrapper for the neutral learned-policy action helper.

    Returns:
        Tuple ``(linear_velocity, angular_velocity, conversion_mode)``.
    """
    if kinematics_model is None:
        kinematics_model = resolve_benchmark_kinematics_model(
            robot_kinematics=robot_kinematics,
            command_limits=cfg,
        )
    return _ppo_action_to_unicycle_impl(
        action,
        obs,
        cfg,
        robot_kinematics=robot_kinematics,
        kinematics_model=kinematics_model,
        project_command=project_command,
    )


# Registry of migrated per-algorithm policy builders (#3384). Consulted before the
# inline dispatch in _build_policy; families not yet migrated fall through to the
# existing if/elif chain.
_POLICY_BUILDERS: dict[str, _policy_builder_registry.PolicyBuilder] = {
    **dict.fromkeys(_goal_policy_builder.GOAL_ALGO_KEYS, _goal_policy_builder.build),
    **dict.fromkeys(
        _adapter_policy_builders.RISK_SURFACE_DWA_KEYS,
        _adapter_policy_builders.build_risk_surface_dwa,
    ),
    **dict.fromkeys(
        _adapter_policy_builders.LIDAR_SOCIAL_FORCE_KEYS,
        _adapter_policy_builders.build_lidar_social_force,
    ),
    **dict.fromkeys(
        _adaptive_proxemic_builder.ADAPTIVE_PROXEMIC_SELECTOR_KEYS,
        _adaptive_proxemic_builder.build,
    ),
    **dict.fromkeys(
        _diffusion_policy_builder.DIFFUSION_POLICY_KEYS,
        _diffusion_policy_builder.build,
    ),
    **dict.fromkeys(
        _distributional_rl_builder.DISTRIBUTIONAL_RL_KEYS,
        _distributional_rl_builder.build,
    ),
    **dict.fromkeys(
        _group_avoidance_builder.GROUP_AVOIDANCE_ALGO_KEYS,
        _group_avoidance_builder.build,
    ),
    **dict.fromkeys(_rule_and_grid_builder.RULE_AND_GRID_KEYS, _rule_and_grid_builder.build),
    **dict.fromkeys(_safety_barrier_builder.ADAPTER_ALGO_KEYS, _safety_barrier_builder.build),
    **dict.fromkeys(_gap_reference_builder.GAP_REFERENCE_KEYS, _gap_reference_builder.build),
    **dict.fromkeys(
        _hybrid_global_rl_builder.HYBRID_GLOBAL_RL_KEYS, _hybrid_global_rl_builder.build
    ),
}


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
    registered_policy = _policy_builder_registry.build_registered_policy(
        algo_key,
        algo_config,
        builders=_POLICY_BUILDERS,
        robot_kinematics=robot_kinematics,
        robot_command_mode=robot_command_mode,
        adapter_impact_eval=adapter_impact_eval,
    )
    if registered_policy is not None:
        return registered_policy

    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    registered_adapter_spec = build_registered_adapter_policy_spec(algo_key, algo_config)
    if registered_adapter_spec is not None:
        meta["algorithm"] = registered_adapter_spec.algo_key
        return _build_adapter_policy(
            algo_key=registered_adapter_spec.algo_key,
            algo_config=registered_adapter_spec.algo_config,
            meta=meta,
            adapter=registered_adapter_spec.adapter,
            adapter_name=registered_adapter_spec.adapter_name,
            robot_kinematics=robot_kinematics,
            normalized_robot_command_mode=normalized_robot_command_mode,
            limitations=registered_adapter_spec.limitations,
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
            "uncertainty_fallback_stop": 0,
            "uncertainty_fallback_slow_down": 0,
            "uncertainty_fallback_configured": 0,
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
        sonic_adapter_cls, sonic_config_builder = _sonic_crowdnav_symbols()
        sonic_adapter = sonic_adapter_cls(
            config=sonic_config_builder(
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
            "uncertainty_fallback_stop": 0,
            "uncertainty_fallback_slow_down": 0,
            "uncertainty_fallback_configured": 0,
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
        crowdnav_adapter_cls, crowdnav_config_builder = _crowdnav_height_symbols()
        adapter = crowdnav_adapter_cls(config=crowdnav_config_builder(algo_config))
    elif algo_key in {"sonic_crowdnav", "sonic_gst"}:
        sonic_adapter_cls, sonic_config_builder = _sonic_crowdnav_symbols()
        adapter = sonic_adapter_cls(config=sonic_config_builder(algo_config))
    elif algo_key in {"gensafenav_ours_gst", "gensafe_ours_gst", "ours_gst"}:
        sonic_adapter_cls, sonic_config_builder = _sonic_crowdnav_symbols()
        adapter = sonic_adapter_cls(
            config=sonic_config_builder(
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
        sonic_adapter_cls, sonic_config_builder = _sonic_crowdnav_symbols()
        adapter = sonic_adapter_cls(
            config=sonic_config_builder(
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


def _sync_episode_compat_overrides() -> None:
    """Propagate legacy monkeypatchable map-runner hooks into the episode module."""
    _map_runner_episode_module._build_env_config = _build_env_config
    _map_runner_episode_module.make_robot_env = make_robot_env
    _map_runner_episode_module.sample_obstacle_points = sample_obstacle_points
    _map_runner_episode_module.compute_shortest_path_length = compute_shortest_path_length
    _map_runner_episode_module.compute_all_metrics = compute_all_metrics
    _map_runner_episode_module.post_process_metrics = post_process_metrics


def _run_map_episode(  # noqa: PLR0913
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
    tracking_precision: dict[str, Any] | None = None,
    synthetic_actuation_profile: dict[str, Any] | None = None,
    latency_stress_profile: dict[str, Any] | None = None,
    safety_wrapper: dict[str, Any] | None = None,
    cbf_safety_filter: dict[str, Any] | None = None,
    record_planner_decision_trace: bool = False,
    record_simulation_step_trace: bool = False,
) -> dict[str, Any]:
    """Run one scenario/seed episode through the extracted episode executor.

    Returns:
        dict[str, Any]: Episode record with metrics, provenance, and planner metadata.
    """
    _sync_episode_compat_overrides()
    return _execute_map_episode(
        scenario,
        seed,
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
        algo=algo,
        scenario_path=scenario_path,
        algo_config=algo_config,
        algo_config_path=algo_config_path,
        adapter_impact_eval=adapter_impact_eval,
        experimental_ped_impact=experimental_ped_impact,
        ped_impact_radius_m=ped_impact_radius_m,
        ped_impact_window_steps=ped_impact_window_steps,
        observation_mode=observation_mode,
        observation_level=observation_level,
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
        observation_noise=observation_noise,
        tracking_precision=tracking_precision,
        synthetic_actuation_profile=synthetic_actuation_profile,
        latency_stress_profile=latency_stress_profile,
        safety_wrapper=safety_wrapper,
        cbf_safety_filter=cbf_safety_filter,
        record_planner_decision_trace=record_planner_decision_trace,
        record_simulation_step_trace=record_simulation_step_trace,
        policy_builder=_build_policy,
    )


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


def _emit_provenance_manifest(  # noqa: PLR0913
    *,
    out_path: Path,
    episode_records: list[dict[str, Any]],
    schema_path: str | Path,
    scenario_path: Path,
    scenarios: list[dict[str, Any]],
    algo: str,
    algo_config_path: str | None,
    benchmark_profile: str,
    suite_key: str,
    total_jobs: int,
    written: int,
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
    active_observation_mode: str | None,
    active_observation_level: str | None,
    noise_hash: str | None = None,
    tracking_precision_hash: str | None = None,
) -> Path:
    """Build and write the provenance manifest for one map-runner batch.

    Returns:
        The path to the written manifest file.
    """
    manifest = _build_result_provenance_manifest(
        out_path=out_path,
        episode_records=episode_records,
        schema_path=schema_path,
        scenario_path=scenario_path,
        scenarios=scenarios,
        algo=algo,
        algo_config_path=algo_config_path,
        benchmark_profile=benchmark_profile,
        suite_key=suite_key,
        total_jobs=total_jobs,
        written=written,
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
        active_observation_mode=active_observation_mode,
        active_observation_level=active_observation_level,
        noise_hash=noise_hash,
        tracking_precision_hash=tracking_precision_hash,
    )
    manifest_path = _provenance_manifest_path(out_path)
    _write_result_provenance_manifest(manifest_path, manifest)
    return manifest_path


def _record_result_provenance_manifest(  # noqa: PLR0913
    *,
    summary: dict[str, Any],
    out_path: Path,
    episode_records: list[dict[str, Any]],
    schema_path: str | Path,
    scenario_path: Path,
    scenarios: list[dict[str, Any]],
    algo: str,
    algo_config_path: str | None,
    benchmark_profile: str,
    suite_key: str,
    total_jobs: int,
    written: int,
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
    active_observation_mode: str | None,
    active_observation_level: str | None,
    noise_hash: str | None = None,
    tracking_precision_hash: str | None = None,
) -> None:
    """Write result provenance sidecar without discarding a completed JSONL artifact."""
    try:
        result_manifest_path = _emit_provenance_manifest(
            out_path=out_path,
            episode_records=episode_records,
            schema_path=schema_path,
            scenario_path=scenario_path,
            scenarios=scenarios,
            algo=algo,
            algo_config_path=algo_config_path,
            benchmark_profile=benchmark_profile,
            suite_key=suite_key,
            total_jobs=total_jobs,
            written=written,
            horizon=horizon,
            dt=dt,
            record_forces=record_forces,
            active_observation_mode=active_observation_mode,
            active_observation_level=active_observation_level,
            noise_hash=noise_hash,
            tracking_precision_hash=tracking_precision_hash,
        )
    except OSError as exc:
        logger.warning("Failed to write result provenance manifest for {}: {}", out_path, exc)
        provenance = summary.setdefault("provenance", {})
        provenance["result_manifest_path"] = str(_provenance_manifest_path(out_path))
        provenance["result_manifest_status"] = "error"
        provenance["result_manifest_error"] = str(exc)
        return

    provenance = summary.setdefault("provenance", {})
    provenance["result_manifest_path"] = str(result_manifest_path)
    provenance["result_manifest_status"] = "available"


def run_map_batch(  # noqa: C901,PLR0912,PLR0913,PLR0915
    scenarios_or_path: list[dict[str, Any]] | str | Path,
    out_path: str | Path,
    schema_path: str | Path,
    *,
    scenario_path: str | Path | None = None,
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
    tracking_precision: dict[str, Any] | None = None,
    synthetic_actuation_profile: dict[str, Any] | None = None,
    latency_stress_profile: dict[str, Any] | None = None,
    safety_wrapper: dict[str, Any] | None = None,
    cbf_safety_filter: dict[str, Any] | None = None,
    record_planner_decision_trace: bool = False,
    record_simulation_step_trace: bool = False,
    multiprocessing_context: Any | None = None,
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
        scenario_path = Path(scenario_path) if scenario_path is not None else Path(".")
        scenarios = list(scenarios_or_path)

    errors = validate_scenario_list([dict(s) for s in scenarios])
    if errors:
        raise ValueError(f"Scenario validation failed: {errors[:3]} (total {len(errors)})")

    suite_seeds = _resolve_seed_list(Path("configs/benchmarks/seed_list_v1.yaml"))
    suite_key = _suite_key(scenario_path)
    noise_spec = normalize_observation_noise_spec(observation_noise)
    noise_hash = observation_noise_hash(noise_spec)
    tracking_precision_spec = normalize_tracking_precision_spec(tracking_precision)
    tracking_precision_spec_hash = tracking_precision_hash(tracking_precision_spec)
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

    # Record the metric-affecting run config (scan_noise, collision regime) once so
    # the emitted result manifest is self-describing about settings that change
    # metric semantics (issue #3701). Fail-soft: never blocks a run.
    metric_affecting_config = _representative_metric_affecting_config(
        filtered, scenario_path=scenario_path
    )

    kinematics_tag, scenario_kinematics = _resolve_batch_kinematics_tag(filtered)
    batch_observation_mode = str(observation_mode).strip() if observation_mode is not None else None
    raw_policy_cfg = _parse_algo_config(algo_config_path)
    _, policy_cfg = _resolve_policy_search_candidate_runtime(
        default_algo=algo,
        algo_config_path=algo_config_path,
        algo_config=raw_policy_cfg,
        scenario={},
    )
    learned_observation_contract = resolve_learned_checkpoint_observation_contract(
        algo,
        policy_cfg,
        observation_mode=batch_observation_mode,
        observation_level=observation_level,
    )
    active_observation_mode = str(learned_observation_contract["active_observation_mode"])
    resolved_observation_level = observation_level
    if resolved_observation_level is None:
        resolved_observation_level = learned_observation_contract.get("observation_level_key")
    algo_contract = enrich_algorithm_metadata(
        algo=algo,
        metadata={},
        robot_kinematics=kinematics_tag,
        adapter_impact_requested=adapter_impact_eval,
        observation_mode=active_observation_mode,
        observation_level=resolved_observation_level,
    )
    algo_contract["learned_checkpoint_observation_contract"] = learned_observation_contract
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
    jobs = _build_seed_jobs(filtered, suite_seeds=suite_seeds, suite_key=suite_key)
    preflight_skipped_jobs = 0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = load_schema(schema_path)
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
            validation_cfg = _apply_scenario_uncertainty_envelope_config(
                validation_algo,
                validation_cfg,
                scenario,
            )
            validation_observation_contract = resolve_learned_checkpoint_observation_contract(
                validation_algo,
                validation_cfg,
                observation_mode=batch_observation_mode,
                observation_level=observation_level,
            )
            validation_observation_mode = str(
                validation_observation_contract["active_observation_mode"]
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
            "tracking_precision": tracking_precision_spec,
            "tracking_precision_hash": tracking_precision_spec_hash,
            "metrics": summarize_collision_metrics([]),
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
            metric_affecting_config=metric_affecting_config,
        )
        summary["benchmark_availability"] = availability_payload(summary)
        # Write not_applicable provenance manifest for skipped preflight path.
        _record_result_provenance_manifest(
            summary=summary,
            out_path=out_path,
            episode_records=[],
            schema_path=schema_path,
            scenario_path=scenario_path,
            scenarios=filtered,
            algo=algo,
            algo_config_path=algo_config_path,
            benchmark_profile=benchmark_profile,
            suite_key=suite_key,
            total_jobs=0,
            written=0,
            horizon=horizon,
            dt=dt,
            record_forces=record_forces,
            active_observation_mode=active_observation_mode,
            active_observation_level=active_observation_level,
            noise_hash=noise_hash,
            tracking_precision_hash=tracking_precision_spec_hash,
        )
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
                identity_cfg = _apply_scenario_uncertainty_envelope_config(
                    identity_algo,
                    identity_cfg,
                    sc,
                )
                identity_observation_contract = resolve_learned_checkpoint_observation_contract(
                    identity_algo,
                    identity_cfg,
                    observation_mode=batch_observation_mode,
                    observation_level=observation_level,
                )
                identity_observation_mode = str(
                    identity_observation_contract["active_observation_mode"]
                )
                identity_observation_level = observation_level
                if identity_observation_level is None:
                    identity_observation_level = identity_observation_contract.get(
                        "observation_level_key"
                    )
                identity_contract = enrich_algorithm_metadata(
                    algo=identity_algo,
                    metadata={},
                    observation_mode=identity_observation_mode,
                    observation_level=identity_observation_level,
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
                    tracking_precision=tracking_precision_spec,
                    synthetic_actuation_profile=(
                        actuation_profile.to_metadata() if actuation_profile is not None else None
                    ),
                    latency_stress_profile=(
                        latency_profile.to_metadata(dt=latency_metadata_dt)
                        if latency_profile is not None
                        else None
                    ),
                    safety_wrapper=dict(safety_wrapper) if safety_wrapper is not None else None,
                    cbf_safety_filter=dict(cbf_safety_filter)
                    if cbf_safety_filter is not None
                    else None,
                    record_planner_decision_trace=record_planner_decision_trace,
                    record_simulation_step_trace=record_simulation_step_trace,
                )
                if _compute_map_episode_id(identity_payload, seed) not in existing:
                    filtered_jobs.append((sc, seed))
            jobs = filtered_jobs

    fixed_params = _build_worker_fixed_params(
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
        algo=algo,
        raw_policy_cfg=raw_policy_cfg,
        algo_config_path=algo_config_path,
        scenario_path=scenario_path,
        adapter_impact_eval=adapter_impact_eval,
        experimental_ped_impact=experimental_ped_impact,
        ped_impact_radius_m=ped_impact_radius_m,
        ped_impact_window_steps=ped_impact_window_steps,
        noise_spec=noise_spec,
        tracking_precision_spec=tracking_precision_spec,
        batch_observation_mode=batch_observation_mode,
        observation_level=observation_level,
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
        actuation_profile_metadata=(
            actuation_profile.to_metadata() if actuation_profile is not None else None
        ),
        latency_profile_metadata=(
            latency_profile.to_metadata(dt=latency_metadata_dt)
            if latency_profile is not None
            else None
        ),
        latency_stress_metrics=(
            not_available_latency_metrics() if latency_profile is not None else None
        ),
        safety_wrapper=dict(safety_wrapper) if safety_wrapper is not None else None,
        cbf_safety_filter=dict(cbf_safety_filter) if cbf_safety_filter is not None else None,
        record_planner_decision_trace=record_planner_decision_trace,
        record_simulation_step_trace=record_simulation_step_trace,
    )

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
        multiprocessing_context=multiprocessing_context,
    )
    wrote = batch_execution.wrote
    episode_records = batch_execution.episode_records
    failures = batch_execution.failures
    adapter_native_steps = batch_execution.adapter_native_steps
    adapter_adapted_steps = batch_execution.adapter_adapted_steps
    adapter_samples_seen = batch_execution.adapter_samples_seen
    runtime_algorithm_contract = batch_execution.runtime_algorithm_contract
    feasibility_totals = batch_execution.feasibility_totals

    summary = _build_completed_batch_summary(
        algo_contract=algo_contract,
        runtime_algorithm_contract=runtime_algorithm_contract,
        preflight=preflight,
        feasibility_totals=feasibility_totals,
        adapter_samples_seen=adapter_samples_seen,
        adapter_native_steps=adapter_native_steps,
        adapter_adapted_steps=adapter_adapted_steps,
        planner_command_space_fallback=_default_robot_command_space(
            kinematics_tag,
            policy_cfg,
            robot_command_mode=robot_command_mode,
        ),
        total_jobs=total_jobs,
        workers=workers,
        batch_runtime_sec=batch_execution.batch_runtime_sec,
        wrote=wrote,
        failures=failures,
        episode_records=episode_records,
        preflight_skipped_jobs=preflight_skipped_jobs,
        out_path=out_path,
        readiness=readiness,
        algo=algo,
        benchmark_profile=benchmark_profile,
        noise_spec=noise_spec,
        noise_hash=noise_hash,
        tracking_precision_spec=tracking_precision_spec,
        tracking_precision_hash=tracking_precision_spec_hash,
        active_observation_mode=active_observation_mode,
        active_observation_level=active_observation_level,
        actuation_profile_metadata=(
            actuation_profile.to_metadata() if actuation_profile is not None else None
        ),
        latency_profile_metadata=(
            latency_profile.to_metadata(dt=latency_metadata_dt)
            if latency_profile is not None
            else None
        ),
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
        schema_path=schema_path,
        scenario_path=scenario_path,
        scenarios=filtered,
        algo_config_path=algo_config_path,
        suite_key=suite_key,
        kinematics_tag=kinematics_tag,
        metric_affecting_config=metric_affecting_config,
    )
    # Emit provenance manifest alongside the JSONL output.
    _record_result_provenance_manifest(
        summary=summary,
        out_path=out_path,
        episode_records=episode_records,
        schema_path=schema_path,
        scenario_path=scenario_path,
        scenarios=filtered,
        algo=algo,
        algo_config_path=algo_config_path,
        benchmark_profile=benchmark_profile,
        suite_key=suite_key,
        total_jobs=total_jobs,
        written=wrote,
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
        active_observation_mode=active_observation_mode,
        active_observation_level=active_observation_level,
        noise_hash=noise_hash,
        tracking_precision_hash=tracking_precision_spec_hash,
    )
    return summary


__all__ = ["run_map_batch"]
