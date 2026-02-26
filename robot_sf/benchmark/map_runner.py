"""Map-based benchmark runner using Gym environments and scenario YAMLs."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from loguru import logger

from robot_sf.baselines.ppo import PPOPlanner
from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    infer_execution_mode_from_counts,
)
from robot_sf.benchmark.algorithm_readiness import BenchmarkProfile, require_algorithm_allowed
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, post_process_metrics
from robot_sf.benchmark.obstacle_sampling import sample_obstacle_points
from robot_sf.benchmark.path_utils import compute_shortest_path_length
from robot_sf.benchmark.scenario_schema import validate_scenario_list
from robot_sf.benchmark.schema_validator import load_schema, validate_episode
from robot_sf.benchmark.thresholds import ensure_metric_parameters
from robot_sf.benchmark.utils import (
    _config_hash,
    _git_hash_fallback,
    index_existing,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig
from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter
from robot_sf.planner.kinematics_model import KinematicsModel, resolve_benchmark_kinematics_model
from robot_sf.planner.socnav import (
    ORCAPlannerAdapter,
    PredictionPlannerAdapter,
    SACADRLPlannerAdapter,
    SamplingPlannerAdapter,
    SocialForcePlannerAdapter,
    SocNavBenchSamplingAdapter,
    SocNavPlannerConfig,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

if TYPE_CHECKING:
    from collections.abc import Callable

    from robot_sf.gym_env.unified_config import RobotSimulationConfig


_SOCNAV_ALGO_KEYS = {
    "socnav_sampling",
    "sampling",
    "orca",
    "sacadrl",
    "prediction_planner",
    "predictive",
    "prediction",
    "sa_cadrl",
    "socnav_bench",
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


def _default_robot_command_space(
    robot_kinematics: str | None,
    algo_config: dict[str, Any],
) -> str:
    """Resolve robot command-space metadata for the current run.

    Returns:
        str: Canonical command-space label.
    """
    kin = str(robot_kinematics or _DEFAULT_KINEMATICS).strip().lower()
    if kin in {"holonomic", "omni", "omnidirectional"}:
        mode = str(algo_config.get("command_mode", "vx_vy")).strip().lower()
        return "holonomic_vxy" if mode == "vx_vy" else "unicycle_vw"
    return "unicycle_vw"


def _init_feasibility_metadata(meta: dict[str, Any]) -> None:
    """Initialize mutable kinematics-feasibility counters in algorithm metadata."""
    meta["kinematics_feasibility"] = {
        "commands_evaluated": 0,
        "infeasible_native_count": 0,
        "projected_count": 0,
        "_sum_abs_delta_linear": 0.0,
        "_sum_abs_delta_angular": 0.0,
        "_max_abs_delta_linear": 0.0,
        "_max_abs_delta_angular": 0.0,
    }


def _project_with_feasibility(
    *,
    model: KinematicsModel,
    command: tuple[float, float],
    meta: dict[str, Any],
) -> tuple[float, float]:
    """Project a command while accumulating feasibility diagnostics.

    Returns:
        tuple[float, float]: Projected command.
    """
    projected = model.project(command)
    feasibility = meta.get("kinematics_feasibility")
    if not isinstance(feasibility, dict):
        return projected
    feasible_native = bool(model.is_feasible(command))
    delta_linear = abs(float(projected[0]) - float(command[0]))
    delta_angular = abs(float(projected[1]) - float(command[1]))
    feasibility["commands_evaluated"] = int(feasibility.get("commands_evaluated", 0)) + 1
    if not feasible_native:
        feasibility["infeasible_native_count"] = (
            int(feasibility.get("infeasible_native_count", 0)) + 1
        )
    if command != projected:
        feasibility["projected_count"] = int(feasibility.get("projected_count", 0)) + 1
    feasibility["_sum_abs_delta_linear"] = float(
        feasibility.get("_sum_abs_delta_linear", 0.0)
    ) + float(delta_linear)
    feasibility["_sum_abs_delta_angular"] = float(
        feasibility.get("_sum_abs_delta_angular", 0.0)
    ) + float(delta_angular)
    feasibility["_max_abs_delta_linear"] = max(
        float(feasibility.get("_max_abs_delta_linear", 0.0)),
        float(delta_linear),
    )
    feasibility["_max_abs_delta_angular"] = max(
        float(feasibility.get("_max_abs_delta_angular", 0.0)),
        float(delta_angular),
    )
    return projected


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


def _parse_algo_config(algo_config_path: str | None) -> dict[str, Any]:
    if not algo_config_path:
        return {}
    path = Path(algo_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Algorithm config file not found: {algo_config_path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("Algorithm config must be a mapping (YAML dict).")
    return data


def _is_socnav_algorithm(algo: str) -> bool:
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
        return False, "quality gate requires numeric min_success_rate and measured_success_rate"
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


def _preflight_policy(  # noqa: C901
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

    def _build_and_close(cfg: dict[str, Any]) -> None:
        effective_kinematics = robot_kinematics
        if effective_kinematics is None:
            effective_kinematics = str(
                cfg.get("robot_kinematics", cfg.get("kinematics", _DEFAULT_KINEMATICS))
            )
        try:
            policy_fn, _meta = _build_policy(
                algo,
                cfg,
                robot_kinematics=effective_kinematics,
            )
        except TypeError as exc:
            # Backward compatibility for tests or monkeypatches with legacy _build_policy signatures.
            if "unexpected keyword argument 'robot_kinematics'" not in str(exc):
                raise
            policy_fn, _meta = _build_policy(algo, cfg)
        planner_close = getattr(policy_fn, "_planner_close", None)
        if callable(planner_close):
            planner_close()

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
        _build_and_close(algo_config)
        return dict(algo_config), {"status": "ok", "learned_policy_contract": learned_contract}
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


def _planner_kinematics_compatibility(
    *,
    algo: str,
    robot_kinematics: str,
    algo_config: dict[str, Any],
) -> tuple[bool, str | None]:
    """Return explicit compatibility status for planner/kinematics combinations."""
    algo_key = algo.strip().lower()
    kin = robot_kinematics.strip().lower()
    if kin in {"holonomic", "omni", "omnidirectional"} and algo_key in {"rvo", "dwa", "teb"}:
        return (
            False,
            f"planner '{algo_key}' is a placeholder adapter and is disabled for '{kin}' runs",
        )
    if algo_key == "ppo" and kin in {"holonomic", "omni", "omnidirectional"}:
        obs_mode = str(algo_config.get("obs_mode", "vector")).strip().lower()
        if obs_mode == "image":
            return (
                False,
                "ppo holonomic runs require non-image obs_mode for map-runner compatibility",
            )
    return True, None


def _build_socnav_config(cfg: dict[str, Any]) -> SocNavPlannerConfig:
    try:
        return SocNavPlannerConfig(**cfg)
    except TypeError:
        return SocNavPlannerConfig()


def _goal_policy(obs: dict[str, Any], *, max_speed: float = 1.0) -> tuple[float, float]:
    robot = obs.get("robot", {})
    goal = obs.get("goal", {})
    robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=float)
    heading = float(np.asarray(robot.get("heading", [0.0]), dtype=float)[0])
    goal_pos = np.asarray(goal.get("current", [0.0, 0.0]), dtype=float)
    vec = goal_pos - robot_pos
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return 0.0, 0.0
    desired_heading = float(np.arctan2(vec[1], vec[0]))
    heading_error = ((desired_heading - heading + np.pi) % (2 * np.pi)) - np.pi
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


def _extract_ppo_pedestrians(pedestrians: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
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
            "radius": robot_radius,
        },
        "agents": agents,
        "obstacles": [],
    }


def _normalize_heading(value: float) -> float:
    """Normalize heading to [-pi, pi].

    Returns:
        Wrapped heading angle in radians.
    """
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


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


def _build_policy(  # noqa: C901, PLR0912, PLR0915
    algo: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build an action policy and algorithm metadata for map-based benchmarking.

    Args:
        algo: Algorithm key to instantiate.
        algo_config: Algorithm configuration payload.
        robot_kinematics: Runtime robot kinematics label for metadata enrichment.
        adapter_impact_eval: Whether to collect native-vs-adapter step counters.

    Returns:
        tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
        Policy callable and enriched metadata dictionary. For PPO, adapter-impact
        counters are mutated in-place in the returned metadata during episode rollout.
    """
    algo_key = algo.lower().strip()
    meta: dict[str, Any] = {"algorithm": algo_key}

    if algo_key in {"goal", "simple", "goal_policy", "simple_policy"}:
        goal_kinematics_model = resolve_benchmark_kinematics_model(
            robot_kinematics=robot_kinematics,
            command_limits=algo_config,
        )

        meta.update(
            {"status": "ok", "config": algo_config, "config_hash": _config_hash(algo_config)}
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
            planner_meta["robot_command_space"] = _default_robot_command_space(
                robot_kinematics,
                algo_config,
            )

        def _policy(obs: dict[str, Any]) -> tuple[float, float]:
            linear, angular = _goal_policy(obs, max_speed=float(algo_config.get("max_speed", 1.0)))
            return _project_with_feasibility(
                model=goal_kinematics_model,
                command=(linear, angular),
                meta=meta,
            )

        return _policy, meta

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
            planner_meta["robot_command_space"] = _default_robot_command_space(
                robot_kinematics,
                algo_config,
            )
        ppo_kinematics_model = resolve_benchmark_kinematics_model(
            robot_kinematics=robot_kinematics,
            command_limits=algo_config,
        )

        def _policy(obs: dict[str, Any]) -> tuple[float, float]:
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
            impact = meta.get("adapter_impact")
            if isinstance(impact, dict) and bool(impact.get("requested", False)):
                if conversion_mode == "native":
                    impact["native_steps"] = int(impact.get("native_steps", 0)) + 1
                else:
                    impact["adapted_steps"] = int(impact.get("adapted_steps", 0)) + 1
                impact["status"] = "collecting"
            return linear, angular

        _policy._planner_close = ppo_planner.close
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
    elif algo_key in {"orca"}:
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        adapter = ORCAPlannerAdapter(config=socnav_cfg, allow_fallback=allow_fallback)
    elif algo_key in {"sacadrl", "sa_cadrl"}:
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        adapter = SACADRLPlannerAdapter(config=socnav_cfg, allow_fallback=allow_fallback)
    elif algo_key in {"prediction_planner", "predictive", "prediction"}:
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        adapter = PredictionPlannerAdapter(config=socnav_cfg, allow_fallback=allow_fallback)
    elif algo_key in {"socnav_bench"}:
        allow_fallback = bool(algo_config.get("allow_fallback", False))
        adapter = SocNavBenchSamplingAdapter(config=socnav_cfg, allow_fallback=allow_fallback)
    elif algo_key in {"rvo", "dwa", "teb"}:
        adapter = SamplingPlannerAdapter(config=socnav_cfg)
        meta.update({"status": "placeholder", "fallback_reason": "unimplemented"})
    else:
        raise ValueError(f"Unknown map-based algorithm '{algo}'.")

    if "status" not in meta:
        meta["status"] = "ok"
    meta["config"] = algo_config
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
        planner_meta["robot_command_space"] = _default_robot_command_space(
            robot_kinematics,
            algo_config,
        )
    adapter_kinematics_model = resolve_benchmark_kinematics_model(
        robot_kinematics=robot_kinematics,
        command_limits=algo_config,
    )

    def _policy(obs: dict[str, Any]) -> tuple[float, float]:
        linear, angular = adapter.plan(obs)
        return _project_with_feasibility(
            model=adapter_kinematics_model,
            command=(float(linear), float(angular)),
            meta=meta,
        )

    return _policy, meta


def _resolve_seed_list(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [int(s) for s in v] for k, v in data.items() if isinstance(v, list)}


def _suite_key(scenario_path: Path) -> str:
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
    seeds = scenario.get("seeds")
    if isinstance(seeds, list) and seeds:
        return [int(s) for s in seeds]
    if suite_seeds.get(suite_key):
        return list(suite_seeds[suite_key])
    if suite_seeds.get("default"):
        return list(suite_seeds["default"])
    return [0]


def _scenario_identity_payload(
    scenario: dict[str, Any],
    *,
    algo: str,
    algo_config: dict[str, Any],
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
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


def _validate_behavior_sanity(scenario: dict[str, Any]) -> list[str]:
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
    if positions.shape[0] < 2:
        return np.zeros_like(positions), np.zeros_like(positions)
    vel = np.gradient(positions, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def _stack_ped_positions(traj: list[np.ndarray], *, fill_value: float = np.nan) -> np.ndarray:
    if not traj:
        return np.zeros((0, 0, 2), dtype=float)
    max_k = max(p.shape[0] for p in traj)
    stacked = np.full((len(traj), max_k, 2), fill_value, dtype=float)
    for i, arr in enumerate(traj):
        if arr.size == 0:
            continue
        stacked[i, : arr.shape[0]] = arr
    return stacked


def _policy_command_to_env_action(
    *,
    env: Any,
    config: RobotSimulationConfig,
    command: tuple[float, float],
) -> np.ndarray:
    """Convert policy unicycle command into the robot's native environment action space.

    Returns:
        np.ndarray: Action vector compatible with ``env.step``.
    """
    sim_robots = getattr(env.simulator, "robots", None)
    if not isinstance(sim_robots, list) or not sim_robots:
        return np.array([float(command[0]), float(command[1])], dtype=float)
    robot = sim_robots[0]
    robot_cfg = getattr(config, "robot_config", None)
    if robot_cfg is None:
        return np.array([command[0], command[1]], dtype=float)

    cls_name = robot_cfg.__class__.__name__.lower()
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
) -> dict[str, Any]:
    ts_start = datetime.now(UTC).isoformat()
    start_time = time.time()
    config = _build_env_config(scenario, scenario_path=scenario_path)
    max_steps = int(scenario.get("simulation_config", {}).get("max_episode_steps", 0) or 0)
    horizon_val = int(horizon) if horizon and horizon > 0 else max_steps
    if horizon_val <= 0:
        horizon_val = 200
    if dt is not None and dt > 0:
        config.sim_config.time_per_step_in_secs = float(dt)

    robot_kinematics = _robot_kinematics_label(config)
    policy_cfg = (
        dict(algo_config) if algo_config is not None else _parse_algo_config(algo_config_path)
    )
    policy_fn, algo_meta = _build_policy(
        algo,
        policy_cfg,
        robot_kinematics=robot_kinematics,
        adapter_impact_eval=adapter_impact_eval,
    )
    planner_close = getattr(policy_fn, "_planner_close", None)

    env = make_robot_env(config=config, seed=int(seed), debug=False)
    obs, _ = env.reset(seed=int(seed))

    robot_positions: list[np.ndarray] = []
    ped_positions: list[np.ndarray] = []
    ped_forces: list[np.ndarray] = []
    reached_goal_step: int | None = None
    termination_reason = "horizon"

    map_def = None
    goal_vec = np.zeros(2, dtype=float)
    try:
        for step_idx in range(horizon_val):
            action_v, action_w = policy_fn(obs)
            action = _policy_command_to_env_action(
                env=env,
                config=config,
                command=(float(action_v), float(action_w)),
            )
            obs, _reward, terminated, truncated, info = env.step(action)

            robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
            peds = np.asarray(env.simulator.ped_pos, dtype=float)
            if record_forces:
                forces = getattr(env.simulator, "last_ped_forces", None)
                if forces is None:
                    forces_arr = np.zeros_like(peds, dtype=float)
                else:
                    forces_arr = np.asarray(forces, dtype=float)
                    if forces_arr.shape != peds.shape:
                        forces_arr = np.zeros_like(peds, dtype=float)
            else:
                forces_arr = np.zeros_like(peds, dtype=float)

            robot_positions.append(robot_pos)
            ped_positions.append(peds)
            ped_forces.append(forces_arr)

            step_success = bool(info.get("success") or info.get("is_success"))
            if reached_goal_step is None and step_success:
                reached_goal_step = step_idx
            # Freeze episode once success is first achieved to avoid later collisions
            # flipping benchmark success labels as horizon changes.
            if step_success:
                termination_reason = "success_reached"
                break
            if terminated or truncated:
                termination_reason = str(
                    info.get("done_reason")
                    or info.get("termination_reason")
                    or ("terminated" if terminated else "truncated")
                )
                break
        if getattr(env, "simulator", None) is not None:
            map_def = env.simulator.map_def
            goal_vec = np.asarray(env.simulator.goal_pos[0], dtype=float)
    finally:
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
    ped_forces_arr = _stack_ped_positions(ped_forces, fill_value=np.nan)

    obstacles = (
        sample_obstacle_points(map_def.obstacles, map_def.bounds) if map_def is not None else None
    )
    if robot_pos_arr.size:
        shortest_path = compute_shortest_path_length(map_def, robot_pos_arr[0], goal_vec)
    else:
        shortest_path = float("nan")

    if robot_pos_arr.size == 0:
        metrics_raw = {"success": 0.0, "time_to_goal_norm": float("nan"), "collisions": 0.0}
    else:
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
        )
        metrics_raw = compute_all_metrics(
            ep,
            horizon=horizon_val,
            shortest_path_len=shortest_path,
            robot_max_speed=_robot_max_speed(config),
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
            )
        else:
            impact["status"] = "not_applicable"
            impact["adapter_fraction"] = 0.0
    _finalize_feasibility_metadata(algo_meta)
    metrics = post_process_metrics(
        metrics_raw,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
    )

    ts_end = datetime.now(UTC).isoformat()
    scenario_id = str(
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )
    scenario_params = _scenario_identity_payload(
        scenario,
        algo=algo,
        algo_config=policy_cfg,
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
    )
    steps_taken = int(robot_pos_arr.shape[0])
    wall_time = float(max(1e-9, time.time() - start_time))
    timing = {"steps_per_second": float(steps_taken) / wall_time if wall_time > 0 else 0.0}
    status = "success" if metrics.get("success") else "failure"
    if metrics.get("collisions"):
        status = "collision"
    record = {
        "version": "v1",
        "episode_id": _compute_map_episode_id(scenario_params, seed),
        "scenario_id": scenario_id,
        "seed": seed,
        "scenario_params": scenario_params,
        "metrics": metrics,
        "algorithm_metadata": algo_meta,
        "algo": algo,
        "config_hash": _config_hash(scenario_params),
        "git_hash": _git_hash_fallback(),
        "timestamps": {"start": ts_start, "end": ts_end},
        "status": status,
        "steps": steps_taken,
        "horizon": horizon_val,
        "wall_time_sec": wall_time,
        "timing": timing,
        "termination_reason": termination_reason,
    }
    ensure_metric_parameters(record)
    return record


def _write_validated(out_path: Path, schema: dict[str, Any], record: dict[str, Any]) -> None:
    validate_episode(record, schema)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _run_map_job_worker(job: tuple[dict[str, Any], int, dict[str, Any]]) -> dict[str, Any]:
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
        scenario_path=Path(params.get("scenario_path")),
        adapter_impact_eval=bool(params.get("adapter_impact_eval", False)),
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
    workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Run map-based scenarios and append episode records.

    Returns:
        Summary payload with counts and failure details.
    """
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
    algo_contract = enrich_algorithm_metadata(
        algo=algo,
        metadata={},
        robot_kinematics=kinematics_tag,
        adapter_impact_requested=adapter_impact_eval,
    )
    planner_meta = algo_contract.get("planner_kinematics")
    if isinstance(planner_meta, dict):
        planner_meta["scenario_kinematics"] = scenario_kinematics
    for scenario in filtered:
        seeds = _select_seeds(scenario, suite_seeds=suite_seeds, suite_key=suite_key)
        for seed in seeds:
            jobs.append((scenario, int(seed)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = load_schema(schema_path)
    policy_cfg = _parse_algo_config(algo_config_path)
    ppo_paper_ready, _paper_reason = (
        _ppo_paper_gate_status(policy_cfg) if algo.strip().lower() == "ppo" else (False, None)
    )
    readiness = require_algorithm_allowed(
        algo=algo,
        benchmark_profile=benchmark_profile,
        ppo_paper_ready=ppo_paper_ready,
    )
    policy_cfg, preflight = _preflight_policy(
        algo=algo,
        algo_config=policy_cfg,
        benchmark_profile=benchmark_profile,
        missing_prereq_policy=socnav_missing_prereq_policy,
        robot_kinematics=kinematics_tag,
    )
    compatible, incompatible_reason = _planner_kinematics_compatibility(
        algo=algo,
        robot_kinematics=kinematics_tag,
        algo_config=policy_cfg,
    )
    if not compatible:
        preflight["status"] = "skipped"
        preflight["compatibility_status"] = "incompatible"
        preflight["compatibility_reason"] = incompatible_reason
    preflight["algorithm_metadata_contract"] = algo_contract
    if preflight.get("status") == "skipped":
        return {
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
        }

    if resume and out_path.exists():
        existing = index_existing(out_path)
        if existing:
            filtered_jobs: list[tuple[dict[str, Any], int]] = []
            for sc, seed in jobs:
                identity_payload = _scenario_identity_payload(
                    sc,
                    algo=algo,
                    algo_config=policy_cfg,
                    horizon=horizon,
                    dt=dt,
                    record_forces=record_forces,
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
        "algo_config": policy_cfg,
        "scenario_path": str(scenario_path),
        "adapter_impact_eval": bool(adapter_impact_eval),
    }

    total_jobs = len(jobs)
    wrote = 0
    failures: list[dict[str, Any]] = []
    adapter_native_steps = 0
    adapter_adapted_steps = 0
    adapter_samples_seen = False
    feasibility_totals = {
        "commands_evaluated": 0,
        "infeasible_native_count": 0,
        "projected_count": 0,
        "sum_abs_delta_linear": 0.0,
        "sum_abs_delta_angular": 0.0,
        "max_abs_delta_linear": 0.0,
        "max_abs_delta_angular": 0.0,
    }
    if workers <= 1:
        for scenario, seed in jobs:
            try:
                rec = _run_map_job_worker((scenario, seed, fixed_params))
                requested_seen, native_steps, adapted_steps = _accumulate_batch_metadata(
                    rec,
                    feasibility_totals=feasibility_totals,
                )
                adapter_samples_seen = adapter_samples_seen or requested_seen
                adapter_native_steps += native_steps
                adapter_adapted_steps += adapted_steps
                _write_validated(out_path, schema, rec)
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
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            future_to_job: dict[Any, tuple[dict[str, Any], int]] = {}
            for scenario, seed in jobs:
                fut = ex.submit(_run_map_job_worker, (scenario, seed, fixed_params))
                future_to_job[fut] = (scenario, seed)
            for fut in as_completed(future_to_job):
                scenario, seed = future_to_job[fut]
                try:
                    rec = fut.result()
                    requested_seen, native_steps, adapted_steps = _accumulate_batch_metadata(
                        rec,
                        feasibility_totals=feasibility_totals,
                    )
                    adapter_samples_seen = adapter_samples_seen or requested_seen
                    adapter_native_steps += native_steps
                    adapter_adapted_steps += adapted_steps
                    _write_validated(out_path, schema, rec)
                    wrote += 1
                except Exception as exc:  # pragma: no cover
                    failures.append(
                        {
                            "scenario_id": scenario.get("name", "unknown"),
                            "seed": seed,
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
            )
        else:
            impact_contract["status"] = "not_applicable"
            impact_contract["adapter_fraction"] = 0.0

    preflight["algorithm_metadata_contract"] = algo_contract
    planner_contract = algo_contract.get("planner_kinematics")
    if isinstance(planner_contract, dict):
        planner_contract["robot_command_space"] = _default_robot_command_space(
            kinematics_tag,
            policy_cfg,
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

    return {
        "total_jobs": total_jobs,
        "written": wrote,
        "successful_jobs": wrote,
        "failed_jobs": len(failures),
        "failures": failures,
        "out_path": str(out_path),
        "algorithm_readiness": {
            "name": readiness.canonical_name if readiness is not None else algo,
            "tier": readiness.tier if readiness is not None else "unknown",
            "profile": benchmark_profile,
        },
        "algorithm_metadata_contract": algo_contract,
        "preflight": preflight,
    }


__all__ = ["run_map_batch"]
