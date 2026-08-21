"""Compatibility facade for Full Classic execution and adaptive sampling.

Implemented incrementally in tasks T026-T029, T027 (parallel), T028 (adaptive iteration),
T029 (full run orchestration skeleton).

Setup, episode record construction, scheduling, and final artifact publication live in the
adjacent ``context``, ``execution``, ``scheduler``, and ``finalizer`` modules.  This facade
retains the historical public and test import paths while coordinating those phases.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.errors import AggregationMetadataError
from robot_sf.benchmark.freeze_manifest import evaluate_freeze_manifest, safe_int
from robot_sf.benchmark.map_runner.map_runner import _signal_state_for_metric_metadata
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, snqi
from robot_sf.benchmark.obstacle_sampling import sample_obstacle_points
from robot_sf.benchmark.path_utils import compute_shortest_path_length
from robot_sf.benchmark.termination_reason import (
    build_outcome_payload,
    metric_scalar,
    outcome_contradictions,
    resolve_termination_reason,
    status_from_termination_reason,
)
from robot_sf.common.math_utils import wrap_angle_pi
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    resolve_map_definition,
)

from . import context as _context
from . import finalizer as _finalizer
from . import scheduler as _scheduler
from .aggregation import aggregate_metrics
from .effects import compute_effect_sizes
from .execution import EpisodeExecutionHooks, execute_episode_record
from .precision import evaluate_precision
from .replay import ReplayCapture  # T021 optional replay capture
from .visuals import generate_visual_artifacts  # new visual artifact integration

BenchmarkManifest = _context.BenchmarkManifest
_prepare_output_dirs = _context._prepare_output_dirs
build_run_context = _context.build_run_context
_update_scaling_efficiency = _finalizer.update_scaling_efficiency
_write_iteration_artifacts = _finalizer.write_iteration_artifacts
_write_json = _finalizer.write_json
_serialize_effects = _finalizer.serialize_effects
_serialize_groups = _finalizer.serialize_groups
_serialize_precision = _finalizer.serialize_precision
_partition_jobs = _scheduler.partition_jobs
_scan_existing_episode_ids = _scheduler.scan_existing_episode_ids
execute_episode_jobs = _scheduler.execute_episode_jobs
finalize_run = _finalizer.finalize_run

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

CLEAR_TRACKING_METADATA_KEY = "clear_tracking_uncertainty"
ROLLOVER_STABILITY_METADATA_KEY = "rollover_stability"

# Import new visualization functions for real plots/videos from episode data
try:
    from robot_sf.benchmark import visualization as _visualization

    VisualizationError = _visualization.VisualizationError
    generate_benchmark_plots = _visualization.generate_benchmark_plots
    validate_visual_artifacts = _visualization.validate_visual_artifacts

    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

# -----------------------------
# Manifest dataclass & helpers
# -----------------------------


def _find_repo_root(path: Path) -> Path:
    """Locate repository root by searching upward for git metadata.

    Returns:
        Repository root path when ``.git`` is found, otherwise a layout fallback.
    """

    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    try:
        return path.resolve().parents[3]
    except IndexError:  # pragma: no cover - defensive fallback
        return start


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def _ensure_algo_metadata(
    record: dict[str, Any],
    *,
    algo: str | None,
    episode_id: str | None,
    logger_ctx=None,
) -> dict[str, Any]:
    """Mirror the algorithm identifier into scenario_params and validate payloads.

    Returns:
        Updated record dictionary with algorithm metadata injected.
    """

    log = logger_ctx or logger
    algo_value = algo.strip() if isinstance(algo, str) else ""
    if not algo_value:
        raise AggregationMetadataError(
            "Episode missing algorithm identifier required for aggregation.",
            episode_id=str(episode_id) if episode_id is not None else None,
            missing_fields=("algo", "scenario_params.algo"),
            advice="Ensure the benchmark configuration sets `algo` before writing episodes.",
        )

    scenario_params = record.get("scenario_params")
    if scenario_params is None:
        scenario_params = {}
        record["scenario_params"] = scenario_params
    elif not isinstance(scenario_params, dict):
        raise AggregationMetadataError(
            "scenario_params must be a mapping to inject algorithm metadata.",
            episode_id=str(episode_id) if episode_id is not None else None,
            missing_fields=("scenario_params", "scenario_params.algo"),
            advice="Regenerate the episode with structured scenario parameters.",
        )

    existing_algo = scenario_params.get("algo")
    log = log.bind(episode_id=episode_id, algo=algo_value)
    if existing_algo is None:
        scenario_params["algo"] = algo_value
        log.bind(event="episode_metadata_injection").debug(
            "Mirrored algorithm metadata into scenario_params",
        )
    elif str(existing_algo) != algo_value:
        scenario_params["algo"] = algo_value
        log.bind(event="episode_metadata_mismatch", previous=str(existing_algo)).warning(
            "Corrected mismatched algorithm metadata for episode",
        )

    algo_meta = record.get("algorithm_metadata")
    if algo_meta is None or not isinstance(algo_meta, dict):
        algo_meta = {}
    algo_meta = enrich_algorithm_metadata(algo=algo_value, metadata=algo_meta)
    record["algorithm_metadata"] = algo_meta

    record["algo"] = algo_value
    return record


def _compute_git_hash(root: Path) -> str:
    """Best‑effort retrieval of current git HEAD short hash.

    Falls back to 'unknown' if repository metadata is inaccessible. Separated to keep
    orchestration function lean (polish phase refactor for C901).

    Returns:
        Short git hash (12 characters) or 'unknown' if not retrievable.
    """
    repo_root = _REPO_ROOT if (_REPO_ROOT / ".git").exists() else root
    commit = _run_git(repo_root, "rev-parse", "--short=12", "HEAD")
    if commit:
        return commit

    git_hash = "unknown"
    try:  # pragma: no cover - environment dependent
        head_ref = repo_root / ".git" / "HEAD"
        if head_ref.exists():
            content = head_ref.read_text(encoding="utf-8").strip()
            if content.startswith("ref:"):
                ref_path = content.split(" ", 1)[1].strip()
                ref_file = repo_root / ".git" / ref_path
                if ref_file.exists():
                    git_hash = ref_file.read_text(encoding="utf-8").strip()[:12]
            else:
                git_hash = content[:12]
    except OSError as exc:  # pragma: no cover - defensive logging fallback
        # Filesystem access errors -> return unknown but log for diagnostics
        logger.debug("_compute_git_hash fs access error: {}", exc)
    except (RuntimeError, TypeError):  # pragma: no cover - defensive
        # Unexpected but plausible runtime/type errors -> log at debug and continue
        logger.debug("_compute_git_hash unexpected error")
    return git_hash


def _run_git(repo_root: Path, *args: str) -> str | None:
    """Run a git command and return stripped stdout on success.

    Returns:
        Command stdout with surrounding whitespace removed, or None on failure.
    """

    try:  # pragma: no cover - depends on external git runtime
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, RuntimeError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out if out else None


def _repo_name_from_remote(remote: str) -> str:
    """Derive a repository name from a git remote string.

    Returns:
        Parsed repository name or "unknown" when unavailable.
    """

    candidate = remote.strip().rstrip("/")
    if candidate.startswith("git@") and ":" in candidate:
        candidate = candidate.split(":", 1)[1]
    tail = candidate.rsplit("/", 1)[-1]
    if tail.endswith(".git"):
        tail = tail[:-4]
    return tail or "unknown"


def _to_iso_utc(ts: float) -> str:
    """Format unix timestamp as canonical UTC ISO-8601 string.

    Returns:
        UTC timestamp in ISO-8601 format with trailing "Z".
    """

    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _build_run_meta(root: Path, cfg, manifest: BenchmarkManifest) -> dict[str, Any]:
    """Build canonical run-level traceability metadata.

    Returns:
        JSON-serializable run metadata payload.
    """

    remote = _run_git(_REPO_ROOT, "config", "--get", "remote.origin.url") or "unknown"
    branch = _run_git(_REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    commit = _run_git(_REPO_ROOT, "rev-parse", "HEAD") or _compute_git_hash(root)

    argv = [str(v) for v in (sys.argv or [])]
    command = argv[0] if argv else "unknown"
    args = argv[1:] if argv else []

    matrix_path_raw = getattr(cfg, "scenario_matrix_path", None)
    matrix_path = "unknown"
    if matrix_path_raw is not None:
        try:
            matrix_path = str(Path(str(matrix_path_raw)).resolve())
        except (OSError, RuntimeError, ValueError, TypeError):
            matrix_path = str(matrix_path_raw)

    base_seed = getattr(cfg, "base_seed", None)
    if base_seed is None:
        base_seed = getattr(cfg, "master_seed", None)
    repeats = getattr(cfg, "repeats", None)
    if repeats is None:
        repeats = getattr(cfg, "initial_episodes", None)

    run_id = root.resolve().name or "unknown"
    run_meta = {
        "run_id": run_id,
        "created_at_utc": _to_iso_utc(float(manifest.created_at)),
        "repo": {
            "name": _repo_name_from_remote(remote) if remote != "unknown" else _REPO_ROOT.name,
            "remote": remote,
            "branch": branch,
            "commit": commit,
        },
        "cli": {
            "command": command,
            "args": args,
        },
        "matrix_path": matrix_path,
        "seed_plan": {
            "base_seed": safe_int(base_seed),
            "repeats": safe_int(repeats),
        },
        "environment": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
        },
    }
    freeze_validation = getattr(manifest, "freeze_validation", {})
    if freeze_validation:
        run_meta["freeze_manifest"] = freeze_validation
    return run_meta


def _write_run_meta_files(root: Path, cfg, manifest: BenchmarkManifest) -> None:
    """Write canonical run metadata file in both local and paper-contract locations."""

    run_meta = _build_run_meta(root, cfg, manifest)
    run_id = str(run_meta.get("run_id", "unknown"))
    paper_path = root / "artifacts" / run_id / "run_meta.json"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(paper_path, run_meta)
    _write_json(root / "run_meta.json", run_meta)


def _init_manifest(
    root: Path,
    episodes_path: Path,
    cfg,
    scenario_matrix_hash: str,
) -> BenchmarkManifest:
    """Initialize a manifest instance for the current run.

    Returns:
        BenchmarkManifest instance.
    """
    return BenchmarkManifest(
        output_root=root,
        git_hash=_compute_git_hash(root),
        scenario_matrix_hash=scenario_matrix_hash,
        config=cfg,
        episodes_path=str(episodes_path),
    )


def _episode_id_from_job(job) -> str:
    """Deterministically derive an episode_id from a job.

    Contract (early phase): scenario_id + '-' + seed. Horizon intentionally excluded
    to keep reproducibility with initial tests; may evolve later when multi‑horizon
    episodes are introduced.

    Returns:
        Episode ID string in format "scenario_id-seed".
    """
    return f"{job.scenario_id}-{job.seed}"


_DEFAULT_SNQI_WEIGHTS = {
    "w_success": 1.0,
    "w_time": 0.7,
    "w_collisions": 1.0,
    "w_near": 0.5,
    "w_comfort": 0.25,
    "w_force_exceed": 0.25,
    "w_jerk": 0.25,
    "w_curvature": 0.25,
}
_STUB_DT_SECONDS = 0.1
_STUB_PATH_EFFICIENCY = 0.9
_STUB_AVG_SPEED = 1.0
_STUB_MAX_STEPS = 5
_STUB_REPLAY_LINEAR_V = 0.05


@lru_cache(maxsize=4)
def _load_snqi_weights(path: str | None):
    """Load SNQI weights from JSON or fall back to defaults.

    Returns:
        Weight mapping dictionary.
    """
    if not path:
        return dict(_DEFAULT_SNQI_WEIGHTS)
    p = Path(path)
    if not p.exists():
        logger.warning("SNQI weights path not found: {}", path)
        return dict(_DEFAULT_SNQI_WEIGHTS)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load SNQI weights from {}: {}", path, exc)
        return dict(_DEFAULT_SNQI_WEIGHTS)


def _resolve_horizon(job, cfg) -> int:
    """Resolve horizon with smoke-mode caps applied.

    Returns:
        Horizon length in steps.
    """
    horizon = int(getattr(job, "horizon", 0) or 0)
    if getattr(cfg, "smoke", False):
        cap = int(getattr(cfg, "smoke_horizon_cap", 40) or 40)
        horizon = min(horizon, cap)
    return max(1, horizon if horizon > 0 else 1)


def _build_env_config(scenario, cfg, horizon: int):
    """Construct a RobotSimulationConfig for a scenario.

    Returns:
        RobotSimulationConfig instance.
    """
    raw = dict(getattr(scenario, "raw", {}))
    matrix_path = Path(cfg.scenario_matrix_path)
    matrix_dir = matrix_path.parent
    map_value = raw.get("map_file")
    candidate = Path(map_value) if map_value else None
    if candidate is not None and not candidate.is_absolute():
        candidate = (matrix_dir / candidate).resolve()
    if candidate is None or not candidate.exists():
        fallback = _REPO_ROOT / "maps" / "svg_maps" / "classic_crossing.svg"
        if fallback.exists():
            raw["map_file"] = str(fallback)
    config = build_robot_config_from_scenario(
        raw,
        scenario_path=matrix_path,
    )
    try:
        dt = float(config.sim_config.time_per_step_in_secs)
    except (ValueError, TypeError, AttributeError):  # pragma: no cover - defensive
        dt = 0.1
    # Ensure sim horizon matches requested horizon
    config.sim_config.sim_time_in_secs = horizon * dt
    return config


def _simple_goal_policy(simulator) -> np.ndarray:
    """Simple goal-seeking controller returning (v, omega).

    Returns:
        Array of [linear, angular] command values.
    """
    robot_pos = np.asarray(simulator.robot_pos[0], dtype=float)
    goal = np.asarray(simulator.goal_pos[0], dtype=float)
    heading = float(simulator.robot_poses[0][1])
    vec = goal - robot_pos
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return np.array([0.0, 0.0], dtype=float)
    desired_heading = math.atan2(vec[1], vec[0])
    heading_err = wrap_angle_pi(desired_heading - heading)
    linear = min(1.0, dist)
    angular = max(min(heading_err, 1.0), -1.0)
    return np.array([linear, angular], dtype=float)


def _stack_ped_positions(series: list[np.ndarray], fill_value: float = 0.0) -> np.ndarray:
    """Stack variable-length pedestrian arrays into a padded tensor.

    Returns:
        Padded array of shape (T, max_peds, 2).
    """
    max_len = max((len(p) for p in series), default=0)
    if max_len == 0:
        return np.zeros((len(series), 0, 2), dtype=float)
    out = np.full((len(series), max_len, 2), fill_value, dtype=float)
    for idx, p in enumerate(series):
        if len(p) == 0:
            continue
        arr = np.asarray(p, dtype=float)
        copy_len = min(arr.shape[0], max_len)
        dim = min(arr.shape[1], out.shape[2]) if arr.ndim >= 2 else 0
        if copy_len > 0 and dim > 0:
            out[idx, :copy_len, :dim] = arr[:copy_len, :dim]
    return out


def _extract_ped_forces(simulator, ped_pos: np.ndarray) -> np.ndarray:
    """Best-effort retrieval of per-pedestrian forces for the current step.

    Returns an array shaped like ``ped_pos``. Missing or mismatched force data is
    filled with NaNs so downstream metrics can flag absent samples instead of
    silently reporting zeros.

    Returns:
        Array of pedestrian forces with same shape as ped_pos, NaN-filled if unavailable.
    """

    forces = getattr(simulator, "last_ped_forces", None)
    if forces is None:
        return np.full_like(ped_pos, np.nan)
    try:
        arr = np.asarray(forces, dtype=float)
    except (ValueError, TypeError):
        return np.full_like(ped_pos, np.nan)

    if arr.shape == ped_pos.shape:
        return np.array(arr, dtype=float, copy=True)

    out = np.full_like(ped_pos, np.nan)
    copy_len = min(arr.shape[0], ped_pos.shape[0])
    dim = min(arr.shape[1], out.shape[1]) if arr.ndim >= 2 else 0
    if arr.ndim >= 2 and copy_len > 0 and dim > 0:
        out[:copy_len, :dim] = arr[:copy_len, :dim]
    return out


def _vel_and_acc(pos: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity and acceleration from positions with simple differencing.

    Returns:
        Tuple of (velocities, accelerations).
    """
    if len(pos) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)
    if len(pos) == 1 or dt <= 0:
        zeros = np.zeros((len(pos), 2), dtype=float)
        return zeros, zeros
    vel = np.diff(pos, axis=0) / dt
    vel = np.vstack([vel[0], vel])
    if len(vel) == 1:
        acc = np.zeros_like(vel)
    else:
        acc = np.diff(vel, axis=0) / dt
        acc = np.vstack([acc[0], acc])
    return vel, acc


def _capture_visual_state(env):
    """Capture optional visualization state for replay artifacts.

    Returns:
        Tuple of (ray_vecs, ped_actions, robot_goal) or (None, None, None).
    """
    if not hasattr(env, "_prepare_visualizable_state"):
        return None, None, None
    try:
        vis_state = env._prepare_visualizable_state()  # type: ignore[attr-defined]
        ray_vecs = (
            [tuple(map(float, r)) for r in np.asarray(vis_state.ray_vecs).reshape(-1, 2)]
            if getattr(vis_state, "ray_vecs", None) is not None
            else None
        )
        ped_actions = (
            [tuple(map(float, r)) for r in np.asarray(vis_state.ped_actions).reshape(-1, 2)]
            if getattr(vis_state, "ped_actions", None) is not None
            else None
        )
        robot_goal = None
        if getattr(vis_state, "robot_action", None) is not None:
            try:
                goal_val = vis_state.robot_action.goal  # type: ignore[attr-defined]
                robot_goal = (float(goal_val[0]), float(goal_val[1]))
            except (AttributeError, TypeError, ValueError, IndexError, KeyError):
                robot_goal = None
        return ray_vecs, ped_actions, robot_goal
    except (ValueError, TypeError, KeyError, AttributeError):
        return None, None, None


def _compute_episode_metrics(  # noqa: PLR0913
    job,
    scenario,
    cfg,
    *,
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    robot_acc: np.ndarray,
    ped_pos: np.ndarray,
    ped_forces: np.ndarray,
    dt: float,
    reached_goal_step: int | None,
    goal: np.ndarray,
    horizon: int,
    robot_radius: float,
    ped_radius: float,
) -> dict[str, float]:
    """Compute episode metrics for the classic benchmark pipeline.

    Returns:
        Mapping of metric name to computed value.
    """
    map_def = None
    map_path = getattr(scenario, "map_path", None)
    if map_path:
        try:
            map_def = resolve_map_definition(str(map_path), scenario_path=Path(str(map_path)))
        except (OSError, ValueError, KeyError):  # pragma: no cover - defensive fallback
            map_def = None
    shortest_path = (
        compute_shortest_path_length(map_def, robot_pos[0], goal)
        if len(robot_pos)
        else float("nan")
    )
    if not math.isfinite(shortest_path):
        logger.bind(
            event="metrics_shortest_path_nan",
            job_id=getattr(job, "job_id", None),
            scenario_id=getattr(job, "scenario_id", None),
            seed=getattr(job, "seed", None),
        ).warning(
            "Shortest path is NaN because the robot trajectory is empty; downstream aggregation may propagate NaN.",
        )
    obstacles = None
    if map_def is not None:
        obstacles = sample_obstacle_points(map_def.obstacles, map_def.bounds)
    ep = EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=ped_pos,
        ped_forces=ped_forces,
        obstacles=obstacles,
        goal=goal,
        dt=dt,
        reached_goal_step=reached_goal_step,
        robot_radius=float(robot_radius),
        ped_radius=float(ped_radius),
        episode_metadata=_episode_metadata_for_metrics(scenario),
    )
    metrics_raw = compute_all_metrics(ep, horizon=horizon, shortest_path_len=shortest_path)
    time_to_goal = (
        dt * float(reached_goal_step)
        if reached_goal_step is not None
        else dt * float(horizon if horizon > 0 else len(robot_pos))
    )
    metrics_raw["time_to_goal"] = time_to_goal
    metrics = dict(metrics_raw)
    metrics["success_rate"] = float(metrics_raw.get("success", 0.0))
    metrics["collision_rate"] = 1.0 if metrics_raw.get("collisions", 0.0) else 0.0
    weights = _load_snqi_weights(getattr(cfg, "snqi_weights_path", None))
    try:
        metrics["snqi"] = snqi(metrics_raw, weights, baseline_stats=None)
    except (ValueError, TypeError, KeyError):  # pragma: no cover - defensive
        metrics["snqi"] = float("nan")
    serializable: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            serializable[key] = float(value)
        else:
            serializable[key] = value
    return serializable


def _signal_contract_state_for_metrics(signal_state: Any) -> dict[str, Any] | None:
    """Return fail-closed signal-state metadata for metric computation.

    The trace-export path can record proxy signal metadata, but metric denominators may only
    include explicit planner-observable benchmark evidence.
    """
    return _signal_state_for_metric_metadata(signal_state)


def _episode_metadata_for_metrics(scenario) -> dict[str, Any] | None:
    """Build optional episode metadata consumed by metric helpers.

    Returns:
        Metric-facing episode metadata for signalized or opt-in instrumentation scenarios.
    """
    raw = getattr(scenario, "raw", {})
    metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    episode_metadata: dict[str, Any] = {}
    signal_state = metadata.get("signal_state") if isinstance(metadata, dict) else None
    metric_signal_state = _signal_contract_state_for_metrics(signal_state)
    if metric_signal_state is not None:
        episode_metadata["signal_state"] = metric_signal_state

    rollover_stability = metadata.get(ROLLOVER_STABILITY_METADATA_KEY)
    if isinstance(rollover_stability, dict) and bool(rollover_stability.get("enabled", False)):
        episode_metadata[ROLLOVER_STABILITY_METADATA_KEY] = dict(rollover_stability)

    clear_tracking = metadata.get(CLEAR_TRACKING_METADATA_KEY)
    if isinstance(clear_tracking, dict) and bool(clear_tracking.get("enabled", False)):
        episode_metadata[CLEAR_TRACKING_METADATA_KEY] = dict(clear_tracking)

    if not episode_metadata:
        return None
    return episode_metadata


def _init_env_for_job(job, cfg, horizon: int, *, episode_id: str, scenario):
    """Initialize the Gym environment and replay capture for a job.

    Returns:
        Tuple of (env, dt, replay_capture, goal_vector).
    """
    config = _build_env_config(scenario, cfg, horizon)
    capture_replay = bool(getattr(cfg, "capture_replay", False))
    record_dir = Path(cfg.output_root)
    replays_dir = record_dir / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = record_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    # Only enable native video recording when not in fast stub/smoke modes.
    record_video_flag = bool(
        getattr(cfg, "record_video", False)
        or (
            capture_replay
            and not getattr(cfg, "fast_stub", False)
            and not getattr(cfg, "smoke", False)
        )
    )
    video_path = videos_dir / f"simview_{episode_id}.mp4" if record_video_flag else None
    env = make_robot_env(
        config=config,
        seed=int(job.seed),
        debug=record_video_flag,
        recording_enabled=capture_replay,
        record_video=record_video_flag,
        video_path=str(video_path) if video_path else None,
        video_fps=float(getattr(cfg, "video_fps", 10) or 10),
        use_jsonl_recording=False,
        recording_dir=str(replays_dir),
        suite_name="classic_full",
        scenario_name=job.scenario_id,
        algorithm_name=str(getattr(cfg, "algo", "unknown")),
        recording_seed=int(job.seed),
    )
    dt = float(getattr(config.sim_config, "time_per_step_in_secs", 0.1))
    replay_cap = (
        ReplayCapture(episode_id=episode_id, scenario_id=job.scenario_id)
        if capture_replay
        else None
    )
    if replay_cap is not None:
        replay_cap.dt = dt
    goal_vec = np.zeros(2, dtype=float)
    sim = getattr(env, "simulator", None)
    if sim is not None:
        try:
            goal_vec = np.asarray(sim.goal_pos[0], dtype=float)
        except (
            ValueError,
            TypeError,
            IndexError,
            KeyError,
            AttributeError,
        ):  # pragma: no cover - defensive fallback
            goal_vec = np.zeros(2, dtype=float)
    return env, dt, replay_cap, goal_vec


def _rollout_episode(env, horizon: int, dt: float, replay_cap):
    """Execute a rollout and collect trajectories and replay metadata.

    Returns:
        Tuple of (robot_positions, ped_positions, ped_forces, reached_goal_step).
    """
    robot_positions: list[np.ndarray] = []
    ped_positions: list[np.ndarray] = []
    ped_forces: list[np.ndarray] = []
    reached_goal_step: int | None = None
    for step_idx in range(horizon):
        action_arr = _simple_goal_policy(env.simulator)
        obs, _reward, terminated, truncated, info = env.step(action_arr)
        _ = obs
        # Snapshot mutable backend buffers to preserve per-step trajectory history.
        robot_pos = np.array(env.simulator.robot_pos[0], dtype=float, copy=True)
        heading = float(env.simulator.robot_poses[0][1])
        peds = np.array(env.simulator.ped_pos, dtype=float, copy=True)
        forces = _extract_ped_forces(env.simulator, peds)
        robot_positions.append(robot_pos)
        ped_positions.append(peds)
        ped_forces.append(forces)
        if replay_cap is not None:
            ray_vecs, ped_actions, robot_goal = _capture_visual_state(env)
            ped_list = [tuple(map(float, row)) for row in peds.tolist()] if peds.size else []
            replay_cap.record(
                t=step_idx * dt,
                x=float(robot_pos[0]),
                y=float(robot_pos[1]),
                heading=heading,
                speed=float(np.linalg.norm(action_arr)),
                ped_positions=ped_list,
                action=(float(action_arr[0]), float(action_arr[1])),
                ray_vecs=ray_vecs,
                ped_actions=ped_actions,
                robot_goal=robot_goal,
            )
        # Render frame if SimulationView recording is active
        try:
            if getattr(env, "sim_ui", None) is not None and getattr(
                env.sim_ui, "record_video", False
            ):
                env.render()
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError, IndexError, KeyError):
            # Rendering is best-effort; ignore to keep rollout running
            pass
        step_meta = info.get("meta", {}) if isinstance(info, dict) else {}
        if reached_goal_step is None and bool(step_meta.get("is_route_complete")):
            reached_goal_step = step_idx
        if terminated or truncated:
            break
    return robot_positions, ped_positions, ped_forces, reached_goal_step


def _close_env(env):
    """Best-effort environment cleanup."""
    try:
        env.exit()
    except (RuntimeError, AttributeError, TypeError, OSError):  # pragma: no cover
        pass
    try:
        env.close()
    except (
        RuntimeError,
        AttributeError,
        TypeError,
        OSError,
    ):  # pragma: no cover - gym close best-effort
        pass


def _make_stub_episode_record(
    job,
    cfg,
    *,
    episode_id: str,
    horizon: int,
) -> dict[str, Any]:
    """Build a synthetic episode record used by fast-stub benchmark runs.

    Args:
        job: Episode job descriptor with scenario/seed identifiers.
        cfg: Benchmark configuration namespace.
        episode_id: Deterministic episode identifier.
        horizon: Episode horizon in simulation steps.

    Returns:
        dict[str, Any]: Synthetic benchmark episode record.
    """
    now = time.time()
    metrics = {
        "collision_rate": 0.0,
        "success_rate": 1.0,
        # Keep deterministic synthetic defaults stable across runs.
        "time_to_goal": float(horizon) * _STUB_DT_SECONDS,
        "path_efficiency": _STUB_PATH_EFFICIENCY,
        "avg_speed": _STUB_AVG_SPEED,
    }
    termination_reason, outcome, contradictions = _termination_payload_from_metrics(metrics)
    record: dict[str, Any] = {
        "version": "v1",
        "episode_id": episode_id,
        "scenario_id": job.scenario_id,
        "seed": job.seed,
        "archetype": job.archetype,
        "density": job.density,
        "status": status_from_termination_reason(termination_reason),
        "termination_reason": termination_reason,
        "outcome": outcome,
        "integrity": {"contradictions": contradictions},
        "metrics": metrics,
        "steps": min(horizon, _STUB_MAX_STEPS),
        "horizon": horizon,
        "wall_time_sec": 0.0,
        "created_at": now,
        "scenario_params": {
            "archetype": job.archetype,
            "density": job.density,
            "max_episode_steps": horizon,
            "scenario_id": job.scenario_id,
            "hash_fragment": getattr(getattr(job, "scenario", None), "hash_fragment", ""),
        },
        "timing": {"steps_per_second": 0.0},
    }
    if bool(getattr(cfg, "capture_replay", False)):
        replay = [
            (i * _STUB_DT_SECONDS, _STUB_REPLAY_LINEAR_V * i, 0.0, 0.0)
            for i in range(record["steps"])
        ]
        record["replay_steps"] = replay
        record["replay_peds"] = [[] for _ in replay]
        record["replay_ped_forces"] = [[] for _ in replay]
        record["replay_actions"] = [(_STUB_REPLAY_LINEAR_V, 0.0) for _ in replay]
        record["replay_dt"] = _STUB_DT_SECONDS
        record["replay_map_path"] = getattr(getattr(job, "scenario", None), "map_path", "")
    return record


def _require_job_scenario(job, *, episode_id: str):
    """Return job.scenario or raise a contract error when missing.

    Args:
        job: Episode job descriptor.
        episode_id: Deterministic episode identifier for diagnostics.

    Returns:
        Scenario descriptor object attached to the job.
    """
    scenario = getattr(job, "scenario", None)
    if scenario is None:
        raise AggregationMetadataError(
            "Episode job missing scenario descriptor.",
            episode_id=episode_id,
            missing_fields=("scenario",),
            advice="Regenerate jobs via plan_scenarios/expand_episode_jobs.",
        )
    return scenario


def _sanitize_episode_metrics(metrics_raw: dict[str, float]) -> dict[str, float]:
    """Drop non-finite metric values prior to serialization.

    Args:
        metrics_raw: Raw metric payload from metric computation.

    Returns:
        dict[str, float]: Serializable metrics with non-finite values removed.
    """
    return {
        key: value
        for key, value in metrics_raw.items()
        if not isinstance(value, float) or math.isfinite(value)
    }


def _status_from_metrics(metrics: dict[str, float]) -> str:
    """Compute episode status from success and collision metrics.

    Args:
        metrics: Episode metric payload.

    Returns:
        Status label (`success`, `failure`, or `collision`).
    """
    success_rate = float(metrics.get("success_rate", 0.0))
    collision_rate = metrics.get("collision_rate")
    status = "success" if success_rate >= 1.0 else "failure"
    if collision_rate:
        status = "collision"
    return status


def _termination_payload_from_metrics(
    metrics: dict[str, float],
) -> tuple[str, dict[str, bool], list[str]]:
    """Derive canonical termination/outcome payload from metrics.

    Returns:
        tuple[str, dict[str, bool], list[str]]: ``(termination_reason, outcome, contradictions)``.
    """
    route_complete = bool(metric_scalar(metrics, "success", "success_rate") > 0.0)
    collision = bool(metric_scalar(metrics, "collisions", "collision_rate") > 0.0)
    ended = route_complete or collision
    termination_reason = resolve_termination_reason(
        terminated=ended,
        truncated=False,
        success=route_complete,
        collision=collision,
        reached_max_steps=not ended,
    )
    outcome = build_outcome_payload(
        route_complete=route_complete,
        collision=collision,
        timeout=not ended,
    )
    contradictions = outcome_contradictions(
        termination_reason=termination_reason,
        outcome=outcome,
        metrics=metrics,
    )
    if contradictions:
        raise ValueError("Episode integrity contradictions detected: " + "; ".join(contradictions))
    return termination_reason, outcome, contradictions


def _orchestrate_real_episode(
    job,
    cfg,
    *,
    episode_id: str,
    scenario,
    horizon: int,
) -> dict[str, Any]:
    """Run a real environment episode and compute finalized metric payload.

    Args:
        job: Episode job descriptor with seed/archetype metadata.
        cfg: Benchmark configuration namespace.
        episode_id: Deterministic episode identifier.
        scenario: Scenario descriptor attached to ``job``.
        horizon: Episode horizon in simulation steps.

    Returns:
        dict[str, Any]: Runtime payload used for record assembly.
    """
    start_time = time.time()
    env, dt, replay_cap, goal_vec = _init_env_for_job(
        job,
        cfg,
        horizon,
        episode_id=episode_id,
        scenario=scenario,
    )
    try:
        env.reset(seed=int(job.seed))
        robot_positions, ped_positions, ped_forces, reached_goal_step = _rollout_episode(
            env,
            horizon,
            dt,
            replay_cap,
        )
    finally:
        _close_env(env)

    steps_taken = len(robot_positions)
    robot_pos_arr = np.asarray(robot_positions, dtype=float)
    robot_vel_arr, robot_acc_arr = _vel_and_acc(robot_pos_arr, dt)
    ped_pos_arr = _stack_ped_positions(ped_positions)
    ped_forces_arr = _stack_ped_positions(ped_forces, fill_value=np.nan)
    if ped_pos_arr.size and np.isnan(ped_forces_arr).all():
        logger.bind(
            event="ped_forces_missing",
            episode_id=episode_id,
            scenario_id=job.scenario_id,
        ).warning("Pedestrian forces unavailable; force-based metrics will be NaN.")
    env_config = getattr(env, "config", None)
    robot_cfg = getattr(env_config, "robot_config", None)
    sim_cfg = getattr(env_config, "sim_config", None)
    metrics_raw = _compute_episode_metrics(
        job,
        scenario,
        cfg,
        robot_pos=robot_pos_arr,
        robot_vel=robot_vel_arr,
        robot_acc=robot_acc_arr,
        ped_pos=ped_pos_arr,
        ped_forces=ped_forces_arr,
        dt=dt,
        reached_goal_step=reached_goal_step,
        goal=goal_vec,
        horizon=horizon,
        robot_radius=float(getattr(robot_cfg, "radius", 1.0)),
        ped_radius=float(getattr(sim_cfg, "ped_radius", 0.4)),
    )
    wall_time = time.time() - start_time
    return {
        "metrics": _sanitize_episode_metrics(metrics_raw),
        "steps_taken": steps_taken,
        "wall_time": wall_time,
        "start_time": start_time,
        "ped_forces": ped_forces,
        "replay_capture": replay_cap,
    }


def _attach_replay_payload(
    record: dict[str, Any],
    *,
    scenario,
    replay_capture: ReplayCapture,
    ped_forces: list[np.ndarray],
) -> None:
    """Attach finalized replay arrays to an episode record in-place.

    Args:
        record: Episode record being assembled.
        scenario: Scenario descriptor for map path metadata.
        replay_capture: Replay capture object used during rollout.
        ped_forces: Per-step pedestrian force snapshots from rollout.
    """
    episode = replay_capture.finalize()
    finalized = episode.steps
    record["replay_steps"] = [(s.t, s.x, s.y, s.heading) for s in finalized]
    record["replay_peds"] = [s.ped_positions or [] for s in finalized]
    record["replay_ped_forces"] = [np.asarray(f, dtype=float).tolist() for f in ped_forces]
    record["replay_actions"] = [s.action for s in finalized]
    record["replay_rays"] = [s.ray_vecs or [] for s in finalized]
    record["replay_ped_actions"] = [s.ped_actions or [] for s in finalized]
    record["replay_goals"] = [s.robot_goal for s in finalized]
    record["replay_dt"] = episode.dt
    record["replay_map_path"] = getattr(scenario, "map_path", "")


def _make_episode_record(job, cfg) -> dict[str, Any]:
    """Execute one episode through the isolated record-assembly phase.

    Returns:
        Episode record with the established v1 schema.
    """
    hooks = EpisodeExecutionHooks(
        episode_id_from_job=_episode_id_from_job,
        resolve_horizon=_resolve_horizon,
        make_stub_episode_record=_make_stub_episode_record,
        require_job_scenario=_require_job_scenario,
        orchestrate_real_episode=_orchestrate_real_episode,
        attach_replay_payload=_attach_replay_payload,
        termination_payload_from_metrics=_termination_payload_from_metrics,
        ensure_algo_metadata=_ensure_algo_metadata,
    )
    return execute_episode_record(job, cfg, hooks)


def run_episode_jobs(jobs: Iterable[object], cfg, manifest) -> Iterator[dict]:  # T026/T027
    """Execute episode jobs through the resume-aware scheduler facade.

    The facade remains in the historical module so existing callers and tests keep their
    import path.  The scheduler owns partitioning, process execution, and deterministic
    parent-side appends; the callback preserves the existing record-builder seam.
    """
    yield from execute_episode_jobs(
        jobs,
        cfg,
        manifest,
        record_builder=_make_episode_record,
    )


def _execute_seq(job_list, existing_ids, episodes_path, cfg, manifest):
    """Compatibility wrapper for the historical sequential scheduler helper."""
    yield from _scheduler.execute_sequential(
        job_list,
        existing_ids,
        episodes_path,
        cfg,
        manifest,
        _make_episode_record,
    )


def _execute_parallel(job_list, existing_ids, episodes_path, cfg, manifest, workers):
    """Compatibility wrapper for the historical process scheduler helper."""
    yield from _scheduler.execute_parallel(
        job_list,
        existing_ids,
        episodes_path,
        cfg,
        manifest,
        workers,
        _make_episode_record,
    )


def _worker_job_wrapper(job, cfg_payload):
    """Compatibility wrapper for the historical process-worker entrypoint.

    Returns:
        Episode record produced by the configured record builder.
    """
    return _scheduler._worker_job_wrapper(job, cfg_payload, _make_episode_record)


def adaptive_sampling_iteration(
    current_records, cfg, scenarios, manifest
) -> tuple[bool, list[Any]]:  # T028
    """Decide whether additional episode jobs are required.

    Minimal implementation for contract phase:
      - Count existing episodes per scenario from current_records.
      - If counts >= cfg.max_episodes (or no scenarios needing more) -> return (True, []).
      - Else create up to cfg.batch_size new synthetic jobs per iteration (evenly per scenario needing more, but simplified here: all remaining for first scenario).

    Future iterations (T034) will incorporate precision evaluation. Seeds are derived
    by extending scenario.planned_seeds with deterministic incremental integers when
    needed, which keeps smoke runs bounded while preserving reproducibility.

    Returns:
        Tuple of (done_flag, new_jobs_list).
    """
    # Touch manifest to avoid unused param lint (future: record iteration stats)
    _ = manifest
    # Gather counts
    per_scenario: dict[str, int] = {}
    for r in current_records:
        sid = r.get("scenario_id")
        if sid is not None:
            per_scenario[sid] = per_scenario.get(sid, 0) + 1

    # Identify scenarios needing more episodes
    needs: list[object] = []
    max_eps = int(getattr(cfg, "max_episodes", 0) or 0)
    batch_size = int(getattr(cfg, "batch_size", 1) or 1)
    for sc in scenarios:
        count = per_scenario.get(sc.scenario_id, 0)
        if count < max_eps:
            needs.append(sc)

    if not needs:
        return True, []

    # Generate new jobs for first needing scenario (simple contract satisfaction)
    target_sc = needs[0]
    existing = per_scenario.get(target_sc.scenario_id, 0)
    remaining = max_eps - existing
    to_create = min(batch_size, remaining)

    # Derive seeds: reuse planned_seeds then extend with increasing integers
    seeds: list[int] = list(getattr(target_sc, "planned_seeds", []))
    # Ensure enough seeds
    while len(seeds) < existing + to_create:
        seeds.append(len(seeds))  # deterministic extension

    # Build lightweight job objects (mirroring EpisodeJob subset) without relying on full dataclass
    jobs = []
    horizon = getattr(cfg, "horizon_override", None) or 100
    start_index = existing
    for i in range(to_create):
        seed = seeds[start_index + i]
        job_id = f"{target_sc.scenario_id}:{seed}:{horizon}"  # simple deterministic id
        job = type("EpisodeJobLite", (), {})()
        job.job_id = job_id
        job.scenario_id = target_sc.scenario_id
        job.seed = seed
        job.archetype = getattr(target_sc, "archetype", "unknown")
        job.density = getattr(target_sc, "density", "unknown")
        job.horizon = horizon
        job.scenario = target_sc
        jobs.append(job)

    done_flag = False  # more iterations likely needed until max reached
    # If after adding this batch we would reach or exceed max for all scenarios mark done next time
    if existing + to_create >= max_eps and len(needs) == 1:
        # After these jobs scenario will be full; check others already full.
        done_flag = all(per_scenario.get(sc.scenario_id, 0) >= max_eps for sc in scenarios)

    return done_flag, jobs


def run_full_benchmark(  # noqa: C901
    cfg,
) -> BenchmarkManifest:  # T029 + T034 integration (refactored in polish phase)
    """Execute classic benchmark with adaptive precision loop.

    Refactored to reduce cyclomatic complexity (extracting helpers for setup, manifest
    initialization, scaling efficiency instrumentation, artifact writes). Public
    semantics preserved for existing tests.

    Returns:
        Final BenchmarkManifest object with execution statistics and artifact paths.
    """
    # Resolve the immutable setup/provenance boundary once.
    context = build_run_context(cfg)
    root = context.root
    episodes_path = context.episodes_path
    raw = list(context.raw_scenarios)
    scenario_matrix_hash = context.scenario_matrix_hash
    scenarios_list = list(context.scenarios)
    jobs = list(context.jobs)

    # Manifest & initial execution
    manifest = _init_manifest(root, episodes_path, cfg, scenario_matrix_hash)
    freeze_manifest_path = getattr(cfg, "freeze_manifest_path", None)
    if freeze_manifest_path:
        manifest.freeze_validation = evaluate_freeze_manifest(
            freeze_manifest_path,
            cfg,
            scenario_matrix_hash=scenario_matrix_hash,
            git_commit=manifest.git_hash,
            raw_scenarios=raw,
        )
        freeze_status = manifest.freeze_validation.get("status")
        if freeze_status == "mismatch":
            logger.warning(
                "Freeze manifest mismatch: {} differences against {}",
                manifest.freeze_validation.get("mismatch_count", 0),
                freeze_manifest_path,
            )
            for mismatch in manifest.freeze_validation.get("mismatches", []):
                logger.warning(
                    "Freeze mismatch {}: expected={} observed={}",
                    mismatch.get("path"),
                    mismatch.get("expected"),
                    mismatch.get("observed"),
                )
        elif freeze_status == "error":
            logger.warning(
                "Freeze manifest validation failed for {}: {}",
                freeze_manifest_path,
                manifest.freeze_validation.get("error"),
            )
        else:
            logger.info("Freeze manifest matched runtime contract: {}", freeze_manifest_path)

    all_records = list(run_episode_jobs(jobs, cfg, manifest))
    max_episodes = int(getattr(cfg, "max_episodes", 0) or 0)

    # Adaptive loop (iteration guard for smoke / tiny budgets)
    iteration_count = 0
    while True:
        groups = aggregate_metrics(all_records, cfg)
        effects = compute_effect_sizes(groups, cfg)
        precision_report = evaluate_precision(groups, cfg)

        # Instrumentation & artifact persistence
        scaling = _update_scaling_efficiency(manifest, cfg)
        try:  # attach for downstream JSON serialization if model allows attribute
            precision_report.scaling_efficiency = scaling  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # precision_report may be a plain dict or a lightweight namespace; ignore
            # absence of attribute or wrong type but do not swallow unrelated errors.
            pass
        _write_iteration_artifacts(root, groups, effects, precision_report)

        # Exit conditions
        if precision_report.final_pass:
            logger.info("Precision criteria met; stopping adaptive loop")
            break
        if max_episodes and sum(g.count for g in groups) >= max_episodes * len(scenarios_list):
            logger.info("Reached max episodes budget; stopping adaptive loop")
            break

        # Additional sampling
        done_flag, new_jobs = adaptive_sampling_iteration(
            all_records,
            cfg,
            scenarios_list,
            manifest,
        )
        if not new_jobs:
            if done_flag:
                logger.info("Adaptive iteration indicated done; no new jobs.")
            break
        new_records = list(run_episode_jobs(new_jobs, cfg, manifest))
        all_records.extend(new_records)
        iteration_count += 1
        # Safety: In smoke mode with very small episode budgets we break after first iteration
        # to prevent runaway loops in early scaffolding stages.
        if getattr(cfg, "smoke", False) and max_episodes <= 2 and iteration_count >= 1:
            logger.info("Early exit guard (smoke small-budget) triggered after first iteration")
            break

    # Close the run through the finalizer boundary.  Optional visualization callables are
    # injected from this facade so existing monkeypatches and dependency availability remain
    # observable without changing the public entrypoint.
    finalize_run(
        root,
        cfg,
        manifest,
        groups=groups,
        all_records=all_records,
        write_run_meta_files_fn=_write_run_meta_files,
        visual_generator=generate_visual_artifacts,
        visualization_available=_VISUALIZATION_AVAILABLE,
        plot_generator=globals().get("generate_benchmark_plots"),
        validation_fn=globals().get("validate_visual_artifacts"),
        visualization_error=globals().get("VisualizationError", _finalizer.VisualizationError),
    )

    return manifest
