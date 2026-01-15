"""Orchestration logic for executing episode jobs and adaptive sampling.

Implemented incrementally in tasks T026-T029, T027 (parallel), T028 (adaptive iteration),
T029 (full run orchestration skeleton).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from robot_sf.benchmark.errors import AggregationMetadataError
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, snqi
from robot_sf.benchmark.obstacle_sampling import sample_obstacle_points
from robot_sf.benchmark.path_utils import compute_shortest_path_length
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    resolve_map_definition,
)

from .aggregation import aggregate_metrics
from .effects import compute_effect_sizes
from .io_utils import append_episode_record, write_manifest
from .planning import expand_episode_jobs, load_scenario_matrix, plan_scenarios
from .precision import evaluate_precision
from .replay import ReplayCapture  # T021 optional replay capture
from .visuals import generate_visual_artifacts  # new visual artifact integration

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# Import new visualization functions for real plots/videos from episode data
try:
    from robot_sf.benchmark.visualization import (
        VisualizationError,
        generate_benchmark_plots,
        validate_visual_artifacts,
    )

    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

# -----------------------------
# Manifest dataclass & helpers
# -----------------------------


@dataclass
class BenchmarkManifest:
    """Manifest metadata for a benchmark run."""

    output_root: Path
    git_hash: str
    scenario_matrix_hash: str
    config: object
    episodes_path: str
    created_at: float = field(default_factory=time.time)
    executed_jobs: int = 0
    skipped_jobs: int = 0
    notes: str = "skeleton_t029"
    runtime_sec: float = 0.0
    episodes_per_second: float = 0.0
    workers: int = 1
    scaling_efficiency: dict = field(default_factory=dict)


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
        record["algorithm_metadata"] = algo_meta
    algo_meta.setdefault("algorithm", algo_value)
    algo_meta.setdefault("status", "ok")

    record["algo"] = algo_value
    return record


def _compute_git_hash(root: Path) -> str:
    """Best‑effort retrieval of current git HEAD short hash.

    Falls back to 'unknown' if repository metadata is inaccessible. Separated to keep
    orchestration function lean (polish phase refactor for C901).

    Returns:
        Short git hash (12 characters) or 'unknown' if not retrievable.
    """
    git_hash = "unknown"
    try:  # pragma: no cover - environment dependent
        head_ref = root / ".git" / "HEAD"
        if head_ref.exists():
            content = head_ref.read_text(encoding="utf-8").strip()
            if content.startswith("ref:"):
                ref_path = content.split(" ", 1)[1].strip()
                ref_file = root / ".git" / ref_path
                if ref_file.exists():
                    git_hash = ref_file.read_text(encoding="utf-8").strip()[:12]
            else:
                git_hash = content[:12]
    except OSError as exc:
        # Filesystem access errors -> return unknown but log for diagnostics
        logger.debug("_compute_git_hash fs access error: %s", exc)
    except (RuntimeError, TypeError):  # pragma: no cover - defensive
        # Unexpected but plausible runtime/type errors -> log at debug and continue
        logger.debug("_compute_git_hash unexpected error")
    return git_hash


def _prepare_output_dirs(cfg):
    """Create and return output directories for benchmark artifacts.

    Returns:
        Tuple of (root, episodes_dir, aggregates_dir, reports_dir, plots_dir).
    """
    root = Path(cfg.output_root)
    episodes_dir = root / "episodes"
    aggregates_dir = root / "aggregates"
    reports_dir = root / "reports"
    plots_dir = root / "plots"
    for d in (episodes_dir, aggregates_dir, reports_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)
    return root, episodes_dir, aggregates_dir, reports_dir, plots_dir


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


def _update_scaling_efficiency(manifest: BenchmarkManifest, cfg):
    """Update runtime, throughput and synthetic parallel efficiency stats in manifest.

    Returns:
        Updated scaling_efficiency dictionary from the manifest.
    """
    now = time.time()
    manifest.runtime_sec = max(0.0, now - manifest.created_at)
    manifest.workers = int(getattr(cfg, "workers", 1) or 1)
    if manifest.runtime_sec > 0:
        manifest.episodes_per_second = manifest.executed_jobs / manifest.runtime_sec
    ideal_rate = manifest.workers * manifest.episodes_per_second if manifest.workers > 0 else 0
    efficiency = 0.0
    if ideal_rate > 0:
        efficiency = manifest.episodes_per_second / ideal_rate
    manifest.scaling_efficiency = {
        "runtime_sec": manifest.runtime_sec,
        "executed_jobs": manifest.executed_jobs,
        "skipped_jobs": manifest.skipped_jobs,
        "episodes_per_second": manifest.episodes_per_second,
        "workers": manifest.workers,
        "parallel_efficiency_placeholder": efficiency,
    }
    return manifest.scaling_efficiency


def _write_iteration_artifacts(root: Path, groups, effects, precision_report):
    """Write aggregate/report JSON artifacts for an iteration."""
    _write_json(root / "aggregates" / "summary.json", _serialize_groups(groups))
    _write_json(root / "reports" / "effect_sizes.json", _serialize_effects(effects))
    _write_json(
        root / "reports" / "statistical_sufficiency.json",
        _serialize_precision(precision_report),
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


def _scan_existing_episode_ids(path: Path) -> set[str]:
    """Return episode_id values already present in an episodes JSONL file."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Lightweight JSON parse (json imported at module top level)
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed episode record line in {}", path)
                    continue
                ep_id = rec.get("episode_id")
                if isinstance(ep_id, str):
                    ids.add(ep_id)
    except OSError as exc:  # pragma: no cover - unlikely on normal FS
        logger.warning("Failed reading existing episodes file {}: {}", path, exc)
    return ids


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
        fallback = (
            Path(__file__).resolve().parents[3] / "maps" / "svg_maps" / "classic_crossing.svg"
        )
        if fallback.exists():
            raw["map_file"] = str(fallback)
    config = build_robot_config_from_scenario(
        raw,
        scenario_path=matrix_path,
    )
    try:
        dt = float(config.sim_config.time_per_step_in_secs)
    except Exception:  # pragma: no cover - defensive
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
    heading_err = (desired_heading - heading + math.pi) % (2 * math.pi) - math.pi
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
    except Exception:
        return np.full_like(ped_pos, np.nan)

    if arr.shape == ped_pos.shape:
        return arr

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
            except Exception:
                robot_goal = None
        return ray_vecs, ped_actions, robot_goal
    except Exception:
        return None, None, None


def _compute_episode_metrics(
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
        except Exception:  # pragma: no cover - defensive fallback
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
    except Exception:  # pragma: no cover - defensive
        metrics["snqi"] = float("nan")
    serializable: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            serializable[key] = float(value)
        else:
            serializable[key] = value
    return serializable


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
        except Exception:  # pragma: no cover - defensive fallback
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
        robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
        heading = float(env.simulator.robot_poses[0][1])
        peds = np.asarray(env.simulator.ped_pos, dtype=float)
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
        except Exception:
            # Rendering is best-effort; ignore to keep rollout running
            pass
        if reached_goal_step is None and bool(info.get("success")):
            reached_goal_step = step_idx
        if terminated or truncated:
            break
    return robot_positions, ped_positions, ped_forces, reached_goal_step


def _close_env(env):
    """Best-effort environment cleanup."""
    try:
        env.exit()
    except Exception:  # pragma: no cover
        pass
    try:
        env.close()
    except Exception:  # pragma: no cover - gym close best-effort
        pass


def _make_episode_record(job, cfg) -> dict[str, Any]:
    """Execute a real episode using the environment factory and compute metrics.

    Returns:
        Episode record dictionary containing metrics, status, and metadata.
    """

    episode_id = _episode_id_from_job(job)
    if bool(getattr(cfg, "fast_stub", False)):
        now = time.time()
        horizon = _resolve_horizon(job, cfg)
        record: dict[str, Any] = {
            "version": "v1",
            "episode_id": episode_id,
            "scenario_id": job.scenario_id,
            "seed": job.seed,
            "archetype": job.archetype,
            "density": job.density,
            "status": "success",
            "metrics": {
                "collision_rate": 0.0,
                "success_rate": 1.0,
                "time_to_goal": float(horizon) * 0.1,
                "path_efficiency": 0.9,
                "avg_speed": 1.0,
            },
            "steps": min(horizon, 5),
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
            replay = [(i * 0.1, 0.05 * i, 0.0, 0.0) for i in range(record["steps"])]
            record["replay_steps"] = replay
            record["replay_peds"] = [[] for _ in replay]
            record["replay_actions"] = [(0.05, 0.0) for _ in replay]
            record["replay_dt"] = 0.1
            record["replay_map_path"] = getattr(getattr(job, "scenario", None), "map_path", "")
        _ensure_algo_metadata(record, algo=getattr(cfg, "algo", None), episode_id=episode_id)
        return record
    scenario = getattr(job, "scenario", None)
    if scenario is None:
        raise AggregationMetadataError(
            "Episode job missing scenario descriptor.",
            episode_id=episode_id,
            missing_fields=("scenario",),
            advice="Regenerate jobs via plan_scenarios/expand_episode_jobs.",
        )
    horizon = _resolve_horizon(job, cfg)
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
    )
    metrics: dict[str, float | None] = {}
    for key, value in metrics_raw.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            continue
        metrics[key] = value
    success_rate = float(metrics.get("success_rate") or 0.0)
    collision_rate = metrics.get("collision_rate")
    status = "success" if success_rate >= 1.0 else "failure"
    if collision_rate:
        status = "collision"
    wall_time = time.time() - start_time
    record: dict[str, Any] = {
        "version": "v1",
        "episode_id": episode_id,
        "scenario_id": job.scenario_id,
        "seed": job.seed,
        "archetype": job.archetype,
        "density": job.density,
        "status": status,
        "metrics": metrics,
        "steps": steps_taken,
        "horizon": horizon,
        "wall_time_sec": wall_time,
        "created_at": start_time,
        "timing": {
            "steps_per_second": float(steps_taken) / wall_time if wall_time > 0 else 0.0,
        },
        "scenario_params": {
            "archetype": job.archetype,
            "density": job.density,
            "max_episode_steps": horizon,
            "scenario_id": job.scenario_id,
            "map_file": getattr(scenario, "map_path", ""),
            "simulation_config": getattr(scenario, "raw", {}).get("simulation_config", {}),
            "metadata": getattr(scenario, "raw", {}).get("metadata", {}),
            "hash_fragment": getattr(scenario, "hash_fragment", ""),
        },
    }
    if bool(getattr(cfg, "capture_replay", False)) and replay_cap is not None:
        episode = replay_cap.finalize()
        episode.map_path = getattr(scenario, "map_path", "")
        finalized = episode.steps
        record["replay_steps"] = [(s.t, s.x, s.y, s.heading) for s in finalized]
        record["replay_peds"] = [s.ped_positions or [] for s in finalized]
        record["replay_actions"] = [s.action for s in finalized]
        record["replay_rays"] = [s.ray_vecs or [] for s in finalized]
        record["replay_ped_actions"] = [s.ped_actions or [] for s in finalized]
        record["replay_goals"] = [s.robot_goal for s in finalized]
        record["replay_dt"] = episode.dt
        record["replay_map_path"] = episode.map_path
    _ensure_algo_metadata(
        record,
        algo=getattr(cfg, "algo", None),
        episode_id=episode_id,
    )
    return record


def _partition_jobs(existing_ids: set[str], job_iter: Iterable[object]) -> tuple[list[object], int]:
    """Split incoming jobs into runnable jobs and skipped count.

    Args:
        existing_ids: Episode IDs already present on disk.
        job_iter: Iterable of job payloads (scenario + seed).

    Returns:
        Tuple of (jobs_to_run, skipped_count).
    """
    run_list: list[object] = []
    skip_count = 0
    for jb in job_iter:
        if _episode_id_from_job(jb) in existing_ids:
            skip_count += 1
        else:
            run_list.append(jb)
    return run_list, skip_count


def _execute_seq(
    job_list: list[object],
    existing_ids: set[str],
    episodes_path: Path,
    cfg,
    manifest,
) -> Iterator[dict]:
    """Execute episode jobs sequentially while appending JSONL output.

    Args:
        job_list: Episode configurations to run.
        existing_ids: Episode ids that already exist on disk (used to skip duplicates).
        episodes_path: Target JSONL path for appending completed episode records.
        cfg: Benchmark configuration namespace.
        manifest: Mutable manifest object for accounting (executed_jobs, etc.).

    Yields:
        dict: Episode record emitted after each job finishes.
    """
    for jb in job_list:
        start = time.time()
        rec = _make_episode_record(jb, cfg)
        end = time.time()
        rec["wall_time_sec"] = end - start
        append_episode_record(episodes_path, rec)
        existing_ids.add(rec["episode_id"])
        if hasattr(manifest, "executed_jobs"):
            manifest.executed_jobs += 1
        yield rec


def _worker_job_wrapper(job, cfg_payload):  # top-level for pickling on spawn
    """Run a single job with a lightweight config payload for multiprocessing.

    Args:
        job: Episode job payload.
        cfg_payload: Serializable config values to build a temp config.

    Returns:
        Episode record dictionary with wall_time_sec added.
    """

    class _TempCfg:
        """Namespace-like wrapper for passing config values into worker jobs."""

        def __init__(self, payload):
            """Populate config fields from a dict payload.

            Args:
                payload: Mapping of config keys to values.
            """
            for k, v in payload.items():
                setattr(self, k, v)

    start = time.time()
    rec = _make_episode_record(job, _TempCfg(cfg_payload))
    rec["wall_time_sec"] = time.time() - start
    return rec


def _execute_parallel(
    job_list: list[object],
    existing_ids: set[str],
    episodes_path: Path,
    cfg,
    manifest,
    workers: int,
) -> Iterator[dict]:
    """Execute episode jobs in parallel worker processes with deterministic appends.

    Args:
        job_list: Episode configurations to run.
        existing_ids: Episode ids to skip (already present on disk).
        episodes_path: Target JSONL path for appending completed episode records.
        cfg: Benchmark configuration namespace.
        manifest: Mutable manifest object for accounting (executed_jobs, etc.).
        workers: Number of process-pool workers to launch.

    Yields:
        dict: Episode record emitted once the parent process appends the result.
    """
    logger.debug("Executing {} jobs in parallel with {} workers", len(job_list), workers)
    cfg_payload = vars(cfg).copy() if hasattr(cfg, "__dict__") else {}
    if "disable_videos" not in cfg_payload:
        cfg_payload["disable_videos"] = True
    results_map: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(_worker_job_wrapper, j, cfg_payload): j for j in job_list}
        for fut in as_completed(future_map):
            rec = fut.result()
            results_map[rec["episode_id"]] = rec
    # Deterministic ordering for append
    for jb in job_list:
        ep_id = _episode_id_from_job(jb)
        rec = results_map[ep_id]
        append_episode_record(episodes_path, rec)
        existing_ids.add(ep_id)
        if hasattr(manifest, "executed_jobs"):
            manifest.executed_jobs += 1
        yield rec


def run_episode_jobs(jobs: Iterable[object], cfg, manifest) -> Iterator[dict]:  # T026/T027
    """Execute episode jobs with resume + optional parallel workers.

    T026 (completed earlier): sequential execution + resume scan.
    T027 extension:
      - If cfg.workers > 1 use a process pool to compute episode records in parallel.
      - Parent process performs file appends (avoids concurrent writes).
      - Update manifest counters: executed_jobs, skipped_jobs.
    """
    episodes_path = Path(manifest.episodes_path)
    existing_ids = _scan_existing_episode_ids(episodes_path)
    logger.debug("Found {} existing episode records (resume)", len(existing_ids))
    to_run, skipped = _partition_jobs(existing_ids, list(jobs))
    if hasattr(manifest, "skipped_jobs"):
        manifest.skipped_jobs += skipped
    workers = int(getattr(cfg, "workers", 1) or 1)
    if workers <= 1 or len(to_run) <= 1:
        yield from _execute_seq(to_run, existing_ids, episodes_path, cfg, manifest)
    else:
        yield from _execute_parallel(to_run, existing_ids, episodes_path, cfg, manifest, workers)


def adaptive_sampling_iteration(current_records, cfg, scenarios, manifest):  # T028
    """Decide whether additional episode jobs are required (placeholder T028).

    Minimal implementation for contract phase:
      - Count existing episodes per scenario from current_records.
      - If counts >= cfg.max_episodes (or no scenarios needing more) -> return (True, []).
      - Else create up to cfg.batch_size new synthetic jobs per iteration (evenly per scenario needing more, but simplified here: all remaining for first scenario).

    Future iterations (T034) will incorporate precision evaluation. Seeds are derived
    by extending scenario.planned_seeds with deterministic incremental integers if
    needed (placeholder logic) to avoid blocking on full seeding strategy.

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


def run_full_benchmark(cfg):  # T029 + T034 integration (refactored in polish phase)  # noqa: C901
    """Execute classic benchmark with adaptive precision loop.

    Refactored to reduce cyclomatic complexity (extracting helpers for setup, manifest
    initialization, scaling efficiency instrumentation, artifact writes). Public
    semantics preserved for existing tests.

    Returns:
        Final BenchmarkManifest object with execution statistics and artifact paths.
    """
    # Output & planning
    root, episodes_dir, _aggregates_dir, _reports_dir, _plots_dir = _prepare_output_dirs(cfg)
    episodes_path = episodes_dir / "episodes.jsonl"
    raw = load_scenario_matrix(cfg.scenario_matrix_path)
    matrix_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    scenario_matrix_hash = hashlib.sha1(matrix_bytes).hexdigest()[:12]
    rng = random.Random(int(getattr(cfg, "master_seed", 123)))
    scenarios = plan_scenarios(raw, cfg, rng=rng)
    jobs = expand_episode_jobs(scenarios, cfg)
    scenarios_list = list(scenarios)
    smoke_limit = bool(getattr(cfg, "smoke_limit_jobs", False))
    if getattr(cfg, "smoke", False) and scenarios_list and smoke_limit:
        scenarios_list = scenarios_list[:1]
        allowed = {sc.scenario_id for sc in scenarios_list}
        jobs = [jb for jb in jobs if jb.scenario_id in allowed]
        jobs = jobs[: max(1, int(getattr(cfg, "smoke_episodes", 1) or 1))]
    if getattr(cfg, "smoke", False):
        horizon_cap = int(getattr(cfg, "smoke_horizon_cap", 40) or 40)
        for jb in jobs:
            try:
                jb.horizon = min(int(getattr(jb, "horizon", horizon_cap)), horizon_cap)
            except Exception:
                jb.horizon = horizon_cap

    # Manifest & initial execution
    manifest = _init_manifest(root, episodes_path, cfg, scenario_matrix_hash)
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

    # Finalize & persist manifest
    _update_scaling_efficiency(manifest, cfg)
    manifest.scaling_efficiency.setdefault("finalized", True)
    write_manifest(manifest, str(root / "manifest.json"))

    # Visual artifacts (plots + videos) generation (post adaptive loop single pass)
    try:
        generate_visual_artifacts(root, cfg, groups, all_records)

        # Also generate real visualizations using new visualization module
        # Skip generating heavy real visualizations when running in smoke mode
        # (smoke mode intentionally keeps runtime small for tests).
        if _VISUALIZATION_AVAILABLE and not getattr(cfg, "smoke", False):
            logger.info("Generating additional real visualizations from episode data")
            try:
                plots_dir = root / "plots"
                videos_dir = root / "videos"
                plots_dir.mkdir(exist_ok=True)
                videos_dir.mkdir(exist_ok=True)

                # Generate real plots from episode data
                plot_artifacts = generate_benchmark_plots(all_records, str(root))
                logger.info(
                    "Generated {} real plots into {}",
                    len(plot_artifacts),
                    plots_dir,
                )

                # Skip legacy episode_*.mp4 generation when SimulationView videos are available
                video_artifacts = []
                logger.debug(
                    "Skipping legacy episode video generation (sim-view videos already produced)"
                )
                logger.info("Generated sim_view videos into {}", videos_dir)

                # Validate all generated artifacts
                all_artifacts = plot_artifacts + video_artifacts
                validation = validate_visual_artifacts(all_artifacts)
                if validation.passed:
                    logger.info("All real visualizations validated successfully")
                else:
                    logger.warning(
                        "Some visualizations failed validation: {} failed artifacts",
                        len(validation.failed_artifacts),
                    )

            except (VisualizationError, FileNotFoundError) as vis_exc:
                logger.warning("Real visualization generation failed (non-fatal): {}", vis_exc)

    except (VisualizationError, FileNotFoundError) as exc:
        logger.warning("Visual artifact generation failed (non-fatal): {}", exc)

    return manifest


def _write_json(path: Path, obj):  # helper
    """Write JSON to disk with a temp file for atomic replace.

    Args:
        path: Output path to write.
        obj: JSON-serializable payload.
    """
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        tmp.replace(path)
    except (OSError, TypeError) as exc:
        logger.warning("Failed writing JSON artifact {}: {}", path, exc)


def _serialize_groups(groups):
    """Serialize aggregation group objects into JSON-friendly dicts.

    Args:
        groups: Iterable of aggregation group objects.

    Returns:
        List of serialized group dictionaries.
    """
    out = []
    for g in groups:
        out.append(
            {
                "archetype": g.archetype,
                "density": g.density,
                "count": g.count,
                "metrics": {
                    k: {
                        "mean": m.mean,
                        "median": m.median,
                        "p95": m.p95,
                        "mean_ci": m.mean_ci,
                        "median_ci": m.median_ci,
                    }
                    for k, m in g.metrics.items()
                },
            },
        )
    return out


def _serialize_effects(effects):
    """Serialize effect-size reports into JSON-friendly dicts.

    Args:
        effects: Iterable of effect report objects.

    Returns:
        List of serialized effect size report dictionaries.
    """
    out = []
    for rep in effects:
        out.append(
            {
                "archetype": rep.archetype,
                "comparisons": [
                    {
                        "metric": c.metric,
                        "density_low": c.density_low,
                        "density_high": c.density_high,
                        "diff": c.diff,
                        "standardized": c.standardized,
                    }
                    for c in rep.comparisons
                ],
            },
        )
    return out


def _serialize_precision(report):
    """Serialize precision report into a JSON-friendly dictionary.

    Args:
        report: Precision report object.

    Returns:
        Dictionary containing serialized precision report.
    """
    return {
        "final_pass": report.final_pass,
        "evaluations": [
            {
                "scenario_id": ev.scenario_id,
                "archetype": ev.archetype,
                "density": ev.density,
                "episodes": ev.episodes,
                "all_pass": ev.all_pass,
                "metric_status": [
                    {
                        "metric": ms.metric,
                        "half_width": ms.half_width,
                        "target": ms.target,
                        "passed": ms.passed,
                    }
                    for ms in ev.metric_status
                ],
            }
            for ev in report.evaluations
        ],
    }
