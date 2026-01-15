"""Episode runner scaffold for the Social Navigation Benchmark.

Responsibilities:
 - Load scenario parameter matrix (YAML list) and expand repeats
 - Deterministically generate scenarios (using `generate_scenario`)
 - Step a simple baseline policy toward each agent's goal (very naive)
 - Collect per-step robot + pedestrian states and (optionally) forces
 - Build EpisodeData and compute metrics + SNQI (weights optional)
 - Validate record against JSON schema and write JSONL

This is an initial lightweight runner; performance optimizations and
multi-processing are future work.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import (
    UTC,  # type: ignore[attr-defined]
    datetime,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import yaml
from loguru import logger

try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    ImageSequenceClip = None  # type: ignore[assignment]

try:
    from robot_sf.baselines import get_baseline, list_baselines
    from robot_sf.baselines.social_force import Observation
except ImportError as exc:  # pragma: no cover - optional baseline dependency
    get_baseline = None  # type: ignore[assignment]
    list_baselines = None  # type: ignore[assignment]
    Observation = None  # type: ignore[assignment]
    _BASELINE_IMPORT_ERROR = exc
else:
    _BASELINE_IMPORT_ERROR = None

from robot_sf.benchmark.constants import EPISODE_SCHEMA_VERSION
from robot_sf.benchmark.manifest import load_manifest, save_manifest
from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, snqi
from robot_sf.benchmark.obstacle_sampling import sample_obstacle_points
from robot_sf.benchmark.scenario_generator import generate_scenario
from robot_sf.benchmark.schema_validator import load_schema, validate_episode
from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _load_baseline_planner(algo: str, algo_config_path: str | None, seed: int):
    """Load and construct a baseline planner from the registry.

    Returns (planner, ObservationCls, config_dict).

    Returns:
        Tuple of (planner instance, Observation class, config dict).
    """
    if get_baseline is None or Observation is None:
        raise RuntimeError(f"Failed to import baseline algorithms: {_BASELINE_IMPORT_ERROR}")

    try:
        planner_class = get_baseline(algo)
    except KeyError:
        available = list_baselines() if list_baselines is not None else []
        raise ValueError(f"Unknown algorithm '{algo}'. Available: {available}")

    # Load configuration if provided
    config = {}
    if algo_config_path:
        config_path = Path(algo_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Algorithm config file not found: {algo_config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            if not isinstance(cfg, dict):
                raise TypeError("Algorithm config must be a mapping (YAML dict).")
            config = cfg
    planner = planner_class(config, seed=seed)
    return planner, Observation, config


def _build_observation(ObservationCls, robot_pos, robot_vel, robot_goal, ped_positions, dt):
    """Build an Observation instance from robot and pedestrian state.

    Args:
        ObservationCls: The Observation class to instantiate.
        robot_pos: Current robot position array.
        robot_vel: Current robot velocity array.
        robot_goal: Robot goal position array.
        ped_positions: Array of pedestrian positions.
        dt: Timestep duration.

    Returns:
        Observation instance with current state data.
    """
    agents = [
        {"position": pos.tolist(), "velocity": [0.0, 0.0], "radius": 0.35} for pos in ped_positions
    ]
    return ObservationCls(
        dt=dt,
        robot={
            "position": robot_pos.tolist(),
            "velocity": robot_vel.tolist(),
            "goal": robot_goal.tolist(),
            "radius": 0.3,
        },
        agents=agents,
        obstacles=[],
    )


# Safety/robustness defaults for any baseline policy
POLICY_STEP_TIMEOUT_SECS: float = 0.2  # step(obs) time budget; fallback to zero action on timeout
FINAL_SPEED_CLAMP: float = 2.0  # m/s cap to prevent unrealistic velocities


def _git_hash_fallback() -> str:
    # Best effort; avoid importing subprocess if not needed later
    """Return the current git hash or 'unknown' if unavailable."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        OSError,
    ) as exc:  # pragma: no cover
        # Best-effort fallback when git is unavailable or fails; retain behavior
        logger.debug("_git_hash_fallback failed: %s", exc)
        return "unknown"


def _config_hash(obj: Any) -> str:
    """Return a stable, deterministic 16-char hash from JSON serialization.

    Args:
        obj: Object to hash (must be JSON-serializable).

    Returns:
        16-character hex hash string.
    """
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def compute_episode_id(scenario_params: dict[str, Any], seed: int) -> str:  # legacy wrapper
    """Backward-compatible wrapper returning deterministic episode id.

    Uses new `make_episode_id` (sha256-based) while preserving previous readable
    prefix by embedding scenario id when available. Existing manifests that
    relied on the old format will naturally diverge (different ids) — identity
    hash below ensures resume manifests invalidate cleanly.

    Returns:
        Episode ID string in format <scenario_id>--<seed>.
    """
    scenario_id = (
        scenario_params.get("id")
        or scenario_params.get("name")
        or scenario_params.get("scenario_id")
        or "unknown"
    )
    # For current test suite compatibility and simple resume semantics we use
    # the legacy id pattern '<scenario_id>--<seed>'. A richer hash-based form
    # can be reintroduced later once all tests & manifests are migrated.
    return f"{scenario_id}--{seed}"


def _episode_identity_hash() -> str:
    """Return a short hash that fingerprints episode identity definition.

    Intentionally small and stable across runs of the same code. If the
    implementation of ``compute_episode_id`` changes, this hash will change
    as well, which invalidates any previously saved manifest sidecars.

    Returns:
        12-character hex hash of compute_episode_id implementation.
    """
    try:
        src = inspect.getsource(compute_episode_id)
    except (OSError, TypeError):
        # Fallback to function name when source isn't available (e.g., pyc only)
        src = compute_episode_id.__name__
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def index_existing(out_path: Path) -> set[str]:
    """Scan an existing JSONL file and return the set of episode_ids found.

    Tolerates malformed lines and missing keys; logs are not emitted here to keep
    the function side-effect free (caller may log if desired).

    Returns:
        Set of episode IDs found in the JSONL file.
    """
    ids: set[str] = set()
    try:
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    eid = rec.get("episode_id")
                    if isinstance(eid, str):
                        ids.add(eid)
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Ignore malformed JSON lines
                    continue
    except FileNotFoundError:
        return set()
    except OSError as exc:
        logger.debug(
            "index_existing unexpected error reading %s: %s", out_path, exc
        )  # pragma: no cover
        return set()
    return ids


def load_scenario_matrix(path: str | Path) -> list[dict[str, Any]]:
    """Load a scenario matrix from YAML (stream, list, or dict with scenarios).

    Returns:
        List of scenario dictionaries.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    # Allow either YAML stream of docs or a single list
    if len(docs) == 1 and isinstance(docs[0], list):
        scenarios = docs[0]
    elif len(docs) == 1 and isinstance(docs[0], dict) and "scenarios" in docs[0]:
        scenarios = docs[0].get("scenarios", [])
    else:
        scenarios = docs
    return [dict(s) for s in scenarios]


def _simple_robot_policy(robot_pos: np.ndarray, goal: np.ndarray, speed: float = 1.0) -> np.ndarray:
    """Return a velocity vector pointing toward goal with capped speed.

    Returns:
        Velocity vector as 2D numpy array.
    """
    vec = goal - robot_pos
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return np.zeros(2)
    dir_unit = vec / dist
    v = dir_unit * min(speed, dist)
    return v


def _prepare_robot_points(
    robot_start: Sequence[float] | None,
    robot_goal: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert robot start/goal to numpy arrays with defaults.

    Args:
        robot_start: Robot starting position or None for default.
        robot_goal: Robot goal position or None for default.

    Returns:
        Tuple of (start position array, goal position array).
    """
    if robot_start is None:
        rs = np.array([0.3, 3.0], dtype=float)
    else:
        rs = np.asarray(robot_start, dtype=float)
    if robot_goal is None:
        rg = np.array([9.7, 3.0], dtype=float)
    else:
        rg = np.asarray(robot_goal, dtype=float)
    return rs, rg


def _stack_or_zero(
    traj: list[np.ndarray],
    *,
    stack_fn: Callable[[Sequence[np.ndarray]], np.ndarray],
    empty_shape: tuple[int, ...],
) -> np.ndarray:
    """Stack recorded trajectory data or return a zero-length array of known shape.

    Note: To avoid unnecessary memory allocation, `empty_shape` should have zero in the first dimension.

    Returns:
        Stacked trajectory array or empty array with specified shape.
    """
    if traj:
        return stack_fn(traj)
    else:
        # Ensure empty_shape[0] == 0 for lazy evaluation
        assert empty_shape[0] == 0, (
            "empty_shape should have zero in the first dimension for lazy evaluation"
        )
        # Return a zero-length array with the correct shape and dtype
        return np.empty(empty_shape)


def _build_episode_data(
    robot_pos_traj: list[np.ndarray],
    robot_vel_traj: list[np.ndarray],
    robot_acc_traj: list[np.ndarray],
    peds_pos_traj: list[np.ndarray],
    ped_forces_traj: list[np.ndarray],
    obstacles: np.ndarray | None,
    goal: np.ndarray,
    dt: float,
    reached_goal_step: int | None,
) -> EpisodeData:
    """Assemble EpisodeData from trajectory buffers and metadata.

    Returns:
        EpisodeData instance.
    """
    robot_pos = _stack_or_zero(robot_pos_traj, stack_fn=np.vstack, empty_shape=(0, 2))
    robot_vel = _stack_or_zero(robot_vel_traj, stack_fn=np.vstack, empty_shape=(0, 2))
    robot_acc = _stack_or_zero(robot_acc_traj, stack_fn=np.vstack, empty_shape=(0, 2))
    peds_pos = _stack_or_zero(peds_pos_traj, stack_fn=np.stack, empty_shape=(0, 0, 2))
    ped_forces = _stack_or_zero(ped_forces_traj, stack_fn=np.stack, empty_shape=(0, 0, 2))

    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        obstacles=obstacles,
        goal=goal,
        dt=dt,
        reached_goal_step=reached_goal_step,
    )


def _create_robot_policy(algo: str, algo_config_path: str | None, seed: int):  # noqa: C901
    """Create a robot policy function based on the specified algorithm.

    Returns:
        Tuple of (policy function, metadata dict).
    """

    def _simple_policy_adapter():
        """Create simple policy that navigates directly toward goal.

        Returns:
            Tuple of (policy function, metadata dict).
        """

        def policy(
            robot_pos: np.ndarray,
            _robot_vel: np.ndarray,
            robot_goal: np.ndarray,
            _ped_positions: np.ndarray,
            _dt: float,
        ) -> np.ndarray:
            """Compute velocity command toward goal.

            Args:
                robot_pos: Current robot position.
                _robot_vel: Current robot velocity (unused).
                robot_goal: Target goal position.
                _ped_positions: Pedestrian positions (unused).
                _dt: Timestep duration (unused).

            Returns:
                Velocity command as 2D array.
            """
            return _simple_robot_policy(robot_pos, robot_goal, speed=1.0)

        return policy, {
            "algorithm": "simple_policy",
            "config": {},
            "config_hash": "na",
            "status": "ok",
        }

    if algo == "simple_policy":
        return _simple_policy_adapter()

    planner, Observation, algo_config = _load_baseline_planner(algo, algo_config_path, seed)

    def _clamp_speed(vel: np.ndarray) -> np.ndarray:
        """Clamp velocity magnitude to maximum allowed speed.

        Args:
            vel: Velocity vector to clamp.

        Returns:
            Clamped velocity vector.
        """
        speed = float(np.linalg.norm(vel))
        if speed > FINAL_SPEED_CLAMP and speed > 1e-9:
            return vel / speed * FINAL_SPEED_CLAMP
        return vel

    def _action_to_velocity(
        action: dict[str, float],
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
    ) -> np.ndarray:
        """Convert planner action dict to velocity vector.

        Args:
            action: Action dict with either (vx, vy) or (v, omega) keys.
            robot_pos: Current robot position.
            robot_vel: Current robot velocity.
            robot_goal: Target goal position.

        Returns:
            Velocity vector as 2D array.
        """
        if "vx" in action and "vy" in action:
            return _clamp_speed(np.array([action["vx"], action["vy"]], dtype=float))
        if "v" in action and "omega" in action:
            v = action["v"]
            current_speed = np.linalg.norm(robot_vel)
            if current_speed > 1e-6:
                vel = robot_vel / current_speed * v
            else:
                goal_dir = robot_goal - robot_pos
                if np.linalg.norm(goal_dir) > 1e-6:
                    vel = goal_dir / np.linalg.norm(goal_dir) * v
                else:
                    vel = np.zeros(2)
            return _clamp_speed(vel)
        raise ValueError(f"Invalid action format from {algo}: {action}")

    def policy_fn(
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
        ped_positions: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Policy function that uses the baseline planner.

        Returns:
            Velocity command as 2D array.
        """
        obs = _build_observation(Observation, robot_pos, robot_vel, robot_goal, ped_positions, dt)

        # Execute planner.step with a small timeout to avoid stalls
        def _do_step():
            """Execute planner step in thread pool.

            Returns:
                Action dict from planner.
            """
            return planner.step(obs)

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_do_step)
                action = fut.result(timeout=POLICY_STEP_TIMEOUT_SECS)
        except FuturesTimeoutError:
            # Safe fallback: zero action in current space
            # Prefer preserving action-space contract when possible
            action = {"vx": 0.0, "vy": 0.0}
        except (RuntimeError, TypeError, ValueError) as exc:
            # Any unexpected planner errors -> fallback but log for diagnostics
            logger.warning("Planner step failed unexpectedly: %s", exc)
            action = {"vx": 0.0, "vy": 0.0}

        # Convert action to velocity (handle both action spaces)
        return _action_to_velocity(action, robot_pos, robot_vel, robot_goal)

    metadata = planner.get_metadata() if hasattr(planner, "get_metadata") else {"algorithm": algo}
    # Ensure consistent metadata schema
    metadata.setdefault("algorithm", algo)
    metadata["config"] = algo_config
    metadata["config_hash"] = _config_hash(algo_config)
    metadata.setdefault("status", "ok")

    return policy_fn, metadata


def _append_video_skip_note(record: dict[str, Any], note: str) -> None:
    """Append a human-readable note to the record's notes field."""
    existing = record.get("notes")
    if existing:
        record["notes"] = f"{existing}; {note}"
    else:
        record["notes"] = note


def _emit_video_skip(
    *,
    record: dict[str, Any],
    episode_id: str,
    scenario_id: str,
    seed: int | None,
    renderer: str,
    reason: str,
    steps: int | None,
    error: str | None = None,
) -> None:
    """Record a video skip reason in logs and episode notes."""
    context = {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "renderer": renderer,
        "reason": reason,
        "steps": steps if steps is not None else -1,
    }
    if error:
        context["error"] = error
    try:
        logger.warning(
            (
                "Video skipped: reason={reason} episode_id={episode_id} "
                "scenario_id={scenario_id} seed={seed} renderer={renderer} steps={steps}"
            ),
            **context,
        )
    except (AttributeError, TypeError):
        # Logging failure -> ignore
        pass

    note_parts = [f"video skipped ({renderer}): {reason}"]
    if steps is not None:
        note_parts.append(f"steps={steps}")
    if error:
        note_parts.append(f"error={error}")
    _append_video_skip_note(record, " ".join(note_parts))


def _try_encode_synthetic_video(
    robot_pos_traj: list[np.ndarray],
    *,
    episode_id: str,
    scenario_id: str,
    out_dir: Path,
    fps: int = 10,
    seed: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Encode a very lightweight synthetic MP4 from robot positions.

    - Draws a simple red dot for the robot position per frame on a black canvas.
    - Uses moviepy ImageSequenceClip if available; returns None when unavailable.

    Returns:
        Tuple of (video metadata dict or None, error info dict or None).
    """
    if ImageSequenceClip is None:
        _skip_info = {
            "reason": "moviepy-missing",
            "renderer": "synthetic",
            "steps": len(robot_pos_traj),
        }
        return None, _skip_info
    # Successfully imported moviepy; proceed with encoding attempt

    N = len(robot_pos_traj)
    if N == 0:
        _skip_info = {
            "reason": "no-frames",
            "renderer": "synthetic",
            "steps": 0,
        }
        return None, _skip_info
    H, W = 128, 128
    # Determine bounds for simple normalization
    xs = np.array([p[0] for p in robot_pos_traj], dtype=float)
    ys = np.array([p[1] for p in robot_pos_traj], dtype=float)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    # Pad 10%
    pad_x = max(1e-6, 0.1 * (max_x - min_x))
    pad_y = max(1e-6, 0.1 * (max_y - min_y))
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    def to_px(x: float, y: float) -> tuple[int, int]:
        # Normalize to [0,1] then scale to pixels; y inverted for image coords
        """Map world coordinates to image pixel coordinates.

        Returns:
            Pixel coordinates (x, y).
        """
        nx = 0.0 if max_x == min_x else (x - min_x) / (max_x - min_x)
        ny = 0.0 if max_y == min_y else (y - min_y) / (max_y - min_y)
        px = int(nx * (W - 1))
        py = int((1.0 - ny) * (H - 1))
        return px, py

    frames: list[np.ndarray] = []
    for i in range(N):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        x, y = robot_pos_traj[i]
        px, py = to_px(float(x), float(y))
        # Draw small 3x3 red square centered at (px,py)
        x0, x1 = max(0, px - 1), min(W, px + 2)
        y0, y1 = max(0, py - 1), min(H, py + 2)
        img[y0:y1, x0:x1, :] = np.array([220, 30, 30], dtype=np.uint8)
        frames.append(img)
    mp4_path = out_dir / f"video_{episode_id}.mp4"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _skip_info = {
            "reason": "unwritable-path",
            "renderer": "synthetic",
            "steps": N,
            "error": str(exc),
        }
        return None, _skip_info
    clip = ImageSequenceClip(frames, fps=fps)  # type: ignore
    try:
        # Keep args minimal for environment compatibility
        clip.write_videofile(str(mp4_path), codec="libx264", fps=fps)
    except OSError as exc:
        _skip_info = {
            "reason": "write-failed",
            "renderer": "synthetic",
            "steps": N,
            "error": str(exc),
        }
        return None, _skip_info
    except (RuntimeError, ValueError) as exc:
        _skip_info = {
            "reason": "encode-failed",
            "renderer": "synthetic",
            "steps": N,
            "error": str(exc),
        }
        return None, _skip_info
    finally:
        try:
            clip.close()  # type: ignore[attr-defined]
        except (AttributeError, OSError):  # pragma: no cover - close best effort
            pass
    try:
        size = mp4_path.stat().st_size
    except (OSError, FileNotFoundError):
        size = 0
    return {
        "status": "success",
        "path": str(mp4_path),
        "format": "mp4",
        "filesize_bytes": int(size),
        "frames": int(N),
        "renderer": "synthetic",
    }, None


def _annotate_and_check_video_perf(
    record: dict[str, Any],
    vid: dict[str, Any],
    perf_start: float,
    enc_start: float,
    enc_end: float,
) -> None:
    """Annotate manifest with encode timing and enforce optional budgets.

    Adds keys: encode_seconds, overhead_ratio.
    Budget env vars:
      - ROBOT_SF_VIDEO_OVERHEAD_SOFT (default 0.10)
      - ROBOT_SF_VIDEO_OVERHEAD_HARD (default 0.50)
      - ROBOT_SF_PERF_ENFORCE (any non-empty to enforce)
    """
    encode_seconds = float(max(0.0, enc_end - enc_start))
    total_elapsed = float(max(1e-9, enc_end - perf_start))
    overhead_ratio = float(encode_seconds / total_elapsed)
    vid["encode_seconds"] = encode_seconds
    vid["overhead_ratio"] = overhead_ratio
    record["video"] = vid

    soft = float(os.getenv("ROBOT_SF_VIDEO_OVERHEAD_SOFT", "0.10"))
    hard = float(os.getenv("ROBOT_SF_VIDEO_OVERHEAD_HARD", "0.50"))
    enforce = bool(os.getenv("ROBOT_SF_PERF_ENFORCE"))
    if overhead_ratio > hard:
        if enforce:
            raise RuntimeError(
                f"video overhead hard breach: ratio={overhead_ratio:.3f} > {hard:.3f}",
            )
        else:
            try:
                logger.warning(
                    (
                        "Video overhead hard breach but continue: "
                        "ratio={ratio:.3f} > {hard:.3f} episode_id={episode_id} "
                        "scenario_id={scenario_id} seed={seed} renderer={renderer}"
                    ),
                    ratio=overhead_ratio,
                    hard=hard,
                    episode_id=record.get("episode_id"),
                    scenario_id=record.get("scenario_id"),
                    seed=record.get("seed"),
                    renderer=vid.get("renderer"),
                )
            except Exception:  # pragma: no cover - logging optional
                pass
    elif overhead_ratio > soft:
        if enforce:
            raise RuntimeError(
                f"video overhead soft breach: ratio={overhead_ratio:.3f} > {soft:.3f}",
            )
        else:
            try:
                logger.warning(
                    (
                        "Video overhead soft breach: "
                        "ratio={ratio:.3f} > {soft:.3f} episode_id={episode_id} "
                        "scenario_id={scenario_id} seed={seed} renderer={renderer}"
                    ),
                    ratio=overhead_ratio,
                    soft=soft,
                    episode_id=record.get("episode_id"),
                    scenario_id=record.get("scenario_id"),
                    seed=record.get("seed"),
                    renderer=vid.get("renderer"),
                )
            except Exception:  # pragma: no cover - logging optional
                pass


def _maybe_encode_video(
    *,
    record: dict[str, Any],
    robot_pos_traj: list[np.ndarray],
    videos_dir: str | None,
    video_enabled: bool,
    video_renderer: str,
    perf_start: float,
) -> None:
    """Best-effort video encoding wrapper with perf annotation and budget checks.

    Swallows all exceptions to keep batch robust.
    """
    if not (video_enabled and str(video_renderer) == "synthetic" and videos_dir is not None):
        return
    episode_id = record["episode_id"]
    scenario_id = record["scenario_id"]
    try:
        enc_t0 = time.perf_counter()
        vid, skip_info = _try_encode_synthetic_video(
            robot_pos_traj,
            episode_id=episode_id,
            scenario_id=scenario_id,
            out_dir=Path(videos_dir),
            fps=10,
            seed=record.get("seed"),
        )
        enc_t1 = time.perf_counter()
        if vid is not None and int(vid.get("filesize_bytes", 0)) > 0:
            _annotate_and_check_video_perf(record, vid, perf_start, enc_t0, enc_t1)
        else:
            reason_payload = skip_info or {
                "reason": "encoder-empty",
                "renderer": str(video_renderer),
                "steps": len(robot_pos_traj),
            }
            _emit_video_skip(
                record=record,
                episode_id=episode_id,
                scenario_id=scenario_id,
                seed=record.get("seed"),
                renderer=str(reason_payload.get("renderer", video_renderer)),
                reason=str(reason_payload.get("reason", "unknown")),
                steps=reason_payload.get("steps"),
                error=reason_payload.get("error"),
            )
    except RuntimeError:
        # Budget enforcement: bubble up to runner to record a failure
        raise
    except (TypeError, ValueError, OSError):  # pragma: no cover - defensive path
        pass


def _simulate_episode_with_policy(
    scenario_params: dict[str, Any],
    seed: int,
    robot_policy: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
    horizon: int,
    dt: float,
    robot_start: Sequence[float] | None,
    robot_goal: Sequence[float] | None,
    record_forces: bool,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[tuple[float, float, float, float]],
    np.ndarray,
    int | None,
]:
    """Run a synthetic episode and return trajectories and metadata.

    Returns:
        Tuple of trajectories, obstacles, goal, and reached-goal step.
    """
    gen = generate_scenario(scenario_params, seed=seed)
    sim = gen.simulator
    if sim is None:
        raise RuntimeError("pysocialforce not available; cannot run episode")
    wrapper = FastPysfWrapper(sim)

    # Determine robot start/goal: defaults if absent
    robot_start_arr, robot_goal_arr = _prepare_robot_points(robot_start, robot_goal)
    robot_pos = robot_start_arr.copy()
    robot_vel = np.zeros(2)

    robot_pos_traj: list[np.ndarray] = []
    robot_vel_traj: list[np.ndarray] = []
    robot_acc_traj: list[np.ndarray] = []
    peds_pos_traj: list[np.ndarray] = []
    ped_forces_traj: list[np.ndarray] = []

    reached_goal_step: int | None = None
    goal_radius = 0.3
    last_vel = robot_vel.copy()

    # Record initial state at t=0 to capture immediate collisions
    initial_ped_pos = sim.peds.pos().copy()
    peds_pos_traj.append(initial_ped_pos)
    if record_forces:
        forces = np.zeros_like(initial_ped_pos, dtype=float)
        for i, p in enumerate(initial_ped_pos):
            forces[i] = wrapper.get_forces_at(p)
        ped_forces_traj.append(forces)
    else:
        ped_forces_traj.append(np.zeros_like(initial_ped_pos, dtype=float))
    robot_pos_traj.append(robot_pos.copy())
    robot_vel_traj.append(robot_vel.copy())
    robot_acc_traj.append(np.zeros_like(robot_vel, dtype=float))

    def _step_robot(
        curr_pos: np.ndarray,
        curr_vel: np.ndarray,
        ped_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance robot state using the policy output.

        Returns:
            Tuple of (new_pos, new_vel, acceleration).
        """
        new_vel = robot_policy(curr_pos, curr_vel, robot_goal_arr, ped_positions, dt)
        new_pos = curr_pos + new_vel * dt
        acc_vec = (new_vel - curr_vel) / dt
        return new_pos, new_vel, acc_vec

    for t in range(horizon):
        ped_pos = peds_pos_traj[-1]

        robot_pos, robot_vel, acc = _step_robot(robot_pos, last_vel, ped_pos)
        last_vel = robot_vel.copy()

        sim.step()

        ped_pos_next = sim.peds.pos().copy()
        peds_pos_traj.append(ped_pos_next)

        if record_forces:
            forces = np.zeros_like(ped_pos_next, dtype=float)
            for i, p in enumerate(ped_pos_next):
                forces[i] = wrapper.get_forces_at(p)
            ped_forces_traj.append(forces)
        else:
            ped_forces_traj.append(np.zeros_like(ped_pos_next, dtype=float))

        robot_pos_traj.append(robot_pos.copy())
        robot_vel_traj.append(robot_vel.copy())
        robot_acc_traj.append(acc.copy())

        if reached_goal_step is None and np.linalg.norm(robot_goal_arr - robot_pos) < goal_radius:
            reached_goal_step = t + 1
            break

    return (
        robot_pos_traj,
        robot_vel_traj,
        robot_acc_traj,
        peds_pos_traj,
        ped_forces_traj,
        gen.obstacles,
        np.asarray(robot_goal_arr, dtype=float),
        reached_goal_step,
    )


def _compute_metrics(
    ep: EpisodeData,
    horizon: int,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    """Compute metrics and optional SNQI post-processing.

    Returns:
        Metrics dictionary for the episode.
    """
    metrics_raw = compute_all_metrics(ep, horizon=horizon)
    return _post_process_metrics(
        metrics_raw,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
    )


def _build_episode_record(
    scenario_params: dict[str, Any],
    seed: int,
    metrics: dict[str, Any],
    algo_metadata: dict[str, Any],
    ts_start: str,
) -> dict[str, Any]:
    """Assemble a JSON-serializable episode record.

    Returns:
        Episode record dictionary.
    """
    episode_id = compute_episode_id(scenario_params, seed)
    record = {
        "version": "v1",
        "episode_id": episode_id,
        "scenario_id": scenario_params.get("id", "unknown"),
        "seed": seed,
        "scenario_params": scenario_params,
        "metrics": metrics,
        "algorithm_metadata": algo_metadata,
        "config_hash": _config_hash(scenario_params),
        "git_hash": _git_hash_fallback(),
        "timestamps": {"start": ts_start, "end": ts_start},
    }
    algo_value = scenario_params.get("algo")
    if algo_value is not None:
        record["algo"] = algo_value
    return record


def run_episode(
    scenario_params: dict[str, Any],
    seed: int,
    *,
    horizon: int = 100,
    dt: float = 0.1,
    robot_start: Sequence[float] | None = None,
    robot_goal: Sequence[float] | None = None,
    record_forces: bool = True,
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
    algo: str = "simple_policy",
    algo_config_path: str | None = None,
    # Video options (optional)
    video_enabled: bool = False,
    video_renderer: str = "none",
    videos_dir: str | None = None,
) -> dict[str, Any]:
    """Run a single episode and return a metrics record dict.

    The robot can use different algorithms based on the 'algo' parameter.

    Returns:
        Episode record dictionary with metrics, trajectories, and metadata.
    """
    # Wall-clock start time for timestamps and perf accounting
    perf_start = time.perf_counter()
    ts_start = datetime.now(UTC).isoformat()
    # Create robot policy based on algorithm
    robot_policy, algo_metadata = _create_robot_policy(algo, algo_config_path, seed)

    # Simulate episode
    trajectories = _simulate_episode_with_policy(
        scenario_params,
        seed,
        robot_policy,
        horizon,
        dt,
        robot_start,
        robot_goal,
        record_forces,
    )
    (
        robot_pos_traj,
        robot_vel_traj,
        robot_acc_traj,
        peds_pos_traj,
        ped_forces_traj,
        obstacle_segments,
        robot_goal_arr,
        reached_goal_step,
    ) = trajectories

    obstacles = sample_obstacle_points(obstacle_segments)

    # Build episode data
    ep = _build_episode_data(
        robot_pos_traj,
        robot_vel_traj,
        robot_acc_traj,
        peds_pos_traj,
        ped_forces_traj,
        obstacles,
        robot_goal_arr,
        dt,
        reached_goal_step,
    )

    # Compute metrics
    metrics = _compute_metrics(ep, horizon, snqi_weights, snqi_baseline)

    # Build record
    scenario_params_record = dict(scenario_params)
    scenario_params_record.setdefault("algo", algo)
    record = _build_episode_record(scenario_params_record, seed, metrics, algo_metadata, ts_start)

    steps_taken = max(0, len(robot_pos_traj) - 1)

    # Handle video
    video_positions = robot_pos_traj[1:] if len(robot_pos_traj) > 1 else []
    _maybe_encode_video(
        record=record,
        robot_pos_traj=video_positions,
        videos_dir=videos_dir,
        video_enabled=video_enabled,
        video_renderer=video_renderer,
        perf_start=perf_start,
    )

    # Update end time
    record["timestamps"]["end"] = datetime.now(UTC).isoformat()
    record["steps"] = steps_taken
    record["horizon"] = horizon
    wall_time = float(max(1e-9, time.perf_counter() - perf_start))
    record["wall_time_sec"] = wall_time
    record["timing"] = {
        "steps_per_second": float(steps_taken) / wall_time if wall_time > 0 else 0.0,
    }
    status = "success" if record.get("metrics", {}).get("success") else "failure"
    if record.get("metrics", {}).get("collisions"):
        status = "collision"
    record["status"] = status
    return record


def validate_and_write(
    record: dict[str, Any],
    schema_path: str | Path,
    out_path: str | Path,
) -> None:
    """Validate an episode record against schema and append to JSONL."""
    schema = load_schema(schema_path)
    validate_episode(record, schema)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


__all__ = [
    "load_scenario_matrix",
    "run_batch",
    "run_episode",
    "validate_and_write",
]


def _post_process_metrics(
    metrics_raw: dict[str, Any],
    *,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    """Normalize metric output and compute SNQI if configured.

    Returns:
        Normalized metrics dictionary.
    """
    metrics: dict[str, Any] = dict(metrics_raw.items())
    metrics["success"] = bool(metrics.get("success", 0.0) == 1.0)
    fq = {k: v for k, v in metrics.items() if k.startswith("force_q")}
    if fq:
        metrics["force_quantiles"] = {
            "q50": float(fq.get("force_q50", float("nan"))),
            "q90": float(fq.get("force_q90", float("nan"))),
            "q95": float(fq.get("force_q95", float("nan"))),
        }
        for k in list(fq.keys()):
            metrics.pop(k, None)
    if snqi_weights is not None:
        snqi_val = snqi(metrics, snqi_weights, baseline_stats=snqi_baseline)
        metrics["snqi"] = float(snqi_val) if math.isfinite(snqi_val) else 0.0
    for count_key in ("collisions", "near_misses", "force_exceed_events"):
        if count_key in metrics and metrics[count_key] is not None:
            try:
                metrics[count_key] = int(metrics[count_key])
            except Exception:  # pragma: no cover
                pass
    return _sanitize_metrics(metrics)


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove NaN/inf metric entries to keep JSON serialization clean.

    Returns:
        Cleaned metrics dictionary with NaN/inf values removed.
    """

    clean: dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, dict):
            nested = _sanitize_metrics(val)
            if nested:
                clean[key] = nested
            continue
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            continue
        clean[key] = val
    return clean


def _expand_jobs(
    scenarios: list[dict[str, Any]],
    base_seed: int = 0,
    repeats_override: int | None = None,
) -> list[tuple[dict[str, Any], int]]:
    """Expand scenarios into (scenario, seed) jobs.

    Returns:
        List of (scenario, seed) tuples.
    """
    jobs: list[tuple[dict[str, Any], int]] = []
    for sc in scenarios:
        reps = int(sc.get("repeats", 1)) if repeats_override is None else int(repeats_override)
        for r in range(reps):
            jobs.append((sc, base_seed + r))
    return jobs


def _run_job_worker(job: tuple[dict[str, Any], int, dict[str, Any]]) -> dict[str, Any]:
    """Top-level worker function to run a single episode.

    Accepts a tuple of (scenario_dict, seed, fixed_params_dict) and returns a record dict.
    This must remain at module top-level for multiprocessing 'spawn' pickling.

    Returns:
        Episode record dictionary.
    """
    sc, seed, params = job
    return run_episode(
        sc,
        seed,
        horizon=int(params["horizon"]),
        dt=float(params["dt"]),
        record_forces=bool(params["record_forces"]),
        snqi_weights=params.get("snqi_weights"),
        snqi_baseline=params.get("snqi_baseline"),
        algo=str(params["algo"]),
        algo_config_path=params.get("algo_config_path"),
        video_enabled=bool(params.get("video_enabled", False)),
        video_renderer=str(params.get("video_renderer", "none")),
        videos_dir=params.get("videos_dir"),
    )


def _write_validated_record(out_path: Path, schema: dict[str, Any], rec: dict[str, Any]) -> None:
    """Validate a record against schema and append to JSONL."""
    validate_episode(rec, schema)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _run_batch_sequential(
    jobs: list[tuple[dict[str, Any], int]],
    *,
    out_path: Path,
    schema: dict[str, Any],
    fixed_params: dict[str, Any],
    progress_cb: Callable[[int, int, dict[str, Any], int, bool, str | None], None] | None,
    fail_fast: bool,
) -> tuple[int, list[dict[str, Any]]]:
    """Run batch jobs sequentially and return write count + failures.

    Returns:
        Tuple of (written_count, failure_records).
    """
    wrote = 0
    failures: list[dict[str, Any]] = []
    total = len(jobs)
    for idx, (sc, seed) in enumerate(jobs, start=1):
        try:
            rec = _run_job_worker((sc, seed, fixed_params))
            _write_validated_record(out_path, schema, rec)
            wrote += 1
            if progress_cb is not None:
                try:
                    progress_cb(idx, total, sc, seed, True, None)
                except Exception:  # pragma: no cover - progress best-effort
                    pass
        except Exception as e:  # pragma: no cover - error path
            failures.append(
                {
                    "scenario_id": sc.get("id", "unknown"),
                    "seed": seed,
                    "error": repr(e),
                },
            )
            if progress_cb is not None:
                try:
                    progress_cb(idx, total, sc, seed, False, repr(e))
                except Exception:  # pragma: no cover
                    pass
            if fail_fast:
                raise
    return wrote, failures


def _run_batch_parallel(
    jobs: list[tuple[dict[str, Any], int]],
    *,
    out_path: Path,
    schema: dict[str, Any],
    fixed_params: dict[str, Any],
    workers: int,
    progress_cb: Callable[[int, int, dict[str, Any], int, bool, str | None], None] | None,
    fail_fast: bool,
) -> tuple[int, list[dict[str, Any]]]:
    """Run batch jobs in parallel and return write count + failures.

    Returns:
        Tuple of (written_count, failure_records).
    """
    wrote = 0
    failures: list[dict[str, Any]] = []
    total = len(jobs)
    # Submit all jobs
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        future_to_job: dict[Any, tuple[int, dict[str, Any], int]] = {}
        for idx, (sc, seed) in enumerate(jobs, start=1):
            fut = ex.submit(_run_job_worker, (sc, seed, fixed_params))
            future_to_job[fut] = (idx, sc, seed)
        for fut in as_completed(future_to_job):
            idx, sc, seed = future_to_job[fut]
            try:
                rec = fut.result()
                _write_validated_record(out_path, schema, rec)
                wrote += 1
                if progress_cb is not None:
                    try:
                        progress_cb(idx, total, sc, seed, True, None)
                    except Exception:  # pragma: no cover
                        pass
            except Exception as e:  # pragma: no cover
                failures.append(
                    {
                        "scenario_id": sc.get("id", "unknown"),
                        "seed": seed,
                        "error": repr(e),
                    },
                )
                if progress_cb is not None:
                    try:
                        progress_cb(idx, total, sc, seed, False, repr(e))
                    except Exception:  # pragma: no cover
                        pass
                if fail_fast:
                    # Cancel remaining futures and re-raise
                    for f in future_to_job:
                        f.cancel()
                    raise
    return wrote, failures


def _prepare_batch_setup(
    scenarios_or_path: list[dict[str, Any]] | str | Path,
    out_path: str | Path,
    schema_path: str | Path,
    append: bool,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    """Prepare scenarios, output path, and schema for batch processing.

    Returns:
        Tuple of (scenarios list, output path, schema dict).
    """
    # Load scenarios
    scenarios_is_path = isinstance(scenarios_or_path, str | Path)
    if scenarios_is_path:
        scenarios = load_scenario_matrix(cast("str | Path", scenarios_or_path))
    else:
        # scenarios_or_path is already list[dict[str, Any]]
        scenarios = cast("list[dict[str, Any]]", scenarios_or_path)

    # Prepare output
    out_path = Path(out_path)
    if not append and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = load_schema(schema_path)
    return scenarios, out_path, schema


def _setup_fixed_params(
    out_path: Path,
    horizon: int,
    dt: float,
    record_forces: bool,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
    video_enabled: bool,
    video_renderer: str,
    algo: str,
    algo_config_path: str | None,
) -> dict[str, Any]:
    """Set up the fixed parameters dict for job execution.

    Returns:
        Dictionary with fixed parameters for all episodes.
    """
    videos_dir = (out_path.parent / "videos").as_posix()
    return {
        "horizon": horizon,
        "dt": dt,
        "record_forces": record_forces,
        "snqi_weights": snqi_weights,
        "snqi_baseline": snqi_baseline,
        "algo": algo,
        "algo_config_path": algo_config_path,
        "video_enabled": bool(video_enabled) and str(video_renderer) != "none",
        "video_renderer": str(video_renderer),
        "videos_dir": videos_dir,
    }


def _filter_resume_jobs(
    jobs: list[tuple[dict[str, Any], int]],
    out_path: Path,
    resume: bool,
) -> list[tuple[dict[str, Any], int]]:
    """Filter jobs based on resume logic, skipping existing episodes.

    Returns:
        Filtered list of (scenario dict, seed) tuples excluding completed episodes.
    """
    if not resume or not out_path.exists():
        return jobs

    # Try fast-path via manifest; fall back to scanning JSONL if stale/missing
    existing_ids = load_manifest(
        out_path,
        expected_identity_hash=_episode_identity_hash(),
        expected_schema_version=EPISODE_SCHEMA_VERSION,
    ) or index_existing(out_path)

    if not existing_ids:
        return jobs

    filtered: list[tuple[dict[str, Any], int]] = []
    for sc, seed in jobs:
        eid = compute_episode_id(sc, seed)
        if eid not in existing_ids:
            filtered.append((sc, seed))
    return filtered


def _run_jobs(
    jobs: list[tuple[dict[str, Any], int]],
    out_path: Path,
    schema: dict[str, Any],
    fixed_params: dict[str, Any],
    workers: int,
    progress_cb: Callable[[int, int, dict[str, Any], int, bool, str | None], None] | None,
    fail_fast: bool,
) -> tuple[int, list[dict[str, Any]]]:
    """Execute jobs using sequential or parallel processing.

    Returns:
        Tuple of (number of records written, list of failed jobs).
    """
    if workers <= 1:
        return _run_batch_sequential(
            jobs,
            out_path=out_path,
            schema=schema,
            fixed_params=fixed_params,
            progress_cb=progress_cb,
            fail_fast=fail_fast,
        )
    else:
        return _run_batch_parallel(
            jobs,
            out_path=out_path,
            schema=schema,
            fixed_params=fixed_params,
            workers=workers,
            progress_cb=progress_cb,
            fail_fast=fail_fast,
        )


def _finalize_batch(
    out_path: Path,
    wrote: int,
    resume: bool,
) -> dict[str, Any]:
    """Finalize batch processing: save manifest and optional performance snapshot.

    Returns:
        Summary dictionary with episode counts and outcome statistics.
    """
    # Save/update manifest to speed up future resume if we wrote anything
    if resume and wrote > 0 and out_path.exists():
        # Re-index by scanning (cheap) to ensure we capture exactly what's on disk
        save_manifest(
            out_path,
            index_existing(out_path),
            identity_hash=_episode_identity_hash(),
            schema_version=EPISODE_SCHEMA_VERSION,
        )

    # Optional: write a small performance snapshot for video encoding if requested
    try:
        if os.getenv("ROBOT_SF_VIDEO_PERF_SNAPSHOT"):
            vids: list[dict] = []
            try:
                with out_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        v = rec.get("video") if isinstance(rec, dict) else None
                        if isinstance(v, dict):
                            vids.append(v)
            except FileNotFoundError:
                vids = []
            total_frames = sum(int(v.get("frames", 0)) for v in vids)
            total_encode = sum(float(v.get("encode_seconds", 0.0)) for v in vids)
            overheads = [float(v.get("overhead_ratio", 0.0)) for v in vids if "overhead_ratio" in v]
            snap = {
                "episodes": len(vids),
                "total_frames": int(total_frames),
                "total_encode_seconds": float(total_encode),
                "encode_ms_per_frame": (1000.0 * total_encode / total_frames)
                if total_frames > 0
                else None,
                "mean_overhead_ratio": (
                    float(sum(overheads) / len(overheads)) if overheads else None
                ),
                "environment": {
                    "os": platform.platform(),
                    "python": platform.python_version(),
                    "processor": platform.processor(),
                },
            }
            perf_path = out_path.parent / "videos" / "perf_snapshot.json"
            perf_path.parent.mkdir(parents=True, exist_ok=True)
            with perf_path.open("w", encoding="utf-8") as f:
                json.dump(snap, f, indent=2)
    except Exception:
        # Best-effort; ignore snapshot errors
        pass

    return {
        "total_jobs": 0,  # Will be set by caller
        "written": wrote,
        "failures": [],  # Will be set by caller
        "out_path": str(out_path),
    }


def run_batch(
    scenarios_or_path: list[dict[str, Any]] | str | Path,
    out_path: str | Path,
    schema_path: str | Path,
    *,
    base_seed: int = 0,
    repeats_override: int | None = None,
    horizon: int = 100,
    dt: float = 0.1,
    record_forces: bool = True,
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
    video_enabled: bool = False,
    video_renderer: str = "none",
    append: bool = True,
    fail_fast: bool = False,
    progress_cb: Callable[[int, int, dict[str, Any], int, bool, str | None], None] | None = None,
    algo: str = "simple_policy",
    algo_config_path: str | None = None,
    workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Run a batch of episodes and write JSONL records.

    scenarios_or_path: either a list of scenario dicts or a YAML file path.
    Returns a summary dict with counts and failures.

    Returns:
        Summary dictionary with episode counts, failures, and execution metadata.
    """
    # Prepare batch setup
    scenarios, out_path, schema = _prepare_batch_setup(
        scenarios_or_path,
        out_path,
        schema_path,
        append,
    )

    # Map-based scenario detection: delegate to map runner
    if scenarios and any("map_file" in sc or "simulation_config" in sc for sc in scenarios):
        return run_map_batch(
            scenarios_or_path,
            out_path,
            schema_path,
            horizon=horizon if horizon > 0 else None,
            dt=dt if dt > 0 else None,
            record_forces=record_forces,
            snqi_weights=snqi_weights,
            snqi_baseline=snqi_baseline,
            algo=algo,
            algo_config_path=algo_config_path,
            workers=workers,
            resume=resume,
        )

    # Expand jobs
    jobs = _expand_jobs(scenarios, base_seed=base_seed, repeats_override=repeats_override)

    # Set up fixed parameters
    fixed_params = _setup_fixed_params(
        out_path,
        horizon,
        dt,
        record_forces,
        snqi_weights,
        snqi_baseline,
        video_enabled,
        video_renderer,
        algo,
        algo_config_path,
    )

    # Filter jobs for resume
    jobs = _filter_resume_jobs(jobs, out_path, resume)

    # Run jobs
    wrote, failures = _run_jobs(
        jobs,
        out_path,
        schema,
        fixed_params,
        workers,
        progress_cb,
        fail_fast,
    )

    # Finalize and return summary
    summary = _finalize_batch(out_path, wrote, resume)
    summary["total_jobs"] = len(jobs)
    summary["failures"] = failures
    return summary
