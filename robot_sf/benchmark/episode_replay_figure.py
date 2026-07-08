"""Episode replay figure generation bridge.

This module provides a CPU-only bridge from persisted campaign episode rows to
replay-derived figure artifacts: stills, filmstrip, and trajectory plots, all
with deterministic replay checks and provenance sidecars.

Claim boundary: figure artifact generation from retained episode rows only.
Does not run campaigns, reinterpret metrics, or promote replay outputs as new
benchmark evidence. Re-simulation is used to render a specific already-recorded
episode row.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from robot_sf.benchmark.full_classic.replay import (
    ReplayEpisode,
    ReplayStep,
    validate_replay_episode,
)

if TYPE_CHECKING:
    from pathlib import Path

try:
    from robot_sf.benchmark.visualization import _ensure_matplotlib_backend
except ImportError:

    def _ensure_matplotlib_backend() -> None:
        """Initialize matplotlib to a headless-safe backend once."""
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")


try:
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

REQUIRED_EPISODE_FIELDS = [
    "episode_id",
    "scenario_id",
    "seed",
]

OPTIONAL_PROVENANCE_FIELDS = [
    "planner",
    "planner_key",
    "algo",
    "algo_config",
    "campaign_id",
    "campaign_root",
    "scenario_matrix_path",
    "config_hash",
    "scenario_matrix_hash",
    "repo_commit",
    "replay_steps",
    "replay_dt",
    "replay_map_path",
    "final_robot_position",
    "final_progress",
    "success",
    "collision",
]


def _finite_floats(*values: Any) -> tuple[float, ...] | None:
    """Coerce values to finite floats, returning ``None`` if any is invalid.

    Used to skip replay steps carrying NaN/Inf or non-numeric coordinates so
    they never reach plot-limit or max-value computations.

    Returns:
        Tuple of finite floats in order, or ``None`` if any value is
        non-numeric or non-finite.
    """
    out: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (ValueError, TypeError):
            return None
        if not math.isfinite(f):
            return None
        out.append(f)
    return tuple(out)


def _parse_final_position(raw: Any) -> tuple[float, float] | None:
    """Coerce a recorded final robot position to a 2-tuple of finite floats.

    Returns:
        ``(x, y)`` of finite floats, or ``None`` when the value is absent or
        malformed so downstream determinism checks skip the comparison
        instead of crashing on a bad row.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        x, y = float(raw[0]), float(raw[1])
    except (ValueError, TypeError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


@dataclass
class EpisodeRow:
    """Validated episode row with provenance."""

    episode_id: str
    scenario_id: str
    seed: int
    planner: str | None = None
    planner_key: str | None = None
    algo: str | None = None
    algo_config: dict[str, Any] | None = None
    campaign_id: str | None = None
    campaign_root: str | None = None
    scenario_matrix_path: str | None = None
    config_hash: str | None = None
    scenario_matrix_hash: str | None = None
    repo_commit: str | None = None
    final_robot_position: tuple[float, float] | None = None
    final_progress: float | None = None
    success: bool | None = None
    collision: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeRow:
        """Create EpisodeRow from dictionary, validating required fields.

        Args:
            data: Episode row dictionary.

        Returns:
            Validated EpisodeRow instance.

        Raises:
            ValueError: If required fields are missing.
        """
        missing = [f for f in REQUIRED_EPISODE_FIELDS if f not in data]
        if missing:
            raise ValueError(f"Episode row missing required fields: {missing}")

        return cls(
            episode_id=str(data["episode_id"]),
            scenario_id=str(data["scenario_id"]),
            seed=int(data["seed"]),
            planner=data.get("planner"),
            planner_key=data.get("planner_key"),
            algo=data.get("algo"),
            algo_config=data.get("algo_config"),
            campaign_id=data.get("campaign_id"),
            campaign_root=data.get("campaign_root"),
            scenario_matrix_path=data.get("scenario_matrix_path"),
            config_hash=data.get("config_hash"),
            scenario_matrix_hash=data.get("scenario_matrix_hash"),
            repo_commit=data.get("repo_commit"),
            final_robot_position=_parse_final_position(data.get("final_robot_position")),
            final_progress=data.get("final_progress"),
            success=data.get("success"),
            collision=data.get("collision"),
            raw=data,
        )


@dataclass
class ReplayResult:
    """Result of episode replay with determinism check."""

    episode: ReplayEpisode
    determinism_check_status: Literal["pass", "fail", "not_evaluable"]
    determinism_details: dict[str, Any] = field(default_factory=dict)
    replay_steps: list[ReplayStep] = field(default_factory=list)


@dataclass
class FigureArtifact:
    """Generated figure artifact with provenance."""

    artifact_type: Literal["still", "filmstrip", "trajectory"]
    path: str
    format: str
    sha256: str
    stamp_text: str


@dataclass
class ProvenanceSidecar:
    """Machine-readable provenance for replay artifacts."""

    campaign_id: str | None
    episode_id: str
    scenario_id: str
    seed: int
    planner_key: str | None
    scenario_matrix_path: str | None
    campaign_config_hash: str | None
    repo_commit: str | None
    replay_command: str
    determinism_check_status: str
    determinism_tolerance: float | None
    source_episodes_jsonl_path: str | None
    source_episodes_jsonl_sha256: str | None
    artifacts: list[dict[str, Any]]
    generated_at: str


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: File path to hash.

    Returns:
        Hex digest string.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of bytes.

    Args:
        data: Bytes to hash.

    Returns:
        Hex digest string.
    """
    return hashlib.sha256(data).hexdigest()


def load_episode_row(episodes_path: Path, episode_id: str) -> EpisodeRow:
    """Load a single episode row from JSONL file by episode_id.

    Args:
        episodes_path: Path to episodes.jsonl file.
        episode_id: Episode ID to search for.

    Returns:
        Validated EpisodeRow.

    Raises:
        FileNotFoundError: If episodes file doesn't exist.
        ValueError: If episode_id not found or row invalid.
    """
    if not episodes_path.exists():
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")

    with open(episodes_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

            if data.get("episode_id") == episode_id:
                return EpisodeRow.from_dict(data)

    raise ValueError(f"Episode ID '{episode_id}' not found in {episodes_path}")


def build_replay_from_episode_row(episode_row: EpisodeRow) -> ReplayEpisode | None:
    """Build ReplayEpisode from episode row if replay data available.

    Args:
        episode_row: Validated episode row.

    Returns:
        ReplayEpisode if replay_steps available, None otherwise.
    """
    replay_steps = episode_row.raw.get("replay_steps")
    if not replay_steps or not isinstance(replay_steps, list):
        return None

    steps = []
    for step_data in replay_steps:
        if isinstance(step_data, dict):
            coords = _finite_floats(
                step_data.get("t", 0.0),
                step_data.get("x", 0.0),
                step_data.get("y", 0.0),
                step_data.get("heading", 0.0),
            )
            if coords is None:
                continue
            t, x, y, heading = coords
            step = ReplayStep(
                t=t,
                x=x,
                y=y,
                heading=heading,
                speed=step_data.get("speed"),
                ped_positions=step_data.get("ped_positions"),
                action=step_data.get("action"),
            )
        elif isinstance(step_data, list | tuple) and len(step_data) >= 4:
            coords = _finite_floats(step_data[0], step_data[1], step_data[2], step_data[3])
            if coords is None:
                continue
            t, x, y, heading = coords
            step = ReplayStep(t=t, x=x, y=y, heading=heading)
        else:
            continue
        steps.append(step)

    if not steps:
        return None

    return ReplayEpisode(
        episode_id=episode_row.episode_id,
        scenario_id=episode_row.scenario_id,
        steps=steps,
        dt=episode_row.raw.get("replay_dt"),
        map_path=episode_row.raw.get("replay_map_path"),
    )


def check_determinism(
    replay_episode: ReplayEpisode,
    episode_row: EpisodeRow,
    tolerance_m: float = 0.1,
) -> tuple[Literal["pass", "fail", "not_evaluable"], dict[str, Any]]:
    """Check if replay matches original episode endpoints.

    Args:
        replay_episode: Replay episode to check.
        episode_row: Original episode row.
        tolerance_m: Position tolerance in meters.

    Returns:
        Tuple of (status, details dict).
    """
    details: dict[str, Any] = {
        "tolerance_m": tolerance_m,
        "checks_performed": [],
        "checks_passed": [],
        "checks_failed": [],
    }

    if not replay_episode.steps:
        return "not_evaluable", {**details, "reason": "no replay steps"}

    final_step = replay_episode.steps[-1]
    final_pos = (final_step.x, final_step.y)
    details["replay_final_position"] = final_pos

    if episode_row.final_robot_position:
        details["checks_performed"].append("final_robot_position")
        original_pos = tuple(episode_row.final_robot_position)
        details["original_final_position"] = original_pos

        dist = np.sqrt(
            (final_pos[0] - original_pos[0]) ** 2 + (final_pos[1] - original_pos[1]) ** 2
        )
        details["position_error_m"] = dist

        if dist <= tolerance_m:
            details["checks_passed"].append("final_robot_position")
        else:
            details["checks_failed"].append(
                f"position error {dist:.3f}m > tolerance {tolerance_m}m"
            )

    if episode_row.final_progress is not None:
        # `final_progress` is recorded for provenance context only. There is no
        # replay-side progress endpoint to compare it against, so it does NOT
        # constitute a determinism check and must not be counted toward a
        # "pass" status — doing so previously let an episode carrying only
        # `final_progress` report determinism "pass" while verifying nothing.
        details.setdefault("checks_informational", []).append("final_progress recorded")
        details["original_final_progress"] = episode_row.final_progress

    if details["checks_failed"]:
        return "fail", details

    if details["checks_passed"]:
        return "pass", details

    return "not_evaluable", {**details, "reason": "no evaluable endpoints in episode row"}


def generate_still(
    replay_episode: ReplayEpisode,
    step_idx: int,
    out_path: Path,
    fmt: str = "png",
    map_path: str | None = None,
) -> FigureArtifact:
    """Generate still frame at specific step.

    Args:
        replay_episode: Episode to render.
        step_idx: Step index to render.
        out_path: Output file path.
        fmt: Output format (png, pdf, svg).
        map_path: Optional map SVG path for background.

    Returns:
        FigureArtifact with metadata.
    """
    _ensure_matplotlib_backend()

    if step_idx < 0 or step_idx >= len(replay_episode.steps):
        raise ValueError(f"step_idx {step_idx} out of range [0, {len(replay_episode.steps)})")

    step = replay_episode.steps[step_idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    if map_path:
        try:
            import matplotlib.image as mpimg  # noqa: PLC0415

            img = mpimg.imread(map_path)
            ax.imshow(img, origin="upper", alpha=0.3)
        except (OSError, ValueError):
            pass

    ax.plot(step.x, step.y, "bo", markersize=15, label="Robot")

    if step.ped_positions:
        ped_x = [p[0] for p in step.ped_positions]
        ped_y = [p[1] for p in step.ped_positions]
        ax.plot(ped_x, ped_y, "ro", markersize=8, label="Pedestrians")

    stamp_text = (
        f"Ep: {replay_episode.episode_id}\nStep: {step_idx}\nScenario: {replay_episode.scenario_id}"
    )
    ax.text(
        0.02,
        0.98,
        stamp_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Still Frame - Step {step_idx}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sha256 = compute_file_sha256(out_path)
    return FigureArtifact(
        artifact_type="still",
        path=str(out_path),
        format=fmt,
        sha256=sha256,
        stamp_text=stamp_text,
    )


def generate_filmstrip(
    replay_episode: ReplayEpisode,
    out_path: Path,
    fmt: str = "png",
    frame_steps: list[int] | None = None,
) -> FigureArtifact:
    """Generate filmstrip showing multiple frames in sequence.

    Args:
        replay_episode: Episode to render.
        out_path: Output file path.
        fmt: Output format (png, pdf).
        frame_steps: List of step indices to show, or None for uniform sampling.

    Returns:
        FigureArtifact with metadata.
    """
    _ensure_matplotlib_backend()

    if not replay_episode.steps:
        raise ValueError("No steps in replay episode")

    if frame_steps is None:
        n_frames = min(8, len(replay_episode.steps))
        indices = np.linspace(0, len(replay_episode.steps) - 1, n_frames, dtype=int)
        frame_steps = indices.tolist()

    if not frame_steps:
        raise ValueError("frame_steps must contain at least one step index")
    out_of_range = [s for s in frame_steps if s < 0 or s >= len(replay_episode.steps)]
    if out_of_range:
        raise ValueError(
            f"frame_steps out of range [0, {len(replay_episode.steps) - 1}]: {out_of_range}"
        )

    n_frames = len(frame_steps)
    fig, axes = plt.subplots(1, n_frames, figsize=(3 * n_frames, 3))
    if n_frames == 1:
        axes = [axes]

    for idx, (ax, step_idx) in enumerate(zip(axes, frame_steps, strict=False)):
        step = replay_episode.steps[step_idx]

        ax.plot(step.x, step.y, "bo", markersize=10)
        if step.ped_positions:
            ped_x = [p[0] for p in step.ped_positions]
            ped_y = [p[1] for p in step.ped_positions]
            ax.plot(ped_x, ped_y, "ro", markersize=6)

        ax.set_title(f"t={step.t:.1f}s")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    stamp_text = f"Filmstrip: {replay_episode.episode_id}"
    fig.suptitle(stamp_text, y=1.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sha256 = compute_file_sha256(out_path)
    return FigureArtifact(
        artifact_type="filmstrip",
        path=str(out_path),
        format=fmt,
        sha256=sha256,
        stamp_text=stamp_text,
    )


def generate_trajectory(
    replay_episode: ReplayEpisode,
    out_path: Path,
    fmt: str = "png",
    map_path: str | None = None,
) -> FigureArtifact:
    """Generate trajectory plot showing robot and pedestrian paths.

    Args:
        replay_episode: Episode to render.
        out_path: Output file path.
        fmt: Output format (png, pdf, svg).
        map_path: Optional map SVG path for background.

    Returns:
        FigureArtifact with metadata.
    """
    _ensure_matplotlib_backend()

    if not replay_episode.steps:
        raise ValueError("No steps in replay episode")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")

    if map_path:
        try:
            import matplotlib.image as mpimg  # noqa: PLC0415

            img = mpimg.imread(map_path)
            ax.imshow(img, origin="upper", alpha=0.3)
        except (OSError, ValueError):
            pass

    robot_x = [s.x for s in replay_episode.steps]
    robot_y = [s.y for s in replay_episode.steps]

    ax.plot(robot_x, robot_y, "b-", linewidth=2, label="Robot trajectory")
    ax.plot(robot_x[0], robot_y[0], "go", markersize=12, label="Start")
    ax.plot(robot_x[-1], robot_y[-1], "rs", markersize=12, label="End")

    ped_trajectories: dict[int, list[tuple[float, float]]] = {}
    for step_idx, step in enumerate(replay_episode.steps):
        if step.ped_positions:
            for ped_idx, pos in enumerate(step.ped_positions):
                if ped_idx not in ped_trajectories:
                    ped_trajectories[ped_idx] = []
                ped_trajectories[ped_idx].append(pos)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(ped_trajectories), 1)))
    for ped_idx, (ped_idx_key, traj) in enumerate(ped_trajectories.items()):
        if len(traj) > 1:
            ped_x = [p[0] for p in traj]
            ped_y = [p[1] for p in traj]
            ax.plot(
                ped_x,
                ped_y,
                "--",
                color=colors[ped_idx % len(colors)],
                linewidth=1.5,
                alpha=0.7,
                label=f"Pedestrian {ped_idx}",
            )

    stamp_text = (
        f"Trajectory: {replay_episode.episode_id}\n"
        f"Scenario: {replay_episode.scenario_id}\n"
        f"Steps: {len(replay_episode.steps)}"
    )
    ax.text(
        0.02,
        0.98,
        stamp_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Trajectory - Episode {replay_episode.episode_id}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sha256 = compute_file_sha256(out_path)
    return FigureArtifact(
        artifact_type="trajectory",
        path=str(out_path),
        format=fmt,
        sha256=sha256,
        stamp_text=stamp_text,
    )


def generate_caption_fragment(
    episode_row: EpisodeRow,
    replay_result: ReplayResult,
    artifacts: list[FigureArtifact],
) -> str:
    """Generate LaTeX caption fragment for artifacts.

    Args:
        episode_row: Source episode row.
        replay_result: Replay result with determinism check.
        artifacts: Generated artifacts.

    Returns:
        LaTeX caption fragment string.
    """
    planner_str = episode_row.planner or episode_row.planner_key or "unknown"
    det_status = replay_result.determinism_check_status

    caption = (
        f"Episode {episode_row.episode_id} "
        f"(scenario: {episode_row.scenario_id}, seed: {episode_row.seed}, "
        f"planner: {planner_str}). "
        f"Determinism check: {det_status}."
    )

    if replay_result.determinism_details.get("position_error_m"):
        caption += f" Position error: {replay_result.determinism_details['position_error_m']:.3f}m."

    return caption


def write_provenance_sidecar(
    out_path: Path,
    sidecar: ProvenanceSidecar,
) -> None:
    """Write provenance sidecar as JSON.

    Args:
        out_path: Output file path.
        sidecar: ProvenanceSidecar to write.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "campaign_id": sidecar.campaign_id,
        "episode_id": sidecar.episode_id,
        "scenario_id": sidecar.scenario_id,
        "seed": sidecar.seed,
        "planner_key": sidecar.planner_key,
        "scenario_matrix_path": sidecar.scenario_matrix_path,
        "campaign_config_hash": sidecar.campaign_config_hash,
        "repo_commit": sidecar.repo_commit,
        "replay_command": sidecar.replay_command,
        "determinism_check_status": sidecar.determinism_check_status,
        "determinism_tolerance": sidecar.determinism_tolerance,
        "source_episodes_jsonl_path": sidecar.source_episodes_jsonl_path,
        "source_episodes_jsonl_sha256": sidecar.source_episodes_jsonl_sha256,
        "artifacts": sidecar.artifacts,
        "generated_at": sidecar.generated_at,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def get_repo_commit() -> str | None:
    """Get current git commit hash.

    Returns:
        Git commit hash string or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _generate_requested_artifacts(
    replay_episode: ReplayEpisode,
    outputs: list[str],
    out_dir: Path,
    fmt: str,
    frame_steps: list[int] | None,
    map_path: str | None,
) -> tuple[list[FigureArtifact], list[dict[str, Any]]]:
    """Generate requested figure artifacts and collect metadata.

    Returns:
        Tuple of (artifacts, artifact metadata dicts).
    """
    artifacts: list[FigureArtifact] = []
    artifact_metadata: list[dict[str, Any]] = []

    if "still" in outputs:
        step_idx = frame_steps[0] if frame_steps else len(replay_episode.steps) // 2
        still_path = out_dir / f"still_{step_idx}.{fmt}"
        still = generate_still(replay_episode, step_idx, still_path, fmt, map_path)
        artifacts.append(still)
        artifact_metadata.append(
            {
                "type": "still",
                "path": still.path,
                "format": still.format,
                "sha256": still.sha256,
                "step_idx": step_idx,
            }
        )

    if "filmstrip" in outputs:
        filmstrip_path = out_dir / f"filmstrip.{fmt}"
        filmstrip = generate_filmstrip(replay_episode, filmstrip_path, fmt, frame_steps)
        artifacts.append(filmstrip)
        artifact_metadata.append(
            {
                "type": "filmstrip",
                "path": filmstrip.path,
                "format": filmstrip.format,
                "sha256": filmstrip.sha256,
                "frame_steps": frame_steps,
            }
        )

    if "trajectory" in outputs:
        trajectory_path = out_dir / f"trajectory.{fmt}"
        trajectory = generate_trajectory(replay_episode, trajectory_path, fmt, map_path)
        artifacts.append(trajectory)
        artifact_metadata.append(
            {
                "type": "trajectory",
                "path": trajectory.path,
                "format": trajectory.format,
                "sha256": trajectory.sha256,
            }
        )

    return artifacts, artifact_metadata


def replay_episode_and_generate_figures(  # noqa: PLR0913
    episode_row: EpisodeRow,
    outputs: list[Literal["still", "filmstrip", "trajectory"]],
    out_dir: Path,
    tolerance_m: float = 0.1,
    frame_steps: list[int] | None = None,
    fmt: str = "png",
    episodes_jsonl_path: Path | None = None,
    campaign_root: Path | None = None,
    scenario_matrix_path: Path | None = None,
    config_hash: str | None = None,
    no_determinism_check: bool = False,
) -> dict[str, Any]:
    """Main entry point: replay episode and generate figure artifacts.

    Args:
        episode_row: Validated episode row.
        outputs: List of output types to generate.
        out_dir: Output directory for artifacts.
        tolerance_m: Determinism check tolerance in meters.
        frame_steps: Frame steps for filmstrip.
        fmt: Output format (png, pdf, svg).
        episodes_jsonl_path: Path to source episodes JSONL.
        campaign_root: Campaign root path.
        scenario_matrix_path: Scenario matrix path.
        config_hash: Campaign config hash.
        no_determinism_check: Skip determinism check (diagnostic only).

    Returns:
        Dictionary with result metadata.

    Raises:
        ValueError: If required inputs missing or invalid.
        RuntimeError: If figure generation fails.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for figure generation")

    out_dir.mkdir(parents=True, exist_ok=True)

    replay_episode = build_replay_from_episode_row(episode_row)
    if not replay_episode:
        raise ValueError(
            f"Episode {episode_row.episode_id} has no replay_steps field in the row; "
            "cannot generate figures without trajectory data. "
            "The episode row must include a 'replay_steps' list with robot states. "
            "Future work will support re-simulation from seed/config when replay_steps "
            "is not available (see issue #4776 re-simulation bridge)."
        )

    if not validate_replay_episode(replay_episode, min_length=2):
        raise ValueError(
            f"Episode {episode_row.episode_id} has insufficient replay steps "
            f"(need >= 2, got {len(replay_episode.steps)})"
        )

    determinism_status, determinism_details = _check_determinism_or_skip(
        replay_episode, episode_row, tolerance_m, no_determinism_check
    )

    replay_result = ReplayResult(
        episode=replay_episode,
        determinism_check_status=determinism_status,
        determinism_details=determinism_details,
    )

    if determinism_status == "fail":
        raise RuntimeError(
            f"Determinism check failed for episode {episode_row.episode_id}: "
            f"{determinism_details.get('checks_failed', ['unknown'])}"
        )

    try:
        map_path = episode_row.raw.get("replay_map_path")
        artifacts, artifact_metadata = _generate_requested_artifacts(
            replay_episode, outputs, out_dir, fmt, frame_steps, map_path
        )

        caption = generate_caption_fragment(episode_row, replay_result, artifacts)
        caption_path = out_dir / "caption_fragment.tex"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)

        source_sha256 = None
        if episodes_jsonl_path and episodes_jsonl_path.exists():
            source_sha256 = compute_file_sha256(episodes_jsonl_path)

        planner_key = (
            episode_row.planner or episode_row.planner_key or episode_row.algo or "unknown"
        )

        sidecar = ProvenanceSidecar(
            campaign_id=episode_row.campaign_id,
            episode_id=episode_row.episode_id,
            scenario_id=episode_row.scenario_id,
            seed=episode_row.seed,
            planner_key=planner_key,
            scenario_matrix_path=(
                str(scenario_matrix_path)
                if scenario_matrix_path
                else episode_row.scenario_matrix_path
            ),
            campaign_config_hash=config_hash or episode_row.config_hash,
            repo_commit=episode_row.repo_commit or get_repo_commit(),
            replay_command=" ".join(sys.argv),
            determinism_check_status=determinism_status,
            determinism_tolerance=tolerance_m if not no_determinism_check else None,
            source_episodes_jsonl_path=str(episodes_jsonl_path) if episodes_jsonl_path else None,
            source_episodes_jsonl_sha256=source_sha256,
            artifacts=artifact_metadata,
            generated_at=datetime.now(UTC).isoformat(),
        )

        sidecar_path = out_dir / "replay_provenance.json"
        write_provenance_sidecar(sidecar_path, sidecar)

        return {
            "episode_id": episode_row.episode_id,
            "scenario_id": episode_row.scenario_id,
            "seed": episode_row.seed,
            "determinism_check_status": determinism_status,
            "artifacts_generated": len(artifacts),
            "artifact_paths": [a.path for a in artifacts],
            "provenance_sidecar": str(sidecar_path),
            "caption_fragment": str(caption_path),
            "output_dir": str(out_dir),
        }

    except Exception as e:
        raise RuntimeError(f"Figure generation failed: {e}") from e


def _check_determinism_or_skip(
    replay_episode: ReplayEpisode,
    episode_row: EpisodeRow,
    tolerance_m: float,
    no_determinism_check: bool,
) -> tuple[str, dict[str, Any]]:
    """Check determinism or return skipped status.

    Returns:
        Tuple of (status, details dict).
    """
    if no_determinism_check:
        return "skipped", {"reason": "determinism check disabled via --no-determinism-check"}
    return check_determinism(replay_episode, episode_row, tolerance_m)


__all__ = [
    "EpisodeRow",
    "FigureArtifact",
    "ProvenanceSidecar",
    "ReplayResult",
    "build_replay_from_episode_row",
    "check_determinism",
    "compute_file_sha256",
    "generate_filmstrip",
    "generate_still",
    "generate_trajectory",
    "load_episode_row",
    "replay_episode_and_generate_figures",
    "write_provenance_sidecar",
]
