"""Multi-planner trajectory overlay figures.

Overlay robot trajectories from multiple planners on the same
scenario + seed into a single provenance-stamped figure.  This is
a visual comparison tool, not benchmark evidence.

Part of issue #4778: exemplar selection + multi-planner overlays.

Usage (library):
    from robot_sf.benchmark.multi_planner_overlay import (
        extract_trajectory_from_episode,
        select_episodes_for_overlay,
        build_overlay_figure,
    )

Usage (CLI):
    uv run python scripts/render_multi_planner_trajectory_overlay.py \\
      --episodes campaign/episodes.jsonl \\
      --scenario-id corridor \\
      --seed 42 \\
      --planners orca,ppo \\
      --out-dir output/overlays
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_sf.benchmark.figures.export import save_publication_figure
from robot_sf.benchmark.figures.provenance import (
    build_caption_fragment,
    build_provenance,
)
from robot_sf.benchmark.figures.style import (
    planner_color,
    publication_style,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


OVERLAY_PROVENANCE_SCHEMA = "multi_planner_overlay.v1"


class MultiPlannerOverlayError(Exception):
    """Raised when overlay inputs are invalid or incomplete."""


@dataclass(frozen=True, slots=True)
class TrajectoryRow:
    """One planner's trajectory for a (scenario, seed) cell."""

    episode_id: str
    planner_key: str
    scenario_id: str
    seed: int
    positions: list[tuple[float, float]]
    pedestrians: list[tuple[float, float]] | None = None
    map_bounds: tuple[float, float, float, float] | None = None
    source_path: str = ""


@dataclass
class OverlayProvenance:
    """Provenance sidecar for a multi-planner overlay figure."""

    schema_version: str = OVERLAY_PROVENANCE_SCHEMA
    scenario_id: str = ""
    seed: int = 0
    planners: list[str] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)
    source_episodes: str = ""
    source_sha256: str = ""
    output_files: list[str] = field(default_factory=list)
    claim_boundary: str = "visual comparison only; not benchmark evidence or metric claim"
    git_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict.

        Returns:
            JSON-serializable dict representation.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------

_TRAJECTORY_PATHS: list[tuple[str, str]] = [
    # (path to positions, path to heading) — heading is optional
    ("trajectory.robot_positions", "trajectory.robot_headings"),
    ("robot_trajectory.positions", "robot_trajectory.headings"),
    ("trace.positions", "trace.headings"),
    ("metrics.trajectory.positions", "metrics.trajectory.headings"),
]


def _get_nested(record: dict[str, Any], path: str, default: Any = None) -> Any:
    """Resolve a dotted-path value from a dict.

    Returns:
        Value at the dotted path, or *default* when any segment is missing.
    """
    cur: Any = record
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _parse_positions(raw: Any) -> list[tuple[float, float]]:
    """Convert position list or dict list to list of (x, y) tuples.

    Returns:
        List of (x, y) float tuples.
    """
    if not isinstance(raw, list) or not raw:
        return []
    positions: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            positions.append((float(item[0]), float(item[1])))
        elif isinstance(item, dict):
            x = item.get("x", item.get("position_x"))
            y = item.get("y", item.get("position_y"))
            if x is not None and y is not None:
                positions.append((float(x), float(y)))
    return positions


def extract_trajectory_from_episode(
    record: dict[str, Any],
) -> TrajectoryRow | None:
    """Extract trajectory data from an episode record.

    Tries several common position field layouts.  Returns ``None`` when
    no trajectory data is found in the record.

    Args:
        record: One episode dict from episodes.jsonl.

    Returns:
        ``TrajectoryRow`` if trajectory positions are found, else ``None``.
    """
    positions: list[tuple[float, float]] = []

    for pos_path, _heading_path in _TRAJECTORY_PATHS:
        raw = _get_nested(record, pos_path)
        if raw is not None:
            positions = _parse_positions(raw)
            if positions:
                break

    if not positions:
        return None

    planner_key = _normalize_planner_key(record)
    scenario_id = str(record.get("scenario_id", ""))
    seed = int(record.get("seed", 0))
    episode_id = str(record.get("episode_id", ""))

    # Pedestrian positions (best-effort)
    ped_raw = _get_nested(record, "pedestrians")
    pedestrians: list[tuple[float, float]] | None = None
    if isinstance(ped_raw, list) and ped_raw:
        pedestrians = []
        for ped in ped_raw:
            if isinstance(ped, dict):
                pos = ped.get("position", ped.get("start_position"))
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    pedestrians.append((float(pos[0]), float(pos[1])))

    # Map bounds (best-effort)
    bounds = _get_nested(record, "scenario_params.map_bounds")
    map_bounds: tuple[float, float, float, float] | None = None
    if isinstance(bounds, (list, tuple)) and len(bounds) >= 4:
        map_bounds = (
            float(bounds[0]),
            float(bounds[1]),
            float(bounds[2]),
            float(bounds[3]),
        )

    source_path = str(record.get("source_path", ""))

    return TrajectoryRow(
        episode_id=episode_id,
        planner_key=planner_key,
        scenario_id=scenario_id,
        seed=seed,
        positions=positions,
        pedestrians=pedestrians or None,
        map_bounds=map_bounds,
        source_path=source_path,
    )


def _normalize_planner_key(record: dict[str, Any]) -> str:
    """Extract planner key from an episode record.

    Returns:
        Planner identifier string.
    """
    for path in ("algo", "scenario_params.algo", "algorithm_metadata.algorithm", "planner"):
        val = _get_nested(record, path)
        if val is not None:
            return str(val).strip()
    return "unknown_planner"


# ---------------------------------------------------------------------------
# Episode selection
# ---------------------------------------------------------------------------


def select_episodes_for_overlay(
    episodes: list[dict[str, Any]],
    *,
    scenario_id: str,
    seed: int,
    planner_keys: Sequence[str],
) -> dict[str, TrajectoryRow]:
    """Select one trajectory row per planner for a given scenario+seed.

    Args:
        episodes: Parsed episode records from JSONL.
        scenario_id: Scenario to filter on.
        seed: Seed to filter on.
        planner_keys: Planner keys that must be present.

    Returns:
        Dict mapping planner key to ``TrajectoryRow``.

    Raises:
        MultiPlannerOverlayError: If a planner key has no matching episode
            with trajectory data.
    """
    result: dict[str, TrajectoryRow] = {}

    for planner_key in planner_keys:
        matched = _find_trajectory_row(episodes, scenario_id, seed, planner_key)
        if matched is None:
            raise MultiPlannerOverlayError(
                f"No trajectory data for planner={planner_key!r}, "
                f"scenario={scenario_id!r}, seed={seed}"
            )
        result[planner_key] = matched

    return result


def _find_trajectory_row(
    episodes: list[dict[str, Any]],
    scenario_id: str,
    seed: int,
    planner_key: str,
) -> TrajectoryRow | None:
    """Find the trajectory row for a specific planner/scenario/seed combo.

    Returns:
        ``TrajectoryRow`` if found, else ``None``.
    """
    for ep in episodes:
        ep_scenario = str(ep.get("scenario_id", ""))
        ep_seed = int(ep.get("seed", 0))
        ep_planner = _normalize_planner_key(ep)
        if ep_scenario == scenario_id and ep_seed == seed and ep_planner == planner_key:
            return extract_trajectory_from_episode(ep)
    return None


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------


def build_overlay_figure(
    trajectory_rows: dict[str, TrajectoryRow],
    *,
    output_base: Path,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Render a multi-planner trajectory overlay figure.

    Overlays robot paths from each planner using the shared planner palette.
    Writes provenance sidecar and caption fragment.

    Args:
        trajectory_rows: Planner key to ``TrajectoryRow`` mapping.
        output_base: Base output path (without extension).
        formats: Output formats to save (``"pdf"``, ``"png"``, ``"svg"``).
        dpi: Resolution for raster formats.

    Returns:
        List of generated file paths (figures plus sidecars).

    Raises:
        MultiPlannerOverlayError: If no trajectory rows provided.
    """
    if not trajectory_rows:
        raise MultiPlannerOverlayError("No trajectory rows to overlay")

    # Collect provenance data
    episode_ids = [row.episode_id for row in trajectory_rows.values()]
    scenario_id = trajectory_rows[next(iter(trajectory_rows))].scenario_id
    seed = trajectory_rows[next(iter(trajectory_rows))].seed
    planners_in_order = list(trajectory_rows.keys())

    # Build the figure
    with publication_style(size="single"):
        fig, ax = plt.subplots()

        _plot_trajectories(ax, trajectory_rows)
        _add_annotations(ax, scenario_id, seed)

        fig.tight_layout()

        # Build provenance
        source_path = _find_source_path(trajectory_rows)

        provenance = build_provenance(
            generator_command="render_multi_planner_trajectory_overlay",
            episode_ids=episode_ids,
            seeds=[seed],
            figure_formats=list(formats),
            config_path=source_path,
            claim_boundary=OverlayProvenance(
                scenario_id=scenario_id,
                seed=seed,
                planners=planners_in_order,
                episode_ids=episode_ids,
                source_episodes=source_path,
                git_hash=_git_sha_short(),
            ).claim_boundary,
        )

        # Add overlay-specific provenance fields
        provenance["planners"] = planners_in_order
        provenance["scenario_id"] = scenario_id
        provenance["seed"] = seed

        caption = build_caption_fragment(
            scenario_id=scenario_id,
            episode_ids=episode_ids,
        )

        # Save with provenance
        saved = save_publication_figure(
            fig,
            output_base=output_base,
            formats=tuple(formats),
            provenance=provenance,
            caption_fragment=caption,
        )

        plt.close(fig)

    return saved


def _plot_trajectories(
    ax: plt.Axes,
    trajectory_rows: dict[str, TrajectoryRow],
) -> None:
    """Plot all planner trajectories and optional pedestrian positions."""
    # Plot each planner trajectory
    for planner_key, row in trajectory_rows.items():
        xs, ys = zip(*row.positions, strict=True)
        color = planner_color(planner_key)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=2,
            label=planner_key,
            zorder=3,
        )
        # Mark start (square) and goal (star)
        if row.positions:
            ax.plot(
                row.positions[0][0],
                row.positions[0][1],
                color=color,
                marker="s",
                markersize=6,
                zorder=4,
                fillstyle="none",
                markeredgewidth=1.5,
            )
            ax.plot(
                row.positions[-1][0],
                row.positions[-1][1],
                color=color,
                marker="*",
                markersize=8,
                zorder=4,
            )

    # Plot pedestrian positions if available from any row
    for row in trajectory_rows.values():
        if row.pedestrians:
            ped_xs, ped_ys = zip(*row.pedestrians, strict=True)
            ax.scatter(
                ped_xs,
                ped_ys,
                color="gray",
                s=15,
                alpha=0.5,
                zorder=1,
                label="_nolegend_",
            )
            break  # Only plot once


def _add_annotations(ax: plt.Axes, scenario_id: str, seed: int) -> None:
    """Add axis labels, title, legend, and claim boundary annotation."""
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Trajectory overlay: {scenario_id} (seed {seed})")

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles, strict=False))
    ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8)

    # Add claim boundary annotation
    ax.text(
        0.01,
        0.01,
        "Visual comparison only - not benchmark evidence",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        alpha=0.6,
    )


def _find_source_path(trajectory_rows: dict[str, TrajectoryRow]) -> str:
    """Find the source path from the first row that has one.

    Returns:
        Source path string or empty string.
    """
    for row in trajectory_rows.values():
        if row.source_path:
            return row.source_path
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha_short(length: int = 7) -> str:
    """Return short git SHA for current HEAD.

    Returns:
        Short git SHA or "unknown" if unavailable.
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", f"--short={length}", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        return sha or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_episodes(path: Path | str) -> list[dict[str, Any]]:
    """Load episodes from a JSONL file.

    Args:
        path: Path to episodes.jsonl.

    Returns:
        List of episode dicts.

    Raises:
        MultiPlannerOverlayError: If file not found.
    """
    episodes: list[dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        raise MultiPlannerOverlayError(f"Episodes file not found: {source}")
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


__all__ = [
    "OVERLAY_PROVENANCE_SCHEMA",
    "MultiPlannerOverlayError",
    "OverlayProvenance",
    "TrajectoryRow",
    "build_overlay_figure",
    "extract_trajectory_from_episode",
    "load_episodes",
    "select_episodes_for_overlay",
]
