"""Representative trajectory panel generation for analysis-workbench traces."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    load_simulation_trace_export,
)
from robot_sf.benchmark.figures.style import publication_style

TRAJECTORY_PANEL_MANIFEST_SCHEMA_VERSION = "trajectory_panel_manifest.v1"
_CATEGORY_ORDER = {
    "collision": 0,
    "near_miss": 1,
    "low_progress": 2,
    "success": 3,
    "other": 4,
}
_NEAR_MISS_MIN_TTC_S = 0.5
_LOW_PROGRESS_DISTANCE_M = 0.25


@dataclass(frozen=True, slots=True)
class RepresentativeEpisode:
    """One selected episode and the trace used to render it."""

    artifact_id: str
    trace_path: Path
    trace: SimulationTraceExport
    category: str
    panel_type: str
    caption: str
    diagnostic_only: bool = True


@dataclass(frozen=True, slots=True)
class PanelArtifact:
    """One rendered trajectory panel artifact."""

    artifact_id: str
    png_path: Path
    pdf_path: Path
    category: str
    panel_type: str


@dataclass(frozen=True, slots=True)
class TrajectoryPanelBundle:
    """Output paths and artifact metadata for a generated panel bundle."""

    output_dir: Path
    selection_csv: Path
    manifest_path: Path
    captions_path: Path
    artifacts: tuple[PanelArtifact, ...]


def select_representative_episodes(
    trace_paths: list[Path],
    *,
    override_csv: Path | None = None,
    limit_per_group: int = 1,
) -> list[RepresentativeEpisode]:
    """Select deterministic representative episodes from trace-export JSON files.

    Returns:
        Stable representative episode rows.
    """

    if override_csv is not None:
        return _select_from_override(override_csv)

    candidates = [_episode_from_trace_path(path) for path in trace_paths]
    grouped: dict[tuple[str, str, str], list[RepresentativeEpisode]] = {}
    for candidate in candidates:
        key = (
            candidate.trace.source.planner_id,
            candidate.trace.source.scenario_id,
            candidate.category,
        )
        grouped.setdefault(key, []).append(candidate)

    selected: list[RepresentativeEpisode] = []
    for key in sorted(grouped, key=lambda item: (_CATEGORY_ORDER.get(item[2], 99), item)):
        rows = sorted(
            grouped[key],
            key=lambda row: (
                row.trace.source.seed,
                row.trace.source.episode_id,
                row.trace.trace_id,
            ),
        )
        selected.extend(rows[:limit_per_group])
    return selected


def generate_trajectory_panel_bundle(
    trace_paths: list[Path],
    *,
    output_dir: Path,
    command: str,
    commit: str,
    override_csv: Path | None = None,
) -> TrajectoryPanelBundle:
    """Render selected trace exports to PNG/PDF panels and write bundle metadata.

    Returns:
        Paths and artifact records for the generated bundle.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_representative_episodes(trace_paths, override_csv=override_csv)

    artifacts: list[PanelArtifact] = []
    for episode in selected:
        panel_dir = (
            output_dir / "failure_mosaics"
            if episode.panel_type == "failure_mosaic"
            else output_dir / "trajectory_panels"
        )
        panel_dir.mkdir(parents=True, exist_ok=True)
        png_path = panel_dir / f"{episode.artifact_id}.png"
        pdf_path = panel_dir / f"{episode.artifact_id}.pdf"
        _render_episode_panel(episode, png_path=png_path, pdf_path=pdf_path)
        artifacts.append(
            PanelArtifact(
                artifact_id=episode.artifact_id,
                png_path=png_path,
                pdf_path=pdf_path,
                category=episode.category,
                panel_type=episode.panel_type,
            )
        )

    selection_csv = output_dir / "representative_episode_selection.csv"
    _write_selection_csv(selection_csv, selected)
    captions_path = output_dir / "captions.md"
    _write_captions(captions_path, selected)
    manifest_path = output_dir / "trajectory_panel_manifest.json"
    _write_manifest(
        manifest_path,
        selected=selected,
        artifacts=artifacts,
        command=command,
        commit=commit,
        captions_path=captions_path,
    )
    return TrajectoryPanelBundle(
        output_dir=output_dir,
        selection_csv=selection_csv,
        manifest_path=manifest_path,
        captions_path=captions_path,
        artifacts=tuple(artifacts),
    )


def _select_from_override(override_csv: Path) -> list[RepresentativeEpisode]:
    """Load reviewer-selected episodes from a CSV file.

    Returns:
        Representative episode rows in CSV order.
    """

    rows: list[RepresentativeEpisode] = []
    with override_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            raw_trace_path = str(row.get("trace_path") or "").strip()
            if not raw_trace_path:
                raise ValueError(f"{override_csv}:{index}: trace_path is required")
            trace_path = Path(raw_trace_path)
            if not trace_path.is_absolute():
                trace_path = override_csv.parent / trace_path
            trace = load_simulation_trace_export(trace_path)
            category = str(row.get("category") or "").strip() or _classify_trace(trace)
            artifact_id = str(row.get("artifact_id") or "").strip() or _artifact_id(
                trace, category=category
            )
            panel_type = str(row.get("panel_type") or "trajectory_panel").strip()
            caption = str(row.get("caption") or "").strip() or _default_caption(trace, category)
            rows.append(
                RepresentativeEpisode(
                    artifact_id=artifact_id,
                    trace_path=trace_path,
                    trace=trace,
                    category=category,
                    panel_type=panel_type,
                    caption=caption,
                )
            )
    return rows


def _episode_from_trace_path(trace_path: Path) -> RepresentativeEpisode:
    """Load one trace path and build its selection metadata.

    Returns:
        Representative episode metadata for the trace.
    """

    trace = load_simulation_trace_export(trace_path)
    category = _classify_trace(trace)
    return RepresentativeEpisode(
        artifact_id=_artifact_id(trace, category=category),
        trace_path=trace_path,
        trace=trace,
        category=category,
        panel_type="failure_mosaic"
        if category in {"collision", "near_miss"}
        else "trajectory_panel",
        caption=_default_caption(trace, category),
    )


def _classify_trace(trace: SimulationTraceExport) -> str:
    """Classify a trace into the representative episode buckets.

    Returns:
        Stable category name for selection and rendering.
    """

    event_text = " ".join(str(frame.planner.get("event", "")).lower() for frame in trace.frames)
    if "collision" in event_text:
        return "collision"
    if _min_numeric_planner_value(trace, "min_ttc") < _NEAR_MISS_MIN_TTC_S:
        return "near_miss"
    if _robot_displacement(trace) < _LOW_PROGRESS_DISTANCE_M:
        return "low_progress"
    if any(token in event_text for token in ("goal", "success", "reached")):
        return "success"
    return "other"


def _min_numeric_planner_value(trace: SimulationTraceExport, key: str) -> float:
    """Return the minimum numeric planner value for ``key`` or infinity."""

    values: list[float] = []
    for frame in trace.frames:
        value = frame.planner.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return min(values) if values else float("inf")


def _robot_displacement(trace: SimulationTraceExport) -> float:
    """Return straight-line displacement between the first and last robot poses."""

    if not trace.frames:
        return 0.0
    try:
        start = _xy(trace.frames[0].robot.get("position"))
        end = _xy(trace.frames[-1].robot.get("position"))
    except (KeyError, TypeError, ValueError, IndexError):
        return 0.0
    return ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5


def _artifact_id(trace: SimulationTraceExport, *, category: str) -> str:
    """Build a stable filesystem-friendly panel artifact ID.

    Returns:
        Normalized artifact identifier.
    """

    parts = [
        "trajectory_panel",
        trace.source.planner_id,
        trace.source.scenario_id,
        category,
        trace.source.episode_id,
    ]
    return "_".join(_slug(part) for part in parts)


def _slug(value: Any) -> str:
    """Normalize a value for stable artifact IDs.

    Returns:
        Filesystem-friendly slug.
    """

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-").lower()
    return slug or "unknown"


def _default_caption(trace: SimulationTraceExport, category: str) -> str:
    """Return a short diagnostic-only caption."""

    return (
        f"{trace.source.planner_id} on {trace.source.scenario_id} episode "
        f"{trace.source.episode_id}; selected as {category}. Diagnostic-only trace panel."
    )


def _render_episode_panel(
    episode: RepresentativeEpisode,
    *,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Render one trace panel to PNG and PDF."""

    trace = episode.trace
    robot_xy = [_xy(frame.robot.get("position")) for frame in trace.frames]
    pedestrian_tracks: dict[str, list[tuple[float, float]]] = {}
    for frame in trace.frames:
        for pedestrian in frame.pedestrians:
            ped_id = str(pedestrian.get("id", "pedestrian"))
            pedestrian_tracks.setdefault(ped_id, []).append(_xy(pedestrian.get("position")))

    # Use publication style for consistent rendering
    with publication_style(size="single"):
        # Override to match original figure size
        matplotlib.rcParams["figure.figsize"] = (6.0, 4.0)
        matplotlib.rcParams["figure.constrained_layout.use"] = True

        fig, ax = plt.subplots()
        if robot_xy:
            xs, ys = zip(*robot_xy, strict=True)
            ax.plot(xs, ys, color="#1f77b4", linewidth=2.0, marker="o", label="robot")
            ax.scatter([xs[0]], [ys[0]], color="#2ca02c", s=55, zorder=3, label="start")
            ax.scatter([xs[-1]], [ys[-1]], color="#d62728", s=55, zorder=3, label="terminal")
        for ped_id, points in sorted(pedestrian_tracks.items()):
            if not points:
                continue
            xs, ys = zip(*points, strict=True)
            ax.plot(xs, ys, color="#7f7f7f", linewidth=1.0, linestyle="--", alpha=0.8)
            ax.scatter(xs, ys, color="#ff7f0e", s=20, alpha=0.8, label=ped_id)

        ax.set_title(f"{trace.source.planner_id} / {trace.source.scenario_id} / {episode.category}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.01,
            0.01,
            "diagnostic-only; map geometry rendered when present in trace inputs",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )
        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles, strict=False))
        ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8)

        png_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=140)
        fig.savefig(pdf_path)
        plt.close(fig)


def _xy(value: Any) -> tuple[float, float]:
    """Return an ``(x, y)`` tuple from a trace position value."""

    if not isinstance(value, list | tuple) or len(value) < 2:
        raise ValueError(f"expected [x, y] position, got {value!r}")
    return float(value[0]), float(value[1])


def _write_selection_csv(path: Path, selected: list[RepresentativeEpisode]) -> None:
    """Write the selected episode table."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "artifact_id",
                "trace_path",
                "planner_id",
                "scenario_id",
                "episode_id",
                "category",
                "panel_type",
                "diagnostic_only",
            ],
        )
        writer.writeheader()
        for episode in selected:
            writer.writerow(
                {
                    "artifact_id": episode.artifact_id,
                    "trace_path": str(episode.trace_path),
                    "planner_id": episode.trace.source.planner_id,
                    "scenario_id": episode.trace.source.scenario_id,
                    "episode_id": episode.trace.source.episode_id,
                    "category": episode.category,
                    "panel_type": episode.panel_type,
                    "diagnostic_only": str(episode.diagnostic_only).lower(),
                }
            )


def _write_captions(path: Path, selected: list[RepresentativeEpisode]) -> None:
    """Write compact captions and evidence-boundary notes."""

    lines = [
        "# Representative Trajectory Panels",
        "",
        (
            "These static panels are diagnostic-only visual artifacts. They preserve trace "
            "evidence boundaries and do not replace aggregate benchmark summaries."
        ),
        "",
    ]
    for episode in selected:
        lines.extend([f"## {episode.artifact_id}", "", episode.caption, ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_manifest(
    path: Path,
    *,
    selected: list[RepresentativeEpisode],
    artifacts: list[PanelArtifact],
    command: str,
    commit: str,
    captions_path: Path,
) -> None:
    """Write a JSON manifest for generated panels."""

    artifact_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    payload = {
        "schema_version": TRAJECTORY_PANEL_MANIFEST_SCHEMA_VERSION,
        "generation_command": command,
        "generation_commit": commit,
        "captions_path": str(captions_path),
        "artifacts": [
            _manifest_artifact(
                episode,
                artifact_by_id[episode.artifact_id],
                command=command,
                commit=commit,
                captions_path=captions_path,
            )
            for episode in selected
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_artifact(
    episode: RepresentativeEpisode,
    artifact: PanelArtifact,
    *,
    command: str,
    commit: str,
    captions_path: Path,
) -> dict[str, Any]:
    """Return one manifest artifact payload."""

    return {
        "artifact_id": artifact.artifact_id,
        "artifact_kind": "figure",
        "panel_type": artifact.panel_type,
        "category": artifact.category,
        "source_kind": "simulation_trace_export",
        "source_files": [
            {
                "path": str(episode.trace_path),
                "source_sha256": _sha256_file(episode.trace_path),
            }
        ],
        "outputs": {
            "png": {"path": str(artifact.png_path), "sha256": _sha256_file(artifact.png_path)},
            "pdf": {"path": str(artifact.pdf_path), "sha256": _sha256_file(artifact.pdf_path)},
        },
        "generation_command": command,
        "generation_commit": commit,
        "claim_boundary": "diagnostic_only",
        "caption_file": {
            "path": str(captions_path),
            "sha256": _sha256_file(captions_path),
        },
        "trace_source": asdict(episode.trace.source),
    }


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for a file.

    Returns:
        Hex-encoded SHA-256 digest.
    """

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
