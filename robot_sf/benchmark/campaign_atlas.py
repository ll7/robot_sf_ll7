"""Campaign atlas and event-aligned ensemble context views (issue #5616).

Orchestrates existing renderers (`figure_qa`, `event_ledger`, `exemplar_selection`,
the publication style) into two publication-facing report surfaces built over an
eligible campaign:

1. **Campaign atlas** — a scenario-family x planner grid showing the seed
   population per cell with outcome counts, Wilson uncertainty intervals, and
   explicit markers for ineligible/missing cells and for selected exemplar cases
   (consumed from a frozen selection manifest).
2. **Event-aligned ensemble context view** — per selected cell, an observed
   medoid trajectory, transparent quantile bands, explicitly marked outliers, and
   a categorical planner-state/predicate sequence strip. The ensemble is aligned
   on a named *event anchor* (event-relative time), never on normalized episode
   duration, and refuses to render when no shared event anchor exists.

Matplotlib owns the authoritative SVG/PDF output (noninteractive Agg backend,
fixed style, stable ordering, hashed outputs). An optional Altair/Vega-Lite HTML
atlas is an exploration convenience only and is parity-checked against the static
summary; the parity check fails closed on disagreement.

Neither surface modifies metric/outcome semantics, selects or re-ranks cases, or
implies causal mechanisms.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from robot_sf.benchmark.event_ledger import EPISODE_EVENT_LEDGER_SCHEMA_VERSION
from robot_sf.common.optional_import import try_import

CAMPAIGN_ATLAS_SCHEMA_VERSION = "campaign_atlas.v1"
EVENT_DETECTOR_VERSION = EPISODE_EVENT_LEDGER_SCHEMA_VERSION
RENDERER_VERSION = "matplotlib"
STYLE_VERSION = "publication_style.v1"

# Colorblind-safe outcome palette (Wong 2011 inspired) with line-style redundancy.
OUTCOME_PALETTE: dict[str, str] = {
    "collision": "#D55E00",
    "success": "#009E73",
    "timeout": "#0072B2",
    "near_miss": "#E69F00",
    "other": "#999999",
}
OUTCOME_HATCH: dict[str, str] = {
    "collision": "",
    "success": "//",
    "timeout": "\\",
    "near_miss": "xx",
    "other": ".",
}
# Categorical predicate palette (colorblind-safe Set2-ish).
PREDICATE_PALETTE: dict[str, str] = {
    "clear": "#009E73",
    "approach": "#56B4E9",
    "evade": "#D55E00",
    "wait": "#CC79A7",
    "occluded": "#E69F00",
    "unknown": "#999999",
}

_ATLAS_RC = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "campaign-atlas-v1",
}
_WILSON_Z = 1.959963984540054


class CampaignAtlasError(ValueError):
    """Raised when campaign atlas inputs are malformed or cannot be rendered."""


class AtlasParityError(CampaignAtlasError):
    """Raised when the HTML exploration atlas disagrees with the static summary."""


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """One ``(time, x, y)`` sample of a robot trajectory."""

    t: float
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class PredicateInterval:
    """One categorical planner-state/predicate interval in episode time."""

    t_start: float
    t_end: float
    label: str


@dataclass(frozen=True, slots=True)
class EpisodeInventoryRow:
    """One episode inventory row consumed by the atlas builder.

    The atlas consumes this as *input*; it does not compute outcomes, event
    anchors, or predicates. Outcome semantics are taken verbatim.
    """

    episode_id: str
    planner: str
    scenario_id: str
    scenario_family: str
    seed: int
    outcome: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    trajectory: tuple[TrajectoryPoint, ...] = field(default_factory=tuple)
    event_anchors: Mapping[str, float] = field(default_factory=dict)
    predicate_timeline: tuple[PredicateInterval, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class AtlasCellSummary:
    """One (scenario-family x planner) cell of the campaign atlas."""

    scenario_family: str
    planner: str
    eligible: bool
    ineligible_reason: str | None
    n_total: int
    outcome_counts: Mapping[str, int]
    outcome_ci: Mapping[str, tuple[float, float, float]]
    exemplar_episode_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AtlasSummary:
    """Normalized, representation-agnostic summary of the campaign atlas.

    Both the static matplotlib atlas and the optional HTML exploration atlas are
    generated from this single object, and the parity check compares the
    per-representation summaries extracted from each output.
    """

    campaign_id: str
    scenario_families: tuple[str, ...]
    planners: tuple[str, ...]
    cells: tuple[AtlasCellSummary, ...]
    metric_definitions: Mapping[str, str]
    event_anchor: str | None
    selection_manifest_hash: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finals(x: float) -> float:
    """Round a finite float for stable, hashable output; raise if non-finite."""
    if not math.isfinite(x):
        raise CampaignAtlasError(f"non-finite value encountered: {x!r}")
    return round(x, 6)


def _wilson_ci(k: int, n: int) -> tuple[float, float, float]:
    """Return ``(p, lo, hi)`` Wilson score interval for ``k`` of ``n`` successes.

    Returns ``(0.0, 0.0, 0.0)`` for an empty cell so the atlas can mark it empty
    without implying a false estimate.
    """
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1.0 + _WILSON_Z * _WILSON_Z / n
    center = (p + _WILSON_Z * _WILSON_Z / (2.0 * n)) / denom
    half = _WILSON_Z * math.sqrt(p * (1.0 - p) / n + _WILSON_Z * _WILSON_Z / (4.0 * n * n)) / denom
    return (_finals(p), _finals(max(0.0, center - half)), _finals(min(1.0, center + half)))


def _sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _save_figure(figure: plt.Figure, out: Path) -> None:
    """Save a matplotlib figure in its requested format.

    Matplotlib injects a ``<dc:date>`` creation timestamp into SVG metadata,
    which breaks byte-identical determinism. We render to SVG text, remove the
    date element, then write the stable bytes. PDF output is written through
    Matplotlib's PDF backend rather than being mislabeled SVG XML.
    """
    if out.suffix.lower() == ".pdf":
        figure.savefig(out, format="pdf", bbox_inches="tight", metadata={"CreationDate": None})
        return

    import io  # noqa: PLC0415 (local import keeps module import-light)

    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    text = re.sub(
        r"<dc:date>.*?</dc:date>", "<dc:date></dc:date>", buffer.getvalue(), flags=re.DOTALL
    )
    out.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sorted_labels(counts: Mapping[str, int]) -> list[str]:
    """Return outcome labels ordered by descending count, then name."""
    return [label for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


# ---------------------------------------------------------------------------
# Atlas model construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AtlasConfig:
    """Deterministic eligibility + inventory configuration for the atlas."""

    campaign_id: str = "campaign"
    min_cell_size: int = 1
    metric_definitions: Mapping[str, str] = field(default_factory=dict)
    eligible_scenario_families: tuple[str, ...] | None = None
    eligible_planners: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Reject configuration that could silently mark every cell eligible."""
        if self.min_cell_size < 1:
            raise CampaignAtlasError("min_cell_size must be at least 1")
        for field_name, values in (
            ("eligible_scenario_families", self.eligible_scenario_families),
            ("eligible_planners", self.eligible_planners),
        ):
            if values is not None and any(not value for value in values):
                raise CampaignAtlasError(f"{field_name} cannot contain empty labels")


def _cell_key(scenario_family: str, planner: str) -> str:
    """Return a deterministic cell key string."""
    return f"{scenario_family}|{planner}"


def build_atlas_summary(
    rows: Sequence[EpisodeInventoryRow],
    *,
    config: AtlasConfig,
    exemplar_episode_ids: Sequence[str] = (),
) -> AtlasSummary:
    """Build a deterministic ``AtlasSummary`` from inventory rows.

    Args:
        rows: Campaign inventory rows (consumed verbatim).
        config: Eligibility + inventory configuration.
        exemplar_episode_ids: Episode ids selected by the frozen selection
            manifest; used only to mark exemplar cells, not to re-rank.

    Returns:
        Deterministic atlas summary covering every eligible cell, with explicit
        markers for ineligible/missing cells.
    """
    exemplar_ids = frozenset(exemplar_episode_ids)
    grouped: dict[str, list[EpisodeInventoryRow]] = {}
    for row in rows:
        grouped.setdefault(_cell_key(row.scenario_family, row.planner), []).append(row)

    scenario_families = (
        sorted(set(config.eligible_scenario_families))
        if config.eligible_scenario_families is not None
        else sorted({row.scenario_family for row in rows})
    )
    planners = (
        sorted(set(config.eligible_planners))
        if config.eligible_planners is not None
        else sorted({row.planner for row in rows})
    )

    cells: list[AtlasCellSummary] = []
    for scenario_family in scenario_families:
        for planner in planners:
            key = _cell_key(scenario_family, planner)
            cell_rows = grouped.get(key, [])
            n_total = len(cell_rows)
            outcome_counts = Counter(row.outcome for row in cell_rows)
            eligible = n_total >= config.min_cell_size
            reason = None
            if n_total == 0:
                reason = "no eligible episodes for this (scenario-family, planner) cell"
            elif not eligible:
                reason = f"cell below minimum cell size ({n_total} < {config.min_cell_size})"
            outcome_ci = {
                label: _wilson_ci(outcome_counts.get(label, 0), n_total)
                for label in _sorted_labels(outcome_counts) or ["other"]
            }
            cell_exemplars = tuple(
                sorted(row.episode_id for row in cell_rows if row.episode_id in exemplar_ids)
            )
            cells.append(
                AtlasCellSummary(
                    scenario_family=scenario_family,
                    planner=planner,
                    eligible=eligible,
                    ineligible_reason=reason,
                    n_total=n_total,
                    outcome_counts=dict(outcome_counts),
                    outcome_ci=outcome_ci,
                    exemplar_episode_ids=cell_exemplars,
                )
            )

    return AtlasSummary(
        campaign_id=config.campaign_id,
        scenario_families=tuple(scenario_families),
        planners=tuple(planners),
        cells=tuple(cells),
        metric_definitions=dict(config.metric_definitions),
        event_anchor=None,
        selection_manifest_hash=_sha256_text(json.dumps(sorted(exemplar_ids))),
    )


# ---------------------------------------------------------------------------
# Static atlas rendering (matplotlib authoritative output)
# ---------------------------------------------------------------------------


def render_atlas_figure(summary: AtlasSummary, out: Path) -> Path:
    """Render the static campaign atlas grid to SVG and return the output path.

    The figure uses small multiples: one panel per (scenario-family, planner)
    cell. Each panel shows a stacked outcome proportion bar with a Wilson 95%
    interval whisker and a marker for the number of selected exemplar cases.
    Ineligible/missing cells are drawn as explicit hatched "N/A" panels.

    Returns:
        Path to the written SVG file (the authoritative static artifact).
    """
    out = Path(out)
    if out.suffix.lower() != ".svg":
        out = out.with_suffix(".svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_rows = max(1, len(summary.scenario_families))
    n_cols = max(1, len(summary.planners))
    cell_index = {(cell.scenario_family, cell.planner): cell for cell in summary.cells}

    with plt.rc_context(_ATLAS_RC):
        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.2 * n_cols + 1.2, 2.6 * n_rows + 1.0),
            squeeze=False,
        )
        for r, scenario_family in enumerate(summary.scenario_families):
            for c, planner in enumerate(summary.planners):
                ax = axes[r][c]
                _draw_atlas_panel(
                    ax,
                    cell_index.get((scenario_family, planner)),
                    scenario_family,
                    planner,
                    figure,
                )
        figure.suptitle(f"Campaign atlas — {summary.campaign_id}", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        _save_figure(figure, out)
        plt.close(figure)
    return out


def _draw_atlas_panel(
    ax: plt.Axes,
    cell: AtlasCellSummary | None,
    scenario_family: str,
    planner: str,
    figure: plt.Figure,
) -> None:
    """Draw one atlas cell panel (or an explicit ineligible marker)."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{scenario_family}\n{planner}", fontsize=10)

    if cell is None or not cell.eligible:
        reason = cell.ineligible_reason if cell is not None else "no eligible episodes"
        ax.add_patch(
            Rectangle((0.04, 0.04), 0.92, 0.92, fill=False, hatch="///", edgecolor="#999999")
        )
        ax.text(
            0.5,
            0.5,
            f"N/A\n{reason}",
            ha="center",
            va="center",
            fontsize=8,
            color="#666666",
            wrap=True,
        )
        return

    labels = _sorted_labels(cell.outcome_counts)
    left = 0.0
    for label in labels:
        count = cell.outcome_counts[label]
        proportion = count / cell.n_total if cell.n_total else 0.0
        color = OUTCOME_PALETTE.get(label, OUTCOME_PALETTE["other"])
        ax.barh(
            [0.55],
            [proportion],
            left=[left],
            height=0.25,
            color=color,
            edgecolor="white",
            hatch=OUTCOME_HATCH.get(label, ""),
        )
        left += proportion
    # Wilson interval whisker on the dominant outcome.
    if labels:
        ci = cell.outcome_ci[labels[0]]
        ax.errorbar(
            ci[0],
            0.55,
            xerr=[[ci[0] - ci[1]], [ci[2] - ci[0]]],
            fmt="none",
            ecolor="#2B2B2B",
            capsize=4,
        )
    # Exemplar marker: a star whose count the reader can resolve back to a case.
    n_ex = len(cell.exemplar_episode_ids)
    ax.text(
        0.5,
        0.18,
        f"n={cell.n_total}",
        ha="center",
        va="center",
        fontsize=9,
        color="#2B2B2B",
    )
    if n_ex:
        ax.scatter(
            [0.5],
            [0.32],
            marker="*",
            s=120,
            color="#C0392B",
            edgecolors="white",
            zorder=5,
        )
        ax.text(
            0.5, 0.40, f"{n_ex} exemplar", ha="center", va="bottom", fontsize=8, color="#C0392B"
        )


# ---------------------------------------------------------------------------
# Event-aligned ensemble context view
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """Outcome of building one event-aligned ensemble context view."""

    cell: AtlasCellSummary
    anchor: str
    status: str  # "rendered" | "unavailable"
    reason: str | None
    aligned_episode_ids: tuple[str, ...]
    medoid_episode_id: str | None
    outlier_episode_ids: tuple[str, ...]
    output_path: Path | None


def render_ensemble_context_view(
    cell_rows: Sequence[EpisodeInventoryRow],
    *,
    anchor: str,
    out: Path,
) -> EnsembleResult:
    """Render an event-aligned ensemble context view for one cell's episodes.

    Trajectories are aligned on the named event anchor (event-relative time,
    ``t_rel = t - anchor_time``). The view shows the observed medoid trajectory,
    transparent quantile bands, explicitly marked outliers, and a categorical
    planner-state/predicate sequence strip.

    Refuses to render when the named event anchor is not shared by every episode
    in the cell: it returns ``status="unavailable"`` with an explicit figure and
    output path, rather than a misleading plot.

    Returns:
        ``EnsembleResult`` with status and the output path (always written).
    """
    if not anchor:
        raise CampaignAtlasError("ensemble anchor must be a non-empty name")
    out = Path(out)
    if out.suffix.lower() not in {".svg", ".pdf"}:
        out = out.with_suffix(".svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    cell = AtlasCellSummary(
        scenario_family=cell_rows[0].scenario_family if cell_rows else "unknown",
        planner=cell_rows[0].planner if cell_rows else "unknown",
        eligible=bool(cell_rows),
        ineligible_reason=None if cell_rows else "no episodes",
        n_total=len(cell_rows),
        outcome_counts={},
        outcome_ci={},
        exemplar_episode_ids=(),
    )

    invalid_rows = _validate_ensemble_rows(cell_rows, anchor=anchor)
    if invalid_rows:
        return _render_ensemble_unavailable(
            cell,
            anchor,
            out,
            reason="invalid ensemble input: " + "; ".join(invalid_rows),
            missing=[],
        )

    aligned = _align_trajectories(cell_rows, anchor=anchor)
    if len(aligned) < 2 or any(len(points) < 2 for _, points in aligned):
        return _render_ensemble_unavailable(
            cell,
            anchor,
            out,
            reason=f"fewer than 2 episodes aligned on anchor '{anchor}'",
            missing=[],
        )

    try:
        (
            time_grid,
            median_path,
            band_radius,
            medoid_id,
            outlier_ids,
            medoid_path,
        ) = _compute_ensemble_geometry(aligned)
    except CampaignAtlasError as exc:
        return _render_ensemble_unavailable(cell, anchor, out, reason=str(exc), missing=[])
    _draw_ensemble_figure(
        out,
        cell,
        anchor,
        time_grid,
        median_path,
        band_radius,
        medoid_id,
        medoid_path,
        outlier_ids,
        aligned,
    )
    return EnsembleResult(
        cell=cell,
        anchor=anchor,
        status="rendered",
        reason=None,
        aligned_episode_ids=tuple(sorted(row.episode_id for row, _ in aligned)),
        medoid_episode_id=medoid_id,
        outlier_episode_ids=tuple(sorted(outlier_ids)),
        output_path=out,
    )


def _validate_ensemble_rows(cell_rows: Sequence[EpisodeInventoryRow], *, anchor: str) -> list[str]:
    """Return deterministic validation errors for event-aligned input rows."""
    errors: list[str] = []
    episode_ids = [row.episode_id for row in cell_rows]
    duplicate_ids = sorted(
        episode_id for episode_id, count in Counter(episode_ids).items() if count > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate episode ids: {', '.join(duplicate_ids)}")

    for row in sorted(cell_rows, key=lambda item: item.episode_id):
        row_errors = _validate_ensemble_row(row, anchor=anchor)
        if row_errors:
            errors.append(f"{row.episode_id}: {', '.join(row_errors)}")
    return errors


def _validate_ensemble_row(row: EpisodeInventoryRow, *, anchor: str) -> list[str]:
    """Return validation errors for one event-aligned inventory row."""
    errors: list[str] = []
    if anchor not in row.event_anchors:
        errors.append(f"missing anchor '{anchor}'")
    elif not math.isfinite(row.event_anchors[anchor]):
        errors.append(f"anchor '{anchor}' is non-finite")

    if len(row.trajectory) < 2:
        errors.append("trajectory needs at least two points")
    else:
        times = [point.t for point in row.trajectory]
        if any(
            not math.isfinite(value)
            for point in row.trajectory
            for value in (point.t, point.x, point.y)
        ):
            errors.append("trajectory contains non-finite values")
        elif any(next_time <= current_time for current_time, next_time in pairwise(times)):
            errors.append("trajectory times must be strictly increasing")

    for interval in row.predicate_timeline:
        if not all(math.isfinite(value) for value in (interval.t_start, interval.t_end)):
            errors.append("predicate timeline contains non-finite values")
            break
        if interval.t_end < interval.t_start:
            errors.append("predicate timeline has a reversed interval")
            break
    return errors


def _align_trajectories(
    cell_rows: Sequence[EpisodeInventoryRow], *, anchor: str
) -> list[tuple[EpisodeInventoryRow, list[tuple[float, float, float]]]]:
    """Return ``(row, [(t_rel, x, y), ...])`` for rows sharing *anchor*."""
    aligned: list[tuple[EpisodeInventoryRow, list[tuple[float, float, float]]]] = []
    for row in sorted(cell_rows, key=lambda item: item.episode_id):
        anchor_time = row.event_anchors[anchor]
        points = [(point.t - anchor_time, point.x, point.y) for point in row.trajectory]
        aligned.append((row, points))
    return aligned


def _compute_ensemble_geometry(
    aligned: list[tuple[EpisodeInventoryRow, list[tuple[float, float, float]]]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str | None,
    set[str],
    np.ndarray,
]:
    """Compute time grid, medoid path, quantile band radius, medoid id, outliers.

    Quantile bands are computed only over the shared event-relative time grid so
    they never span semantically unaligned episodes. Outliers are episodes whose
    per-time distance from the median path exceeds the median + 1.5*IQR band.
    """
    t_min = max(points[0][0] for _, points in aligned)
    t_max = min(points[-1][0] for _, points in aligned)
    if t_max <= t_min:
        raise CampaignAtlasError(
            "no common event-relative time interval exists across the aligned episodes"
        )
    time_grid = np.linspace(t_min, t_max, 120)
    xs = np.zeros((len(aligned), len(time_grid)))
    ys = np.zeros((len(aligned), len(time_grid)))
    for i, (_, points) in enumerate(aligned):
        ts = np.array([p[0] for p in points])
        xs[i] = np.interp(time_grid, ts, np.array([p[1] for p in points]))
        ys[i] = np.interp(time_grid, ts, np.array([p[2] for p in points]))
    median_x = np.median(xs, axis=0)
    median_y = np.median(ys, axis=0)
    distances = np.sqrt((xs - median_x) ** 2 + (ys - median_y) ** 2)
    band_median = np.median(distances, axis=0)
    q1 = np.percentile(distances, 25, axis=0)
    q3 = np.percentile(distances, 75, axis=0)
    band_radius = band_median + 1.5 * (q3 - q1)

    medoid_id: str | None = None
    medoid_index: int | None = None
    best_key: tuple[float, str] | None = None
    outlier_ids: set[str] = set()
    for i, (row, _) in enumerate(aligned):
        mean_dist = float(np.mean(distances[i]))
        candidate_key = (mean_dist, row.episode_id)
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            medoid_id = row.episode_id
            medoid_index = i
        if float(np.max(distances[i])) > float(np.max(band_radius)):
            outlier_ids.add(row.episode_id)
    median_path = np.column_stack((median_x, median_y))
    if medoid_index is None:
        raise CampaignAtlasError("could not select an observed medoid trajectory")
    medoid_path = np.column_stack((xs[medoid_index], ys[medoid_index]))
    return time_grid, median_path, band_radius, medoid_id, outlier_ids, medoid_path


def _draw_ensemble_figure(
    out: Path,
    cell: AtlasCellSummary,
    anchor: str,
    time_grid: np.ndarray,
    median_path: np.ndarray,
    band_radius: np.ndarray,
    medoid_id: str | None,
    medoid_path: np.ndarray,
    outlier_ids: set[str],
    aligned: list[tuple[EpisodeInventoryRow, list[tuple[float, float, float]]]],
) -> None:
    """Draw the ensemble context figure: medoid, quantile band, outliers, predicates."""
    with plt.rc_context(_ATLAS_RC):
        figure, (ax_map, ax_strip) = plt.subplots(
            2,
            1,
            figsize=(7.2, 6.0),
            gridspec_kw={"height_ratios": (4, 1.2)},
            squeeze=True,
        )
        # Quantile band as a translucent tube around the median path.
        outer = median_path + np.column_stack((band_radius, band_radius)) * np.array([1.0, 1.0])
        lower = median_path - np.column_stack((band_radius, band_radius)) * np.array([1.0, 1.0])
        band_poly = np.vstack([outer, lower[::-1]])
        ax_map.add_patch(
            Polygon(band_poly, closed=True, facecolor="#56B4E9", alpha=0.18, edgecolor="none")
        )
        # Individual aligned trajectories (faint), outliers marked red.
        for row, points in aligned:
            xs = np.array([p[1] for p in points])
            ys = np.array([p[2] for p in points])
            is_outlier = row.episode_id in outlier_ids
            ax_map.plot(
                xs,
                ys,
                color="#C0392B" if is_outlier else "#999999",
                linewidth=1.6 if is_outlier else 0.7,
                alpha=0.9 if is_outlier else 0.5,
                zorder=3 if is_outlier else 1,
            )
        # Medoid path bold.
        if medoid_id is not None:
            ax_map.plot(
                medoid_path[:, 0],
                medoid_path[:, 1],
                color="#009E73",
                linewidth=2.2,
                zorder=4,
                label=f"medoid ({medoid_id})",
            )
        ax_map.set_xlabel("x (m)")
        ax_map.set_ylabel("y (m)")
        ax_map.set_aspect("equal", adjustable="datalim")
        ax_map.set_title(
            f"Ensemble context — {cell.scenario_family} / {cell.planner}\n"
            f"aligned on event anchor '{anchor}' (event-relative time)"
        )
        ax_map.legend(loc="best", fontsize=8)

        # Categorical planner-state / predicate sequence strip.
        for row_index, (row, _) in enumerate(aligned):
            anchor_time = row.event_anchors.get(anchor, 0.0)
            for interval in row.predicate_timeline:
                t_start = interval.t_start - anchor_time
                t_end = interval.t_end - anchor_time
                color = PREDICATE_PALETTE.get(interval.label, PREDICATE_PALETTE["unknown"])
                ax_strip.add_patch(
                    Rectangle(
                        (t_start, row_index),
                        max(t_end - t_start, 1e-6),
                        1.0,
                        facecolor=color,
                        edgecolor="none",
                    )
                )
        ax_strip.set_yticks(range(len(aligned)))
        ax_strip.set_yticklabels([row.episode_id for row, _ in aligned], fontsize=8)
        ax_strip.set_xlabel("event-relative time (s)")
        ax_strip.set_ylim(0, len(aligned))
        ax_strip.set_title("planner-state / predicate sequence", fontsize=9)

        figure.tight_layout()
        _save_figure(figure, out)
        plt.close(figure)


def _render_ensemble_unavailable(
    cell: AtlasCellSummary,
    anchor: str,
    out: Path,
    *,
    reason: str,
    missing: list[str],
) -> EnsembleResult:
    """Write an explicit 'unavailable' ensemble figure instead of a misleading plot."""
    with plt.rc_context(_ATLAS_RC):
        figure, ax = plt.subplots(figsize=(7.2, 3.0))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            0.6,
            "Ensemble context view unavailable",
            ha="center",
            va="center",
            fontsize=14,
            color="#C0392B",
        )
        ax.text(
            0.5,
            0.35,
            reason,
            ha="center",
            va="center",
            fontsize=9,
            color="#666666",
            wrap=True,
        )
        ax.set_title(
            f"{cell.scenario_family} / {cell.planner} — anchor '{anchor}'",
            fontsize=10,
        )
        figure.tight_layout()
        _save_figure(figure, out)
        plt.close(figure)
    return EnsembleResult(
        cell=cell,
        anchor=anchor,
        status="unavailable",
        reason=reason,
        aligned_episode_ids=(),
        medoid_episode_id=None,
        outlier_episode_ids=(),
        output_path=out,
    )


# ---------------------------------------------------------------------------
# Optional HTML exploration atlas (Altair when available, else static table)
# ---------------------------------------------------------------------------


def _summary_parity_dict(summary: AtlasSummary) -> dict[str, Any]:
    """Return the representation-agnostic fields used for the parity check."""
    return {
        "schema_version": CAMPAIGN_ATLAS_SCHEMA_VERSION,
        "campaign_id": summary.campaign_id,
        "scenario_families": list(summary.scenario_families),
        "planners": list(summary.planners),
        "event_anchor": summary.event_anchor,
        "selection_manifest_hash": summary.selection_manifest_hash,
        "metric_definitions": dict(summary.metric_definitions),
        "cells": [
            {
                "scenario_family": cell.scenario_family,
                "planner": cell.planner,
                "eligible": cell.eligible,
                "ineligible_reason": cell.ineligible_reason,
                "n_total": cell.n_total,
                "outcome_counts": dict(cell.outcome_counts),
                "outcome_ci": {k: list(v) for k, v in cell.outcome_ci.items()},
                "exemplar_episode_ids": list(cell.exemplar_episode_ids),
            }
            for cell in summary.cells
        ],
    }


def render_html_atlas_exploration(summary: AtlasSummary, out: Path) -> Path:
    """Render an optional HTML exploration atlas carrying the same summary.

    When Altair is importable the figure is a Vega-Lite spec; otherwise a plain
    table is rendered. In both cases the normalized parity summary is embedded as
    JSON so ``check_atlas_parity`` can compare it to the static output.

    Returns:
        Path to the written HTML file.
    """
    out = Path(out)
    if out.suffix.lower() != ".html":
        out = out.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    parity = _summary_parity_dict(summary)

    cells = parity["cells"]
    altair = try_import("altair")
    if altair is not None:
        body = _render_altair_html(cells, altair)
    else:
        body = _render_table_html(cells)

    embedded = json.dumps(parity, sort_keys=True, indent=2)
    rendered_data = json.dumps({"cells": cells}, sort_keys=True, indent=2)
    html = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>Campaign atlas (exploration) — {summary.campaign_id}</title>\n"
        "</head>\n<body>\n"
        '<script id="campaign-atlas-summary" type="application/json">\n'
        f"{embedded}\n</script>\n"
        '<script id="campaign-atlas-render-data" type="application/json">\n'
        f"{rendered_data}\n</script>\n"
        f"{body}\n</body>\n</html>\n"
    )
    out.write_text(html, encoding="utf-8")
    return out


def _render_table_html(cells: Sequence[dict[str, Any]]) -> str:
    """Return a plain HTML table fallback for the exploration atlas."""
    rows_html = []
    for cell in cells:
        outcomes = ", ".join(
            f"{label}={count}" for label, count in sorted(cell["outcome_counts"].items())
        )
        rows_html.append(
            "<tr>"
            f"<td>{cell['scenario_family']}</td>"
            f"<td>{cell['planner']}</td>"
            f"<td>{'yes' if cell['eligible'] else 'no'}</td>"
            f"<td>{cell['n_total']}</td>"
            f"<td>{outcomes}</td>"
            f"<td>{', '.join(cell['exemplar_episode_ids'])}</td>"
            "</tr>"
        )
    return (
        "<h1>Campaign atlas (exploration view)</h1>\n"
        '<table border="1">\n'
        "<tr><th>scenario-family</th><th>planner</th><th>eligible</th>"
        "<th>n</th><th>outcomes</th><th>exemplars</th></tr>\n"
        f"{''.join(rows_html)}\n</table>\n"
    )


def _render_altair_html(cells: Sequence[dict[str, Any]], alt: Any) -> str:
    """Return an Altair/Vega-Lite HTML block for the exploration atlas."""
    data = [
        {
            **cell,
            "exemplars": len(cell["exemplar_episode_ids"]),
        }
        for cell in cells
    ]
    chart = (
        alt.Chart(alt.Data(values=data))
        .mark_bar()
        .encode(
            x=alt.X("planner:N"),
            y=alt.Y("n_total:Q"),
            color=alt.Color("planner:N"),
            column=alt.Column("scenario_family:N"),
        )
    )
    return chart.to_html(embed_options={"renderer": "svg"})


def check_atlas_parity(html_path: Path, summary_path: Path) -> None:
    """Fail closed if the HTML exploration data disagrees with the static summary.

    Raises:
        AtlasParityError: When the embedded HTML summary does not match the
            static atlas summary JSON.
    """
    html_text = Path(html_path).read_text(encoding="utf-8")
    match = re.search(
        r"<script id=\"campaign-atlas-summary\" type=\"application/json\">(.*?)</script>",
        html_text,
        re.DOTALL,
    )
    if match is None:
        raise AtlasParityError("HTML atlas is missing the embedded summary block")
    html_summary = json.loads(match.group(1))
    render_match = re.search(
        r"<script id=\"campaign-atlas-render-data\" type=\"application/json\">(.*?)</script>",
        html_text,
        re.DOTALL,
    )
    if render_match is None:
        raise AtlasParityError("HTML atlas is missing the rendered-data block")
    render_data = json.loads(render_match.group(1))
    static_summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

    if html_summary != static_summary:
        diff_fields = [
            key
            for key in set(html_summary) | set(static_summary)
            if html_summary.get(key) != static_summary.get(key)
        ]
        raise AtlasParityError(
            "HTML exploration atlas disagrees with static atlas summary on fields: "
            + ", ".join(sorted(diff_fields))
        )

    if render_data.get("cells") != static_summary.get("cells"):
        raise AtlasParityError("HTML exploration data disagrees with static atlas cell data")


# ---------------------------------------------------------------------------
# Full build (atlas + per-cell ensemble views + manifest binding)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AtlasBuildResult:
    """Paths and structured results from a full atlas build."""

    atlas_svg: Path
    atlas_summary_json: Path
    ensemble_views: tuple[EnsembleResult, ...]
    catalog_path: Path | None


def build_campaign_atlas(
    rows: Sequence[EpisodeInventoryRow],
    *,
    out_dir: Path,
    config: AtlasConfig,
    exemplar_episode_ids: Sequence[str] = (),
    selection_manifest_hash: str = "",
    ensemble_anchor: str | None = None,
    render_html: bool = False,
    command: str = "",
    commit: str = "",
    source_inventory: Path | None = None,
) -> AtlasBuildResult:
    """Build the campaign atlas plus per-cell ensemble views and a manifest.

    Writes the authoritative static SVG atlas, a sidecar summary JSON (the parity
    source of truth), per-cell ensemble context views, and an
    ``artifact_catalog.v1`` manifest binding campaign id, the scenario/planner/
    seed inventory, event-detector versions, the selection-manifest hash, and
    source + output content hashes.

    Returns:
        ``AtlasBuildResult`` with all written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_atlas_summary(rows, config=config, exemplar_episode_ids=exemplar_episode_ids)
    if selection_manifest_hash or ensemble_anchor is not None:
        summary = AtlasSummary(
            campaign_id=summary.campaign_id,
            scenario_families=summary.scenario_families,
            planners=summary.planners,
            cells=summary.cells,
            metric_definitions=summary.metric_definitions,
            event_anchor=ensemble_anchor,
            selection_manifest_hash=selection_manifest_hash or summary.selection_manifest_hash,
        )

    atlas_svg = render_atlas_figure(summary, out_dir / "campaign_atlas.svg")
    atlas_summary_json = out_dir / "campaign_atlas_summary.json"
    atlas_summary_json.write_text(
        json.dumps(_summary_parity_dict(summary), sort_keys=True, indent=2),
        encoding="utf-8",
    )

    ensemble_views: list[EnsembleResult] = []
    if ensemble_anchor is not None:
        cells = _group_rows_by_cell(rows)
        eligible_cells = {
            (cell.scenario_family, cell.planner) for cell in summary.cells if cell.eligible
        }
        for (scenario_family, planner), cell_rows in sorted(cells.items()):
            if (scenario_family, planner) not in eligible_cells:
                continue
            result = render_ensemble_context_view(
                cell_rows,
                anchor=ensemble_anchor,
                out=out_dir / f"ensemble__{scenario_family}__{planner}.svg",
            )
            ensemble_views.append(result)

    if render_html:
        render_html_atlas_exploration(summary, out_dir / "campaign_atlas_exploration.html")

    catalog_path = _write_atlas_catalog(
        out_dir,
        summary,
        rows=rows,
        atlas_svg=atlas_svg,
        atlas_summary_json=atlas_summary_json,
        ensemble_views=tuple(ensemble_views),
        source_inventory=source_inventory,
        command=command,
        commit=commit,
    )
    return AtlasBuildResult(
        atlas_svg=atlas_svg,
        atlas_summary_json=atlas_summary_json,
        ensemble_views=tuple(ensemble_views),
        catalog_path=catalog_path,
    )


def _group_rows_by_cell(
    rows: Sequence[EpisodeInventoryRow],
) -> dict[tuple[str, str], list[EpisodeInventoryRow]]:
    """Group inventory rows by (scenario-family, planner) cell."""
    grouped: dict[tuple[str, str], list[EpisodeInventoryRow]] = {}
    for row in rows:
        grouped.setdefault((row.scenario_family, row.planner), []).append(row)
    return grouped


def _write_atlas_catalog(
    out_dir: Path,
    summary: AtlasSummary,
    *,
    rows: Sequence[EpisodeInventoryRow],
    source_inventory: Path | None,
    atlas_svg: Path,
    atlas_summary_json: Path,
    ensemble_views: tuple[EnsembleResult, ...],
    command: str,
    commit: str,
) -> Path:
    """Write an ``artifact_catalog.v1`` manifest binding the atlas artifacts."""
    from robot_sf.benchmark.artifact_catalog import (  # noqa: PLC0415
        ARTIFACT_CATALOG_SCHEMA_VERSION,
    )

    seed_inventory = [
        {
            "episode_id": row.episode_id,
            "scenario_id": row.scenario_id,
            "scenario_family": row.scenario_family,
            "planner": row.planner,
            "seed": row.seed,
        }
        for row in sorted(
            rows,
            key=lambda item: (
                item.scenario_family,
                item.planner,
                item.scenario_id,
                item.seed,
                item.episode_id,
            ),
        )
    ]
    provenance = {
        "campaign_id": summary.campaign_id,
        "scenario_families": list(summary.scenario_families),
        "planners": list(summary.planners),
        "seed_inventory": seed_inventory,
        "event_anchor": summary.event_anchor,
        "event_detector_version": EVENT_DETECTOR_VERSION,
        "renderer_version": RENDERER_VERSION,
        "style_version": STYLE_VERSION,
        "selection_manifest_hash": summary.selection_manifest_hash,
        "metric_definitions": dict(summary.metric_definitions),
    }
    provenance_path = out_dir / "campaign_atlas_provenance.json"
    provenance_path.write_text(json.dumps(provenance, sort_keys=True, indent=2), encoding="utf-8")
    caption_path = out_dir / "campaign_atlas_caption.md"
    caption_path.write_text(
        "# Campaign atlas figures\n\n"
        f"Campaign: `{summary.campaign_id}`.\n\n"
        "Claim boundary: diagnostic tooling output only; these figures are not benchmark-success "
        "or paper-facing evidence. Outcome labels and event anchors are consumed from the "
        "versioned campaign inventory.\n\n"
        f"Event anchor: `{summary.event_anchor or 'not requested'}`.\n",
        encoding="utf-8",
    )

    source_files = [
        {
            "path": atlas_summary_json.name,
            "sha256": _sha256_file(atlas_summary_json),
        },
        {
            "path": provenance_path.name,
            "sha256": _sha256_file(provenance_path),
        },
    ]
    if source_inventory is not None:
        source_inventory = Path(source_inventory)
        if not source_inventory.is_file():
            raise CampaignAtlasError(f"source inventory does not exist: {source_inventory}")
        source_files.append(
            {
                "path": source_inventory.as_posix(),
                "sha256": _sha256_file(source_inventory),
            }
        )

    artifacts: list[dict[str, Any]] = [
        {
            "artifact_id": "campaign_atlas",
            "artifact_kind": "figure",
            "source_kind": "campaign_inventory",
            "source_files": source_files,
            "outputs": {"svg": {"path": atlas_svg.name, "sha256": _sha256_file(atlas_svg)}},
            "generation_command": command,
            "generation_commit": commit,
            "claim_boundary": "diagnostic_only",
            "caption_file": {
                "path": caption_path.name,
                "sha256": _sha256_file(caption_path),
            },
        }
    ]
    for view in ensemble_views:
        if view.output_path is None:
            continue
        artifacts.append(
            {
                "artifact_id": f"ensemble_{view.cell.scenario_family}_{view.cell.planner}",
                "artifact_kind": "figure",
                "source_kind": "campaign_inventory",
                "source_files": source_files,
                "outputs": {
                    "svg": {
                        "path": view.output_path.name,
                        "sha256": _sha256_file(view.output_path),
                    }
                },
                "generation_command": command,
                "generation_commit": commit,
                "claim_boundary": "diagnostic_only",
                "caption_file": {
                    "path": caption_path.name,
                    "sha256": _sha256_file(caption_path),
                },
            }
        )

    catalog = {
        "schema_version": ARTIFACT_CATALOG_SCHEMA_VERSION,
        "catalog_id": f"campaign_atlas_{summary.campaign_id}",
        "artifacts": artifacts,
    }
    catalog_path = out_dir / "campaign_atlas_catalog.yaml"
    import yaml  # noqa: PLC0415

    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=True), encoding="utf-8")
    return catalog_path


__all__ = [
    "CAMPAIGN_ATLAS_SCHEMA_VERSION",
    "AtlasCellSummary",
    "AtlasConfig",
    "AtlasParityError",
    "AtlasSummary",
    "CampaignAtlasError",
    "EnsembleResult",
    "EpisodeInventoryRow",
    "PredicateInterval",
    "TrajectoryPoint",
    "build_atlas_summary",
    "build_campaign_atlas",
    "check_atlas_parity",
    "render_atlas_figure",
    "render_ensemble_context_view",
    "render_html_atlas_exploration",
]
