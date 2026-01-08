#!/usr/bin/env python3
"""Preview single-pedestrian trajectories on top of scenario map geometry.

Usage examples:
  uv run python scripts/tools/preview_scenario_trajectories.py \
    --scenario configs/scenarios/classic_interactions.yaml \
    --scenario-id classic_head_on_corridor

  uv run python scripts/tools/preview_scenario_trajectories.py \
    --scenario configs/scenarios/francis2023.yaml \
    --all

  uv run python scripts/tools/preview_scenario_trajectories.py \
    --scenario configs/scenarios/classic_interactions.yaml \
    --scenario-id classic_head_on_corridor \
    --ped ped_1 --ped ped_2 \
    --show
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

import matplotlib.pyplot as plt
from loguru import logger

from robot_sf.common.logging import configure_logging
from robot_sf.maps import create_map_figure, render_map_definition
from robot_sf.nav.map_config import SinglePedestrianDefinition
from robot_sf.training.scenario_loader import (
    apply_single_pedestrian_overrides,
    load_scenarios,
    resolve_map_definition,
    select_scenario,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to the scenario YAML file.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        help="Scenario name or id to preview (defaults to the first entry).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render previews for every scenario in the file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output PNG path (defaults to output/preview/scenario_trajectories/...).",
    )
    parser.add_argument(
        "--style",
        choices=["clean", "full"],
        default="clean",
        help="Map render style: 'clean' hides routes/POIs/zone labels (default).",
    )
    parser.add_argument(
        "--legend",
        choices=["none", "inside", "outside"],
        default="none",
        help="Legend placement for overlay items (default: none).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the preview interactively after rendering.",
    )
    parser.add_argument(
        "--ped",
        action="append",
        default=[],
        help="Filter by pedestrian id (repeat or comma-separated).",
    )
    return parser


def _slugify(value: str) -> str:
    """Convert a string into a filesystem-friendly slug."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "scenario_preview"


def _default_output_path(scenario_name: str) -> Path:
    """Generate a default output path under the canonical output root."""
    slug = _slugify(scenario_name)
    return Path("output/preview/scenario_trajectories") / f"{slug}.png"


def _parse_ped_ids(values: list[str]) -> list[str]:
    """Parse a list of --ped inputs into a list of ids."""
    ids: list[str] = []
    for value in values:
        for entry in value.split(","):
            entry = entry.strip()
            if entry:
                ids.append(entry)
    return ids


def _filter_pedestrians(
    pedestrians: list[SinglePedestrianDefinition],
    requested_ids: list[str],
) -> list[SinglePedestrianDefinition]:
    """Filter pedestrians by id, preserving requested order when provided."""
    if not requested_ids:
        return pedestrians
    ped_by_id = {ped.id: ped for ped in pedestrians}
    missing = [pid for pid in requested_ids if pid not in ped_by_id]
    if missing:
        raise ValueError(f"Requested pedestrian id(s) not found: {', '.join(missing)}")
    return [ped_by_id[pid] for pid in requested_ids]


def _annotation_text(ped: SinglePedestrianDefinition) -> str:
    """Build annotation text for a pedestrian."""
    parts = [ped.id]
    if ped.speed_m_s is not None:
        parts.append(f"{ped.speed_m_s:.2f} m/s")
    if ped.note:
        parts.append(ped.note)
    return " | ".join(parts)


def _plot_single_pedestrians(
    ax,
    pedestrians: list[SinglePedestrianDefinition],
) -> None:
    """Plot single-pedestrian starts, goals, and trajectories."""
    if not pedestrians:
        return
    cmap = plt.get_cmap("tab10")

    for idx, ped in enumerate(pedestrians):
        color = cmap(idx % cmap.N)
        start_x, start_y = ped.start
        ax.scatter(
            [start_x],
            [start_y],
            color=color,
            marker="o",
            s=60,
            label=f"{ped.id} start",
            zorder=5,
        )
        ax.text(
            start_x,
            start_y,
            _annotation_text(ped),
            color=color,
            fontsize=8,
            ha="left",
            va="bottom",
        )

        if ped.goal is not None:
            goal_x, goal_y = ped.goal
            ax.scatter(
                [goal_x],
                [goal_y],
                color=color,
                marker="X",
                s=70,
                label=f"{ped.id} goal",
                zorder=5,
            )
            ax.plot(
                [start_x, goal_x],
                [start_y, goal_y],
                color=color,
                linestyle=":",
                linewidth=1.0,
                label=f"{ped.id} goal path",
            )

        if ped.trajectory:
            points = [ped.start, *ped.trajectory]
            xs, ys = zip(*points, strict=False)
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle="--",
                linewidth=1.5,
                marker="o",
                markersize=3,
                label=f"{ped.id} trajectory",
            )

        wait_label_added = False
        for rule in ped.wait_at or []:
            if not ped.trajectory:
                continue
            waypoint = ped.trajectory[rule.waypoint_index]
            label = f"{ped.id} wait" if not wait_label_added else None
            ax.scatter(
                [waypoint[0]],
                [waypoint[1]],
                marker="s",
                s=70,
                facecolor="none",
                edgecolor=color,
                linewidths=1.5,
                label=label,
                zorder=6,
            )
            wait_text = f"wait {rule.wait_s:g}s"
            if rule.note:
                wait_text = f"{wait_text} | {rule.note}"
            ax.text(
                waypoint[0],
                waypoint[1],
                wait_text,
                color=color,
                fontsize=7,
                ha="left",
                va="top",
            )
            wait_label_added = True


def _add_legend(ax, *, style: str) -> None:
    """Add a legend for overlay elements based on the requested style."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    by_label = dict(zip(labels, handles, strict=False))
    if style == "inside":
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    elif style == "outside":
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
        )


def _resolve_output_path(
    scenario_name: str,
    *,
    output_root: Path | None,
    multi: bool,
) -> Path:
    """Resolve the output path for a scenario preview.

    Returns:
        Path: Output path for the rendered preview.
    """
    if output_root is None:
        return _default_output_path(scenario_name)
    if multi:
        return output_root / f"{_slugify(scenario_name)}.png"
    return output_root


def _render_preview(
    scenario: Mapping[str, Any],
    *,
    scenario_path: Path,
    args: argparse.Namespace,
    output_root: Path | None,
    multi: bool,
) -> None:
    """Render and save a preview image for a single scenario."""
    map_file = scenario.get("map_file")
    if not map_file:
        raise ValueError("Scenario entry missing required map_file")
    map_def = resolve_map_definition(map_file, scenario_path=scenario_path)
    if map_def is None:
        raise ValueError(f"Failed to load map definition from '{map_file}'")

    apply_single_pedestrian_overrides(map_def, scenario.get("single_pedestrians"))

    pedestrians = map_def.single_pedestrians or []
    if not pedestrians:
        logger.warning("No single pedestrians defined in map or overrides.")

    requested_ids = _parse_ped_ids(args.ped)
    pedestrians = _filter_pedestrians(pedestrians, requested_ids)

    scenario_name = str(scenario.get("name") or scenario.get("scenario_id") or Path(map_file).stem)

    fig, ax = create_map_figure(map_def)
    use_clean = args.style == "clean"
    render_map_definition(
        map_def,
        ax,
        title=scenario_name,
        show_legend=False,
        show_zone_labels=not use_clean,
        show_routes=not use_clean,
        show_pois=not use_clean,
    )
    _plot_single_pedestrians(ax, pedestrians)
    if args.legend != "none":
        _add_legend(ax, style=args.legend)

    output_path = _resolve_output_path(
        scenario_name,
        output_root=output_root,
        multi=multi,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    logger.info("Saved scenario preview to {}", output_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for scenario trajectory previews."""
    args = _build_parser().parse_args(argv)
    configure_logging()

    scenario_path = args.scenario.resolve()
    scenarios = load_scenarios(scenario_path)
    if args.all and args.scenario_id:
        raise ValueError("Use --all or --scenario-id, not both.")

    output_root = args.output
    if args.all and output_root is not None and output_root.suffix:
        raise ValueError("--output must be a directory when using --all")

    if args.all:
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
        if args.show:
            logger.warning("Showing all previews will open multiple windows.")
        for scenario in scenarios:
            _render_preview(
                scenario,
                scenario_path=scenario_path,
                args=args,
                output_root=output_root,
                multi=True,
            )
    else:
        scenario = select_scenario(scenarios, args.scenario_id)
        _render_preview(
            scenario,
            scenario_path=scenario_path,
            args=args,
            output_root=output_root,
            multi=False,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
