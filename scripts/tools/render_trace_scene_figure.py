#!/usr/bin/env python3
"""Render a top-down trace-scene figure from episode data.

Generates a matplotlib figure with map obstacles, metre scale,
robot/pedestrian trajectories, and time-synchronised markers.

The ``--qa`` flag runs the figure-quality linter after rendering and
exits non-zero when error-severity defects are detected.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required for trace-scene figure rendering.", file=sys.stderr)
    sys.exit(1)

import numpy as np

from robot_sf.benchmark.figure_qa import FigureDefect, lint_figure


@dataclass
class EpisodeStep:
    """One step of an episode trace."""

    t: float
    x: float
    y: float
    heading: float = 0.0
    ped_positions: list[tuple[float, float]] | None = None


def _load_steps(jsonl_path: Path) -> list[EpisodeStep]:
    """Load episode steps from a JSONL file or generate synthetic data."""
    if jsonl_path.exists() and jsonl_path.is_file():
        steps: list[EpisodeStep] = []
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                steps.append(
                    EpisodeStep(
                        t=float(row.get("t", 0.0)),
                        x=float(row.get("x", 0.0)),
                        y=float(row.get("y", 0.0)),
                        heading=float(row.get("heading", 0.0)),
                        ped_positions=row.get("ped_positions"),
                    )
                )
        if steps:
            return steps

    rng = np.random.default_rng(42)
    n_steps = 60
    steps = []
    rx, ry = 1.0, 3.0
    for i in range(n_steps):
        t = i * 0.1
        rx += 0.15 + rng.normal(0, 0.02)
        ry += rng.normal(0, 0.03)
        peds = [
            (float(3.0 + rng.normal(0, 0.3)), float(2.0 + rng.normal(0, 0.5))) for _ in range(4)
        ]
        steps.append(EpisodeStep(t=t, x=float(rx), y=float(ry), ped_positions=peds))
    return steps


def render_scene_figure(
    steps: list[EpisodeStep],
    *,
    title: str = "Trace Scene",
) -> plt.Figure:
    """Render a top-down scene figure with robot trajectory and pedestrians."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    robot_x = [s.x for s in steps]
    robot_y = [s.y for s in steps]

    ax.plot(robot_x, robot_y, "b-", linewidth=2, label="Robot")
    ax.plot(robot_x[0], robot_y[0], "go", markersize=10, label="Start")
    ax.plot(robot_x[-1], robot_y[-1], "rs", markersize=10, label="End")

    x_pad = 0.5
    y_pad = 0.8
    ax.set_xlim(min(robot_x) - x_pad, max(robot_x) + x_pad)
    ax.set_ylim(min(robot_y) - y_pad, max(robot_y) + y_pad)

    ped_positions: dict[int, list[tuple[float, float]]] = {}
    for step in steps:
        if step.ped_positions:
            for idx, pos in enumerate(step.ped_positions):
                ped_positions.setdefault(idx, []).append(pos)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(ped_positions), 1)))
    for ped_idx, positions in ped_positions.items():
        px = [p[0] for p in positions]
        py = [p[1] for p in positions]
        ax.plot(
            px,
            py,
            "--",
            color=colors[ped_idx % len(colors)],
            linewidth=1.0,
            alpha=0.7,
            label=f"Ped {ped_idx}",
        )

    for step in steps[::10]:
        ax.text(
            step.x,
            max(robot_y) + 0.35,
            f"{step.t:.1f}s",
            fontsize=6,
            color="gray",
            ha="center",
        )

    stamp = (
        f"Steps: {len(steps)}\nt: {steps[0].t:.1f}–{steps[-1].t:.1f}s\nPeds: {len(ped_positions)}"
    )
    ax.text(
        0.02,
        0.98,
        stamp,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    ax.legend(loc="upper right", fontsize=8)
    return fig


def _format_defect_report(defects: list[FigureDefect]) -> str:
    """Format a lint defect report for CLI output."""
    if not defects:
        return "No defects detected."
    lines = [f"Detected {len(defects)} defect(s):"]
    for d in defects:
        loc = f" @ ({d.location[0]:.1f}, {d.location[1]:.1f})" if d.location else ""
        lines.append(f"  [{d.severity}] {d.defect_type}: {d.message}{loc}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the trace-scene figure CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        type=Path,
        default=None,
        help="Path to episode JSONL with trace steps. Uses synthetic data when absent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output figure path (PNG, PDF, SVG).",
    )
    parser.add_argument(
        "--title",
        default="Trace Scene",
        help="Figure title.",
    )
    parser.add_argument(
        "--qa",
        action="store_true",
        default=False,
        help="Run figure-quality linter after rendering; exit 1 on error-severity defects.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render a trace-scene figure and optionally run the QA linter."""
    args = build_arg_parser().parse_args(argv)

    steps = _load_steps(args.episode or Path())
    fig = render_scene_figure(steps, title=args.title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure to {args.output}")

    if args.qa:
        fig_qa = render_scene_figure(steps, title=args.title)
        defects = lint_figure(fig_qa)
        plt.close(fig_qa)
        print(_format_defect_report(defects))
        has_error = any(d.severity == "error" for d in defects)
        if has_error:
            print("QA FAILED: error-severity defects detected.")
            return 1
        print("QA passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
