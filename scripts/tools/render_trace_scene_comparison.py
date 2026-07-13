#!/usr/bin/env python3
"""Render print-ready trace-scene figures from one or two exemplar episode directories.

Complements the single-scene renderer in ``render_trace_scene_figure.py``: this tool drives
``robot_sf.benchmark.trace_scene_figure``, which adds a same-scenario **comparison mode** (two
episodes, shared extents), an attention-hierarchy palette (muted crowd + highlighted focal
pedestrian), proxemic reference lines on the coupled distance/speed timeline, and adaptive
marker-label decluttering. Pass one episode directory for a single scene, two for a comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from robot_sf.benchmark.figure_qa import lint_figure
from robot_sf.benchmark.trace_scene_figure import (
    TraceSchemaError,
    load_episode,
    render_comparison,
    render_scene,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dirs", nargs="+", type=Path, help="one or two episode directories")
    parser.add_argument("-o", "--out", type=Path, required=True, help="output PDF path")
    parser.add_argument(
        "--marker-interval",
        type=float,
        default=None,
        help="marker interval in seconds (default: 2, or 4 for episodes longer than 25 s)",
    )
    parser.add_argument(
        "--ped-radius", type=float, default=8.0, help="pedestrian focus radius in metres"
    )
    parser.add_argument(
        "--no-highlight-focal",
        action="store_true",
        help="use the legacy per-pedestrian color cycle",
    )
    parser.add_argument(
        "--collision-envelope",
        type=float,
        default=1.4,
        help="collision-envelope reference distance in metres",
    )
    parser.add_argument(
        "--comfort-distance",
        type=float,
        default=1.2,
        help="personal-space reference distance in metres",
    )
    parser.add_argument("--no-timeline", action="store_true", help="omit timeline panels")
    parser.add_argument("--png", action="store_true", help="also render a PNG beside the PDF")
    parser.add_argument(
        "--qa",
        action="store_true",
        help="lint rendered artists and fail on error-severity defects",
    )
    return parser


def _output_paths(out: Path, include_png: bool) -> tuple[Path, Path | None]:
    if out.suffix.lower() == ".png":
        pdf = out.with_suffix(".pdf")
    elif out.suffix:
        pdf = out
    else:
        pdf = out.with_suffix(".pdf")
    png = pdf.with_suffix(".png") if include_png else None
    return pdf, png


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, render the requested figure, and return an exit code."""

    args = _parser().parse_args(argv)
    if len(args.episode_dirs) not in {1, 2}:
        print("error: provide exactly one episode directory or two for comparison", file=sys.stderr)
        return 2

    try:
        episodes = [load_episode(path) for path in args.episode_dirs]
        pdf_out, png_out = _output_paths(args.out, args.png)
        render = render_scene if len(episodes) == 1 else render_comparison
        subject = episodes[0] if len(episodes) == 1 else episodes
        render_result = render(
            subject,
            pdf_out,
            marker_interval_s=args.marker_interval,
            ped_focus_radius_m=args.ped_radius,
            highlight_focal=not args.no_highlight_focal,
            collision_envelope_m=args.collision_envelope,
            comfort_distance_m=args.comfort_distance,
            timeline=not args.no_timeline,
            return_figure=args.qa,
        )
        if args.qa:
            if not isinstance(render_result, tuple):
                raise RuntimeError("renderer did not return a Figure for QA")
            _, figure = render_result
            defects = lint_figure(figure)
            if defects:
                for defect in defects:
                    print(f"{defect.severity.upper()} {defect.kind}: {defect.message}")
            else:
                print("QA clean: no defects")
            if png_out is not None:
                figure.savefig(png_out, dpi=300, bbox_inches="tight")
            plt.close(figure)
            if any(defect.severity == "error" for defect in defects):
                return 1
        elif png_out is not None:
            render(
                subject,
                png_out,
                marker_interval_s=args.marker_interval,
                ped_focus_radius_m=args.ped_radius,
                highlight_focal=not args.no_highlight_focal,
                collision_envelope_m=args.collision_envelope,
                comfort_distance_m=args.comfort_distance,
                timeline=not args.no_timeline,
            )
    except (TraceSchemaError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
