#!/usr/bin/env python3
"""Convert a staged SocNavBench traversible pickle into a Robot SF SVG map.

This is the narrow issue #1134 conversion path: it consumes the official
``traversibles/ETH/data.pkl`` staging contract and produces a reviewable SVG
map artifact. It never downloads data and it refuses to write a placeholder map
when the staged source pickle is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from scripts.tools.manage_external_data import resolve_asset_local_path_by_id

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "maps" / "svg_maps" / "socnavbench" / "socnavbench_eth.svg"
EXIT_BLOCKED = 2
SOCNAV_ASSET_ID = "socnavbench-s3dis-eth"
ETH_TRAVERSIBLE_RELATIVE_PATH = (
    Path("sd3dis") / "stanford_building_parser_dataset" / "traversibles" / "ETH" / "data.pkl"
)


@dataclass(frozen=True)
class GridPoint:
    """A traversible grid point in row/column coordinates."""

    row: int
    col: int


@dataclass(frozen=True)
class RectRun:
    """Axis-aligned run of obstacle cells in half-open grid coordinates."""

    row_start: int
    row_end: int
    col_start: int
    col_end: int


@dataclass(frozen=True)
class TraversibleMap:
    """Validated SocNavBench traversible payload ready for SVG conversion."""

    resolution: float
    traversible: np.ndarray
    source_path: Path

    @property
    def cell_size(self) -> float:
        """Return cell edge length in map units."""

        return 1.0 / self.resolution

    @property
    def width(self) -> float:
        """Return map width in Robot SF SVG units."""

        return float(self.traversible.shape[1]) * self.cell_size

    @property
    def height(self) -> float:
        """Return map height in Robot SF SVG units."""

        return float(self.traversible.shape[0]) * self.cell_size


class SocNavBenchConversionError(RuntimeError):
    """Raised when staged SocNavBench data cannot be converted safely."""


def sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_traversible(path: Path) -> TraversibleMap:
    """Load and validate a SocNavBench traversible ``data.pkl`` file."""

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:  # pragma: no cover - exact pickle failure varies.
        raise SocNavBenchConversionError(f"Could not load traversible pickle: {path}") from exc

    if not isinstance(payload, dict):
        raise SocNavBenchConversionError("Traversible pickle must contain a mapping payload.")
    if "resolution" not in payload or "traversible" not in payload:
        raise SocNavBenchConversionError(
            "Traversible pickle missing required keys: resolution, traversible."
        )

    resolution = float(payload["resolution"])
    if not np.isfinite(resolution) or resolution <= 0:
        raise SocNavBenchConversionError("Traversible resolution must be finite and positive.")

    traversible = np.asarray(payload["traversible"])
    if traversible.ndim != 2 or 0 in traversible.shape:
        raise SocNavBenchConversionError("Traversible array must be non-empty and two-dimensional.")
    if np.issubdtype(traversible.dtype, np.number) and not np.isfinite(traversible).all():
        raise SocNavBenchConversionError("Traversible array contains non-finite values.")

    free_mask = traversible.astype(bool)
    if not free_mask.any():
        raise SocNavBenchConversionError("Traversible array contains no free cells.")
    return TraversibleMap(resolution=resolution, traversible=free_mask, source_path=path)


def _neighbors(point: GridPoint, shape: tuple[int, int]) -> list[GridPoint]:
    """Return four-connected in-bounds neighbors."""

    rows, cols = shape
    candidates = (
        GridPoint(point.row - 1, point.col),
        GridPoint(point.row + 1, point.col),
        GridPoint(point.row, point.col - 1),
        GridPoint(point.row, point.col + 1),
    )
    return [
        candidate
        for candidate in candidates
        if 0 <= candidate.row < rows and 0 <= candidate.col < cols
    ]


def largest_component(mask: np.ndarray) -> set[GridPoint]:
    """Return the largest four-connected free-space component."""

    seen: set[GridPoint] = set()
    best: set[GridPoint] = set()
    rows, cols = mask.shape
    for row in range(rows):
        for col in range(cols):
            start = GridPoint(row, col)
            if not mask[row, col] or start in seen:
                continue
            component: set[GridPoint] = set()
            queue: deque[GridPoint] = deque([start])
            seen.add(start)
            while queue:
                point = queue.popleft()
                component.add(point)
                for neighbor in _neighbors(point, mask.shape):
                    if mask[neighbor.row, neighbor.col] and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            if len(component) > len(best):
                best = component
    if not best:
        raise SocNavBenchConversionError("No connected traversible component found.")
    return best


def select_route_endpoints(component: set[GridPoint]) -> tuple[GridPoint, GridPoint]:
    """Select deterministic left-to-right endpoints inside a free-space component."""

    ordered = sorted(component, key=lambda point: (point.col, point.row))
    start_col = ordered[0].col
    goal_col = ordered[-1].col
    start_candidates = [point for point in ordered if point.col == start_col]
    goal_candidates = [point for point in ordered if point.col == goal_col]
    median_row = sorted(point.row for point in component)[len(component) // 2]
    start = min(start_candidates, key=lambda point: (abs(point.row - median_row), point.row))
    goal = min(goal_candidates, key=lambda point: (abs(point.row - median_row), point.row))
    if start == goal:
        raise SocNavBenchConversionError("Largest traversible component is too small for a route.")
    return start, goal


def shortest_path(mask: np.ndarray, start: GridPoint, goal: GridPoint) -> list[GridPoint]:
    """Return a deterministic four-connected shortest path between two free cells."""

    queue: deque[GridPoint] = deque([start])
    previous: dict[GridPoint, GridPoint | None] = {start: None}
    while queue:
        point = queue.popleft()
        if point == goal:
            break
        for neighbor in _neighbors(point, mask.shape):
            if mask[neighbor.row, neighbor.col] and neighbor not in previous:
                previous[neighbor] = point
                queue.append(neighbor)
    if goal not in previous:
        raise SocNavBenchConversionError("No traversible path connects selected route endpoints.")

    path: list[GridPoint] = []
    cursor: GridPoint | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    return list(reversed(path))


def simplify_path(path: list[GridPoint]) -> list[GridPoint]:
    """Drop intermediate collinear grid points from a route path."""

    if len(path) <= 2:
        return path
    simplified = [path[0]]
    prev_direction: tuple[int, int] | None = None
    for left, right in pairwise(path):
        direction = (right.row - left.row, right.col - left.col)
        if prev_direction is not None and direction != prev_direction:
            simplified.append(left)
        prev_direction = direction
    simplified.append(path[-1])
    return simplified


def obstacle_runs(mask: np.ndarray) -> list[RectRun]:
    """Merge non-traversible cells into row-aligned rectangular obstacle runs."""

    active: dict[tuple[int, int], RectRun] = {}
    finished: list[RectRun] = []
    for row in range(mask.shape[0]):
        row_runs: list[tuple[int, int]] = []
        col = 0
        while col < mask.shape[1]:
            if mask[row, col]:
                col += 1
                continue
            start_col = col
            while col < mask.shape[1] and not mask[row, col]:
                col += 1
            row_runs.append((start_col, col))

        next_active: dict[tuple[int, int], RectRun] = {}
        for run in row_runs:
            previous = active.pop(run, None)
            if previous is None:
                next_active[run] = RectRun(row, row + 1, run[0], run[1])
            else:
                next_active[run] = RectRun(
                    previous.row_start, row + 1, previous.col_start, previous.col_end
                )
        finished.extend(active.values())
        active = next_active
    finished.extend(active.values())
    return finished


def grid_center(point: GridPoint, cell_size: float) -> tuple[float, float]:
    """Return SVG-space center coordinate for a grid cell."""

    return ((point.col + 0.5) * cell_size, (point.row + 0.5) * cell_size)


def _format_float(value: float) -> str:
    """Format floats compactly while keeping SVG stable."""

    return f"{value:.6f}".rstrip("0").rstrip(".")


def _path_d(points: list[GridPoint], cell_size: float) -> str:
    """Return SVG path data from grid points."""

    coords = [grid_center(point, cell_size) for point in points]
    head = coords[0]
    tail = coords[1:]
    parts = [f"M {_format_float(head[0])},{_format_float(head[1])}"]
    parts.extend(f"L {_format_float(x)},{_format_float(y)}" for x, y in tail)
    return " ".join(parts)


def _zone_rect(point: GridPoint, cell_size: float) -> tuple[float, float, float, float]:
    """Return a one-cell SVG rect centered on ``point``."""

    center_x, center_y = grid_center(point, cell_size)
    return center_x - cell_size / 2.0, center_y - cell_size / 2.0, cell_size, cell_size


def render_svg(
    map_data: TraversibleMap, *, map_id: str = "socnavbench_eth"
) -> tuple[str, dict[str, Any]]:
    """Render traversible data to SVG text and conversion metadata."""

    component = largest_component(map_data.traversible)
    start, goal = select_route_endpoints(component)
    route = simplify_path(shortest_path(map_data.traversible, start, goal))
    runs = obstacle_runs(map_data.traversible)
    cell = map_data.cell_size

    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        (
            f'<svg width="{_format_float(map_data.width)}" height="{_format_float(map_data.height)}" '
            f'viewBox="0 0 {_format_float(map_data.width)} {_format_float(map_data.height)}" '
            f'version="1.1" id="svg_{escape(map_id)}" '
            'xmlns="http://www.w3.org/2000/svg" '
            'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">'
        ),
        f'  <g id="{escape(map_id)}" inkscape:label="SocNavBench ETH converted map">',
        (
            "    <!-- Generated from SocNavBench traversible data.pkl; "
            "raw staged data remains outside git. -->"
        ),
    ]

    for idx, run in enumerate(runs):
        x = run.col_start * cell
        y = run.row_start * cell
        width = (run.col_end - run.col_start) * cell
        height = (run.row_end - run.row_start) * cell
        lines.append(
            "    "
            f'<rect id="obstacle_{idx}" x="{_format_float(x)}" y="{_format_float(y)}" '
            f'width="{_format_float(width)}" height="{_format_float(height)}" '
            'style="fill:#000000" inkscape:label="obstacle" />'
        )

    zone_specs = (
        ("robot_spawn_zone_0", start, "#ffdf00"),
        ("robot_goal_zone_0", goal, "#ff6c00"),
        ("ped_spawn_zone_0", goal, "#29a35a"),
        ("ped_goal_zone_0", start, "#246fe0"),
    )
    for label, point, fill in zone_specs:
        x, y, width, height = _zone_rect(point, cell)
        lines.append(
            "    "
            f'<rect id="{label}" x="{_format_float(x)}" y="{_format_float(y)}" '
            f'width="{_format_float(width)}" height="{_format_float(height)}" '
            f'style="fill:{fill}" inkscape:label="{label}" />'
        )

    robot_path = _path_d(route, cell)
    ped_path = _path_d(list(reversed(route)), cell)
    lines.extend(
        [
            (
                f'    <path id="robot_route_0_0" d="{escape(robot_path)}" '
                'style="fill:none;stroke:#0310b4;stroke-width:0.08" '
                'inkscape:label="robot_route_0_0" />'
            ),
            (
                f'    <path id="ped_route_0_0" d="{escape(ped_path)}" '
                'style="fill:none;stroke:#00a05a;stroke-width:0.08" '
                'inkscape:label="ped_route_0_0" />'
            ),
            "  </g>",
            "</svg>",
            "",
        ]
    )
    metadata = {
        "map_id": map_id,
        "source_path": str(map_data.source_path),
        "resolution": map_data.resolution,
        "traversible_shape": list(map_data.traversible.shape),
        "map_width": map_data.width,
        "map_height": map_data.height,
        "free_cell_count": int(map_data.traversible.sum()),
        "obstacle_rect_count": len(runs),
        "route_point_count": len(route),
        "route_start_cell": [start.row, start.col],
        "route_goal_cell": [goal.row, goal.col],
    }
    return "\n".join(lines), metadata


def resolve_source(args: argparse.Namespace) -> Path:
    """Resolve the input traversible pickle path from CLI arguments."""

    if args.input_pkl is not None:
        return args.input_pkl.expanduser().resolve()
    socnav_root = (
        args.socnav_root.expanduser().resolve()
        if args.socnav_root is not None
        else resolve_asset_local_path_by_id(SOCNAV_ASSET_ID).expanduser().resolve()
    )
    return socnav_root / ETH_TRAVERSIBLE_RELATIVE_PATH


def write_report(path: Path | None, report: dict[str, Any]) -> None:
    """Write optional JSON report."""

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socnav-root", type=Path, default=None)
    parser.add_argument("--input-pkl", type=Path, default=None)
    parser.add_argument("--output-svg", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--map-id", default="socnavbench_eth")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-obstacle-rects",
        type=int,
        default=50000,
        help="Fail closed if conversion would emit more obstacle rectangles.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the SocNavBench traversible-to-SVG converter."""

    args = build_arg_parser().parse_args(argv)
    source = resolve_source(args)
    output_svg = args.output_svg.expanduser().resolve()
    if not source.is_file():
        report = {
            "status": "blocked_missing_traversible",
            "conversion_ready": False,
            "source_path": str(source),
            "output_svg": str(output_svg),
            "next_action": (
                "Stage official SocNavBench ETH traversible data.pkl via the external-data "
                "contract, then re-run this converter. No placeholder SVG was written."
            ),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        write_report(args.report_json, report)
        return EXIT_BLOCKED

    try:
        map_data = load_traversible(source)
        svg_text, metadata = render_svg(map_data, map_id=args.map_id)
    except SocNavBenchConversionError as exc:
        report = {
            "status": "blocked_invalid_traversible",
            "conversion_ready": False,
            "source_path": str(source),
            "output_svg": str(output_svg),
            "error": str(exc),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        write_report(args.report_json, report)
        return EXIT_BLOCKED

    if metadata["obstacle_rect_count"] > args.max_obstacle_rects:
        report = {
            **metadata,
            "status": "blocked_too_many_obstacle_rects",
            "conversion_ready": False,
            "source_sha256": sha256_file(source),
            "output_svg": str(output_svg),
            "max_obstacle_rects": args.max_obstacle_rects,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        write_report(args.report_json, report)
        return EXIT_BLOCKED

    report = {
        **metadata,
        "status": "dry_run_ready" if args.dry_run else "converted",
        "conversion_ready": True,
        "source_sha256": sha256_file(source),
        "output_svg": str(output_svg),
        "wrote_svg": not args.dry_run,
    }
    if not args.dry_run:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        output_svg.write_text(svg_text, encoding="utf-8")
        report["output_sha256"] = sha256_file(output_svg)

    print(json.dumps(report, indent=2, sort_keys=True))
    write_report(args.report_json, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
