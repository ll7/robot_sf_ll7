#!/usr/bin/env python3
"""Import a small Stanford Drone Dataset annotation slice as Robot SF scenarios.

The importer intentionally targets the original SDD annotation text format:

``track_id xmin ymin xmax ymax frame lost occluded generated label``

It does not download or redistribute SDD. Users provide a local annotation file obtained under the
dataset license and this script writes a Robot SF map YAML, scenario YAML, and provenance JSON.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SDD_SOURCE_URL = "https://cvgl.stanford.edu/projects/uav_data/"
SDD_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0"


@dataclass(frozen=True)
class SddPoint:
    """One usable SDD annotation center point."""

    track_id: str
    frame: int
    x_px: float
    y_px: float
    label: str
    raw_label: str


@dataclass(frozen=True)
class ImportOptions:
    """Versioned normalization and extraction settings for one SDD import."""

    dataset_id: str
    source_annotation: Path
    meters_per_pixel: float
    frame_rate_hz: float
    min_track_points: int
    max_pedestrians: int
    stride: int
    max_waypoints: int
    margin_m: float
    y_flip_height_px: float | None = None


def normalize_sdd_label(raw_label: str) -> str:
    """Normalize an SDD annotation label for user-facing comparison and storage."""
    label = str(raw_label).strip()
    if len(label) >= 2 and label[0] == label[-1] and label[0] in {"'", '"'}:
        return label[1:-1].strip()
    return label


def _parse_annotation_line(line: str, *, line_number: int) -> SddPoint | None:
    """Parse one SDD annotation line.

    Returns:
        SddPoint | None: Parsed point, or ``None`` for blank/comment lines.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fields = stripped.split()
    if len(fields) != 10:
        raise ValueError(
            f"SDD annotation line {line_number} must have 10 fields, got {len(fields)}."
        )
    track_id, xmin, ymin, xmax, ymax, frame, lost, _occluded, _generated, label = fields
    if int(lost) != 0:
        return None
    x_center = (float(xmin) + float(xmax)) / 2.0
    y_center = (float(ymin) + float(ymax)) / 2.0
    return SddPoint(
        track_id=str(track_id),
        frame=int(frame),
        x_px=x_center,
        y_px=y_center,
        label=normalize_sdd_label(label),
        raw_label=str(label),
    )


def load_sdd_points(path: Path, *, label: str) -> list[SddPoint]:
    """Load pedestrian points from an SDD annotation file."""
    target_label = normalize_sdd_label(label)
    target_label_key = target_label.casefold()
    points: list[SddPoint] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            point = _parse_annotation_line(line, line_number=line_number)
            if point is None:
                continue
            if point.label.casefold() == target_label_key:
                points.append(point)
    if not points:
        raise ValueError(f"No usable '{target_label}' annotations found in {path}.")
    return points


def _trajectory_points(
    points: list[SddPoint],
    *,
    meters_per_pixel: float,
    x_offset_m: float,
    y_offset_m: float,
    y_flip_height_px: float | None,
    stride: int,
    max_waypoints: int,
) -> list[list[float]]:
    """Convert ordered SDD points into Robot SF meter coordinates."""
    trajectory: list[list[float]] = []
    for point in sorted(points, key=lambda item: item.frame)[::stride]:
        y_px = y_flip_height_px - point.y_px if y_flip_height_px is not None else point.y_px
        trajectory.append(
            [
                round(point.x_px * meters_per_pixel + x_offset_m, 4),
                round(y_px * meters_per_pixel + y_offset_m, 4),
            ]
        )
        if len(trajectory) >= max_waypoints:
            break
    return trajectory


def build_import_payload(
    points: list[SddPoint],
    options: ImportOptions,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build map, scenario, and provenance payloads from parsed SDD points."""
    grouped: dict[str, list[SddPoint]] = defaultdict(list)
    for point in points:
        grouped[point.track_id].append(point)
    selected = [
        track_points
        for _track_id, track_points in sorted(grouped.items(), key=lambda item: item[0])
        if len(track_points) >= options.min_track_points
    ][: options.max_pedestrians]
    if not selected:
        raise ValueError(
            f"No tracks have at least {options.min_track_points} usable points after "
            "label/lost filtering."
        )

    raw_x = [point.x_px * options.meters_per_pixel for track in selected for point in track]
    raw_y = [
        (
            (options.y_flip_height_px - point.y_px)
            if options.y_flip_height_px is not None
            else point.y_px
        )
        * options.meters_per_pixel
        for track in selected
        for point in track
    ]
    x_offset = options.margin_m - min(raw_x)
    y_offset = options.margin_m - min(raw_y)
    width = max(raw_x) + x_offset + options.margin_m
    height = max(raw_y) + y_offset + options.margin_m

    pedestrians: list[dict[str, Any]] = []
    for track_points in selected:
        trajectory = _trajectory_points(
            track_points,
            meters_per_pixel=options.meters_per_pixel,
            x_offset_m=x_offset,
            y_offset_m=y_offset,
            y_flip_height_px=options.y_flip_height_px,
            stride=options.stride,
            max_waypoints=options.max_waypoints,
        )
        if len(trajectory) < 2:
            continue
        frames = [point.frame for point in track_points]
        pedestrians.append(
            {
                "id": f"sdd_{track_points[0].track_id}",
                "start": trajectory[0],
                "trajectory": trajectory[1:],
                "metadata": {
                    "source_track_id": track_points[0].track_id,
                    "source_frame_start": min(frames),
                    "source_frame_end": max(frames),
                    "source_label": track_points[0].label,
                    "source_raw_label": track_points[0].raw_label,
                },
            }
        )
    if not pedestrians:
        raise ValueError("No selected tracks retained at least two trajectory points.")

    zone_size = max(1.0, min(width, height) * 0.1)
    robot_spawn = [
        [options.margin_m, options.margin_m],
        [options.margin_m + zone_size, options.margin_m],
        [options.margin_m + zone_size, options.margin_m + zone_size],
    ]
    robot_goal = [
        [
            max(options.margin_m, width - options.margin_m - zone_size),
            max(options.margin_m, height - options.margin_m - zone_size),
        ],
        [
            max(options.margin_m, width - options.margin_m),
            max(options.margin_m, height - options.margin_m - zone_size),
        ],
        [
            max(options.margin_m, width - options.margin_m),
            max(options.margin_m, height - options.margin_m),
        ],
    ]
    map_payload: dict[str, Any] = {
        "x_margin": [0.0, round(width, 4)],
        "y_margin": [0.0, round(height, 4)],
        "obstacles": [],
        "robot_spawn_zones": [robot_spawn],
        "robot_goal_zones": [robot_goal],
        "ped_spawn_zones": [],
        "ped_goal_zones": [],
        "ped_crowded_zones": [],
        "robot_routes": [
            {"spawn_id": 0, "goal_id": 0, "waypoints": [robot_spawn[0], robot_goal[2]]}
        ],
        "ped_routes": [],
        "single_pedestrians": pedestrians,
    }
    scenario_payload: dict[str, Any] = {
        "scenarios": [
            {
                "name": options.dataset_id,
                "map_file": f"{options.dataset_id}.map.yaml",
                "simulation_config": {
                    "max_episode_steps": max(100, int(options.max_waypoints * options.stride)),
                    "ped_density": 0.0,
                },
                "single_pedestrians": [
                    {
                        "id": ped["id"],
                        "metadata": {
                            "real_world_dataset": "stanford_drone_dataset",
                            "importer": "sdd_annotations_v1",
                            **ped["metadata"],
                        },
                    }
                    for ped in pedestrians
                ],
                "robot_config": {},
                "metadata": {
                    "dataset": "stanford_drone_dataset",
                    "dataset_id": options.dataset_id,
                    "license": SDD_LICENSE,
                    "source_url": SDD_SOURCE_URL,
                    "source_annotation": str(options.source_annotation),
                    "normalization": "bbox_center_pixels_to_local_meters",
                    "meters_per_pixel": options.meters_per_pixel,
                    "frame_rate_hz": options.frame_rate_hz,
                    "trajectory_stride": options.stride,
                },
                "seeds": [0],
            }
        ]
    }
    provenance = {
        "dataset": "stanford_drone_dataset",
        "dataset_id": options.dataset_id,
        "source_url": SDD_SOURCE_URL,
        "license": SDD_LICENSE,
        "source_annotation": str(options.source_annotation),
        "meters_per_pixel": options.meters_per_pixel,
        "frame_rate_hz": options.frame_rate_hz,
        "min_track_points": options.min_track_points,
        "max_pedestrians": options.max_pedestrians,
        "stride": options.stride,
        "max_waypoints": options.max_waypoints,
        "pedestrians": [ped["metadata"] for ped in pedestrians],
    }
    return map_payload, scenario_payload, provenance


def write_import_outputs(
    *,
    out_dir: Path,
    dataset_id: str,
    map_payload: dict[str, Any],
    scenario_payload: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Path]:
    """Write generated importer outputs and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "map": out_dir / f"{dataset_id}.map.yaml",
        "scenario": out_dir / f"{dataset_id}.scenario.yaml",
        "provenance": out_dir / f"{dataset_id}.provenance.json",
    }
    paths["map"].write_text(yaml.safe_dump(map_payload, sort_keys=False), encoding="utf-8")
    paths["scenario"].write_text(
        yaml.safe_dump(scenario_payload, sort_keys=False),
        encoding="utf-8",
    )
    paths["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, required=True, help="Path to SDD annotations.txt"
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for generated files")
    parser.add_argument("--dataset-id", default="sdd_import_v1", help="Scenario/map id prefix")
    parser.add_argument("--label", default="Pedestrian", help="SDD label to import")
    parser.add_argument("--meters-per-pixel", type=float, required=True)
    parser.add_argument("--frame-rate-hz", type=float, default=30.0)
    parser.add_argument("--min-track-points", type=int, default=8)
    parser.add_argument("--max-pedestrians", type=int, default=4)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-waypoints", type=int, default=24)
    parser.add_argument("--margin-m", type=float, default=2.0)
    parser.add_argument("--y-flip-height-px", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the SDD importer."""
    args = parse_args()
    if args.meters_per_pixel <= 0:
        raise ValueError("--meters-per-pixel must be > 0")
    if args.frame_rate_hz <= 0:
        raise ValueError("--frame-rate-hz must be > 0")
    if args.min_track_points <= 1 or args.max_pedestrians <= 0:
        raise ValueError("--min-track-points must be > 1 and --max-pedestrians must be > 0")
    if args.stride <= 0 or args.max_waypoints <= 1:
        raise ValueError("--stride must be > 0 and --max-waypoints must be > 1")
    points = load_sdd_points(args.annotations, label=args.label)
    options = ImportOptions(
        dataset_id=args.dataset_id,
        source_annotation=args.annotations,
        meters_per_pixel=args.meters_per_pixel,
        frame_rate_hz=args.frame_rate_hz,
        min_track_points=args.min_track_points,
        max_pedestrians=args.max_pedestrians,
        stride=args.stride,
        max_waypoints=args.max_waypoints,
        margin_m=args.margin_m,
        y_flip_height_px=args.y_flip_height_px,
    )
    map_payload, scenario_payload, provenance = build_import_payload(
        points,
        options,
    )
    paths = write_import_outputs(
        out_dir=args.out_dir,
        dataset_id=args.dataset_id,
        map_payload=map_payload,
        scenario_payload=scenario_payload,
        provenance=provenance,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
