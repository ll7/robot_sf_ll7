"""Audit identical geometry and trace first physical overlaps without replaying episodes."""

import json
import os
from pathlib import Path

import numpy as np
from shapely import wkt
from shapely.geometry import LineString, Point

root = Path(os.environ["WALL_CONTACT_ARTIFACT_ROOT"])
versions = [
    "legacy",
    "range_only",
    "physical_margin",
    "contact_stiff",
    "wall10x",
    "small_dt",
    "no_social",
    "multi_segment",
    "physical_margin10x",
    "physical_margin10x_small_dt",
]
report = {}
for version in versions:
    slot = root / "measurements" / version / "70"
    if not (slot / "force-trace.npz").exists():
        continue
    trajectory = np.load(slot / "trajectory.npz")
    trace = np.load(slot / "force-trace.npz")
    meta = json.loads((slot / "geometry.json").read_text())
    measurement = json.loads((slot / "measurement.json").read_text())
    radius = measurement["physical_radius_m"]
    edges = trace["raw_edges"]
    polygons = [wkt.loads(value) for value in meta["polygons_wkt"]]
    events = []
    for ped_id in range(measurement["pedestrians"]):
        negative = np.flatnonzero(trajectory["wall_clearances"][:, ped_id] < -1e-9)
        if not len(negative):
            continue
        tick = int(negative[0])
        source_tick = max(0, tick - 1)
        previous = trajectory["positions"][source_tick, ped_id]
        point = trajectory["positions"][tick, ped_id]
        swept = LineString([previous, point]) if tick else Point(point)
        distances = np.linalg.norm(
            trace["closest_points"][source_tick, ped_id]
            - trace["force_states"][source_tick, ped_id, :2],
            axis=-1,
        )
        nearest = int(np.argmin(distances))
        penetrated = [
            i
            for i, edge in enumerate(edges)
            if swept.distance(LineString(edge[:4].reshape(2, 2))) < radius - 1e-9
        ]
        per_segment = trace["segment_forces"][source_tick, ped_id]
        distinct_points = []
        applied_segments = []
        for segment_id, nearest_point in enumerate(trace["closest_points"][source_tick, ped_id]):
            if version == "legacy":
                applied_segments.append(segment_id)
            elif version != "multi_segment":
                if segment_id == nearest:
                    applied_segments.append(segment_id)
            elif distances[segment_id] < measurement["force_radius_m"] + 0.2:
                if not any(
                    np.linalg.norm(nearest_point - value) <= 1e-9 for value in distinct_points
                ):
                    distinct_points.append(nearest_point)
                    applied_segments.append(segment_id)
        components = {
            name: value[ped_id].tolist()
            for name, value in zip(
                meta["component_names"], trace["force_components"][source_tick], strict=True
            )
        }
        event = {
            "ped_id": ped_id,
            "first_tick": tick,
            "time_s": tick * measurement["dt_s"],
            "initial_overlap": tick == 0,
            "overlap_samples": len(negative),
            "previous_position": previous.tolist(),
            "position": point.tolist(),
            "nearest_segment": nearest,
            "penetrated_segments": penetrated,
            "nearest_segment_was_penetrated": nearest in penetrated,
            "physical_radius_m": radius,
            "force_radius_m": measurement["force_radius_m"],
            "physical_clearance_before_m": float(np.min(distances) - radius),
            "force_clearance_before_m": float(np.min(distances) - measurement["force_radius_m"]),
            "center_inside_polygon": any(p.contains(Point(point)) for p in polygons),
            "components": components,
            "previous_velocity": trace["integration"][source_tick, 0, ped_id].tolist(),
            "uncapped_velocity": trace["integration"][source_tick, 1, ped_id].tolist(),
            "applied_velocity": trace["integration"][source_tick, 2, ped_id].tolist(),
            "position_velocity": trace["integration"][source_tick, 3, ped_id].tolist(),
            "nearby_segments": [
                {
                    "id": i,
                    "edge": edges[i].tolist(),
                    "distance_before_m": float(distances[i]),
                    "force": per_segment[i].tolist(),
                    "penetrated": i in penetrated,
                    "selected_nearest": i == nearest,
                }
                for i in range(len(edges))
                if distances[i] < radius + 0.55 or i in penetrated
            ],
        }
        events.append(event)
        for segment in event["nearby_segments"]:
            segment["applied_force"] = (
                segment["force"] if segment["id"] in applied_segments else [0.0, 0.0]
            )
        event["observed_nearest_segment"] = int(
            np.argmin([Point(point).distance(LineString(edge[:4].reshape(2, 2))) for edge in edges])
        )
        event["pre_behavior_velocity"] = trajectory["velocities"][source_tick, ped_id].tolist()
    report[version] = {
        "geometry_sha256": measurement["geometry_sha256"],
        "physical_radius_m": radius,
        "force_radius_m": measurement["force_radius_m"],
        "wall_factor": meta["wall_factor"],
        "integration_scheme": meta["integration_scheme"],
        "penetrating_pedestrians": len(events),
        "blocked_pedestrians": measurement["blocked_pedestrians"],
        "wall_penetration_samples": measurement["wall_penetration_pedestrian_steps"],
        "events": events,
    }
    report[version]["per_pedestrian_geometry"] = [
        {
            "id": i,
            "center_enters_polygon": any(
                p.contains(Point(point))
                for point in trajectory["positions"][:, i]
                for p in polygons
            ),
            "straight_assigned_path_intersects_polygon_interior": any(
                LineString(
                    [trajectory["positions"][0, i], trajectory["assigned_goals"][i]]
                ).relate_pattern(p, "T********")
                for p in polygons
            ),
            "minimum_physical_clearance_m": float(np.min(trajectory["wall_clearances"][:, i])),
            "final_physical_clearance_m": float(trajectory["wall_clearances"][-1, i]),
            "final_goal_distance_m": measurement["per_pedestrian"][i]["final_goal_distance_m"],
        }
        for i in range(measurement["pedestrians"])
    ]
    print(
        json.dumps(
            {
                "version": version,
                "penetrators": len(events),
                "blocked": measurement["blocked_pedestrians"],
                "force_radius": measurement["force_radius_m"],
                "events": [
                    {
                        k: e[k]
                        for k in (
                            "ped_id",
                            "first_tick",
                            "nearest_segment",
                            "penetrated_segments",
                            "physical_clearance_before_m",
                            "components",
                        )
                    }
                    for e in events
                ],
            }
        )
    )
(root / "trace-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
assert len({row["geometry_sha256"] for row in report.values()}) == 1
assert len({row["physical_radius_m"] for row in report.values()}) == 1
