"""Summarize issue #10017 wall-law variant measurements."""

import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from pysocialforce.config import ObstacleForceConfig
from pysocialforce.forces import ObstacleForce

root = Path(os.environ["WALL_CONTACT_ARTIFACT_ROOT"])
plan = json.loads((root / "measurement-manifest.json").read_text())
records = [
    json.loads(path.read_text()) for path in (root / "measurements").glob("*/*/receipt.json")
]
indexed = {(row["version"], row["index"]): row for row in records}
assert len(indexed) == len(records)


class _Peds:
    agent_radius = 0.35

    @staticmethod
    def pos() -> np.ndarray:
        return np.array([[6.5, 2.0]], dtype=float)


class _DoorwaySimulation:
    peds = _Peds()

    @staticmethod
    def get_raw_obstacles() -> np.ndarray:
        return np.array(
            [
                [8.0, 2.6, 8.0, 4.0, -1.0, 0.0],
                [8.0, 0.0, 8.0, 1.4, -1.0, 0.0],
            ],
            dtype=float,
        )


def doorway_braking_force(law: str, scale: float) -> float:
    """Evaluate the declared 1.2 m doorway approach point with the runtime factor."""
    config = ObstacleForceConfig(law_version=law)
    config.factor *= scale
    force = ObstacleForce(config, _DoorwaySimulation())()[0, 0]
    return -float(force)


def width_crossing(version: str, index: int) -> tuple[bool, float]:
    """Extract whole-body crossing and near-door stops from the retained trajectory."""
    task = plan["tasks"][index]
    source = Path(plan["roots"][version])
    manifest = source / task["manifest"]
    scenario = next(
        row
        for row in yaml.safe_load(manifest.read_text())["scenarios"]
        if row["name"] == task["scenario"]
    )
    svg = ET.parse((manifest.parent / scenario["map_file"]).resolve())
    posts = [node for node in svg.iter() if node.tag.endswith("rect") and node.get("x") == "15"]
    assert len(posts) == 2 and all(float(node.get("width")) == 1 for node in posts)
    low = float(posts[0].get("y")) + float(posts[0].get("height"))
    high = float(posts[1].get("y"))
    receipt = indexed[(version, index)]
    path = root / "measurements" / version / str(index) / "trajectory.npz"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == receipt["measurement"]["trajectory_sha256"]
    )
    trace = np.load(path)
    positions = trace["positions"][:, 0]
    velocities = trace["velocities"][:, 0]
    radius = receipt["measurement"]["physical_radius_m"]
    exited = positions[:, 0] <= 15 - radius
    first_exit = int(np.flatnonzero(exited)[0]) if np.any(exited) else None
    in_posts = (positions[:, 0] >= 15 - radius) & (positions[:, 0] <= 16 + radius)
    inside_gap = bool(
        np.all((positions[in_posts, 1] >= low + radius) & (positions[in_posts, 1] <= high - radius))
    )
    run = maximum = 0
    for p, v in zip(
        positions[:first_exit] if first_exit else positions,
        velocities[:first_exit] if first_exit else velocities,
        strict=True,
    ):
        stopped = 15 - radius < p[0] < 19 and np.linalg.norm(v) < 0.05
        run = run + 1 if stopped else 0
        maximum = max(maximum, run)
    return first_exit is not None and inside_gap, maximum * 0.1


def penetrating_pedestrians(version: str, index: int) -> int:
    """Count any-time physical body overlaps, including pre-existing spawn overlap."""
    receipt = indexed[(version, index)]
    path = root / "measurements" / version / str(index) / "trajectory.npz"
    trace = np.load(path)
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == receipt["measurement"]["trajectory_sha256"]
    )
    return int(np.count_nonzero(np.any(trace["wall_clearances"] < -1e-9, axis=0)))


summary = []
versions = [
    "legacy",
    "range_only",
    "physical_margin",
    "contact_stiff",
    "multi_segment",
    "wall10x",
    "physical_margin10x",
]
for version in versions:
    rows = [indexed[(version, index)] for index in range(len(plan["tasks"]))]
    measured = [row for row in rows if row["classification"] == "MEASURED"]
    aperture = [
        row
        for row in measured
        if row["task"]["kind"] == "aperture"
        and row["measurement"]["crossed_gap"]
        and row["measurement"]["max_pre_gap_stop_s"] < 2.0
        and row["measurement"]["wall_penetration_pedestrian_steps"] == 0
    ]
    width_total = width_crossed = 0
    width_stops = []
    for index, task in enumerate(plan["tasks"]):
        if task["kind"] == "scenario" and task["scenario"].startswith(
            "francis2023_narrow_doorway_width_"
        ):
            width_total += 1
            crossed, stop_s = width_crossing(version, index)
            width_crossed += int(
                crossed
                and stop_s < 2.0
                and indexed[(version, index)]["measurement"]["wall_penetration_pedestrian_steps"]
                == 0
            )
            width_stops.append(stop_s)
    penetrators = 0
    blocked = 0
    wall_steps = 0
    runtimes = []
    by_fixture = defaultdict(lambda: {"penetrating_pedestrians": 0, "blocked": 0, "slots": 0})
    for index, row in enumerate(rows):
        if row["classification"] != "MEASURED":
            continue
        measurement = row["measurement"]
        penetrating = penetrating_pedestrians(version, index)
        penetrators += penetrating
        blocked += int(measurement["blocked_pedestrians"])
        wall_steps += int(measurement["wall_penetration_pedestrian_steps"])
        runtimes.append(float(row["runtime_s"]))
        fixture = (
            f"lone_aperture_{row['task']['width_m']:.1f}"
            if row["task"]["kind"] == "aperture"
            else row["task"]["scenario"]
        )
        by_fixture[fixture]["penetrating_pedestrians"] += penetrating
        by_fixture[fixture]["blocked"] += int(measurement["blocked_pedestrians"])
        by_fixture[fixture]["slots"] += 1
    item = {
        "variant": version,
        "law": plan["laws"][version],
        "source_sha": plan["sources"][version],
        "measured_slots": len(measured),
        "penetrating_pedestrians": penetrators,
        "wall_penetration_pedestrian_steps": wall_steps,
        "blocked_after_70s": blocked,
        "lone_aperture_pass": len(aperture),
        "lone_aperture_total": 40,
        "three_width_crossings": width_crossed,
        "three_width_total": width_total,
        "three_width_max_stop_s": max(width_stops) if width_stops else math.nan,
        "doorway_braking_force_mps2": doorway_braking_force(
            plan["laws"][version], plan.get("controls", {}).get(version, {}).get("wall_scale", 1.0)
        ),
        "bottleneck_blocked": by_fixture["classic_realworld_double_bottleneck_high"]["blocked"],
        "bottleneck_goals_reached": indexed[(version, 70)]["measurement"]["assigned_goal_reached"],
        "runtime_total_s": round(sum(runtimes), 3),
        "runtime_max_s": round(max(runtimes), 3) if runtimes else math.nan,
        "by_fixture": dict(by_fixture),
    }
    summary.append(item)

table = [
    "| Variant | Law | Penetrating pedestrians | Blocked after 70s | Lone aperture | Three-width | Doorway braking m/s^2 | Runtime total s |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in summary:
    table.append(
        "| {variant} | `{law}` | {penetrating_pedestrians} | {blocked_after_70s} | "
        "{lone_aperture_pass}/{lone_aperture_total} | {three_width_crossings}/{three_width_total} | "
        "{doorway_braking_force_mps2:.6f} | {runtime_total_s:.3f} |".format(**row)
    )

result = {"sources": plan["sources"], "summary": summary}
(root / "variant-summary.json").write_text(json.dumps(result, indent=2) + "\n")
(root / "variant-table.md").write_text("\n".join(table) + "\n")
print(json.dumps(summary, indent=2))
