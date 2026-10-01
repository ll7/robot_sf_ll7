"""Separate wall-law development cases, issue 10061; diagnostic evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numba import njit
from pysocialforce.config import SURFACE_DISTANCE_UNIT_NORMAL_V2
from pysocialforce.forces import ObstacleForce

from robot_sf.evidence.writers import write_json
from scripts.validation import calibrate_obstacle_force_10061 as harness

SEEDS = tuple(range(1001, 1011))
RADIUS = harness.FOOTPRINT_RADIUS
ORIGINAL_FORCE = ObstacleForce.__call__
BASE_CONFIG = harness.config


@njit
def exponential_kernel(positions, obstacles, amplitude, length, radius):
    """Negative gradient of U=A*B*exp(-(d-radius)/B), per segment."""
    output = np.zeros_like(positions)
    for i in range(len(positions)):
        for obstacle in obstacles:
            ax, ay, bx, by = obstacle[:4]
            vx, vy = bx - ax, by - ay
            denominator = vx * vx + vy * vy
            t = (
                min(
                    1.0,
                    max(
                        0.0,
                        ((positions[i, 0] - ax) * vx + (positions[i, 1] - ay) * vy) / denominator,
                    ),
                )
                if denominator
                else 0.0
            )
            dx, dy = positions[i, 0] - ax - t * vx, positions[i, 1] - ay - t * vy
            distance = np.sqrt(dx * dx + dy * dy)
            if distance > 0:
                strength = amplitude * np.exp(min(100.0, (radius - distance) / length)) / distance
                output[i, 0] += strength * dx
                output[i, 1] += strength * dy
    return output


def exponential_force(self):
    """Diagnostic exponential with the physical body edge as distance origin."""
    return exponential_kernel(
        self.get_peds(), self.get_obstacles(), self.config.factor, self.config.threshold, RADIUS
    )


def candidate_config(candidate, speed):
    """Select an actual kernel and desired speed without changing defaults."""
    law, amplitude, offset = candidate
    ObstacleForce.__call__ = exponential_force if law == "exponential_edge" else ORIGINAL_FORCE
    config = BASE_CONFIG(amplitude, offset)
    config.scene_config.max_speed_multiplier = 1.3
    if law == "true_gradient":
        config.obstacle_force_config.law_version = SURFACE_DISTANCE_UNIT_NORMAL_V2
    # Scene state maximum speed is 1.3 * initial velocity; initialize at speed/1.3.
    return config


def wall_metrics(positions, segments):
    """Measure both centre and physical-edge distance, without wall clipping."""
    distances = harness.distance(positions, segments)
    minimum = float(distances.min())
    return {
        "min_center_distance_m": minimum,
        "min_edge_distance_m": minimum - RADIUS,
        "wall_penetration_m": max(0.0, RADIUS - minimum),
    }


def aperture(candidate, seed, width, speed):
    """Lone thick aperture, separately measured approach and crossing speeds."""
    rng = np.random.default_rng(seed)
    segments = [(-1.0, 3.0, 17.0, 3.0), (-1.0, -3.0, 17.0, -3.0)]
    for a, b in [(width / 2, 3.0), (-3.0, -width / 2)]:
        segments.extend([(8.0, a, 8.0, b), (8.0, a, 8.1, a), (8.1, a, 8.1, b), (8.1, b, 8.0, b)])
    state = np.array([[2.0, rng.uniform(-0.03, 0.03), speed / 1.3, 0.0, 15.0, 0.0, 0.5]])
    positions, velocities = harness.simulate(
        state, segments, candidate_config(candidate, speed), 1000, stop_x=9.2
    )
    x = positions[1:, 0, 0]
    approach = velocities[(x >= 4.0) & (x <= 6.0), 0]
    crossing = velocities[(x >= 7.0) & (x <= 9.1), 0]
    approach_speed = float(approach.mean()) if len(approach) else None
    minimum_speed = float(crossing.min()) if len(crossing) else None
    return {
        "seed": seed,
        "width_m": width,
        "desired_speed_m_s": speed,
        "passed": bool(x[-1] > 9.2),
        "crossing_min_speed_m_s": minimum_speed,
        "approach_speed_m_s": approach_speed,
        "speed_drop_m_s": approach_speed - minimum_speed
        if minimum_speed is not None and approach_speed is not None
        else None,
        "final_x_m": float(x[-1]),
        **wall_metrics(positions, segments),
    }


def corridor(candidate, seed, speed, direct=False):
    """Free wall-adjacent walking and a separate perpendicular approach stress case."""
    rng = np.random.default_rng(seed)
    y = 2.0 - RADIUS - 0.25 + rng.uniform(-0.01, 0.01)
    segments = [(-1.0, 2.0, 101.0, 2.0)]
    if direct:
        state = np.array([[2.0, 1.0 + rng.uniform(-0.01, 0.01), 0.0, speed / 1.3, 2.0, 4.0, 0.5]])
        update = None
    else:
        state = np.array([[2.0, y, speed / 1.3, 0.0, 100.0, y, 0.5]])

        def update(sim):
            sim.peds.state[0, 4:6] = [sim.peds.pos()[0, 0] + 1.0, y]

    positions, velocities = harness.simulate(
        state, segments, candidate_config(candidate, speed), 400, goal_update=update
    )
    distances = harness.distance(positions[100:], segments)
    return {
        "seed": seed,
        "desired_speed_m_s": speed,
        "direct_wall_approach": direct,
        "steady_center_distance_m": float(np.median(distances)),
        "steady_edge_distance_m": float(np.median(distances) - RADIUS),
        "steady_speed_m_s": float(velocities[100:].mean()),
        **wall_metrics(positions, segments),
    }


def screen(candidate):
    """Screen all families against the same independent cases and ten dev seeds."""
    rows = [
        aperture(candidate, s, w, speed)
        for s in SEEDS
        for w in [0.8, 1.0, 1.2, 2.0, 2.2, 2.8, 3.6]
        for speed in [0.65, 1.3]
    ]
    walls = [
        corridor(candidate, s, speed, direct)
        for s in SEEDS
        for speed in [0.65, 1.3]
        for direct in [False, True]
    ]
    return {"candidate": candidate, "aperture": rows, "wall_cases": walls}


def bottleneck(candidate, seed, width, wide=False):
    """Finite-N flow with explicit experiment geometry/supply and density mismatch."""
    rng = np.random.default_rng(seed)
    n = 350 if wide else 60
    length = 1.0 if wide else 2.8
    # A soft-model 3.3/m² initial bulk, not feasible nonoverlapping 0.40 m discs.
    density = 3.0 if wide else 3.3
    if wide:
        # Liao holding semicircle: radius 8.618 m, N=350, density 3/m².
        # Deterministic hex sites within the semicircle; jitter is seed controlled.
        radius = 8.618
        spacing = (2 / (np.sqrt(3) * density)) ** 0.5 * 0.94
        sites = []
        for row in range(-18, 19):
            yy = row * spacing * np.sqrt(3) / 2
            for column in range(-20, 1):
                xx = (column + (row % 2) * 0.5) * spacing
                if xx < -0.42 and xx * xx + yy * yy < radius * radius:
                    sites.append((xx, yy))
        sites = np.asarray(sites)[rng.choice(len(sites), n, replace=False)]
        if len(sites) != n:
            raise ValueError("insufficient semicircle sites")
        x, y = sites[:, 0], sites[:, 1]
        half = 10.0
    else:
        columns = 6
        rows = n // columns
        spacing = 1 / np.sqrt(density)
        xx, yy = np.meshgrid(np.arange(columns) * spacing, np.arange(rows) * spacing)
        x = xx.ravel() - (columns - 1) * spacing / 2 - 3.0
        y = yy.ravel() - (rows - 1) * spacing / 2
        half = 3.5
    x = x + rng.uniform(-0.01, 0.01, n)
    y = y + rng.uniform(-0.01, 0.01, n)
    downstream = length + 20.0
    segments = [
        (-50.0, half, downstream, half),
        (-50.0, -half, downstream, -half),
        (0.0, width / 2, 0.0, half),
        (0.0, -half, 0.0, -width / 2),
        (0.0, width / 2, length, width / 2),
        (0.0, -width / 2, length, -width / 2),
    ]
    state = np.column_stack(
        [
            x,
            y,
            np.full(n, (1.55 if wide else 1.3) / 1.3),
            np.zeros(n),
            np.full(n, length + 0.6),
            np.clip(y, -max(0.0, width / 2 - RADIUS - 0.05), max(0.0, width / 2 - RADIUS - 0.05)),
            np.full(n, 0.5),
        ]
    )

    def update(sim):
        passed = sim.peds.pos()[:, 0] > length + 0.2
        sim.peds.state[passed, 4] = downstream

    positions, _ = harness.simulate(
        state,
        segments,
        candidate_config(candidate, 1.55 if wide else 1.3),
        1600,
        goal_update=update,
    )
    # Seyfried Table 2 samples 0.4 m inside; wide opening at exit.
    plane = length if wide else 0.4
    times = []
    for j in range(n):
        index = np.flatnonzero(positions[:, j, 0] >= plane)
        if len(index):
            k = int(index[0])
            previous = positions[k - 1, j, 0] if k else plane
            fraction = (plane - previous) / (positions[k, j, 0] - previous) if k else 0.0
            times.append(float((k - 1 + fraction) * 0.1))
    times.sort()
    count = len(times)
    flow = count / (times[-1] - times[0]) if count >= 2 else 0.0
    central = times[int(0.2 * count) : int(0.8 * count)]
    central_flow = (len(central) - 1) / (central[-1] - central[0]) if len(central) >= 2 else 0.0
    delta = state[:, None, :2] - state[None, :, :2]
    pair = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(pair, np.inf)
    return {
        "candidate": candidate,
        "seed": seed,
        "width_m": width,
        "bottleneck_length_m": length,
        "persons": n,
        "initial_density_m2": density,
        "desired_speed_m_s": 1.55 if wide else 1.3,
        "initial_min_pair_distance_m": float(pair.min()),
        "initial_footprint_pair_overlaps": int((pair < 2 * RADIUS).sum() // 2),
        "crossing_plane_m": plane,
        "crossed": count,
        "all_crossed": count == n,
        "flow_persons_s": flow,
        "specific_flow_persons_m_s": flow / width,
        "central_20_80_flow_persons_s": central_flow,
        "central_period_is_proven_stationary": False,
        "crossing_times_s": times,
        **wall_metrics(positions, segments),
    }


def flow_task(task):
    """Evaluate one profile/seed/width; retain failed zero-flow cases."""
    candidate, seed, width, wide = task
    return bottleneck(candidate, seed, width, wide)


def lanes(candidate):
    """Canonical lane harness, with the same explicit wall-order conversion."""
    original_config = harness.config

    def configured(factor, offset):
        return candidate_config(candidate, 0.65)

    harness.config = configured
    try:
        return {"candidate": candidate, "result": harness.lanes(candidate[1:])}
    finally:
        harness.config = original_config


def main():
    """Write immutable-case receipts with explicit seed and source identities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen", "flow", "lanes"], required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be 1..16")
    grid = json.loads(args.grid.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    tasks = (
        grid
        if args.stage != "flow"
        else [
            (c, s, w, wide)
            for c in grid
            for s in SEEDS
            for wide, widths in [(False, [0.8, 1.0, 1.2]), (True, [2.4, 3.0, 3.6, 5.0])]
            for w in widths
        ]
    )
    worker = {"screen": screen, "flow": flow_task, "lanes": lanes}[args.stage]
    print("RESOLVED SEEDS", SEEDS, "GRID", grid, flush=True)
    records = []
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(worker, t) for t in tasks]):
            record = future.result()
            records.append(record)
            write_json(args.out / f"case_{len(records):04}.json", record)
            print("DONE", len(records), "/", len(tasks), record["candidate"], flush=True)
    write_json(
        args.out / "manifest.json",
        {
            "diagnostic_only": True,
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "grid_sha256": hashlib.sha256(args.grid.read_bytes()).hexdigest(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "seeds": list(SEEDS),
            "footprint_radius_m": RADIUS,
            "stage": args.stage,
            "cases": len(records),
        },
    )


if __name__ == "__main__":
    main()
