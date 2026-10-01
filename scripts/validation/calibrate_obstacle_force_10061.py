"""Frozen diagnostic grid, issue 10061; dev seeds 1001..1010 only."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pysocialforce as pysf
from pysocialforce.config import SimulatorConfig

from robot_sf.evidence.writers import write_json
from robot_sf.research.emergent_phenomena import (
    LITERATURE_CALIBRATION,
    RELEASED_DEFAULT_CALIBRATION,
    ScenarioConfig,
    run_scenario,
)

SEEDS = list(range(1001, 1011))
DOORS = [1.2, 2.0, 2.2, 2.8, 3.6]
FACTORS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
OFFSETS = [-0.57, -0.3, -0.1, 0.0, 0.1]
GRID = [(10.0, -0.57)] + [(f, b) for b in OFFSETS for f in FACTORS if (f, b) != (10.0, -0.57)]


def config(f, b):
    """Build a raw parameter candidate without changing the released default.

    Returns:
        Measured output for the requested configuration."""
    c = SimulatorConfig()
    c.scene_config.enable_group = False
    c.obstacle_force_config.factor = f
    c.obstacle_force_config.threshold = b
    return c


def distance(pos, segments):
    """Measure Euclidean centre distance to the closest physical wall segment.

    Returns:
        Measured output for the requested configuration."""
    p = np.asarray(pos)[..., None, :]
    s = np.asarray(segments)
    a = s[:, :2]
    v = s[:, 2:] - a
    t = np.clip(((p - a) * v).sum(-1) / (v * v).sum(-1), 0, 1)
    return np.linalg.norm(p - a - t[..., None] * v, axis=-1).min(-1)


def simulate(state, segs, c, steps, goal_update=None, stop_x=None):
    # Builders use xyxy; EnvState requires xxyy. Explicitly checked against bytes.
    """Step the real substrate with an explicit xyxy to xxyy wall conversion.

    Returns:
        Measured output for the requested configuration."""
    walls = [(a, c_, b, d) for a, b, c_, d in segs]
    sim = pysf.Simulator(state=state.copy(), obstacles=walls, config=c)
    traj = [sim.peds.pos().copy()]
    speeds = []
    for i in range(steps):
        if goal_update is not None:
            goal_update(sim)
        sim.step()
        traj.append(sim.peds.pos().copy())
        speeds.append(np.linalg.norm(sim.peds.vel(), axis=1))
        if stop_x is not None and sim.peds.pos()[0, 0] > stop_x:
            break
    return np.asarray(traj), np.asarray(speeds)


def lone(f, b, seed, width, thickness=0.0):
    """Measure passage through an opening, including a finite-thickness wall.

    Returns:
        Measured output for the requested configuration."""
    rng = np.random.default_rng(seed)
    segs = [(-1, 2, 17, 2), (-1, -2, 17, -2), (8, width / 2, 8, 3), (8, -width / 2, 8, -3)]
    if thickness:
        for a, z in [(width / 2, 2.0), (-2.0, -width / 2)]:
            segs.extend(
                [
                    (8, a, 8 + thickness, a),
                    (8 + thickness, a, 8 + thickness, z),
                    (8 + thickness, z, 8, z),
                ]
            )
    state = np.array([[2.0, rng.uniform(-0.05, 0.05), 0.5, 0, 15.0, 0, 0.5]])
    p, v = simulate(state, segs, config(f, b), 1000, stop_x=9.0)
    return {
        "seed": seed,
        "width": width,
        "thickness": thickness,
        "passed": bool(p[-1, 0, 0] > 9),
        "final_x": float(p[-1, 0, 0]),
        "speed": float(v[-100:].mean()),
        "min_center_distance": float(distance(p, segs).min()),
        "steps": len(p) - 1,
    }


def corridor(f, b, seed):
    """Measure steady clearance on a wall-adjacent route after the transient.

    Returns:
        Measured output for the requested configuration."""
    rng = np.random.default_rng(seed)
    y = 1.4 + rng.uniform(-0.01, 0.01)
    segs = [(-1, 2, 101, 2), (-1, -2, 101, -2)]
    state = np.array([[2, y, 0.5, 0, 100, y, 0.5]])

    def update(s):
        s.peds.state[0, 4:6] = [s.peds.pos()[0, 0] + 1.0, y]

    p, v = simulate(state, segs, config(f, b), 400, goal_update=update)
    d = distance(p, segs)[:, 0]
    return {
        "seed": seed,
        "min_center_distance": float(d.min()),
        "steady_body_clearance": float(d[100:].min() - 0.35),
        "steady_speed": float(v[100:].mean()),
    }


def screen(pair):
    """Apply the five-door passage and free-corridor clearance hard gates.

    Returns:
        Measured output for the requested configuration."""
    f, b = pair
    rows = [lone(f, b, s, w, t) for s in SEEDS for w in DOORS for t in [0.0, 0.1]]
    cr = [corridor(f, b, s) for s in SEEDS]
    ok = all(r["passed"] and r["min_center_distance"] >= 0.35 for r in rows) and all(
        r["min_center_distance"] >= 0.35 and 0.2 <= r["steady_body_clearance"] <= 0.5 for r in cr
    )
    return {"factor": f, "offset": b, "screen_pass": ok, "lone": rows, "corridor": cr}


def crowd(f, b, seed, width, speed):
    """Measure first-to-last outflow of a finite 60-person unidirectional queue.

    Returns:
        Measured output for the requested configuration."""
    rng = np.random.default_rng(seed)
    n = 60
    xs, ys = np.meshgrid(np.arange(6) * 0.75 + 0.8, np.linspace(-3.5, 3.5, 10))
    x = xs.ravel() + rng.uniform(-0.015, 0.015, n)
    y = ys.ravel() + rng.uniform(-0.015, 0.015, n)
    # Unidirectional queue; centreline waypoint 1m beyond plane, then clear downstream.
    state = np.column_stack(
        [x, y, np.full(n, speed / 1.3), np.zeros(n), np.full(n, 9.0), np.zeros(n), np.full(n, 0.5)]
    )
    segs = [
        (-1, 4, 25, 4),
        (-1, -4, 25, -4),
        (-1, -4, -1, 4),
        (8, width / 2, 8, 5),
        (8, -width / 2, 8, -5),
    ]

    def update(s):
        crossed = s.peds.pos()[:, 0] > 8.5
        s.peds.state[crossed, 4] = 24.0

    p, _v = simulate(state, segs, config(f, b), 1200, goal_update=update)
    crossed = p[:, :, 0] >= 8
    times = []
    for j in range(n):
        tt = np.flatnonzero(crossed[:, j])
        if len(tt):
            times.append(float(tt[0] * 0.1))
    times.sort()
    count = len(times)
    # Same first-to-last convention as published table, finite-N diagnostic.
    flow = count / (times[-1] - times[0]) if count >= 2 and times[-1] > times[0] else 0.0
    return {
        "seed": seed,
        "width": width,
        "desired_speed": speed,
        "crossed": count,
        "flow": flow,
        "specific_flow": flow / width,
        "min_center_distance": float(distance(p, segs).min()),
        "crossing_times": times,
    }


def flow_task(task):
    """Evaluate both walking speeds and five widths for one dev seed.

    Returns:
        Measured output for the requested configuration."""
    f, b, seed = task
    return {
        "factor": f,
        "offset": b,
        "seed": seed,
        "flow": [crowd(f, b, seed, w, sp) for w in [0.8, 1.0, 1.2, 2.0, 3.6] for sp in [0.65, 1.3]],
    }


def lanes(pair):
    # Use canonical builder/order parameter with an explicit obstacle layout conversion.

    """Measure the canonical lane order parameter with correctly placed walls.

    Returns:
        Measured output for the requested configuration."""
    import robot_sf.research.emergent_phenomena as ep

    orig = ep._BUILDERS["bidirectional_corridor"]

    def corrected(c, cal):
        state, walls, dirs = orig(c, cal)
        return state, [(a, c_, b, d) for a, b, c_, d in walls], dirs

    ep._BUILDERS["bidirectional_corridor"] = corrected
    f, b = pair
    rows = []
    for s in SEEDS:
        sc = ScenarioConfig("bidirectional_corridor", 24.0, 2.5, 24, s, 400)
        for cal in [RELEASED_DEFAULT_CALIBRATION, LITERATURE_CALIBRATION]:
            r = run_scenario(sc, cal, config(f, b))
            rows.append(
                {
                    "seed": s,
                    "calibration": cal.name,
                    "order_parameters": r.order_parameters,
                    "min_center_distance": float(
                        distance(
                            r.trajectory.positions, [(-1, 2.5, 25, 2.5), (-1, -2.5, 25, -2.5)]
                        ).min()
                    ),
                }
            )
    ep._BUILDERS["bidirectional_corridor"] = orig
    return {"factor": f, "offset": b, "lanes": rows}


def main():
    """Run one diagnostic stage with bounded workers and recorded provenance."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "flow", "lanes"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pairs", type=Path)
    a = ap.parse_args()
    assert 1 <= a.workers <= 16
    assert all(1001 <= s <= 1010 for s in SEEDS)
    pairs = GRID if a.pairs is None else [tuple(p) for p in json.loads(a.pairs.read_text())]
    if any(not np.isfinite(f) or not np.isfinite(b) or f <= 0 for f, b in pairs):
        raise ValueError("grid factors must be positive and all parameters finite")
    print("RESOLVED SEEDS", SEEDS, "PAIRS", pairs, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    task = {"screen": screen, "flow": flow_task, "lanes": lanes}[a.stage]
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
        tasks = [(f, b, s) for f, b in pairs for s in SEEDS] if a.stage == "flow" else pairs
        futures = {pool.submit(task, p): p for p in tasks}
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            pair = futures[fut]
            key = "_".join(str(v) for v in pair)
            write_json(a.out / f"{a.stage}_{key}.json", row)
            print("DONE", a.stage, pair, row.get("screen_pass"), flush=True)
    write_json(
        a.out / f"{a.stage}_manifest.json",
        {
            "seeds": SEEDS,
            "grid": pairs,
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "workers": a.workers,
        },
    )


if __name__ == "__main__":
    main()
