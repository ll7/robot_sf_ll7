#!/usr/bin/env python3
"""Diagnostic clearance distributions and passage flow for the corrected wall-law fixture.

Reuses #10304's seeded simulator and termination unchanged. This sparse,
bidirectional diagnostic does not estimate saturated empirical capacity.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import shapely

from scripts.validation import issue_10304_wall_law_matrix as matrix

ARMS = ("legacy", "physical_margin10x")
OPENINGS = ((20.0, 4.0), (40.0, 5.0))


def crossing_times(trace: np.ndarray, dt: float, plane: float) -> np.ndarray:
    """First crossing per pedestrian, in the authored travel direction; interpolate."""
    direction = np.sign(trace[-1, :, 0] - trace[0, :, 0])
    before = (trace[:-1, :, 0] - plane) * direction
    after = (trace[1:, :, 0] - plane) * direction
    times = []
    for ped in range(trace.shape[1]):
        matches = np.flatnonzero((before[:, ped] < 0) & (after[:, ped] >= 0))
        if not len(matches):
            raise ValueError(f"pedestrian {ped} did not cross x={plane}")
        tick = int(matches[0])
        fraction = -before[tick, ped] / (after[tick, ped] - before[tick, ped])
        times.append((tick + fraction) * dt)
    return np.asarray(times)


def clearance_stats(values: np.ndarray) -> dict:
    """Exact circular body clearance to nearest solid, pooled over pedestrian-steps."""
    values = np.asarray(values).ravel()
    assert values.size and np.isfinite(values).all()
    return {
        "pedestrian_steps": int(values.size),
        "median_clearance_m": float(np.quantile(values, 0.5, method="linear")),
        "p10_clearance_m": float(np.quantile(values, 0.1, method="linear")),
        "steps_below_10cm": int(np.count_nonzero(values < 0.10)),
        "share_below_10cm": float(np.mean(values < 0.10)),
        "minimum_clearance_m": float(values.min()),
    }


def run_trial(payload: tuple[str, int, str]) -> dict:
    """Observe the existing runner without altering force, integration, or termination."""
    variant, seed, artifact_dir = payload
    matrix._validate_seed(seed)
    original_overlap = matrix._overlap_state
    original_init = matrix.init_simulators
    positions = []
    timing = {}
    polygons = []

    def observe_overlap(current, obstacles):
        positions.append(np.asarray(current).copy())
        if not polygons:
            polygons.extend(obstacles)
        return original_overlap(current, obstacles)

    def observe_init(config, *args, **kwargs):
        timing["dt_s"] = config.sim_config.time_per_step_in_secs
        simulators = original_init(config, *args, **kwargs)
        return simulators

    matrix._overlap_state = observe_overlap
    matrix.init_simulators = observe_init
    try:
        baseline = matrix._run_one((variant, seed))
    finally:
        matrix._overlap_state = original_overlap
        matrix.init_simulators = original_init
    trace = np.asarray(positions)
    assert trace.shape == (baseline.steps + 1, baseline.pedestrians, 2)
    # Distance from center to solid minus body radius gives the exact disc edge
    # clearance (unlike the historical polygonal circle approximation).
    centers = shapely.points(trace[1:])
    distances = (
        np.min(np.stack([shapely.distance(centers, obstacle) for obstacle in polygons]), axis=0)
        - matrix.BODY_RADIUS_M
    )
    crossings = []
    for plane, width in OPENINGS:
        times = crossing_times(trace, timing["dt_s"], plane)
        duration = float(np.ptp(times))
        assert duration > 0
        crossings.append(
            {
                "plane_x_m": plane,
                "width_m": width,
                "times_s_by_pedestrian": times.tolist(),
                "headway_count": len(times) - 1,
                "first_to_last_s": duration,
                "specific_flow_persons_per_m_s": (len(times) - 1) / duration / width,
            }
        )
    destination = Path(artifact_dir) / f"{variant}-{seed}.npz"
    np.savez_compressed(destination, positions=trace, clearance_m=distances)
    return {
        "baseline": asdict(baseline),
        "dt_s": timing["dt_s"],
        "clearance": clearance_stats(distances),
        "crossings": crossings,
        "raw_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def main() -> None:
    """Run only authorized dev seeds and write reproducible diagnostic evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1001-1200")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seeds = matrix._parse_seed_spec(args.seeds)
    if len(seeds) != len(set(seeds)):
        parser.error("duplicate seeds would distort the paired comparison")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [(arm, seed, str(args.output_dir)) for arm in ARMS for seed in seeds]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, 8)) as pool:
        rows = list(pool.map(run_trial, payloads))
    summaries = []
    for arm in ARMS:
        selected = [row for row in rows if row["baseline"]["variant"] == arm]
        samples = []
        for seed in seeds:
            with np.load(args.output_dir / f"{arm}-{seed}.npz") as raw:
                samples.append(raw["clearance_m"].ravel())
        flow = []
        for index, (plane, width) in enumerate(OPENINGS):
            passages = [row["crossings"][index] for row in selected]
            rates = [row["specific_flow_persons_per_m_s"] for row in passages]
            flow.append(
                {
                    "plane_x_m": plane,
                    "width_m": width,
                    "pooled_specific_flow_persons_per_m_s": sum(
                        row["headway_count"] for row in passages
                    )
                    / sum(row["first_to_last_s"] for row in passages)
                    / width,
                    "mean_per_seed_specific_flow": float(np.mean(rates)),
                    "p10_per_seed_specific_flow": float(np.quantile(rates, 0.1)),
                    "p90_per_seed_specific_flow": float(np.quantile(rates, 0.9)),
                }
            )
        summaries.append(
            {
                **matrix._summarize([matrix.TrialResult(**row["baseline"]) for row in selected])[0],
                "clearance_distribution": clearance_stats(np.concatenate(samples)),
                "flow": flow,
            }
        )
    paths = [Path(__file__), Path(matrix.__file__), Path("uv.lock")]
    paths.extend(sorted(Path("configs/scenarios").glob("*.yaml")))
    paths.append(
        Path(
            "maps/successor_svg_maps/issue_9762_classic_realworld_bottleneck_goal_zone_entry_v2.svg"
        )
    )
    result = {
        "evidence_tier": "diagnostic-only",
        "execution_mode": "native",
        "fallback_or_degraded_rows": 0,
        "git_head": matrix._git_head(),
        "dirty_paths": subprocess.check_output(["git", "status", "--short"], text=True),
        "seeds": seeds,
        "input_sha256": {
            str(path.relative_to(Path.cwd()) if path.is_absolute() else path): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        },
        "clearance_window": "every pedestrian after every integration step until all goals; no tick zero",
        "clearance_geometry": "center distance to nearest solid polygon minus 0.40m circular body radius",
        "flow_estimator": "pooled (N-1)/sum(last-first crossing seconds)/physical opening width; first directed crossings",
        "summary": summaries,
        "trials": rows,
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
