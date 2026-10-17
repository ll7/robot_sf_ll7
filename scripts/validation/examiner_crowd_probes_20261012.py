"""Check encounter-window lanes and a goal-aligned unidirectional 1.2 m exit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pysocialforce as pysf
from loguru import logger

from robot_sf.research.emergent_phenomena import (
    RELEASED_DEFAULT_CALIBRATION,
    ScenarioConfig,
    TrajectoryRecord,
    build_high_density_exit,
    lane_purity,
    lane_segregation_index,
    released_default_config,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/context/robsweep/evidence"


def run() -> None:
    """Retain an independent exit fixture and shorter active-flow lane windows."""
    logger.remove()
    lanes, doors = [], []
    for seed in range(1061, 1091):
        bank = np.load(EVIDENCE / "sanity" / f"bidirectional_corridor_{seed}.npz")
        for begin, end in ((5, 15), (10, 30), (30, 60)):
            # Canonical estimators average the latter half: slice an equal-sized
            # prefix so their second half is exactly the requested window.
            start = max(0, 2 * begin - end)
            trajectory = TrajectoryRecord(
                bank["positions"][start * 10 : end * 10 + 1],
                bank["velocities"][start * 10 : end * 10 + 1],
                bank["directions"],
                np.arange((end - start) * 10 + 1) * 0.1,
                0.1,
            )
            lanes.append(
                {
                    "seed": seed,
                    "requested_window_s": [begin, end],
                    "actual_window_s": [(start + end) / 2, end],
                    "segregation": lane_segregation_index(trajectory),
                    "purity": lane_purity(trajectory),
                }
            )
        cfg = ScenarioConfig(
            "high_density_exit", 12.0, 2.5, 40, seed, 600, {"exit_half_width": 0.6}
        )
        state, obstacles, directions = build_high_density_exit(cfg, RELEASED_DEFAULT_CALIBRATION)
        rng = np.random.default_rng(seed)
        gx, gy = np.meshgrid(np.arange(1.0, 9.0), np.linspace(-1.8, 1.8, 5))
        centers = np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-0.04, 0.04, (40, 2))
        distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if distances.min() < 0.8 or np.abs(centers[:, 1]).max() > 2.1:
            raise ValueError("Initial pedestrian placement violates physical clearance")
        state[:, :2] = centers
        state[:, 4] = 16.0  # target beyond the door, not on the exit plane.
        state[:, 5] = 0.0
        physics = released_default_config()
        physics.social_force_config.kernel_version = "wrapped_v2"
        sim = pysf.Simulator(state=state, obstacles=obstacles, config=physics)
        positions = [sim.peds.pos().copy()]
        for _ in range(cfg.n_steps):
            sim.step()
            positions.append(sim.peds.pos().copy())
        pos = np.asarray(positions)
        before, after = pos[:-1], pos[1:]
        crossing = (before[:, :, 0] < 12.0) & (after[:, :, 0] >= 12.0)
        alpha = np.zeros(crossing.shape)
        np.divide(
            12.0 - before[:, :, 0], after[:, :, 0] - before[:, :, 0], out=alpha, where=crossing
        )
        y = before[:, :, 1] + alpha * (after[:, :, 1] - before[:, :, 1])
        valid = crossing & (np.abs(y) <= 0.2)
        # Physical benchmark pedestrian radius is 0.4; substrate force radius 0.35.
        wall_clearances = []
        for x1, x2, y1, y2 in obstacles:
            start = np.array([x1, y1])
            edge = np.array([x2 - x1, y2 - y1])
            fraction = np.clip(((pos - start) * edge).sum(axis=-1) / (edge @ edge), 0, 1)
            wall_clearances.append(
                np.linalg.norm(pos - start - fraction[:, :, None] * edge, axis=-1) - 0.4
            )
        wall = np.min(wall_clearances, axis=0)
        count = int(valid.any(axis=0).sum())
        pair_distances = np.linalg.norm(pos[0, :, None, :] - pos[0, None, :, :], axis=-1)
        np.fill_diagonal(pair_distances, np.inf)
        clear_actors = (wall >= -1e-6).all(axis=0)
        doors.append(
            {
                "seed": seed,
                "valid_unique_crossings": count,
                "raw_unique_crossings": int(crossing.any(axis=0).sum()),
                "throughput_person_s": count / 60.0,
                "finite": bool(np.isfinite(pos).all()),
                "wall_overlap_ped_steps": int((wall < -1e-6).sum()),
                "min_wall_clearance_m": float(wall.min()),
                "initial_min_wall_clearance_m": float(wall[0].min()),
                "initial_min_pair_surface_clearance_m": float(pair_distances.min() - 0.8),
                "wall_clear_unique_crossings": int((valid.any(axis=0) & clear_actors).sum()),
            }
        )
        np.savez_compressed(
            EVIDENCE / "sanity" / f"unidirectional_door_{seed}.npz",
            positions=pos,
            directions=directions,
        )
    payload = {
        "evidence_tier": "diagnostic-only",
        "lane_windows": lanes,
        "unidirectional_door": doors,
        "door_width_m": 1.2,
        "preferred_speed_m_s": 0.65,
        "duration_s": 60.0,
        "population": 40,
        "initial_region_m": [0.96, 8.04, -1.84, 1.84],
        "goal_m": [16.0, 0.0],
        "placement": "clear_jittered_grid_v1; 8 columns, 5 rows, uniform +/-0.04 m jitter; independently checked physical clearance",
        "force_radius_m": 0.35,
        "social_force_kernel_version": str(physics.social_force_config.kernel_version),
        "physical_check_radius_m": 0.4,
        "claim_boundary": "No robot, route-aligned exit. Finite population; no empirical throughput fit or sustained-flow qualification.",
    }
    (EVIDENCE / "crowd_probes.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "unidirectional_mean_person_s": float(
                    np.mean([r["throughput_person_s"] for r in doors])
                ),
                "unidirectional_total_crossings": sum(r["valid_unique_crossings"] for r in doors),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    EVIDENCE = args.output_dir.resolve()
    if (EVIDENCE / "crowd_probes.json").exists():
        raise FileExistsError("Existing crowd probe output; select a fresh --output-dir")
    run()
