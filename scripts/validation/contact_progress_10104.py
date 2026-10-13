"""Development-only contact progress, swept exclusion and paired cost diagnostics.

This is an implementation diagnostic, not empirical or release qualification.
The off timing arm receives exactly the admissible inputs of the contact arm.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pysocialforce

from scripts.validation import pedestrian_validation_10074 as suite


def swept_pair_minimum(positions: np.ndarray) -> float:
    """Return the analytic minimum centre distance throughout linear steps."""
    first, second = np.triu_indices(positions.shape[1], 1)
    minimum = np.inf
    for start in range(0, len(positions) - 1, 32):
        before = positions[start : start + 32]
        after = positions[start + 1 : start + 33]
        delta = before[:, first] - before[:, second]
        change = (after[:, first] - after[:, second]) - delta
        denominator = np.sum(change * change, axis=-1)
        alpha = np.clip(-np.sum(delta * change, axis=-1) / np.maximum(denominator, 1e-30), 0, 1)
        minimum = min(
            minimum, float(np.linalg.norm(delta + alpha[..., None] * change, axis=-1).min())
        )
    return float(minimum)


def run_diagnostics(destination: Path, seed: int = 1001) -> list[dict[str, Any]]:  # noqa: C901
    """Run the four review workloads and preserve every timing sample and state.

    Returns:
        Diagnostic rows with crossings, physical minima and measured cost.
    """
    if not 1001 <= seed <= 1030:
        raise ValueError("the CALFIT diagnostic requires an approved development seed")
    destination.mkdir(parents=True, exist_ok=True)
    original = pysocialforce.Simulator.step
    records: list[dict[str, Any]] = []
    off: Any = None
    snapshots: list[np.ndarray] = []
    timings: list[list[int]] = [[], []]
    walls = np.empty((0, 6))

    def paired(sim, *args, **kwargs):
        nonlocal off, walls
        if off is None:
            config = copy.deepcopy(sim.config)
            for key in (
                "pedestrian_contact_rule",
                "pedestrian_contact_passes",
                "_contact_step_receipts",
            ):
                if hasattr(config, key):
                    delattr(config, key)
            walls = sim.get_raw_obstacles().copy()
            obstacles = [(float(w[0]), float(w[2]), float(w[1]), float(w[3])) for w in walls]
            off = pysocialforce.Simulator(sim.peds.state.copy(), obstacles=obstacles, config=config)
            off_peds = cast("Any", off.peds)
            off_peds.assign_desired_speeds(sim.peds.max_speeds.copy())
            integrate = off_peds.step

            def capped(force, *integration_args, **integration_kwargs):
                desired = np.asarray(off_peds.max_speeds, dtype=float).copy()
                off_peds.max_speeds = np.full(off_peds.size(), 3.0)
                try:
                    return integrate(force, *integration_args, **integration_kwargs)
                finally:
                    off_peds.max_speeds = desired

            off_peds.step = capped
        before = sim.peds.state.copy()
        snapshots.append(before)
        off.peds.state[:] = before
        if len(timings[0]) % 160 == 0:
            np.testing.assert_array_equal(sim.compute_forces(), off.compute_forces())
        arms, result = [off, sim], None
        for arm in [0, 1] if len(timings[0]) % 2 == 0 else [1, 0]:
            start = time.perf_counter_ns()
            value = original(arms[arm], *args, **kwargs)
            timings[arm].append(time.perf_counter_ns() - start)
            if arm == 1:
                result = value
        return result

    pysocialforce.Simulator.step = paired
    try:
        for case, width, passes in (
            ("V3", "1.0", 3),
            ("V3", "1.0", 10),
            ("V4", "2.4", 3),
            ("V4", "2.4", 10),
        ):
            off, snapshots, timings = None, [], [[], []]
            options = {
                "calfit": True,
                "speed_tier": "literature",
                "execution_cap_m_s": 3.0,
                "wall_profile": "body_edge_v3",
                "pedestrian_contact_rule": "projection_v2",
                "pedestrian_contact_passes": passes,
            }
            row = suite.run_task((case, seed, width, 0.28, "radius", options))
            raw = np.asarray(timings, dtype=np.int64)
            off_ns, on_ns = raw[:, 10:].sum(axis=1)
            count = raw.shape[1] - 10
            record = {
                "case": case,
                "passes": passes,
                "seed": seed,
                "crossed": row["crossed"],
                "steps": len(row["_positions"]) - 1,
                "runtime": row["step_runtime"],
                "pair_overlap": row["pair_overlap"],
                "minimum_swept_centre_distance_m": swept_pair_minimum(row["_positions"]),
                "wall_penetration_m": row["wall_penetration_m"],
                "timing_samples_per_arm": count,
                "warmup_excluded_per_arm": 10,
                "off_ns_per_step": float(off_ns / count),
                "on_ns_per_step": float(on_ns / count),
                "overhead_percent": float((on_ns / off_ns - 1) * 100),
                "off_ns_blocks": [float(v.mean()) for v in np.array_split(raw[0, 10:], 20)],
                "on_ns_blocks": [float(v.mean()) for v in np.array_split(raw[1, 10:], 20)],
            }
            np.savez_compressed(
                destination / f"{case}-{passes}.npz",
                positions=row["_positions"],
                states=np.asarray(snapshots),
                times_ns=raw,
                walls=walls,
                desired=row["desired_speeds_m_s"],
            )
            records.append(record)
            (destination / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
            print(
                f"{case}, passes={passes}: crossed={record['crossed']}, cost={record['overhead_percent']:.2f}%",
                flush=True,
            )
    finally:
        pysocialforce.Simulator.step = original
    return records


def main() -> int:
    """Write reproducible diagnostic output; return failure if any target fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = run_diagnostics(args.output, args.seed)
    (args.output / "source.json").write_text(
        json.dumps({"head": head, "substrate": pysocialforce.__file__}, indent=2) + "\n"
    )
    manifest = []
    for artifact in sorted(args.output.iterdir()):
        if artifact.is_file() and artifact.name != "SHA256SUMS":
            manifest.append(
                f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  {artifact.name}\n"
            )
    (args.output / "SHA256SUMS").write_text("".join(manifest))
    qualified = all(
        row["crossed"] >= 0.9 * (60 if row["case"] == "V3" else 350)
        and row["minimum_swept_centre_distance_m"] >= 0.56 - 1e-10
        and row["wall_penetration_m"] <= 1e-10
        and row["runtime"][0]["fallback_rate"] < 0.01
        and row["runtime"][0]["unresolved_count"] == 0
        and row["runtime"][0]["over_cap_samples"] == 0
        and row["overhead_percent"] <= 15.0
        for row in rows
    )
    return 0 if qualified else 3


if __name__ == "__main__":
    raise SystemExit(main())
