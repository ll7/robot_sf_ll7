"""Bounded PEDCONTACT successor: complete banks and ranking, with no item pruning."""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from robot_sf.evidence.writers import write_json
from scripts.validation import calfit_search_10074 as search
from scripts.validation import pedestrian_validation_10074 as suite

ROOT = Path(__file__).resolve().parents[2]
DEADLINE = datetime(2026, 10, 7, 18, 32, tzinfo=UTC)
SEEDS = [1001, 1002, 1003]
# 48 main scenarios + 3 doorway widths + 6 probes, five arms, ten dev seeds.
ROBOT_GATE_PAIRS = (48 + 3 + 6) * 5 * 10 * 3


def identity(point):
    """Hash the complete explicit candidate without its previous identifier.

    Returns:
        Copy with a deterministic candidate identifier.
    """
    point = dict(point)
    point.pop("id", None)
    point["id"] = hashlib.sha256(json.dumps(point, sort_keys=True).encode()).hexdigest()[:12]
    return point


def points():
    """Declare the bounded successor before any fit results are available.

    Returns:
        All 108 explicit candidate settings.
    """
    result = []
    for radius, cap, a, b, rng in itertools.product(
        [0.25, 0.28, 0.30], [2.0, 3.0], [3.0, 6.0, 9.0], [0.04, 0.08, 0.16], [0.2, 0.5]
    ):
        p = search.candidate("calibrated_v2", 0.003, 0.375, radius, cap)
        p.update(
            pedcontact_measurement=True,
            pedestrian_contact_rule="projection_v1",
            pedestrian_wall_rule="bounded_edge_v1",
            wall_contact_parameters={"amplitude_m_s2": a, "decay_m": b, "range_m": rng},
        )
        result.append(identity(p))
    return result


def group(point):
    """Identify settings with identical wall-free V6 dynamics.

    Returns:
        Radius and execution-cap class identifier.
    """
    return f"r{point['radius_m']:.2f}_cap{point['cap_m_s']:.1f}"


def freeze(root):
    """Freeze the grid only after complete physical and robot-gate evidence."""
    comparison_path = root.parent / "step4_comparison.json"
    robot_path = root.parent / "robot_gate_summary.json"
    comparison = json.loads(comparison_path.read_text())
    robot = json.loads(robot_path.read_text())
    if not comparison["fit_admitted"] or robot["pairs"] != ROBOT_GATE_PAIRS:
        raise ValueError("complete step-4 qualification and robot gate required")
    ps = points()
    blob = {
        "schema": "pedcontact.bounded_fit.v2",
        "seeds": SEEDS,
        "deadline_utc": DEADLINE.isoformat(),
        "points": ps,
        "groups": sorted({group(p) for p in ps}),
        "screen": "V2 and V5 evaluated for every wall setting; all 18 cases measured, no pruning",
        "invariance": "V6 has no walls; its physical threshold is independent of deterministic baseline noise. No invariance is used to prune candidates.",
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "admission_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (comparison_path, robot_path)
        },
    }
    write_json(root / "grid.json", blob)
    print("FROZEN", len(ps), "points", blob["groups"], flush=True)


def acquire(root, index):
    """Measure a complete 18-item bank for one point and dev seed, without pruning."""
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 1:
        raise ValueError("one-CPU Slurm acquisition required")
    if datetime.now(UTC) >= DEADLINE:
        raise RuntimeError("original five-day deadline reached")
    grid = json.loads((root / "grid.json").read_text())
    if index is None or not 0 <= index < len(grid["points"]) * len(SEEDS):
        raise ValueError("explicit complete-bank index required")
    if (
        grid["source_sha"]
        != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ):
        raise ValueError("frozen producer changed")
    if grid["driver_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("frozen driver changed")
    point = grid["points"][index // len(SEEDS)]
    seed = SEEDS[index % len(SEEDS)]
    search.run_candidate(point, [seed], root / "banks" / point["id"] / str(seed), workers=1)


def collect(root):
    """Rank every completely measured setting by passed items and range residual.

    Returns:
        Full per-item values, intervals, physical counters and declared grid ranking.
    """
    from scripts.validation.pedcontact_10101 import interval

    grid = json.loads((root / "grid.json").read_text())
    cfg = suite.load_config(suite.DEFAULT_CONFIG)
    cfg["seeds"] = SEEDS
    ranked = []
    for point in grid["points"]:
        rows = []
        for seed in SEEDS:
            directory = root / "banks" / point["id"] / str(seed)
            search.verify_run(directory)
            live = search.read_json(directory / "identity.json")
            if (
                live["source_sha"] != grid["source_sha"]
                or live["candidate"] != point
                or live["seeds"] != [seed]
            ):
                raise ValueError("bank differs from frozen fit grid")
            rows.extend(search.read_json(path) for path in sorted(directory.glob("case_*.json")))
        gate = suite.acceptance_gate(rows, config=cfg, require_complete=True)
        checks = gate["checks"]
        values = []
        fields = {
            "V1": "fitted_desired_speed_m_s",
            "V2": "speed_drop_m_s",
            "V3": "specific_flow_persons_m_s",
            "V4": "all_data_specific_flow_persons_m_s",
            "V5": "lateral_cm_to_edge_m",
            "V6": "onset_m",
        }
        for check in checks:
            bank = [
                r for r in rows if r["case"] == check["case"] and r["variant"] == check["variant"]
            ]
            value = interval([r.get(fields[check["case"]]) for r in bank])
            if check["case"] == "V4" and "width slope" in check["variant"]:
                slopes = []
                for seed in SEEDS:
                    wide = sorted(
                        [r for r in rows if r["case"] == "V4" and r["seed"] == seed],
                        key=lambda r: float(r["variant"]),
                    )
                    quantity = check["variant"].removesuffix(" width slope")
                    if all(r.get(quantity) is not None for r in wide):
                        widths = np.asarray([float(r["variant"]) for r in wide])
                        flows = np.asarray([r[quantity] for r in wide])
                        slopes.append(float(widths @ flows / (widths @ widths)))
                value = interval(slopes)
            values.append(dict(check, observed_interval=value))
        # Missing/censored estimates rank behind measured range misses.
        residual = sum(
            float(c.get("distance_outside_tolerance") or 0)
            if c.get("estimate") is not None
            else 1e6
            for c in checks
        )
        passed = sum(c["status"] == "PASS" for c in checks)
        ranked.append(
            {
                "point": point,
                "passed_items": passed,
                "total_items": len(checks),
                "producible_items": gate["producible_items"],
                "unproducible_items": [c for c in checks if c["status"] not in {"PASS", "FAIL"}],
                "range_residual": residual,
                "gate_exit": gate["exit_code"],
                "values": values,
                "overlap_pair_steps": sum(r["pair_overlap"]["all"]["below_2r_count"] for r in rows),
                "wall_penetration_ped_steps": sum(r["wall_penetration_ped_steps"] for r in rows),
                "screen_V2_V5": [c for c in checks if c["case"] in {"V2", "V5"}],
                "physical_violations": gate["physical_violations"],
            }
        )
    ranked.sort(key=lambda r: (-r["passed_items"], r["range_residual"], r["point"]["id"]))
    result = {
        "schema": "pedcontact.bounded_fit_result.v2",
        "source_sha": grid["source_sha"],
        "seeds": SEEDS,
        "points": len(ranked),
        "ranking": ranked,
        "pruned_points": 0,
        "claim": "complete declared bounded grid; engineering-equivalent V6; no global optimum or empirical validation claim",
        "deadline_utc": grid["deadline_utc"],
        "accepted_full_settings": [r["point"] for r in ranked if r["gate_exit"] == 0],
    }
    write_json(root / "fit_result.json", result)
    print(
        "COMPLETE",
        len(ranked),
        "BEST",
        ranked[0]["passed_items"],
        "/",
        ranked[0]["total_items"],
        flush=True,
    )
    return result


def main():
    """Acquire or audit complete bounded dev-only banks."""
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["freeze", "run", "collect"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if a.mode == "freeze":
        a.root.mkdir(parents=True, exist_ok=True)
        freeze(a.root)
    elif a.mode == "run":
        acquire(a.root, a.index)
    else:
        collect(a.root)


if __name__ == "__main__":
    main()
