"""Bounded PEDCONTACT successor: full V6 necessary-condition screening before costly full banks."""

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
    if not comparison["fit_admitted"] or robot["pairs"] != 7200:
        raise ValueError("complete step-4 qualification and robot gate required")
    ps = points()
    blob = {
        "schema": "pedcontact.bounded_fit.v1",
        "seeds": SEEDS,
        "deadline_utc": DEADLINE.isoformat(),
        "points": ps,
        "groups": sorted({group(p) for p in ps}),
        "screen": "V6 all three controlled speeds; necessary-condition pruning only",
        "invariance": "V6 has no obstacles, wall force is exactly zero for all wall parameters; other parameters fixed within each radius/cap group",
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


def acquire(root, index):  # noqa: C901
    """Measure all V6 cases for one frozen equivalence class on Slurm."""
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 1:
        raise ValueError("one-CPU Slurm acquisition required")
    if datetime.now(UTC) >= DEADLINE:
        raise RuntimeError("original five-day deadline reached")
    pending = subprocess.check_output(
        ["squeue", "--states=PENDING", "--noheader", "--format=%j"], text=True
    )
    if any("0.0.8" in s or "008" in s or "sealed" in s.lower() for s in pending.splitlines()):
        raise RuntimeError("priority 0.0.8 campaign pending")
    grid = json.loads((root / "grid.json").read_text())
    if index is None or not 0 <= index < len(grid["groups"]):
        raise ValueError("explicit bounded screen index required")
    if (
        grid["source_sha"]
        != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        or grid["driver_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("frozen producer or driver changed")
    key = grid["groups"][index]
    p = next(
        p
        for p in grid["points"]
        if group(p) == key
        and p["wall_contact_parameters"] == {"amplitude_m_s2": 3.0, "decay_m": 0.04, "range_m": 0.2}
    )
    cfg = suite.load_config(suite.DEFAULT_CONFIG)
    cfg["seeds"] = SEEDS
    options = {
        "calfit": True,
        "speed_tier": "literature",
        "wall_candidate": p,
        "wall_profile": "legacy_v1",
        "execution_cap_m_s": p["cap_m_s"],
        "shoulder_width_m": cfg["V2"]["shoulder_proxy_m"],
        "equivalence": {f"V{i}": cfg[f"V{i}"]["equivalence"] for i in range(1, 7)},
    }
    for name in [
        "pedestrian_contact_rule",
        "pedestrian_wall_rule",
        "wall_contact_parameters",
        "pedcontact_measurement",
    ]:
        options[name] = p[name]
    tasks = [t for t in suite.protocol_tasks(cfg, p["radius_m"], "radius", options) if t[0] == "V6"]
    out = root / "screens" / key
    out.mkdir(parents=True, exist_ok=False)
    live = search._identity(p, SEEDS, cfg)
    live.update(
        grid_sha256=hashlib.sha256((root / "grid.json").read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="V6 necessary condition only",
    )
    write_json(out / "identity.json", live)
    print("RESOLVED SEEDS", SEEDS, "GROUP", key, "TASKS", len(tasks), flush=True)
    rows = []
    for i, task in enumerate(tasks):
        if datetime.now(UTC) >= DEADLINE:
            raise RuntimeError("five-day deadline reached")
        row = suite.run_task(task)
        arrays = {k[1:]: row.pop(k) for k in list(row) if k.startswith("_")}
        raw = out / f"trajectory_{i:04}.npz"
        np.savez_compressed(raw, **arrays)
        row.update(
            raw_trajectory=raw.name,
            raw_trajectory_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        )
        if row.get("wall_penetration_ped_steps") != 0:
            raise RuntimeError("unexpected V6 wall geometry")
        # Independent empty-wall negative control using the effective force object.
        from pysocialforce import Simulator
        from pysocialforce.config import SimulatorConfig

        sc = SimulatorConfig()
        sc.obstacle_force_config.wall_contact_rule = "bounded_edge_v1"
        for name, value in p["wall_contact_parameters"].items():
            setattr(sc.obstacle_force_config, "wall_contact_" + name, value)
        sim = Simulator(np.array([[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.5]]), obstacles=[], config=sc)
        if not np.array_equal(sim.forces[2](), np.zeros((1, 2))):
            raise RuntimeError("empty-wall invariance failed")
        write_json(out / f"case_{i:04}.json", row)
        rows.append(row)
        print("DONE", i + 1, "/", len(tasks), row["variant"], row["seed"], flush=True)
    gate = suite.acceptance_gate(rows, config=cfg, require_complete=False)
    write_json(out / "gate.json", gate)
    members = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(out.iterdir())
        if f.is_file()
    }
    write_json(
        out / "completed.json",
        {
            "group": key,
            "members_sha256": members,
            "v6_checks": gate["checks"],
            "v6_all_pass": all(c["status"] == "PASS" for c in gate["checks"]),
            "physical_violations": gate["physical_violations"],
        },
    )


def collect(root):
    """Verify complete raw banks before pruning a necessary-condition failure."""
    grid = json.loads((root / "grid.json").read_text())
    screens = {}
    for key in grid["groups"]:
        directory = root / "screens" / key
        d = json.loads((directory / "completed.json").read_text())
        for name, digest in d["members_sha256"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError("screen member changed")
        identity = json.loads((directory / "identity.json").read_text())
        if identity["grid_sha256"] != hashlib.sha256((root / "grid.json").read_bytes()).hexdigest():
            raise ValueError("grid identity drift")
        if (
            identity["source_sha"] != grid["source_sha"]
            or identity["driver_sha256"] != grid["driver_sha256"]
        ):
            raise ValueError("screen producer differs from frozen grid")
        rows = [json.loads(p.read_text()) for p in sorted(directory.glob("case_*.json"))]
        cfg = suite.load_config(suite.DEFAULT_CONFIG)
        cfg["seeds"] = SEEDS
        expected = {
            (t[0], t[1], t[2])
            for t in suite.protocol_tasks(cfg, identity["candidate"]["radius_m"], "radius", {})
            if t[0] == "V6"
        }
        observed = [(r["case"], r["seed"], r["variant"]) for r in rows]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError("incomplete or duplicate V6 screen")
        gate = suite.acceptance_gate(rows, config=cfg, require_complete=False)
        if gate["checks"] != d["v6_checks"]:
            raise ValueError("screen gate disagrees with raw case measurements")
        screens[key] = d
    rejected = [p["id"] for p in grid["points"] if not screens[group(p)]["v6_all_pass"]]
    survivors = [p for p in grid["points"] if screens[group(p)]["v6_all_pass"]]
    result = {
        "schema": "pedcontact.fit_screen_result.v1",
        "source_sha": grid["source_sha"],
        "points": len(grid["points"]),
        "screen_groups": screens,
        "rejected_necessary_condition_n": len(rejected),
        "rejected_ids": rejected,
        "survivors_for_full_banks": survivors,
        "accepted_full_settings": [],
        "claim": "necessary-condition pruning, not full-grid fitness or global optimum",
        "deadline_utc": grid["deadline_utc"],
    }
    write_json(root / "screen_result.json", result)
    print("REJECTED", len(rejected), "SURVIVORS", len(survivors), flush=True)


def main():
    """Acquire or audit the bounded dev-only necessary-condition screen."""
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["freeze", "screen", "collect"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if a.mode == "freeze":
        a.root.mkdir(parents=True, exist_ok=True)
        freeze(a.root)
    elif a.mode == "screen":
        acquire(a.root, a.index)
    else:
        collect(a.root)


if __name__ == "__main__":
    main()
