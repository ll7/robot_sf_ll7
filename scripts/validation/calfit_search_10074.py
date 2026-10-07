"""Bounded opt-in CALFIT grid, raw acquisition, Pareto analysis and one refinement.

No experiment runs without explicit dev seeds and an allocation receipt. Raw
outputs and engineering verdicts remain separate from release admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pysocialforce

from robot_sf.evidence.writers import write_json, write_text
from scripts.validation import pedestrian_validation_10074 as suite


def candidate(family, factor, offset, radius, cap) -> dict[str, object]:
    """Return a deterministic candidate with actual law/parameter identity."""
    data = {
        "family": family,
        "factor": float(factor),
        "offset_m": float(offset),
        "radius_m": float(radius),
        "cap_m_s": float(cap),
        "desired_mean_m_s": 1.29,
        "desired_sd_m_s": 0.19,
    }
    data["id"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]
    return data


def coarse_grid() -> list[dict[str, object]]:
    """Return the 114-point grid recorded before the author ruling."""
    points = []
    for radius in [0.25, 0.28, 0.30]:
        for cap in [2.0, 3.0]:
            points.append(candidate("legacy_v1", 10.0, -0.57, radius, cap))
            points += [
                candidate("calibrated_v2", f, b, radius, cap)
                for f in [0.0003, 0.003, 0.03]
                for b in [0.25, 0.375]
            ]
            points += [
                candidate("gradient_v3", f, radius, radius, cap) for f in [0.0003, 0.001, 0.003]
            ]
            points += [
                candidate("exponential_edge", a, b, radius, cap)
                for a in [5.0, 15.0, 40.0]
                for b in [0.02, 0.04, 0.08]
            ]
    assert len(points) == 114 and len({p["id"] for p in points}) == 114
    return points


def read_json(path):
    """Read a preserved acquisition or grid JSON object."""
    return json.loads(Path(path).read_text())


def _identity(point, seeds, config):
    substrate = Path(pysocialforce.__file__).parent
    files = {}
    names = ["forces.py", "config.py", "scene.py"]
    if point.get("pedcontact_measurement"):
        names += ["contact.py", "simulator.py"]
    for name in names:
        source = suite.ROOT / "fast-pysf/pysocialforce" / name
        installed = substrate / name
        digest = hashlib.sha256(installed.read_bytes()).hexdigest()
        if digest != hashlib.sha256(source.read_bytes()).hexdigest():
            raise RuntimeError("installed force substrate differs from source: " + name)
        files[name] = digest
    return {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=suite.ROOT, text=True
        ).strip(),
        "candidate": point,
        "seeds": seeds,
        "config": config,
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "installed_files": files,
        "python": subprocess.check_output([sys.executable, "--version"], text=True).strip(),
        "source_files": {
            str(p.relative_to(suite.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(suite.__file__),
                Path(suite.estimators.__file__),
                suite.ROOT / "robot_sf/research/pedestrian_acceptance.py",
                suite.ROOT / "robot_sf/research/pedestrian_initial_state.py",
            ]
        },
    }


def run_candidate(point, seeds, out, workers=2, config_path=None) -> dict[str, object]:
    """Acquire the complete V1-V6 bank and retain lossless raw measurements."""
    if (
        not os.environ.get("SLURM_JOB_ID")
        or workers > 2
        or workers > int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    ):
        raise ValueError("CALFIT acquisition requires Slurm and at most two workers per task")
    if (
        not seeds
        or any(type(s) is not int or not 1001 <= s <= 1030 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("CALFIT episodes require unique dev seeds1001-1030")
    if (
        not 0.25 <= point["radius_m"] <= 0.30
        or point["desired_mean_m_s"] != 1.29
        or point["desired_sd_m_s"] != 0.19
    ):
        raise ValueError("candidate violates the author calibration contract")
    # Refuse an already admitted sealed campaign before the simulation batch.
    queue = subprocess.check_output(
        ["squeue", "--states=PENDING", "--noheader", "--format=%j"], text=True
    )
    if any(
        ("0.0.8" in line or "008" in line or "sealed" in line.lower())
        for line in queue.splitlines()
    ):
        raise RuntimeError("priority sealed campaign pending; hold CALFIT before simulation")
    config = read_json(config_path) if config_path else suite.load_config(suite.DEFAULT_CONFIG)
    config["seeds"] = seeds
    options = {
        "calfit": True,
        "speed_tier": "literature",
        "wall_candidate": point,
        "wall_profile": "gradient_v3" if point["family"] == "gradient_v3" else "legacy_v1",
        "execution_cap_m_s": point["cap_m_s"],
        "shoulder_width_m": config["V2"]["shoulder_proxy_m"],
        "yaw_threshold_rad_s": config["V6"].get("yaw_threshold_rad_s", 0.05),
        "analysis_window_m": config["V6"].get("analysis_window_m", 3.0),
        "onset_persistence_s": config["V6"].get("persistence_s", 0.3),
        "equivalence": {f"V{i}": config[f"V{i}"]["equivalence"] for i in range(1, 7)},
    }
    for key in (
        "pedestrian_contact_rule",
        "pedestrian_wall_rule",
        "wall_contact_parameters",
        "pedcontact_measurement",
    ):
        if key in point:
            options[key] = point[key]
    grid = suite.protocol_tasks(config, point["radius_m"], "radius", options)
    print("RESOLVED SEEDS", seeds, "CANDIDATE", point, "CASES", len(grid), flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    identity = _identity(point, seeds, config)
    write_json(out / "identity.json", identity)
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(suite.run_task, t): i for i, t in enumerate(grid)}
        for future in as_completed(futures):
            row = future.result()
            i = futures[future]
            arrays = {key[1:]: row.pop(key) for key in list(row) if key.startswith("_")}
            trace = out / f"trajectory_{i:04}.npz"
            np.savez_compressed(trace, **arrays)
            row["raw_trajectory"] = trace.name
            row["raw_trajectory_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            write_json(out / f"case_{i:04}.json", row)
            rows.append(row)
            print("DONE", len(rows), "/", len(grid), row["case"], row["seed"], flush=True)
    observed = {(r["case"], r["seed"], r["variant"]) for r in rows}
    if len(rows) != len(grid) or observed != {(t[0], t[1], t[2]) for t in grid}:
        raise RuntimeError("acquisition grid is incomplete or duplicate")
    gate = suite.acceptance_gate(rows, config=config, require_complete=True)
    write_json(out / "gate.json", gate)
    summary = candidate_summary(point, rows, gate)
    write_json(out / "summary.json", summary)
    rendered = "| case | variant | estimate | target | difference | tolerance range | status | source |\n|---|---|---|---|---|---|---|---|\n"
    for c in gate["checks"]:
        rendered += (
            "| "
            + " | ".join(
                str(c[k])
                for k in [
                    "case",
                    "variant",
                    "estimate",
                    "target",
                    "difference",
                    "tolerance_range",
                    "status",
                    "source",
                ]
            )
            + " |\n"
        )
    rendered += "\nEvery tolerance: author-delegated engineering tolerance, 2026-10-02.\n"
    rendered += (
        "Physical violations: "
        + str(len(gate["physical_violations"]))
        + "; model limitation: "
        + str(gate["model_limitations"])
        + "\n"
    )
    write_text(out / "gate_table.md", rendered, issue_ref="#10074")
    write_text(
        out / "SHA256SUMS",
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
            for p in sorted(out.iterdir())
            if p.is_file()
        ),
        issue_ref="#10074",
    )
    verify_run(out)
    write_json(
        out / "completed.json",
        {"candidate_id": point["id"], "episode_n": len(rows), "gate_exit": gate["exit_code"]},
    )
    return summary


def verify_run(out) -> dict[str, object]:
    """Verify raw members, source scope and complete bank before using a point."""
    out = Path(out)
    identity = read_json(out / "identity.json")
    if any(not 1001 <= s <= 1030 for s in identity["seeds"]):
        raise ValueError("non-dev identity")
    hashes = {}
    for line in (out / "SHA256SUMS").read_text().splitlines():
        if not line or line.startswith("<!--"):
            continue
        digest, name = line.split("  ", 1)
        if (
            Path(name).name != name
            or name in hashes
            or hashlib.sha256((out / name).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("acquisition digest mismatch: " + name)
        hashes[name] = digest
    paths = sorted(out.glob("case_*.json"))
    rows = [read_json(p) for p in paths]
    expected = suite.protocol_tasks(
        identity["config"], identity["candidate"]["radius_m"], "radius", {"calfit": True}
    )
    observed = [(r["case"], r["seed"], r["variant"]) for r in rows]
    if len(observed) != len(expected) or set(observed) != {(t[0], t[1], t[2]) for t in expected}:
        raise ValueError("incomplete or duplicate acquisition")
    for path, row in zip(paths, rows, strict=True):
        if (
            path.name not in hashes
            or row["raw_trajectory"] not in hashes
            or row["raw_trajectory_sha256"] != hashes[row["raw_trajectory"]]
        ):
            raise ValueError("missing/mismatched raw receipt")
    if not {"identity.json", "gate.json", "summary.json", "gate_table.md"} <= hashes.keys():
        raise ValueError("manifest omits acquisition metadata")
    return identity


def candidate_summary(point, rows, gate) -> dict[str, object]:
    """Report narrow widths, censoring and dense-wall safety without fake zeros."""
    narrow = [r for r in rows if r["case"] == "V3"]
    wide = [r for r in rows if r["case"] == "V4"]
    flows = {}
    for width in [0.8, 0.9, 1.0, 1.1, 1.2]:
        bank = [r for r in narrow if float(r["variant"]) == width]
        observed = [
            r["specific_flow_persons_m_s"]
            for r in bank
            if r["specific_flow_persons_m_s"] is not None
        ]
        flows[str(width)] = {
            "mean": float(np.mean(observed)) if observed else None,
            "observed_n": len(observed),
            "attempted_n": len(bank),
            "mean_crossed": float(np.mean([r["crossed"] for r in bank])) if bank else None,
        }
    complete = bool(narrow) and all(r["all_crossed"] for r in narrow)
    targets = [1.61, 1.86, 1.90, 1.93, 1.97]
    return {
        "candidate": point,
        "gate_exit": gate["exit_code"],
        "narrow": flows,
        "narrow_complete": complete,
        "flow_score": min(
            flows[str(w)]["mean"] / q
            for w, q in zip([0.8, 0.9, 1.0, 1.1, 1.2], targets, strict=True)
        )
        if complete
        else None,
        "dense_wall_penetration_m": max((r["wall_penetration_m"] for r in wide), default=None),
        "dense_pair_overlap_count": sum(r["pair_overlap"]["all"]["below_2r_count"] for r in wide),
        "narrow_pair_overlap_count": sum(
            r["pair_overlap"]["all"]["below_2r_count"] for r in narrow
        ),
        "numeric_failures": gate["numeric_failures"],
        "measurement_missing": gate["measurement_missing"],
        "physical_violation_rows": len(gate["physical_violations"]),
        "complete_narrow_measurements": sum(
            r["specific_flow_persons_m_s"] is not None for r in narrow
        ),
    }


def pareto_front(summaries) -> list[dict[str, object]]:
    """Return nondominated points with complete finite-N evidence at every width."""
    eligible = [
        s for s in summaries if s["narrow_complete"] and s["dense_wall_penetration_m"] is not None
    ]
    return [
        a
        for a in eligible
        if not any(
            b["dense_wall_penetration_m"] <= a["dense_wall_penetration_m"]
            and b["flow_score"] >= a["flow_score"]
            and (
                b["dense_wall_penetration_m"] < a["dense_wall_penetration_m"]
                or b["flow_score"] > a["flow_score"]
            )
            for b in eligible
        )
    ]


def aggregate(root) -> dict[str, object]:
    """Read only completed, digest-verified candidate acquisitions."""
    summaries = []
    for path in sorted(Path(root).glob("*/completed.json")):
        verify_run(path.parent)
        summaries.append(read_json(path.parent / "summary.json"))
    return {
        "schema": "calfit.search.v1",
        "completed_n": len(summaries),
        "summaries": summaries,
        "pareto": pareto_front(summaries),
        "censored_n": sum(not s["narrow_complete"] for s in summaries),
    }


def refinement(summaries) -> dict[str, object]:  # noqa: C901
    """Prepare one bounded factor/radius refinement, deduplicating coarse points."""
    frontier = pareto_front(summaries)
    ranked = sorted(
        frontier or summaries,
        key=lambda s: (
            -s["complete_narrow_measurements"],
            s["dense_wall_penetration_m"],
            -(s["flow_score"] or 0),
            s["candidate"]["id"],
        ),
    )
    parents = []
    # Spread the four boundary points across measured law families where possible.
    for s in ranked:
        if s["candidate"]["family"] not in {p["candidate"]["family"] for p in parents}:
            parents.append(s)
        if len(parents) == 4:
            break
    for s in ranked:
        if len(parents) == 4:
            break
        if s not in parents:
            parents.append(s)
    seen = {s["candidate"]["id"] for s in summaries}
    points = []
    for s in parents:
        p = s["candidate"]
        for scale in [0.5, 1.0, 2.0]:
            for r in [
                max(0.25, round(p["radius_m"] - 0.01, 3)),
                p["radius_m"],
                min(0.30, round(p["radius_m"] + 0.01, 3)),
            ]:
                q = candidate(
                    p["family"],
                    p["factor"] * scale,
                    r if p["family"] == "gradient_v3" else p["offset_m"],
                    r,
                    p["cap_m_s"],
                )
                if q["id"] not in seen:
                    seen.add(q["id"])
                    points.append(q)
    return {
        "schema": "calfit.grid.v1",
        "stage": "refinement",
        "seeds": [1001, 1002, 1003],
        "parents": [s["candidate"]["id"] for s in parents],
        "selection": "complete-flow Pareto survivors"
        if frontier
        else "censored boundary screen; no numeric Pareto claim",
        "candidates": points,
    }


def main(argv=None) -> int:
    """Write a grid/summary or acquire one explicitly indexed Slurm candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["grid", "run", "summarize", "refine", "verify"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "grid":
        write_json(
            args.out,
            {
                "schema": "calfit.grid.v1",
                "stage": "coarse",
                "seeds": [1001, 1002, 1003],
                "candidates": coarse_grid(),
            },
        )
    elif args.mode == "run":
        grid = read_json(args.grid)
        point = grid["candidates"][args.index]
        run_candidate(point, grid["seeds"], args.out / point["id"], config_path=args.config)
    elif args.mode == "verify":
        verify_run(args.out)
    else:
        result = aggregate(args.root)
        write_json(args.out, refinement(result["summaries"]) if args.mode == "refine" else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
