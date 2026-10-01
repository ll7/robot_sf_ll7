"""Source-bound pedestrian diagnostics; one command, no empirical-fit admission.

Reuses #10073's aperture, corrected Seyfried-shaped and Liao harnesses verbatim.
Unverified measurement definitions produce null comparisons, never inferred targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pysocialforce

from robot_sf.evidence.writers import write_json, write_text
from robot_sf.research import emergent_phenomena as ep
from scripts.validation import compare_obstacle_laws_10061 as reused

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/benchmarks/pedestrian_validation_0_0_9.json"
CANDIDATE = ("legacy_refit", 10.0, -0.57)


def load_config(path):
    """Reject invalid seeds and any changed baseline interaction law before stepping."""
    data = json.loads(Path(path).read_text())
    seeds = data["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 1001 <= s <= 1030 for s in seeds)
    ):
        raise ValueError("new episodes require unique dev seeds 1001..1030")
    if data["model"] != {
        "wall_law": "legacy_v1",
        "wall_factor": 10.0,
        "wall_offset_m": -0.57,
        "wall_sigma": 0.0,
        "pair_kernel": "wrapped_v2",
        "force_radius_m": 0.35,
        "baseline_physical_radius_m": 0.40,
        "native_free_speed_m_s": 0.65,
        "robot_activation_threshold_m": 2.0,
    }:
        raise ValueError("baseline model overrides are forbidden")
    return data


def statistics(values):
    """Mean, sample standard deviation and n over observed finite values only."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite simulated measurement")
    return {
        "mean": float(arr.mean()) if len(arr) else None,
        "spread_sd": float(arr.std(ddof=1)) if len(arr) > 1 else (0.0 if len(arr) else None),
        "n": len(arr),
    }


def free_or_encounter(seed, kind, radius, speed=0.65):
    """Retain robot-free native walking or encounter traces without inventing onset."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(-0.03, 0.03)
    state = np.array([[0.0, y, speed / 1.3, 0.0, 100.0, y, 0.5]])
    segments = []
    if kind == "V5":
        segments = [
            (8.0, -0.5, 8.0, 0.5),
            (8.0, 0.5, 9.0, 0.5),
            (9.0, 0.5, 9.0, -0.5),
            (9.0, -0.5, 8.0, -0.5),
        ]
    elif kind == "V6":
        state = np.vstack([state, [14.0, -y, -speed / 1.3, 0.0, -100.0, -y, 0.5]])
    positions, speeds = reused.harness.simulate(
        state, segments, reused.candidate_config(CANDIDATE, speed), 200
    )
    result = {"speed_m_s": float(speeds[100:200, 0].mean()), "prescribed_speed_m_s": speed}
    if kind == "V5":
        alongside = positions[(positions[:, 0, 0] >= 8.0) & (positions[:, 0, 0] <= 9.0), 0]
        result["lateral_cm_to_edge_m"] = (
            float((np.abs(alongside[:, 1]) - 0.5).min()) if len(alongside) else None
        )
        result["passed"] = bool(positions[-1, 0, 0] > 9.0)
        result.update(reused.wall_metrics(positions, segments))
    if kind == "V6":
        result["min_pair_centre_m"] = float(
            np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1).min()
        )
        result["min_pair_edge_m"] = result["min_pair_centre_m"] - 2 * radius
        result["onset_m"] = None
    result["trajectory_sha256"] = hashlib.sha256(positions.tobytes()).hexdigest()
    return result


def lanes(seed, width, radius):
    """Use the existing corridor builder and segregation measure with physical margins."""
    scenario = ep.ScenarioConfig("bidirectional_corridor", 24.0, width / 2, 48, seed, 400)
    state, walls, directions = ep.build_bidirectional_corridor(
        scenario, ep.RELEASED_DEFAULT_CALIBRATION
    )
    margin = width / 2 - radius - 0.05
    state[:, 1] = np.clip(state[:, 1], -margin, margin)
    state[:, 5] = state[:, 1]
    positions, _ = reused.harness.simulate(
        state, walls, reused.candidate_config(CANDIDATE, 0.65), 400
    )
    trajectory = ep.TrajectoryRecord(
        positions, np.zeros_like(positions), directions, np.arange(401) * 0.1, 0.1
    )
    pair = np.linalg.norm(state[:, None, :2] - state[None, :, :2], axis=-1)
    np.fill_diagonal(pair, np.inf)
    return {
        "segregation": ep.lane_segregation_index(trajectory),
        "initial_pair_overlaps": int((pair < 2 * radius).sum() // 2),
        **reused.wall_metrics(positions, walls),
    }


def run_task(task):
    """Run one dev episode in an isolated worker, retaining the original force defaults."""
    case, seed, variant, radius, mode = task
    original_config = reused.candidate_config

    # Preserve the released .35 force/.40 physical mismatch in the baseline only.
    # All candidate cases use one radius; legacy sigma=0 has no radius-dependent wall term.
    def configured(candidate, speed):
        config = original_config(candidate, speed)
        if mode == "radius":
            config.scene_config.agent_radius = radius
        return config

    reused.candidate_config = configured
    reused.RADIUS = radius
    row = {"case": case, "seed": seed, "variant": variant, "radius_m": radius, "mode": mode}
    if case in {"V1", "V5", "V6"}:
        row.update(free_or_encounter(seed, case, radius, float(variant) if case == "V6" else 0.65))
    elif case == "V2":
        row.update(reused.aperture(CANDIDATE, seed, float(variant), 0.65))
        row["aperture_ratio_basis"] = (
            "archived physical width, source participant shoulder width unavailable"
        )
    elif case in {"V3", "V4"}:
        row.update(reused.bottleneck(CANDIDATE, seed, float(variant), wide=case == "V4"))
    elif case == "V7":
        row.update(lanes(seed, float(variant), radius))
    elif case == "V8":
        row.update(reused.corridor(CANDIDATE, seed, 0.65))
    else:
        raise ValueError(f"unknown case {case}")
    reused.candidate_config = original_config
    return row


def tasks(config, radius, mode):
    """Materialize the complete V1-V8 grid in a deterministic order."""
    variants = {
        "V1": ["native"],
        "V2": config["V2"]["diagnostic_widths_m"],
        "V3": config["V3"]["widths_m"],
        "V4": config["V4"]["widths_m"],
        "V5": ["diagnostic"],
        "V6": config["V6"]["speeds_m_s"],
        "V7": config["V7"]["widths_m"],
        "V8": ["both origins"],
    }
    return [
        (case, seed, str(v), radius, mode)
        for case, values in variants.items()
        for v in values
        for seed in config["seeds"]
    ]


def summary(rows, config):
    """Separate actual source estimators from diagnostics and missing definitions."""
    table = []
    keys = {
        "V1": "speed_m_s",
        "V2": "speed_drop_m_s",
        "V3": "specific_flow_persons_m_s",
        "V4": "flow_persons_s",
        "V5": "lateral_cm_to_edge_m",
        "V6": "onset_m",
        "V7": "segregation",
        "V8": "steady_center_distance_m",
    }
    for case, variant in sorted({(r["case"], r["variant"]) for r in rows}):
        group = [r for r in rows if (r["case"], r["variant"]) == (case, variant)]
        src = config[case]
        stat = statistics([r.get(keys[case]) for r in group])
        published, difference = None, None
        reason = src.get(
            "setup_status",
            "native released preferred speed 0.65 m/s; population speed distribution is not reproduced",
        )
        if case == "V1":
            published = {
                "mean_m_s": 1.29,
                "sd_m_s": 0.19,
                "slow_normal_fast_m_s": [1.15, 1.42, 1.78],
            }
            difference = stat["mean"] - 1.29
        if case == "V3":
            published = src["published_specific_flow_persons_m_s"][
                src["widths_m"].index(float(variant))
            ]
            difference = stat["mean"] - published if stat["mean"] is not None else None
            reason += f"; complete runs {stat['n']}/{len(group)}; nominal density may require initial disc overlaps"
        if case == "V2":
            published = src["published_drop_m_s"]
        if case == "V4":
            published = src["published_slopes_persons_m_s"]
        if case == "V5":
            published = {"longitudinal_m": 2.0, "lateral_m": 0.5}
        if case == "V6":
            published = src["published_onset_m"][src["speeds_m_s"].index(float(variant))]
        if case == "V7":
            published = "target not yet verified"
        if case == "V8":
            published = 0.4
            stat["edge"] = statistics([r["steady_edge_distance_m"] for r in group])
        table.append(
            {
                "case": case,
                "variant": variant,
                "quantity": keys[case],
                "published_value": published,
                "source": src["source"],
                "location": src["location"],
                "simulated": stat,
                "difference": difference,
                "known_reason": reason,
                "attempted_n": len(group),
                "crossed": statistics(r.get("crossed") for r in group),
                "censored_specific_flow": statistics(
                    r.get("censored_specific_flow_persons_m_s") for r in group
                ),
                "initial_pair_overlaps": statistics(
                    r.get("initial_footprint_pair_overlaps", r.get("initial_pair_overlaps"))
                    for r in group
                ),
                "seeds": [r["seed"] for r in group],
                "wall_penetration_max_m": max(r.get("wall_penetration_m", 0.0) for r in group),
                "diagnostic_passed_n": sum(r.get("passed", False) for r in group),
            }
        )
    table.append(wide_slope_summary(rows, config))
    return table


def wide_slope_summary(rows, config):
    """Retain per-seed wide-flow regression estimators without a stationarity claim."""
    wide = [r for r in rows if r["case"] == "V4"]
    slopes, intercept_slopes, central_slopes = [], [], []
    for seed in config["seeds"]:
        group = sorted((r for r in wide if r["seed"] == seed), key=lambda r: r["width_m"])
        if len(group) != len(config["V4"]["widths_m"]) or not all(r["all_crossed"] for r in group):
            continue
        widths = np.asarray([r["width_m"] for r in group])
        flows = np.asarray([r["flow_persons_s"] for r in group])
        central = np.asarray([r["central_20_80_flow_persons_s"] for r in group])
        slopes.append(float(widths @ flows / (widths @ widths)))
        intercept_slopes.append(float(np.polyfit(widths, flows, 1)[0]))
        central_slopes.append(float(widths @ central / (widths @ widths)))
    return {
        "case": "V4",
        "variant": "diagnostic width slope",
        "quantity": "persons/(m s)",
        "published_value": config["V4"]["published_slopes_persons_m_s"],
        "source": config["V4"]["source"],
        "location": config["V4"]["location"],
        "simulated": statistics(slopes),
        "difference": None,
        "known_reason": "exit-plane first-to-last fit through origin; original line/window unverified; only complete 350-person runs; fitted-intercept and central-period slopes retained; central period is not proven stationary",
        "fitted_intercept_slope": statistics(intercept_slopes),
        "central_period_slope": statistics(central_slopes),
        "attempted_n": len(config["seeds"]),
        "seeds": config["seeds"],
    }


def markdown(table):
    """Render the requested five columns while keeping units and missingness visible."""
    lines = [
        "| case | published value (source, table/figure) | simulated value (mean, spread, n) | difference | known reason |",
        "|---|---|---|---|---|",
    ]
    for row in table:
        simulated = row["simulated"]
        text = f"{simulated['mean']} ± {simulated['spread_sd']}; n={simulated['n']}/{row['attempted_n']}; {row['quantity']}"
        if "edge" in simulated:
            text += f"; edge {simulated['edge']}"
        if row.get("crossed", {}).get("mean") is not None:
            text += f"; crossed {row['crossed']}; partial-count Js {row['censored_specific_flow']}"
        if row["case"] == "V2":
            text += f"; passed {row['diagnostic_passed_n']}/{row['attempted_n']}; max wall penetration {row['wall_penetration_max_m']} m"
        cells = [
            row["case"] + " " + row["variant"],
            f"{row['published_value']} ({row['source']}, {row['location']})",
            text,
            "unavailable" if row["difference"] is None else str(row["difference"]),
            row["known_reason"],
        ]
        lines.append(
            "| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in cells) + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Execute all V1-V8 cases, saving raw measurements and exact execution identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "radius"], default="baseline")
    parser.add_argument("--radius", type=float, default=0.4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.radius not in config["radii_m"] or (args.mode == "baseline" and args.radius != 0.4):
        parser.error("baseline radius must stay .40; radius sweep must use the declared grid")
    max_workers = min(16, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    if not 1 <= args.workers <= max_workers:
        parser.error(f"workers must be 1..{max_workers}; local limit is two")
    grid = tasks(config, args.radius, args.mode)
    print("RESOLVED SEEDS", config["seeds"], "CASES", len(grid), flush=True)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "identity.json",
        {
            "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "harness_sha256": hashlib.sha256(Path(reused.__file__).read_bytes()).hexdigest(),
            "force_sha256": hashlib.sha256(
                (ROOT / "fast-pysf/pysocialforce/forces.py").read_bytes()
            ).hexdigest(),
            "substrate_path": pysocialforce.__file__,
            "model": config["model"],
            "mode": args.mode,
            "radius_m": args.radius,
            "resolved_force_radius_m": 0.35 if args.mode == "baseline" else args.radius,
            "seeds": config["seeds"],
            "episode_n": len(grid),
            "workers": args.workers,
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "diagnostic_only": True,
        },
    )
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_task, t): i for i, t in enumerate(grid)}
        for future in as_completed(futures):
            row = future.result()
            write_json(args.out / f"case_{futures[future]:04}.json", row)
            rows.append(row)
            print("DONE", len(rows), "/", len(grid), row["case"], row["seed"], flush=True)
    rows.sort(key=lambda r: (r["case"], r["variant"], r["seed"]))
    table = summary(rows, config)
    write_json(
        args.out / "table.json",
        {
            "schema": "pedval.table.v1",
            "identity": json.loads((args.out / "identity.json").read_text()),
            "rows": table,
        },
    )
    write_text(args.out / "table.md", markdown(table), issue_ref="#10074")


if __name__ == "__main__":
    main()
