"""Source-bound pedestrian release measurements and fail-closed admission checks.

Runs published or explicitly documented equivalent protocols on unchanged legacy
and successor force profiles. Censored observations remain null, never zero.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pysocialforce

from robot_sf.evidence.writers import write_json, write_text
from robot_sf.research import emergent_phenomena as ep
from robot_sf.research import pedestrian_validation as estimators
from robot_sf.sim.obstacle_force_profile import apply_obstacle_force_profile
from robot_sf.sim.sim_config import SimulationSettings
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


def legacy_run_task(task):
    """Run one dev episode in an isolated worker, retaining the original force defaults."""
    case, seed, variant, radius, mode = task
    original_config = reused.candidate_config
    original_radius = reused.RADIUS

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
    reused.RADIUS = original_radius
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


def legacy_main(argv=None):
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


# Source protocol runner. Historical helpers above remain callable for default-byte proof.


def protocol_simulate(
    state,
    segments,
    config,
    steps,
    *,
    interferer=False,
    speed_cap_m_s=None,
    goal_update=None,
    stop_x=None,
):
    """Step actual forces; optionally prescribe a straight nonreactive second walker.

    Only a literature-speed variant uses a separate 3 m/s execution cap. The
    goal-driving desired speeds are restored by PedState after each integration.


    Returns:
        Measurement values with explicit missingness and units.
    """
    walls = [(a, c, b, d) for a, b, c, d in segments]
    sim = pysocialforce.Simulator(state=state.copy(), obstacles=walls, config=config)
    positions, speeds = [sim.peds.pos().copy()], []
    desired = sim.peds.max_speeds.copy()
    for k in range(steps):
        if goal_update is not None:
            goal_update(sim)
        if speed_cap_m_s is None:
            sim.step()
        else:
            force = np.zeros_like(sim.peds.pos())
            for component in sim.forces:
                force += component()
            sim.peds.max_speeds = np.full(sim.peds.size(), speed_cap_m_s)
            sim.peds.step(force)
        if interferer:
            # Subject forces saw the exact prescribed position/velocity at step start.
            sim.peds.state[1, :2] = (
                state[1, :2] + (k + 1) * config.scene_config.dt_secs * state[1, 2:4]
            )
            sim.peds.state[1, 2:4] = state[1, 2:4]
        positions.append(sim.peds.pos().copy())
        speeds.append(np.linalg.norm(sim.peds.vel(), axis=1))
        if stop_x is not None and sim.peds.pos()[0, 0] > stop_x:
            break
    return np.asarray(positions), np.asarray(speeds), desired


def run_task(task):  # noqa: C901, PLR0915
    """One source-protocol dev episode, or historical byte-compatibility episode."""
    case, seed, variant, radius, mode = task[:5]
    options = task[5] if len(task) > 5 else {}
    if not 1001 <= seed <= 1030:
        raise ValueError("new episodes require dev seeds 1001..1030")
    if options.get("protocol") == "legacy":
        return legacy_run_task(task[:5])
    profile = options.get("wall_profile", "legacy_v1")
    literature = options.get("speed_tier") == "literature"
    original_config, original_simulate, original_radius = (
        reused.candidate_config,
        reused.harness.simulate,
        reused.RADIUS,
    )
    traces = []

    def configured(candidate, speed):
        cfg = original_config(candidate, speed)
        settings = SimulationSettings(pedestrian_radius_m=radius if mode == "radius" else None)
        if settings.pedestrian_radius_m is not None:
            cfg.scene_config.agent_radius = settings.pedestrian_radius_m
        apply_obstacle_force_profile(
            cfg.obstacle_force_config, profile, settings.pedestrian_radius_m
        )
        if literature:
            cfg.scene_config.desired_speed_mean = 1.3
            cfg.scene_config.desired_speed_std = 0.2
            cfg.scene_config.desired_speed_seed = seed
        return cfg

    def captured(state, segments, config, steps, **kwargs):
        p, v, desired = protocol_simulate(
            state, segments, config, steps, speed_cap_m_s=3.0 if literature else None, **kwargs
        )
        traces.append((p, v, desired))
        return p, v

    reused.candidate_config, reused.harness.simulate, reused.RADIUS = configured, captured, radius
    row = {
        "case": case,
        "seed": seed,
        "variant": variant,
        "radius_m": radius,
        "mode": mode,
        "wall_profile": profile,
        "speed_tier": "literature" if literature else "native",
        "groups": [],
        "estimator_status": options.get("equivalence", {}).get(case, "documented_equivalent"),
    }
    rng = np.random.default_rng(seed)
    try:
        if case == "V1":
            y = rng.uniform(-0.03, 0.03)
            state = np.array([[0.0, y, 0.0, 0.0, 100.0, y, 0.5]])
            cfg = configured(CANDIDATE, 0.65)
            if not literature:
                cfg.scene_config.desired_speed_mean, cfg.scene_config.desired_speed_std = 0.65, 0.0
            p, v = captured(state, [], cfg, 200)
            # The simulator does not model cue reaction. Shift the clock by the
            # source delay, making t=.35 the known start of model acceleration.
            t = np.arange(len(p)) * 0.1 + 0.35
            row.update(estimators.acceleration_fit(np.r_[0.0, v[:, 0]], t))
            row["speed_m_s"] = float(v[100:, 0].mean())
        elif case == "V2":
            ratio = float(variant)
            shoulder = options.get("shoulder_width_m", 0.46)
            width = ratio * shoulder
            row.update(reused.aperture(CANDIDATE, seed, width, 0.65))
            p = traces[-1][0]
            row["legacy_spatial_speed_drop_m_s"] = row.pop("speed_drop_m_s")
            row.update(estimators.aperture_drop(p[:, 0], np.arange(len(p)) * 0.1, plane_m=8.0))
            row.update(
                aperture_shoulder_ratio=ratio,
                shoulder_width_m=shoulder,
                shoulder_basis="fixed proxy; independent of disc diameter; measured acromion widths unavailable",
            )
        elif case in {"V3", "V4"}:
            row.update(
                reused.bottleneck(
                    CANDIDATE,
                    seed,
                    float(variant),
                    wide=case == "V4",
                    physical_radius_m=radius if mode == "radius" else None,
                )
            )
            p = traces[-1][0]
            if case == "V4":
                row.update(
                    estimators.bottleneck_flow(
                        p,
                        np.arange(len(p)) * 0.1,
                        width_m=float(variant),
                        plane_m=1.0,
                        expected_n=350,
                    )
                )
        elif case == "V5":
            centre, obstacle_radius = (8.0, 0.0), 0.25
            angles = np.linspace(0, 2 * np.pi, 33)
            vertices = np.column_stack(
                [8 + obstacle_radius * np.cos(angles), obstacle_radius * np.sin(angles)]
            )
            segments = [tuple(np.r_[a, b]) for a, b in itertools.pairwise(vertices)]
            y = 0.05 + rng.uniform(-0.005, 0.005)
            state = np.array([[0.0, y, 0.5, 0.0, 16.0, y, 0.5]])
            p, _ = captured(state, segments, configured(CANDIDATE, 0.65), 1000)
            row.update(
                estimators.circumvention_clearance(
                    p[:, 0],
                    np.arange(len(p)) * 0.1,
                    centre_xy=centre,
                    obstacle_radius_m=obstacle_radius,
                )
            )
            row.update(reused.wall_metrics(p, segments))
            row["obstacle_polygon_sides"] = 32
            row["obstacle_radius_m"] = obstacle_radius
        elif case == "V6":
            speed = float(variant)
            y = 0.01 + rng.uniform(0, 0.005)
            state = np.array(
                [
                    [0.0, y, speed, 0.0, 100.0, y, 0.5],
                    [12.0, 0.0, -speed, 0.0, -100.0, 0.0, 0.5],
                ]
            )
            cfg = configured(CANDIDATE, speed)
            # Source speed conditions are controlled, even in the literature tier.
            cfg.scene_config.desired_speed_mean, cfg.scene_config.desired_speed_std = speed, 0.0
            steps = int(np.ceil(12 / speed / 0.1))
            p, v, desired = protocol_simulate(state, [], cfg, steps, interferer=True)
            traces.append((p, v, desired))
            baselines = []
            for _ in range(5):
                bp, _bv, _bd = protocol_simulate(state[:1], [], cfg, steps)
                baselines.append(bp[:, 0])
            row.update(
                estimators.turning_onset(p[:, 0], p[:, 1], np.arange(len(p)) * 0.1, baselines)
            )
            row["interferer_nonreactive"] = True
            row["baseline_trial_n"] = 5
            row["baseline_identical_reason"] = "deterministic CM model; no trunk sway/noise"
            row["min_pair_centre_m"] = float(np.linalg.norm(p[:, 0] - p[:, 1], axis=-1).min())
            row["_baselines"] = np.asarray(baselines)
        elif case == "V7":
            row.update(lanes(seed, float(variant), radius))
        elif case == "V8":
            row.update(reused.corridor(CANDIDATE, seed, 0.65))
        else:
            raise ValueError(f"unknown case {case}")
        p, v, desired = traces[-1]
        row["pair_overlap"] = estimators.pair_overlap(p, radius, row["groups"])
        row["desired_speeds_m_s"] = desired.tolist()
        row["execution_cap_m_s"] = (
            3.0 if literature and case != "V6" else "legacy desired-speed cap"
        )
        row["trajectory_sha256"] = hashlib.sha256(p.tobytes()).hexdigest()
        row["_positions"], row["_speeds"] = p, v
        return row
    finally:
        reused.candidate_config, reused.harness.simulate, reused.RADIUS = (
            original_config,
            original_simulate,
            original_radius,
        )


def protocol_tasks(config, radius, mode, options):
    """Same source case grid for each model; all seven shoulder ratios explicit."""
    grid = tasks(config, radius, mode)
    grid = [t for t in grid if t[0] != "V2"]
    grid += [
        ("V2", seed, str(ratio), radius, mode)
        for ratio in config["V2"]["ratios"]
        for seed in config["seeds"]
    ]
    return [(*t, options) for t in grid]


def protocol_summary(rows, config):  # noqa: C901
    """Add source-estimator eligibility, specific flow and pair-step denominators."""
    table = summary(rows, config)
    lookup = {(r["case"], r["variant"]): r for r in table}
    for case, variant in sorted({(r["case"], r["variant"]) for r in rows}):
        group = [r for r in rows if (r["case"], r["variant"]) == (case, variant)]
        entry = lookup[case, variant]
        key = {"V1": "fitted_desired_speed_m_s", "V4": "all_data_specific_flow_persons_m_s"}.get(
            case, entry["quantity"]
        )
        entry["quantity"] = key
        entry["simulated"] = statistics(r.get(key) for r in group)
        entry["estimator_status"] = group[0]["estimator_status"]
        entry["known_reason"] = config[case]["protocol_reason"]
        entry["unavailable_reasons"] = sorted({r["reason"] for r in group if r.get("reason")})
        entry["difference"] = None
        if case == "V1" and entry["simulated"]["mean"] is not None:
            entry["difference"] = entry["simulated"]["mean"] - 1.29
        if case == "V2":
            ratio = float(variant)
            entry["published_value"] = (
                0.4
                if ratio == 0.9
                else ([0.12, 0.16] if ratio >= 1.3 else "graphical value not extracted")
            )
            entry["passed_n"] = sum(r["passed"] for r in group)
            if (
                isinstance(entry["published_value"], float)
                and entry["simulated"]["mean"] is not None
            ):
                entry["difference"] = entry["simulated"]["mean"] - entry["published_value"]
        if case == "V3":
            mean = entry["simulated"]["mean"]
            entry["difference"] = None if mean is None else mean - entry["published_value"]
        if case == "V4":
            entry["steady_specific_flow"] = statistics(
                r["steady_specific_flow_persons_m_s"] for r in group
            )
        if case == "V5":
            entry["published_value"] = {
                "lateral_m": 0.5,
                "longitudinal_m": "2 m ellipse radius; not measured by lateral clearance",
            }
            if entry["simulated"]["mean"] is not None:
                entry["difference"] = entry["simulated"]["mean"] - 0.5
        if case == "V6" and entry["simulated"]["mean"] is not None:
            entry["difference"] = entry["simulated"]["mean"] - entry["published_value"]
        if case == "V8":
            entry["simulated"]["edge"] = statistics(r["steady_edge_distance_m"] for r in group)
        entry["pair_overlap"] = {}
        for split in ("group", "non_group", "all"):
            banks = [r["pair_overlap"][split] for r in group]
            denominator = sum(r["pair_steps"] for r in banks)
            entry["pair_overlap"][split] = {
                "pair_steps": denominator,
                "share_below_2r": sum(r["below_2r_count"] for r in banks) / denominator
                if denominator
                else None,
                "share_below_0_45": sum(r["below_0_45_count"] for r in banks) / denominator
                if denominator
                else None,
                "minimum_centre_distance_m": min(
                    (
                        r["minimum_centre_distance_m"]
                        for r in banks
                        if r["minimum_centre_distance_m"] is not None
                    ),
                    default=None,
                ),
            }
        entry["known_reason"] += "; " + entry["estimator_status"]
        if entry["unavailable_reasons"]:
            entry["known_reason"] += "; " + "; ".join(entry["unavailable_reasons"])
    # Replace inherited diagnostic slope with both declared estimator slopes.
    table = [r for r in table if r["variant"] != "diagnostic width slope"]
    for quantity, published in (("all_data_flow_persons_s", 2.3), ("steady_flow_persons_s", 2.5)):
        values = []
        for seed in config["seeds"]:
            bank = sorted(
                [r for r in rows if r["case"] == "V4" and r["seed"] == seed],
                key=lambda r: r["width_m"],
            )
            if len(bank) == len(config["V4"]["widths_m"]) and all(
                r[quantity] is not None for r in bank
            ):
                widths, flows = (
                    np.array([r["width_m"] for r in bank]),
                    np.array([r[quantity] for r in bank]),
                )
                values.append(float(widths @ flows / (widths @ widths)))
        stat = statistics(values)
        table.append(
            {
                "case": "V4",
                "variant": quantity + " width slope",
                "quantity": "persons/(m s)",
                "published_value": published,
                "source": config["V4"]["source"],
                "location": config["V4"]["location"],
                "simulated": stat,
                "difference": None if stat["mean"] is None else stat["mean"] - published,
                "known_reason": config["V4"]["protocol_reason"]
                + "; documented_equivalent; through-origin fit; complete width banks only",
                "attempted_n": len(config["seeds"]),
                "estimator_status": "documented_equivalent",
            }
        )
    return table


def acceptance_gate(rows):
    """Fail closed on censored primary measurements, body contact or wall penetration.

    Exit 2 denotes incomplete measurements; 3 physical infeasibility; 5 missing
    numerical acceptance tolerances/domain approval. Published SDs are not
    acceptance tolerances. No release pass is inferred from complete measurements.
    """
    missing = [
        f"{r['case']}/{r['variant']}/{r['seed']}"
        for r in rows
        if r.get(
            {
                "V1": "fitted_desired_speed_m_s",
                "V2": "speed_drop_m_s",
                "V3": "specific_flow_persons_m_s",
                "V4": "steady_specific_flow_persons_m_s",
                "V5": "lateral_cm_to_edge_m",
                "V6": "onset_m",
            }.get(r["case"], "seed")
        )
        is None
    ]
    physical = [
        f"{r['case']}/{r['variant']}/{r['seed']}"
        for r in rows
        if r.get("wall_penetration_m", 0.0) > 1e-6 or r["pair_overlap"]["all"]["below_2r_count"] > 0
    ]
    return {
        "schema": "valsuite.gate.v1",
        "measurement_missing": missing,
        "physical_violations": physical,
        "exit_code": 2 if missing else (3 if physical else 5),
        "release_admission": "blocked pending domain review and numeric tolerances for documented equivalents",
    }


def verify_acquisition(out, config, config_path):  # noqa: C901
    """Verify complete, unique case grid and byte manifests before judging a model."""
    identity = json.loads((out / "identity.json").read_text())
    if identity["protocol"] != "source":
        raise ValueError("compatibility protocol cannot admit a release")
    if identity["config_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError("acquisition config differs from requested gate config")
    expected = {
        (t[0], t[1], t[2])
        for t in protocol_tasks(config, identity["radius_m"], identity["mode"], {})
    }
    paths = sorted(out.glob("case_*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    observed = [(r["case"], r["seed"], r["variant"]) for r in rows]
    if len(rows) != len(expected) or set(observed) != expected:
        raise ValueError("incomplete or duplicate acquisition case grid")
    if identity["episode_n"] != len(expected) or identity["seeds"] != config["seeds"]:
        raise ValueError("acquisition identity does not match the declared dev grid")
    manifest = {}
    for line in (out / "SHA256SUMS").read_text().splitlines():
        if line.startswith("<!--") or not line.strip():
            continue
        digest, name = line.split("  ", 1)
        if Path(name).name != name or name in manifest:
            raise ValueError("invalid manifest member")
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"manifest digest mismatch: {name}")
        manifest[name] = digest
    required = {"identity.json", "table.json", "table.md", "gate.json"}
    required.update(p.name for p in paths)
    required.update(r["raw_trajectory"] for r in rows)
    if not required <= manifest.keys():
        raise ValueError("manifest omits required raw acquisition files")
    for row in rows:
        if manifest[row["raw_trajectory"]] != row["raw_trajectory_sha256"]:
            raise ValueError("row trajectory digest differs from manifest")
    return rows


def source_main(argv=None):  # noqa: C901, PLR0915, PLR0912
    """Run the full source case grid unchanged for baseline and successor profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "radius"], default="baseline")
    parser.add_argument("--radius", type=float, default=0.4)
    parser.add_argument("--speed-tier", choices=["native", "literature"], default="native")
    parser.add_argument(
        "--wall-profile", choices=["legacy_v1", "calibrated_v2", "gradient_v3"], default="legacy_v1"
    )
    parser.add_argument("--protocol", choices=["source", "legacy"], default="source")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="verify saved acquisition and return model gate exit",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.gate_only:
        try:
            rows = verify_acquisition(args.out, config, args.config)
        except (OSError, ValueError, KeyError) as error:
            print(f"INVALID ACQUISITION: {error}", file=sys.stderr)
            return 4
        return acceptance_gate(rows)["exit_code"]
    if args.radius not in config["radii_m"] or (args.mode == "baseline" and args.radius != 0.4):
        parser.error("baseline radius must stay .40; radius must be in declared grid")
    max_workers = min(16, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    if not 1 <= args.workers <= max_workers:
        parser.error(f"workers must be 1..{max_workers}; local limit is two")
    if args.protocol == "legacy" and (
        args.speed_tier != "native" or args.wall_profile != "legacy_v1"
    ):
        parser.error("legacy compatibility protocol requires native speed and unchanged wall law")
    options = {
        "protocol": args.protocol,
        "wall_profile": args.wall_profile,
        "speed_tier": args.speed_tier,
        "shoulder_width_m": config["V2"]["shoulder_proxy_m"],
        "equivalence": {f"V{i}": config[f"V{i}"]["equivalence"] for i in range(1, 9)},
    }
    grid = (
        tasks(config, args.radius, args.mode)
        if args.protocol == "legacy"
        else protocol_tasks(config, args.radius, args.mode, options)
    )
    if args.protocol == "legacy":
        grid = [(*t, options) for t in grid]
    print("RESOLVED SEEDS", config["seeds"], "CASES", len(grid), flush=True)
    if args.check_only:
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    substrate = Path(pysocialforce.__file__).parent
    identity = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model": config["model"],
        "mode": args.mode,
        "radius_m": args.radius,
        "force_radius_m": 0.35 if args.mode == "baseline" else args.radius,
        "speed_tier": args.speed_tier,
        "wall_profile": args.wall_profile,
        "protocol": args.protocol,
        "seeds": config["seeds"],
        "episode_n": len(grid),
        "baseline_trial_n": len(config["seeds"]) * len(config["V6"]["speeds_m_s"]) * 5
        if args.protocol == "source"
        else 0,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "workers": args.workers,
        "source_files": {},
        "installed_files": {},
        "diagnostic_only": True,
    }
    for path in [
        Path(__file__),
        Path(estimators.__file__),
        Path(reused.__file__),
        ROOT / "fast-pysf/pysocialforce/forces.py",
        ROOT / "fast-pysf/pysocialforce/config.py",
    ]:
        identity["source_files"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for name in ("forces.py", "config.py", "scene.py"):
        identity["installed_files"][name] = hashlib.sha256(
            (substrate / name).read_bytes()
        ).hexdigest()
        if (
            identity["installed_files"][name]
            != hashlib.sha256((ROOT / "fast-pysf/pysocialforce" / name).read_bytes()).hexdigest()
        ):
            raise RuntimeError(f"installed substrate differs from producer source: {name}")
    write_json(args.out / "identity.json", identity)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_task, task): i for i, task in enumerate(grid)}
        for future in as_completed(futures):
            row = future.result()
            arrays = {key[1:]: row.pop(key) for key in list(row) if key.startswith("_")}
            index = futures[future]
            if arrays:
                trace = args.out / f"trajectory_{index:04}.npz"
                np.savez_compressed(trace, **arrays)
                row["raw_trajectory"] = trace.name
                row["raw_trajectory_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            write_json(args.out / f"case_{index:04}.json", row)
            rows.append(row)
            print("DONE", len(rows), "/", len(grid), row["case"], row["seed"], flush=True)
    rows.sort(key=lambda r: (r["case"], r["variant"], r["seed"]))
    table = summary(rows, config) if args.protocol == "legacy" else protocol_summary(rows, config)
    write_json(
        args.out / "table.json", {"schema": "pedval.table.v2", "identity": identity, "rows": table}
    )
    rendered = markdown(table)
    if args.protocol == "source":
        rendered += "\n| case | pair class | pair-steps | share d < 2r | share d < 0.45 m | minimum centre distance (m) |\n|---|---|---|---|---|---|\n"
        for entry in table:
            for split, measurements in entry.get("pair_overlap", {}).items():
                rendered += f"| {entry['case']} {entry['variant']} | {split} | {measurements['pair_steps']} | {measurements['share_below_2r']} | {measurements['share_below_0_45']} | {measurements['minimum_centre_distance_m']} |\n"
        gate = acceptance_gate(rows)
        write_json(args.out / "gate.json", gate)
    else:
        gate = {"exit_code": 0, "compatibility_only": True}
        write_json(args.out / "gate.json", gate)
    write_text(args.out / "table.md", rendered, issue_ref="#10074")
    manifest = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
        for p in sorted(args.out.iterdir())
        if p.is_file()
    )
    write_text(args.out / "SHA256SUMS", manifest, issue_ref="#10074")
    print("GATE EXIT", gate["exit_code"], flush=True)
    # Acquisition succeeded even when the model fails the separately recorded gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(source_main())
