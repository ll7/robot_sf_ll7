"""Analyse the guard / crossing-pedestrian diagnostic (contract.yaml is the authority).

Two stages, both deterministic:
  project : raw run directory (episodes/*.jsonl + traj/*.npz) -> evidence/episodes.csv.gz
            (one row per robot episode or no-robot run, with the paired pedestrian measures)
  analyze : evidence/episodes.csv.gz -> evidence/rates.csv, evidence/paired_effects.csv,
            evidence/summary.json (seed-block bootstrap, predeclared outcome rule)

Usage (repo root; stdlib + numpy):
  python docs/context/guard_crossing_pedestrians/analyze.py project --raw <run dir> --out <dir>
  python docs/context/guard_crossing_pedestrians/analyze.py analyze --episodes <csv.gz> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools"))

CELLS = [
    "classic_doorway_medium",
    "classic_doorway_high",
    "classic_bottleneck_medium",
    "classic_bottleneck_high",
]
CONDS = ["traverse", "outside"]
ROBOT_ARMS = ["G1E1", "G0E1", "orca"]
EXPECTED_ALGO = {"G1E1": "hybrid_rule_local_planner", "G0E1": "hybrid_rule_local_planner", "orca": "orca"}
DT = 0.1
WINDOW_STEPS = 500
WINDOW_S = WINDOW_STEPS * DT
BOOT = 10_000
BOOT_RNG = 1001

COLUMNS = [
    "arm", "cell", "condition", "seed", "mode", "fallback_count", "fallback_or_degraded_marker",
    "guard_disabled", "status", "git_hash", "steps", "termination_reason", "parking",
    "success", "ped_contact", "wall_contact", "timeout",
    "n_peds", "passages", "first_passage_s", "nr_passers", "nr_passers_blocked",
    "mean_delay_censored_s", "throughput_loss", "min_robot_ped_dist_m",
    "t0_ped_match", "sim_config_match", "source_line_sha256",
]


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAILED: {msg}")


# --------------------------------------------------------------------------- project
def _load_traj(path: Path) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def project(raw: Path, out: Path) -> None:
    import xped_lib as L  # noqa: PLC0415
    from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker  # noqa: PLC0415

    seeds_seen: set[int] = set()
    rows = []
    none_cache: dict[tuple[str, str, int], dict] = {}

    def none_traj(cell, cond, seed):
        key = (cell, cond, seed)
        if key not in none_cache:
            p = raw / "traj" / f"none__{cell}__{cond}__s{seed}.npz"
            require(p.exists(), f"missing no-robot run {p.name}")
            none_cache[key] = _load_traj(p)
        return none_cache[key]

    def ped_measures(cell, traj_robot, traj_none):
        xy = traj_robot["ped_xy"].astype(np.float64)
        require(xy.shape[0] == WINDOW_STEPS + 1, "observation window length differs")
        cr = L.gate_crossings(cell, xy)
        nr = L.gate_crossings(cell, traj_none["ped_xy"].astype(np.float64))
        firsts = [c[0] * DT for c in cr if c]
        passers = [i for i, c in enumerate(nr) if c]
        blocked = [i for i in passers if not cr[i]]
        delays = [
            (cr[i][0] * DT if cr[i] else WINDOW_S) - nr[i][0] * DT for i in passers
        ]
        return {
            "n_peds": xy.shape[1],
            "passages": sum(len(c) for c in cr),
            "first_passage_s": round(min(firsts), 1) if firsts else "",
            "nr_passers": len(passers),
            "nr_passers_blocked": len(blocked),
            "mean_delay_censored_s": round(float(np.mean(delays)), 3) if delays else "",
            "throughput_loss": sum(len(c) for c in nr) - sum(len(c) for c in cr),
        }

    for path in sorted((raw / "traj").glob("none__*.npz")):
        _, cell, cond, s = path.stem.split("__")
        seed = int(s[1:])
        seeds_seen.add(seed)
        t = none_traj(cell, cond, seed)
        m = ped_measures(cell, t, t)
        rows.append({
            "arm": "none", "cell": cell, "condition": cond, "seed": seed, "mode": "no_robot",
            "fallback_count": 0, "fallback_or_degraded_marker": "", "guard_disabled": "",
            "status": "ok", "git_hash": "", "steps": 0, "termination_reason": "no_robot",
            "parking": str(t["parking"]), "success": "", "ped_contact": "", "wall_contact": "",
            "timeout": "", **m, "min_robot_ped_dist_m": "", "t0_ped_match": "",
            "sim_config_match": "", "source_line_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    for path in sorted((raw / "episodes").glob("*.jsonl")):
        arm, scen = path.stem.split("__", 1)
        require(arm in ROBOT_ARMS, f"unknown arm {arm}")
        cell, cond = scen.split("__")
        with path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                raw_row = json.loads(line)
                seed = int(raw_row["seed"])
                require(1001 <= seed <= 1200, "non-development seed rejected")
                seeds_seen.add(seed)
                meta = raw_row["algorithm_metadata"]
                rt = dict(meta)
                rt.pop("simulation_step_trace", None)
                rt.pop("paired_effect_metric_producer", None)
                marker = runtime_fallback_or_degraded_marker(rt, expected_algorithm=EXPECTED_ALGO[arm])
                metrics, outcome = raw_row["metrics"], raw_row["outcome"]
                ped = metrics["ped_collision_count"] > 0
                wall = metrics["obstacle_collision_count"] > 0
                collision = bool(outcome["collision_event"])
                success = bool(outcome["route_complete"]) and not collision
                require(bool(metrics["success"]) == success, "success/outcome contradiction")
                timeout = bool(outcome["timeout_event"]) and not collision and not success
                tpath = raw / "traj" / f"{arm}__{scen}__s{seed}.npz"
                require(tpath.exists(), f"missing trajectory {tpath.name}")
                t = _load_traj(tpath)
                require(int(t["robot_steps"]) == int(raw_row["steps"]), f"step mismatch {tpath.name}")
                tn = none_traj(cell, cond, seed)
                m = ped_measures(cell, t, tn)
                steps = int(raw_row["steps"])
                pxy = t["ped_xy"][: steps + 1].astype(np.float64)
                rxy = t["robot_xy"][: steps + 1].astype(np.float64)
                dmin = float(np.min(np.linalg.norm(pxy - rxy[:, None, :], axis=2))) if pxy.shape[1] else ""
                rows.append({
                    "arm": arm, "cell": cell, "condition": cond, "seed": seed,
                    "mode": meta["planner_kinematics"]["execution_mode"],
                    "fallback_count": meta.get("planner_runtime", {}).get("fallback_count", ""),
                    "fallback_or_degraded_marker": marker or "",
                    "guard_disabled": meta.get("config", {}).get("ablation_disable_hard_guard", ""),
                    "status": meta.get("status", ""), "git_hash": raw_row.get("git_hash", ""),
                    "steps": steps, "termination_reason": raw_row["termination_reason"],
                    "parking": str(t["parking"]), "success": int(success), "ped_contact": int(ped),
                    "wall_contact": int(wall), "timeout": int(timeout), **m,
                    "min_robot_ped_dist_m": round(dmin, 3) if dmin != "" else "",
                    "t0_ped_match": int(np.array_equal(t["ped_xy"][0], tn["ped_xy"][0])),
                    "sim_config_match": int(str(t["sim_config_sha256"]) == str(tn["sim_config_sha256"])),
                    "source_line_sha256": hashlib.sha256(line).hexdigest(),
                })
    require(all(not (50000 <= s < 60000) for s in seeds_seen), "sealed-band seed present")
    rows.sort(key=lambda r: (r["arm"], r["cell"], r["condition"], r["seed"]))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode(), 9, mtime=0))
    print(f"projected {len(rows)} rows")


# --------------------------------------------------------------------------- analyze
def _f(v):
    return np.nan if v == "" else float(v)


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as fh:
        return list(csv.DictReader(fh))


def build_tensor(rows, arms, seeds, endpoint):
    """Array [seed, cell, cond, arm] of an episode endpoint (nan where undefined)."""
    idx = {(r["arm"], r["cell"], r["condition"], int(r["seed"])): r for r in rows}
    out = np.full((len(seeds), len(CELLS), len(CONDS), len(arms)), np.nan)
    for si, s in enumerate(seeds):
        for ci, c in enumerate(CELLS):
            for ki, k in enumerate(CONDS):
                for ai, a in enumerate(arms):
                    r = idx.get((a, c, k, s))
                    require(r is not None, f"missing row {a} {c} {k} {s}")
                    out[si, ci, ki, ai] = _f(r[endpoint])
    return out


def boot_ci(per_seed: np.ndarray, rng_seed: int = BOOT_RNG):
    """per_seed: [n_seed] seed-block statistic (cell-averaged). Percentile 95% CI of the mean."""
    rng = np.random.default_rng(rng_seed)
    n = per_seed.shape[0]
    draws = rng.integers(0, n, size=(BOOT, n))
    means = np.nanmean(per_seed[draws], axis=1)
    return float(np.nanmean(per_seed)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(episodes: Path, out: Path) -> None:
    rows = load_rows(episodes)
    seeds = sorted({int(r["seed"]) for r in rows})
    arms = ROBOT_ARMS
    robot_rows = [r for r in rows if r["arm"] in arms]
    # Admission checks on the rows themselves.
    fallback = sum(1 for r in robot_rows if r["fallback_or_degraded_marker"] or str(r["fallback_count"]) not in ("0", ""))
    modes = sorted({(r["arm"], r["mode"]) for r in robot_rows})
    guard_flags = sorted({(r["arm"], r["guard_disabled"]) for r in robot_rows})
    t0 = [r["t0_ped_match"] for r in robot_rows]
    simc = [r["sim_config_match"] for r in robot_rows]

    endpoints = {
        "success": "success", "ped_contact": "ped_contact", "wall_contact": "wall_contact",
        "timeout": "timeout", "blocked_frac": None, "mean_delay_censored_s": "mean_delay_censored_s",
        "throughput_loss": "throughput_loss",
    }
    tensors = {}
    for name, col in endpoints.items():
        if col:
            tensors[name] = build_tensor(rows, arms, seeds, col)
    passers = build_tensor(rows, arms, seeds, "nr_passers")
    blocked = build_tensor(rows, arms, seeds, "nr_passers_blocked")
    with np.errstate(invalid="ignore", divide="ignore"):
        tensors["blocked_frac"] = np.where(passers > 0, blocked / passers, np.nan)
    pct = {"success", "ped_contact", "wall_contact", "timeout", "blocked_frac"}

    rate_rows, effect_rows = [], []
    summary = {"seeds": len(seeds), "seed_range": [seeds[0], seeds[-1]], "effects": {}}

    def cell_stat(T, ai, ki, cells):
        return np.nanmean(T[:, cells, ki, ai], axis=1)

    scopes = {"pooled": list(range(len(CELLS))), **{c: [i] for i, c in enumerate(CELLS)}}
    for name, T in tensors.items():
        scale = 100.0 if name in pct else 1.0
        for scope, cells in scopes.items():
            for ki, k in enumerate(CONDS):
                for ai, a in enumerate(arms):
                    v = cell_stat(T, ai, ki, cells) * scale
                    n = int(np.sum(~np.isnan(T[:, cells, ki, ai])))
                    est, lo, hi = boot_ci(v)
                    rate_rows.append([name, scope, k, a, n, round(est, 3), round(lo, 3), round(hi, 3)])
                pairs = {"guard": (0, 1), "hybrid_minus_orca": (0, 2)}
                for label, (a1, a0) in pairs.items():
                    d = (cell_stat(T, a1, ki, cells) - cell_stat(T, a0, ki, cells)) * scale
                    n = int(np.sum(~np.isnan(T[:, cells, ki, a1] - T[:, cells, ki, a0])))
                    est, lo, hi = boot_ci(d)
                    effect_rows.append([name, scope, k, label, n, round(est, 3), round(lo, 3), round(hi, 3)])
                    summary["effects"][f"{name}|{scope}|{k}|{label}"] = [est, lo, hi, n]
            # Interaction: guard effect in traverse minus guard effect in outside.
            g = lambda ki: cell_stat(T, 0, ki, cells) - cell_stat(T, 1, ki, cells)  # noqa: E731
            d = (g(0) - g(1)) * scale
            est, lo, hi = boot_ci(d)
            n = int(np.sum(~np.isnan(d)))
            effect_rows.append([name, scope, "traverse_minus_outside", "guard", n, round(est, 3), round(lo, 3), round(hi, 3)])
            summary["effects"][f"{name}|{scope}|traverse_minus_outside|guard"] = [est, lo, hi, n]
            # Robot-arm presence vs no-robot for the guarded arm: contact of outside vs traverse.
            if name == "ped_contact":
                d = (cell_stat(T, 0, 0, cells) - cell_stat(T, 0, 1, cells)) * scale
                est, lo, hi = boot_ci(d)
                effect_rows.append([name, scope, "traverse_minus_outside", "G1E1_level", len(seeds), round(est, 3), round(lo, 3), round(hi, 3)])
                summary["effects"][f"ped_contact|{scope}|traverse_minus_outside|G1E1_level"] = [est, lo, hi, len(seeds)]

    # No-robot passage (qualification-type) counts.
    nr = {}
    for c in CELLS:
        for k in CONDS:
            rr = [r for r in rows if r["arm"] == "none" and r["cell"] == c and r["condition"] == k]
            nr[f"{c}|{k}"] = {
                "runs": len(rr),
                "runs_with_passage": sum(1 for r in rr if int(r["passages"]) > 0),
                "passages_total": sum(int(r["passages"]) for r in rr),
                "peds_total": sum(int(r["n_peds"]) for r in rr),
            }
    summary["no_robot_passage"] = nr
    summary["admission"] = {
        "robot_episodes": len(robot_rows),
        "no_robot_runs": len(rows) - len(robot_rows),
        "fallback_or_degraded_rows": fallback,
        "arm_modes": modes,
        "arm_guard_flags": guard_flags,
        "t0_pedestrian_state_match_rows": f"{sum(int(x) for x in t0)}/{len(t0)}",
        "sim_config_match_rows": f"{sum(int(x) for x in simc)}/{len(simc)}",
    }
    summary["outcome_rule"] = outcome_rule(summary["effects"], nr)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "rates.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["endpoint", "scope", "condition", "arm", "n", "estimate", "ci_low", "ci_high"])
        w.writerows(rate_rows)
    with (out / "paired_effects.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["endpoint", "scope", "condition", "contrast", "n_pairs", "estimate", "ci_low", "ci_high"])
        w.writerows(effect_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary["admission"], default=str))
    print(json.dumps(summary["outcome_rule"], indent=1))


def outcome_rule(E: dict, nr: dict) -> dict:
    """Predeclared rule (contract.yaml: outcome_rule). Intervals: seed-block bootstrap 95%."""
    get = lambda k: E[k]  # noqa: E731
    s_tr = get("success|pooled|traverse|guard")
    s_int = get("success|pooled|traverse_minus_outside|guard")
    d_tr = get("mean_delay_censored_s|pooled|traverse|guard")
    b_tr = get("blocked_frac|pooled|traverse|guard")
    pc_g = get("ped_contact|pooled|traverse|guard")
    pc_lvl = get("ped_contact|pooled|traverse_minus_outside|G1E1_level")
    triggered = []
    qual_ok = all(
        nr[f"{c}|traverse"]["runs_with_passage"] >= 0.9 * nr[f"{c}|traverse"]["runs"]
        and nr[f"{c}|outside"]["passages_total"] == 0
        for c in CELLS
    )
    if not qual_ok:
        triggered.append("O4_pedestrians_fail_to_pass_without_robot")
    else:
        if s_tr[2] <= 0 or s_int[2] < 0 or pc_g[1] > 0 or pc_lvl[1] > 0:
            triggered.append("O3_benefit_shrinks_reverses_or_contacts_increase")
        if s_tr[1] > 0 and (d_tr[1] > 0 or b_tr[1] > 0):
            triggered.append("O2_success_improves_but_pedestrian_cost_rises")
        if s_tr[1] > 0 and d_tr[2] <= 2.0 and b_tr[2] <= 5.0 and not triggered:
            triggered.append("O1_benefit_survives_pedestrian_progress_preserved")
        if not triggered:
            triggered.append("O0_inconclusive_limitation_stays_open")
    return {
        "triggered": triggered,
        "inputs": {
            "guard_success_traverse_pp": s_tr[:3],
            "guard_success_interaction_pp": s_int[:3],
            "guard_delay_traverse_s": d_tr[:3],
            "guard_blocked_traverse_pp": b_tr[:3],
            "guard_pedcontact_traverse_pp": pc_g[:3],
            "guarded_pedcontact_traverse_minus_outside_pp": pc_lvl[:3],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("project")
    a.add_argument("--raw", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("analyze")
    b.add_argument("--episodes", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "project":
        project(args.raw, args.out)
    else:
        analyze(args.episodes, args.out)


if __name__ == "__main__":
    main()
