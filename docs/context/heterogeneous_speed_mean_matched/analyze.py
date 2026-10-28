"""Analysis for the mean-matched heterogeneous pedestrian speed diagnostic.

Two stages:
  extract  raw run tree -> episodes.csv (one row per episode, incl. pedestrian
           delay) + pairing_checks.json (realized-input verification)
  analyze  episodes.csv -> paired_effects.csv, rates.csv, summary.json

Usage:
    python analyze.py extract --runs <run-dir> --out <evidence-dir>
    python analyze.py analyze --episodes <evidence-dir>/episodes.csv --contract contract.yaml \
        --out <evidence-dir>

The ``analyze`` stage needs only NumPy, PyYAML and the standard library.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

PLANNERS = ["hybrid_v4_continuous", "orca", "social_force", "ppo_release"]
CELLS = ["crossing", "group_crossing", "head_on_corridor", "overtaking"]
CONDS = ["uniform", "mixed"]
MIXED_SORTED = [0.325] * 3 + [0.65] * 6 + [0.975] * 3
DT = 0.1
NEAR_M = 3.0  # robot-centre to pedestrian-centre distance (robot radius 1.0 m) for the robot-near delay share
TOL = 1e-9

# ----------------------------------------------------------------------------- extract


def _counter_state(value: object) -> str:
    """Return zero, positive, or unknown without coercing missing/invalid counters."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return "unknown"
    return "zero" if value == 0 else "positive"


def _planner_fallback_indicators(am: dict, pr: dict) -> tuple[list[str], list[str], list[str]]:
    """Distinguish recorded fallback evidence from missing planner telemetry."""
    reasons_fb, reasons_dg, unknown = [], [], []
    fallback_count = pr.get("fallback_count")
    if fallback_count is None:
        unknown.append("planner_fallback_counter_unavailable")
    elif _counter_state(fallback_count) == "positive":
        reasons_fb.append(f"planner_fallback_count={fallback_count}")
    elif _counter_state(fallback_count) == "unknown":
        unknown.append("planner_fallback_counter_invalid")
    if am.get("fallback_used") or am.get("used_fallback"):
        reasons_fb.append("algorithm_fallback_used")
    algorithm_status = am.get("status")
    if algorithm_status is None:
        unknown.append("algorithm_status_unavailable")
    elif "fallback" in str(algorithm_status).lower():
        reasons_fb.append(f"algorithm_status={algorithm_status}")
    elif str(algorithm_status).lower() in ("error", "failed", "failed_run"):
        reasons_dg.append(f"algorithm_status={algorithm_status}")
    elif str(algorithm_status).lower() != "ok":
        unknown.append(f"algorithm_status_unrecognized={algorithm_status}")
    if am.get("fallback_reason") or pr.get("fallback_reason"):
        reasons_fb.append("recorded_fallback_reason")
    return reasons_fb, reasons_dg, unknown


def _mode(rec: dict) -> tuple[str, str]:  # noqa: C901
    """Classify recorded indicators and retain unknown telemetry in the reason."""
    am = rec.get("algorithm_metadata", {}) or {}
    pr = am.get("planner_runtime", {}) or {}
    reasons_fb, reasons_dg, unknown = _planner_fallback_indicators(am, pr)
    cp = pr.get("checkpoint_provenance") or {}
    if cp:
        if cp.get("fallback_triggered"):
            reasons_fb.append("checkpoint_fallback")
        if cp.get("load_succeeded") is False:
            reasons_fb.append("checkpoint_not_loaded")
    for key in ("goal_posterior_avoidance", "proxemic_costmap"):
        if (pr.get(key) or {}).get("fallback_or_degraded"):
            reasons_dg.append(f"{key}_fallback_or_degraded")
    if ((rec.get("integrity") or {}).get("effective_view") or {}).get("degraded"):
        reasons_dg.append("effective_view_degraded")
    if (rec.get("pedestrian_model") or {}).get("fallback_degraded_status", "native") != "native":
        reasons_dg.append("pedestrian_model_not_native")
    if (rec.get("spawn_validity") or {}).get("invalid_run"):
        reasons_dg.append("invalid_spawn")
    if str(rec.get("status")) in ("error", "failed_run"):
        reasons_dg.append("status_error")
    if reasons_fb:
        return "fallback", ";".join(reasons_fb + reasons_dg + unknown)
    if reasons_dg:
        return "degraded", ";".join(reasons_dg + unknown)
    return "native", ";".join(unknown)


def _ped_delay(npz_path: str, labels: list[str]) -> dict:
    """Per-episode pedestrian delay from realized per-step state.

    Lost time of pedestrian i: sum over steps of dt * (1 - v.e / v0), e = unit vector to its
    current goal, v0 = its preferred speed; steps with v0 == 0 (start delay) are skipped.
    ``delay_rate`` divides by the episode duration (fraction of time lost); the robot-near
    share counts only steps with the robot centre within NEAR_M of the pedestrian.
    """
    z = np.load(npz_path)
    pos, vel, goal, v0 = z["ped_pos"], z["ped_vel"], z["ped_goal"], z["ped_max_speed"]
    rob = z["robot_pos"]
    # state k>=1 is after step k; use steps 1..T (velocity realized by the step)
    pos, vel, goal, v0, rob = pos[1:], vel[1:], goal[1:], v0[1:], rob[1:]
    T = pos.shape[0]
    out: dict = {"ped_steps": int(T)}
    if T == 0:
        return out
    d = goal - pos
    dn = np.linalg.norm(d, axis=2)
    e = np.divide(d, dn[..., None], out=np.zeros_like(d), where=dn[..., None] > 1e-6)
    prog = np.sum(vel * e, axis=2)
    active = (v0 > 1e-6) & np.isfinite(prog) & (dn > 1e-6)
    lost = np.where(active, DT * (1.0 - prog / np.where(v0 > 1e-6, v0, 1.0)), 0.0)
    rdist = np.linalg.norm(pos - rob[:, None, :], axis=2)
    near = active & (rdist < NEAR_M)
    dur = T * DT
    lab = np.asarray(labels)
    out["ped_delay_rate_all"] = float(lost.sum(axis=0).mean() / dur)
    out["ped_near_delay_rate_all"] = float(np.where(near, lost, 0.0).sum(axis=0).mean() / dur)
    out["ped_near_time_frac_all"] = float(near.sum(axis=0).mean() / T)
    for s in ("slow", "standard", "fast"):
        m = lab == s
        if m.any():
            out[f"ped_delay_rate_{s}"] = float(lost[:, m].sum(axis=0).mean() / dur)
            out[f"ped_near_delay_rate_{s}"] = float(
                np.where(near[:, m], lost[:, m], 0.0).sum(axis=0).mean() / dur
            )
    # speed stability: preferred speed never changed after reset except start-delay zeros
    ref = z["ped_max_speed"][0]
    nz = v0 > 1e-6
    out["max_speed_changed"] = bool(np.any(np.abs(np.where(nz, v0, ref) - ref) > 1e-6))
    out["max_speed_start_delay_steps"] = int((~nz).sum())
    return out


def _mixed_labels(n: int, seed: int) -> list[str]:
    from robot_sf.ped_npc.ped_archetypes import assign_archetype_labels

    return [
        str(x)
        for x in assign_archetype_labels(
            n, {"slow": 0.25, "standard": 0.5, "fast": 0.25}, seed=seed
        )
    ]


def extract(runs: str, out: str) -> None:
    """Extract episode rows and realized-input checks from raw observations."""
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    rows, receipts = [], {}
    for jf in sorted(glob.glob(os.path.join(runs, "*", "*", "*__*.jsonl"))):
        chunk_dir = os.path.dirname(jf)
        planner = os.path.basename(os.path.dirname(chunk_dir))
        with open(jf) as fh:
            for line in fh:
                rec = json.loads(line)
                scen = rec["scenario_id"]
                seed = int(rec["seed"])
                stem = os.path.join(chunk_dir, "observer", f"{scen}__s{seed}")
                rc = json.load(open(stem + ".receipt.json"))
                cell, cond = rc["cell"], rc["condition"]
                m = rec.get("metrics", {}) or {}
                oc = rec.get("outcome", {}) or {}
                mode, reason = _mode(rec)
                # stratum labels: mixed uses its own; uniform uses the labels the mixed arm
                # assigns to the same slots (same archetype seed), for identity-paired delay
                labels = (
                    rc["stratum_labels"] if cond == "mixed" else _mixed_labels(rc["n_peds"], seed)
                )
                pd = _ped_delay(stem + ".npz", labels)
                am = rec.get("algorithm_metadata", {}) or {}
                pr = am.get("planner_runtime", {}) or {}
                cp = pr.get("checkpoint_provenance") or {}
                success = bool(m.get("success"))
                row = {
                    "planner": planner,
                    "cell": cell,
                    "condition": cond,
                    "seed": seed,
                    "mode": mode,
                    "mode_reason": reason,
                    "algorithm_status": am.get("status"),
                    "planner_fallback_count": pr.get("fallback_count"),
                    "planner_fallback_counter_state": _counter_state(pr.get("fallback_count")),
                    "command_execution_mode": (am.get("planner_kinematics") or {}).get(
                        "execution_mode"
                    ),
                    "status": rec.get("status"),
                    "termination_reason": rec.get("termination_reason"),
                    "steps": int(rec.get("steps", 0)),
                    "success": int(success),
                    "collision": int(bool(oc.get("collision_event"))),
                    "ped_contact": int(float(m.get("ped_collision_count", 0) or 0) > 0),
                    "obstacle_contact": int(float(m.get("obstacle_collision_count", 0) or 0) > 0),
                    "agent_contact": int(float(m.get("agent_collision_count", 0) or 0) > 0),
                    "timeout": int(bool(oc.get("timeout_event"))),
                    "time_to_goal_s": round(int(rec.get("steps", 0)) * DT, 3) if success else "",
                    "min_distance_m": m.get("min_distance"),
                    "path_length_m": m.get("socnavbench_path_length"),
                    "wall_time_s": rec.get("wall_time_sec"),
                    "captured_steps": rc["captured_steps"],
                    "observer_error": rc.get("observer_error", ""),
                    "git_hash": (rec.get("provenance") or {}).get("git_hash"),
                    "algo_config_hash": (rec.get("scenario_params") or {}).get("algo_config_hash"),
                    "checkpoint_load": cp.get("load_status", ""),
                    "checkpoint_load_succeeded": cp.get("load_succeeded"),
                    "checkpoint_fallback_triggered": cp.get("fallback_triggered"),
                }
                row.update(pd)
                rows.append(row)
                receipts[(planner, cell, cond, seed)] = rc
    checks = pairing_checks(rows, receipts)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    order = list(rows[0].keys()) + [k for k in keys if k not in rows[0]]
    rows.sort(key=lambda r: (r["planner"], r["cell"], r["seed"], r["condition"]))
    with open(out_p / "episodes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=order, restval="")
        w.writeheader()
        w.writerows(rows)
    _write_mode_counts(out_p / "mode_counts.csv", rows)
    (out_p / "pairing_checks.json").write_text(json.dumps(checks, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in checks.items() if k != "failures"}, indent=1))


def _write_mode_counts(path: Path, rows: list[dict]) -> None:
    """Record classified modes and observed versus unavailable planner counters."""
    grouped: dict = {}
    for row in rows:
        key = (row["planner"], row["cell"], row["condition"])
        if key not in grouped:
            grouped[key] = dict.fromkeys(
                (
                    "native",
                    "fallback",
                    "degraded",
                    "total",
                    "counter_zero",
                    "counter_positive",
                    "counter_unknown",
                ),
                0,
            )
        counts = grouped[key]
        counts[row["mode"]] += 1
        counts["total"] += 1
        value = row["planner_fallback_count"]
        counter = _counter_state(value)
        counts[f"counter_{counter}"] += 1
    _write_csv(
        path,
        [
            {"planner": p, "cell": c, "condition": cond, **counts}
            for (p, c, cond), counts in sorted(grouped.items())
        ],
    )


def _same(a, b) -> bool:
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=1e-6)


def _relocated_rows(rc: dict) -> set:
    rel = (rc.get("spawn_relocation") or {}).get("relocated") or {}
    return {int(k) for k in rel}


def pairing_checks(rows: list[dict], receipts: dict) -> dict:  # noqa: C901, PLR0912
    """Verify observed conditions, planner resets, and episode capture."""
    fails: list[str] = []
    n_pairs = 0
    relocation_pairs: set = set()
    for (planner, cell, cond, seed), rc in receipts.items():
        ms = sorted(rc["max_speeds"])
        if rc["n_peds"] != 12:
            fails.append(f"{planner}/{cell}/{cond}/{seed}: n_peds={rc['n_peds']}")
        if rc["desired_speed_mean"] is not None or rc["explicit_desired_speeds"] is not None:
            fails.append(f"{planner}/{cell}/{cond}/{seed}: desired-speed override present")
        target = MIXED_SORTED if cond == "mixed" else [0.65] * 12
        if not np.allclose(ms, target, atol=TOL):
            fails.append(f"{planner}/{cell}/{cond}/{seed}: max_speeds {ms}")
        if abs(float(np.mean(rc["max_speeds"])) - 0.65) > TOL:
            fails.append(f"{planner}/{cell}/{cond}/{seed}: mean speed != 0.65")
        if cond == "mixed":
            fac = {"slow": 0.325, "standard": 0.65, "fast": 0.975}
            if not _same([fac[x] for x in rc["stratum_labels"]], rc["max_speeds"]):
                fails.append(f"{planner}/{cell}/{cond}/{seed}: labels do not match speeds")
        if cond != "uniform":
            continue
        other = receipts.get((planner, cell, "mixed", seed))
        if other is None:
            fails.append(f"{planner}/{cell}/{seed}: missing mixed partner")
            continue
        n_pairs += 1
        moved = _relocated_rows(rc) | _relocated_rows(other)
        dpos = np.linalg.norm(np.asarray(rc["start_pos"]) - np.asarray(other["start_pos"]), axis=1)
        diff_rows = {int(i) for i in np.nonzero(dpos > 1e-6)[0]}
        if diff_rows:
            if diff_rows <= moved:
                relocation_pairs.add((planner, cell, seed))
            else:
                fails.append(f"{planner}/{cell}/{seed}: start_pos differs outside relocated rows")
        for k in ("goal", "tau", "robot_pose", "robot_goal"):
            if not _same(rc[k], other[k]):
                fails.append(f"{planner}/{cell}/{seed}: {k} differs between conditions")
        for k in ("groups", "behaviors", "agent_radius", "n_peds"):
            if rc[k] != other[k]:
                fails.append(f"{planner}/{cell}/{seed}: {k} differs between conditions")
        # initial velocity differs only by each pedestrian's speed factor
        fac = np.asarray(other["max_speeds"]) / 0.65
        keep = [i for i in range(rc["n_peds"]) if i not in moved]
        if not _same(
            (np.asarray(rc["start_vel"]) * fac[:, None])[keep], np.asarray(other["start_vel"])[keep]
        ):
            fails.append(f"{planner}/{cell}/{seed}: start_vel not factor-scaled")
    # Compare start_pos, goal, max_speeds, robot_pose and robot_goal across planners.
    cross = 0
    for (planner, cell, cond, seed), rc in receipts.items():
        ref = receipts.get((PLANNERS[1], cell, cond, seed))
        if ref is None or planner == PLANNERS[1]:
            continue
        cross += 1
        for k in ("start_pos", "goal", "max_speeds", "robot_pose", "robot_goal"):
            if not _same(rc[k], ref[k]):
                fails.append(f"{planner} vs orca/{cell}/{cond}/{seed}: {k} differs across planners")
    mode_counts: dict = defaultdict(int)
    for r in rows:
        mode_counts[r["mode"]] += 1
    extra = {
        "speed_changed_after_reset": sum(1 for r in rows if r.get("max_speed_changed")),
        "observer_errors": sum(1 for r in rows if r.get("observer_error")),
        "captured_steps_mismatch": sum(
            1 for r in rows if int(r["captured_steps"]) != int(r["steps"]) + 1
        ),
    }
    for r in rows:
        r["pair_relocation_differs"] = int((r["planner"], r["cell"], r["seed"]) in relocation_pairs)
    return {
        "episodes": len(rows),
        "pairs_with_speed_coupled_relocation_difference": len(relocation_pairs),
        "condition_pairs_checked": n_pairs,
        "cross_planner_resets_checked": cross,
        "mode_counts": dict(mode_counts),
        **extra,
        "n_failures": len(fails),
        "failures": fails[:200],
        "pass": not fails
        and extra["speed_changed_after_reset"] == 0
        and extra["observer_errors"] == 0
        and extra["captured_steps_mismatch"] == 0,
    }


# ----------------------------------------------------------------------------- analyze

ENDPOINTS = [
    # name, column, kind ('rate' in pp, 'mean'), condition on success in both arms
    ("success", "success", "rate", False),
    ("collision", "collision", "rate", False),
    ("timeout", "timeout", "rate", False),
    ("ped_contact", "ped_contact", "rate", False),
    ("obstacle_contact", "obstacle_contact", "rate", False),
    ("time_to_goal_s", "time_to_goal_s", "mean", True),
    ("min_distance_m", "min_distance_m", "mean", False),
    ("ped_delay_rate_all", "ped_delay_rate_all", "mean", False),
    ("ped_delay_rate_slow", "ped_delay_rate_slow", "mean", False),
    ("ped_delay_rate_fast", "ped_delay_rate_fast", "mean", False),
    ("ped_near_delay_rate_all", "ped_near_delay_rate_all", "mean", False),
    ("ped_near_delay_rate_slow", "ped_near_delay_rate_slow", "mean", False),
    ("ped_near_delay_rate_fast", "ped_near_delay_rate_fast", "mean", False),
]


def _f(x):
    if x in ("", None):
        return math.nan
    return float(x)


def _telemetry_summary(rows: list[dict]) -> dict:
    """Summarize observable indicators without counting missing counters as zero."""
    counters = {p: {"zero": 0, "positive": 0, "unknown": 0} for p in PLANNERS}
    statuses = {"ok": 0, "other": 0, "unknown": 0}
    checkpoints = {"loaded_without_checkpoint_fallback": 0, "failed_or_fallback": 0, "unknown": 0}
    for row in rows:
        state = row.get("planner_fallback_counter_state", "unknown")
        if state not in ("zero", "positive", "unknown"):
            state = "unknown"
        counters[row["planner"]][state] += 1
        status = row.get("algorithm_status")
        statuses["ok" if status == "ok" else "other" if status else "unknown"] += 1
        if row["planner"] == "ppo_release":
            loaded, fallback = (
                row.get("checkpoint_load_succeeded"),
                row.get("checkpoint_fallback_triggered"),
            )
            if loaded == "True" and fallback == "False":
                checkpoints["loaded_without_checkpoint_fallback"] += 1
            elif loaded in ("True", "False") and fallback in ("True", "False"):
                checkpoints["failed_or_fallback"] += 1
            else:
                checkpoints["unknown"] += 1
    return {
        "planner_fallback_counters": counters,
        "algorithm_status": statuses,
        "ppo_checkpoint": checkpoints,
        "native_interpretation": "Classification reflects recorded indicators; unknown counters are not zero.",
    }


def analyze(episodes: str, contract: str, out: str, exclude_relocation_pairs: bool = False) -> None:  # noqa: C901, PLR0915
    """Compute the preregistered paired effects, intervals, and decision."""
    import yaml

    con = yaml.safe_load(open(contract))
    an = con["analysis"]
    seeds_main = [int(s) for s in con["seeds"]["main"]]
    rows = list(csv.DictReader(open(episodes)))
    rows = [r for r in rows if int(r["seed"]) in set(seeds_main)]
    if exclude_relocation_pairs:
        rows = [r for r in rows if str(r.get("pair_relocation_differs", "0")) != "1"]
    seeds = sorted({int(r["seed"]) for r in rows})
    S, P, C = len(seeds), len(PLANNERS), len(CELLS)
    sidx = {s: i for i, s in enumerate(seeds)}
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    # tensor [endpoint][planner, cell, cond, seed]
    X = {}
    for name, col, _kind, _cs in ENDPOINTS:
        X[name] = np.full((P, C, 2, S), np.nan)
    native = np.zeros((P, C, 2, S), dtype=bool)
    present = np.zeros((P, C, 2, S), dtype=bool)
    for r in rows:
        i, j = PLANNERS.index(r["planner"]), CELLS.index(r["cell"])
        k, s = CONDS.index(r["condition"]), sidx[int(r["seed"])]
        present[i, j, k, s] = True
        native[i, j, k, s] = r["mode"] == "native"
        for name, col, _kind, _cs in ENDPOINTS:
            X[name][i, j, k, s] = _f(r.get(col))
    # only native episodes enter effects; non-native counted and reported
    for name in X:
        X[name][~native] = np.nan
    scale = {name: (100.0 if kind == "rate" else 1.0) for name, _c, kind, _s in ENDPOINTS}
    cond_succ = {name: cs for name, _c, _k, cs in ENDPOINTS}
    rng = np.random.default_rng(int(an["bootstrap_seed"]))
    B = int(an["bootstrap_draws"])
    draws = rng.integers(0, S, size=(B, S))

    def cell_effect(name, sel):
        """Paired mixed-uniform per (planner, cell) over seed index array ``sel``."""
        x = X[name][:, :, :, sel] * scale[name]
        if cond_succ[name]:
            both = ~np.isnan(x[:, :, 0, :]) & ~np.isnan(x[:, :, 1, :])
            u = np.where(both, x[:, :, 0, :], np.nan)
            m = np.where(both, x[:, :, 1, :], np.nan)
        else:
            u, m = x[:, :, 0, :], x[:, :, 1, :]
        with np.errstate(all="ignore"):
            d = np.nanmean(m - u, axis=-1)
            lu = np.nanmean(u, axis=-1)
            lm = np.nanmean(m, axis=-1)
        return d, lu, lm  # (P, C)

    full = np.arange(S)
    effects = []
    pooled = {}
    for name, _c, _k, _cs in ENDPOINTS:
        d, lu, lm = cell_effect(name, full)
        bd = np.stack([cell_effect(name, draws[b])[0] for b in range(B)])  # (B, P, C)
        with np.errstate(all="ignore"):
            pooled_pt = np.nanmean(d, axis=1)
            pooled_b = np.nanmean(bd, axis=2)
        for i, p in enumerate(PLANNERS):
            for j, c in enumerate(CELLS):
                x = X[name][i, j]
                pairs = int(np.sum(~np.isnan(x[0]) & ~np.isnan(x[1])))
                lo, hi = np.nanpercentile(bd[:, i, j], [2.5, 97.5])
                effects.append(_eff_row(name, p, c, lu[i, j], lm[i, j], d[i, j], lo, hi, pairs))
            lo, hi = np.nanpercentile(pooled_b[:, i], [2.5, 97.5])
            with np.errstate(all="ignore"):
                effects.append(
                    _eff_row(
                        name,
                        p,
                        "ALL_equal_cell",
                        np.nanmean(lu[i]),
                        np.nanmean(lm[i]),
                        pooled_pt[i],
                        lo,
                        hi,
                        int(np.sum(~np.isnan(X[name][i, :, 0]) & ~np.isnan(X[name][i, :, 1]))),
                    )
                )
            pooled[(name, p)] = (pooled_pt[i], lo, hi)
    # planner-pair interactions on success and collision: (A-B)_mixed - (A-B)_uniform
    inter = []
    order = {}
    for name in ("success", "collision"):
        d, lu, lm = cell_effect(name, full)
        bstack = [cell_effect(name, draws[b]) for b in range(B)]
        level_u = np.nanmean(lu, axis=1)
        level_m = np.nanmean(lm, axis=1)
        order[name] = {
            "uniform": [PLANNERS[i] for i in np.argsort(-level_u, kind="stable")],
            "mixed": [PLANNERS[i] for i in np.argsort(-level_m, kind="stable")],
            "uniform_levels_pp": dict(zip(PLANNERS, map(float, level_u), strict=True)),
            "mixed_levels_pp": dict(zip(PLANNERS, map(float, level_m), strict=True)),
        }
        for a in range(P):
            for b in range(a + 1, P):
                pt = float(np.nanmean(d[a] - d[b]))
                bs = np.array([np.nanmean(bb[0][a] - bb[0][b]) for bb in bstack])
                du = float(level_u[a] - level_u[b])
                bu = np.array([np.nanmean(bb[1][a] - bb[1][b]) for bb in bstack])
                dm = float(level_m[a] - level_m[b])
                bm = np.array([np.nanmean(bb[2][a] - bb[2][b]) for bb in bstack])
                inter.append(
                    {
                        "endpoint": name,
                        "planner_a": PLANNERS[a],
                        "planner_b": PLANNERS[b],
                        "diff_uniform_pp": du,
                        "diff_uniform_lo": float(np.nanpercentile(bu, 2.5)),
                        "diff_uniform_hi": float(np.nanpercentile(bu, 97.5)),
                        "diff_mixed_pp": dm,
                        "diff_mixed_lo": float(np.nanpercentile(bm, 2.5)),
                        "diff_mixed_hi": float(np.nanpercentile(bm, 97.5)),
                        "interaction_pp": pt,
                        "interaction_lo": float(np.nanpercentile(bs, 2.5)),
                        "interaction_hi": float(np.nanpercentile(bs, 97.5)),
                    }
                )
    _write_csv(out_p / "paired_effects.csv", effects)
    _write_csv(out_p / "planner_pair_interactions.csv", inter)
    counts = {
        "seeds": S,
        "episodes_expected": P * C * 2 * S,
        "episodes_present": int(present.sum()),
        "episodes_native": int(native.sum()),
        "episodes_non_native": int(present.sum() - native.sum()),
    }
    verdict = decide(con["outcome_rule"], pooled, inter, order)
    summary = {
        "counts": counts,
        "telemetry": _telemetry_summary(rows),
        "order": order,
        "verdict": verdict,
        "pooled_effects": {f"{n}|{p}": list(map(float, v)) for (n, p), v in pooled.items()},
    }
    (out_p / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({"counts": counts, "verdict": verdict, "order": order}, indent=1))


def _eff_row(name, p, c, lu, lm, d, lo, hi, pairs):  # noqa: PLR0913
    return {
        "endpoint": name,
        "planner": p,
        "cell": c,
        "uniform": _r(lu),
        "mixed": _r(lm),
        "effect_mixed_minus_uniform": _r(d),
        "ci_lo": _r(lo),
        "ci_hi": _r(hi),
        "paired_n": pairs,
    }


def _r(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 6)


def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def decide(rule: dict, pooled: dict, inter: list, order: dict) -> dict:  # noqa: C901
    """Apply the predeclared outcome rule (contract.yaml: outcome_rule)."""
    mat = float(rule["material_pp"])
    wide = float(rule["inconclusive_ci_width_pp"])
    persist_bound = float(rule["persist_bound_pp"])
    triggers = []
    # B: order change among pairs whose uniform difference is resolved
    for it in inter:
        if it["endpoint"] != "success":
            continue
        resolved = it["diff_uniform_lo"] > 0 or it["diff_uniform_hi"] < 0
        flipped = np.sign(it["diff_uniform_pp"]) != np.sign(it["diff_mixed_pp"])
        if resolved and flipped:
            triggers.append(f"order_change:{it['planner_a']}>{it['planner_b']}")
        if (it["interaction_lo"] > 0 or it["interaction_hi"] < 0) and abs(
            it["interaction_pp"]
        ) >= mat:
            triggers.append(f"selective_interaction:{it['planner_a']}-{it['planner_b']}")
    for (name, p), (pt, lo, hi) in pooled.items():
        if name in ("success", "collision") and (lo > 0 or hi < 0) and abs(pt) >= mat:
            triggers.append(f"selective_deterioration:{name}:{p}")
    widths = {
        f"{n}|{p}": hi - lo
        for (n, p), (pt, lo, hi) in pooled.items()
        if n in ("success", "collision")
    }
    too_wide = [k for k, w in widths.items() if w > wide]
    within = all(
        lo >= -persist_bound and hi <= persist_bound
        for (n, p), (pt, lo, hi) in pooled.items()
        if n in ("success", "collision")
    )
    ped = []
    for p in PLANNERS:
        pt, lo, hi = pooled[("ped_near_delay_rate_slow", p)]
        if lo > 0:
            ped.append(p)
    if triggers:
        outcome = "B_ORDER_CHANGE_OR_SELECTIVE_DETERIORATION"
    elif too_wide:
        outcome = "D_INCONCLUSIVE_RETAIN_LIMITATION"
    elif within:
        outcome = "A_PERSIST_FINITE_SENSITIVITY"
    else:
        outcome = "D_INCONCLUSIVE_RETAIN_LIMITATION"
    qualifier = (
        "C_SLOW_PED_DELAY_QUALIFIER"
        if ped and outcome in ("A_PERSIST_FINITE_SENSITIVITY", "D_INCONCLUSIVE_RETAIN_LIMITATION")
        else None
    )
    return {
        "outcome": outcome,
        "qualifier": qualifier,
        "triggers": triggers,
        "too_wide": too_wide,
        "within_persist_bound": within,
        "secondary_observations": {"slow_ped_near_delay_increase_planners": ped},
    }


def main() -> None:
    """Dispatch extraction or analysis from the command line."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--runs", required=True)
    e.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--episodes", required=True)
    a.add_argument("--contract", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--exclude-relocation-pairs", action="store_true")
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.runs, args.out)
    else:
        analyze(args.episodes, args.contract, args.out, args.exclude_relocation_pairs)


if __name__ == "__main__":
    main()
