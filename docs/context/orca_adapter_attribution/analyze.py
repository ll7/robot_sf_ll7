"""Predeclared analysis for the ORCA adapter-attribution diagnostic.

    python analyze.py --contract contract.yaml --evidence evidence/ --output <dir>

Reads ``episodes.csv`` and ``steps.csv.gz`` (from extract.py) and writes
``cell_table.csv``, ``paired_effects.csv`` and ``summary.json``.  Needs only the
standard library and NumPy.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import yaml

if TYPE_CHECKING:
    from typing import Any


class Effect(TypedDict):
    """One paired effect and its seed-block interval."""

    effect: str
    treatment: str
    control: str
    scope: str
    endpoint: str
    n_seeds: int
    treatment_count: int
    control_count: int
    denominator: int
    estimate: float
    ci95: list[float]


DRAWS = 10_000
RNG_SEED = 1001


def _ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _f(value: str) -> float | None:
    return None if value in ("", "None", None) else float(value)


def main() -> int:  # noqa: C901, PLR0915
    """Compute the predeclared tables, effects and rule outcomes.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", default="main")
    parser.add_argument("--steps", default="steps.csv.gz", help="step table inside --evidence")
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    th = contract["thresholds"]
    eps = float(th["eps_mps"])
    seeds = [int(s) for s in contract["seeds"][args.stage]]
    cells = list(contract["cells"])
    deficit = list(contract["deficit_cells"])
    configs = [c["key"] for c in contract["configurations"]]
    args.output.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.evidence / "episodes.csv", encoding="utf-8")))
    rows = [r for r in rows if int(r["seed"]) in set(seeds)]
    by = {(r["config"], r["cell"], int(r["seed"])): r for r in rows}

    # Completeness and mode accounting.
    expected = len(configs) * len(cells) * len(seeds)
    modes = Counter(r["mode"] for r in rows)
    init_ids = defaultdict(set)
    for r in rows:
        init_ids[(r["cell"], int(r["seed"]))].add(
            (
                round(float(r["init_x"]), 6),
                round(float(r["init_y"]), 6),
                round(float(r["init_heading"]), 6),
            )
        )
    init_matched = all(len(v) == 1 for v in init_ids.values())

    # Cell table.
    cell_rows: list[dict[str, Any]] = []
    for cfg in configs:
        for cell in cells:
            sub = [by[(cfg, cell, s)] for s in seeds if (cfg, cell, s) in by]
            failed = [r for r in sub if r["success"] == "0"]
            dom = Counter(r["tail_dominant_stall"] for r in failed)
            cell_rows.append(
                {
                    "config": cfg,
                    "cell": cell,
                    "n": len(sub),
                    "success": sum(int(r["success"]) for r in sub),
                    "collision": sum(int(r["collision"]) for r in sub),
                    "timeout": sum(int(r["timeout"]) for r in sub),
                    "deadlock_metric": sum(int(r["deadlock_metric"]) for r in sub),
                    "crossed_wall": sum(int(r["crossed_wall"]) for r in sub),
                    "terminal_x_median": float(np.median([float(r["terminal_x"]) for r in sub]))
                    if sub
                    else None,
                    "min_ped_dist_min": min(float(r["min_ped_dist"]) for r in sub) if sub else None,
                    "failed": len(failed),
                    **{
                        f"failed_dominant_{k}": dom.get(k, 0)
                        for k in (
                            "pref_zero",
                            "solver_zero",
                            "projection_zero",
                            "plant_zero",
                            "command_zero",
                            "none",
                        )
                    },
                    "native": sum(r["mode"] == "native" for r in sub),
                    "fallback": sum(r["mode"] == "fallback" for r in sub),
                    "degraded": sum(r["mode"] == "degraded" for r in sub),
                }
            )

    # Step-level measures (A3, M1).
    a3_steps = Counter()
    a3_eps = defaultdict(set)
    gaps = defaultdict(list)
    proj_steps = Counter()
    with gzip.open(args.evidence / args.steps, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["seed"]) not in set(seeds):
                continue
            key = (r["config"], r["cell"])
            g = _f(r.get("door_gap_m", ""))
            if g is not None:
                gaps[key].append(g)
            if r["config"] != "orca_dd":
                continue
            sx, sy = _f(r["solver_vx"]), _f(r["solver_vy"])
            gx, gy = _f(r["goal_x"]), _f(r["goal_y"])
            sel = _f(r["sel_a"])
            if sx is None or sy is None or gx is None or gy is None or sel is None:
                continue
            dx, dy = gx - float(r["x"]), gy - float(r["y"])
            norm = math.hypot(dx, dy) or 1.0
            goalward = (sx * dx + sy * dy) / norm
            if goalward >= eps and abs(sel) < eps:
                a3_steps[r["cell"]] += 1
                a3_eps[r["cell"]].add(int(r["seed"]))
            if r["stall_class"] == "projection_zero":
                proj_steps[r["cell"]] += 1

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(seeds), size=(DRAWS, len(seeds)))

    def seed_matrix(cfg: str, cell_list: list[str], field: str) -> np.ndarray:
        return np.array(
            [[float(by[(cfg, c, s)][field]) for c in cell_list] for s in seeds], dtype=float
        )

    effects: list[Effect] = []
    for treat, ctrl, label in (("orca_holo", "orca_dd", "E1"), ("sf_holo", "sf_dd", "E2")):
        for scope, cl in [*((c, [c]) for c in cells), ("deficit_pooled", deficit)]:
            for field in ("success", "crossed_wall", "deadlock_metric", "collision"):
                d = seed_matrix(treat, cl, field).mean(axis=1) - seed_matrix(ctrl, cl, field).mean(
                    axis=1
                )
                boot = d[idx].mean(axis=1)
                effects.append(
                    {
                        "effect": label,
                        "treatment": treat,
                        "control": ctrl,
                        "scope": scope,
                        "endpoint": field,
                        "n_seeds": len(seeds),
                        "treatment_count": int(seed_matrix(treat, cl, field).sum()),
                        "control_count": int(seed_matrix(ctrl, cl, field).sum()),
                        "denominator": len(seeds) * len(cl),
                        "estimate": float(d.mean()),
                        "ci95": _ci(boot),
                    }
                )

    # A1/A2 shares over failed orca_dd episodes in deficit cells, seed-block bootstrap.
    def share(kinds: set[str]) -> dict:
        per_seed_num = np.zeros(len(seeds))
        per_seed_den = np.zeros(len(seeds))
        for i, s in enumerate(seeds):
            for c in deficit:
                r = by[("orca_dd", c, s)]
                if r["success"] == "0":
                    per_seed_den[i] += 1
                    per_seed_num[i] += r["tail_dominant_stall"] in kinds
        num, den = per_seed_num.sum(), per_seed_den.sum()
        bn, bd = per_seed_num[idx].sum(axis=1), per_seed_den[idx].sum(axis=1)
        boot = np.where(bd > 0, bn / np.maximum(bd, 1), np.nan)
        return {
            "count": int(num),
            "denominator": int(den),
            "share": float(num / den) if den else None,
            "ci95": _ci(boot[~np.isnan(boot)]) if den else None,
        }

    a1 = share({"solver_zero", "pref_zero"})
    a2 = share({"projection_zero"})
    m1 = {
        f"{cfg}|{cell}": (float(np.median(gaps[(cfg, cell)])) if gaps[(cfg, cell)] else None)
        for cfg in ("orca_dd", "orca_holo")
        for cell in cells
    }

    def eff(label: str, scope: str, field: str = "success") -> Effect:
        return next(
            e
            for e in effects
            if e["effect"] == label and e["scope"] == scope and e["endpoint"] == field
        )

    ctab = {(r["config"], r["cell"]): r for r in cell_rows}
    passable = int(th["passable_successes"])
    rules = {
        "R1_native_removes_deficit": all(
            ctab[("orca_holo", c)]["success"] >= passable for c in deficit
        )
        and eff("E1", "deficit_pooled")["ci95"][0] > 0,
        "R2_projection_suppresses": (a2["share"] or 0) >= 0.5,
        "R3_solver_requests_zero": (a1["share"] or 0) >= 0.8,
        "R4_native_worsens": any(eff("E1", s)["ci95"][1] < 0 for s in [*cells, "deficit_pooled"]),
        "R4_cells": [s for s in [*cells, "deficit_pooled"] if eff("E1", s)["ci95"][1] < 0],
        "R5_little_difference": -0.10 <= eff("E1", "deficit_pooled")["ci95"][0]
        and eff("E1", "deficit_pooled")["ci95"][1] <= 0.10,
        "R5_shared_floor": all(
            ctab[(k, c)]["success"] == 0 for k in ("orca_dd", "orca_holo") for c in deficit
        ),
    }
    rules["stage2_triggered"] = not (
        rules["R2_projection_suppresses"] or rules["R3_solver_requests_zero"]
    )

    summary = {
        "schema": "xorca-summary.v1",
        "interpretation": (
            "New development-seed runs on reconstructed geometry show the immediate RVO2 "
            "output conditional on the executed adapter's inputs. In the terminal windows "
            "of the relevant failures, solver speed was below 0.05 m/s and preferred "
            "velocity was generally goalward. This does not prove that RVO2 or obstacle "
            "inflation intrinsically caused the historical doorway failures. The obstacle "
            "gap is descriptive; no exact-geometry intervention was run."
        ),
        "stage": args.stage,
        "rows": len(rows),
        "expected_rows": expected,
        "modes": dict(modes),
        "initial_state_matched_across_configs": init_matched,
        "A1_solver_or_pref_zero_dominant": a1,
        "A2_projection_zero_dominant": a2,
        "A3_goalward_request_suppressed_steps": {c: a3_steps.get(c, 0) for c in cells},
        "A3_episodes_with_any": {c: len(a3_eps.get(c, ())) for c in cells},
        "orca_dd_projection_zero_steps": {c: proj_steps.get(c, 0) for c in cells},
        "M1_median_door_gap_m": m1,
        "rules": rules,
        "bootstrap": {
            "draws": DRAWS,
            "rng": f"numpy default_rng({RNG_SEED})",
            "unit": "seed block",
        },
    }
    with open(args.output / "cell_table.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cell_rows[0]))
        w.writeheader()
        w.writerows(cell_rows)
    with open(args.output / "paired_effects.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "effect",
                "treatment",
                "control",
                "scope",
                "endpoint",
                "n_seeds",
                "denominator",
                "treatment_count",
                "control_count",
                "estimate",
                "ci95_low",
                "ci95_high",
            ],
        )
        w.writeheader()
        for e in effects:
            w.writerow(
                {
                    **{k: v for k, v in e.items() if k != "ci95"},
                    "estimate": f"{e['estimate']:.4f}",
                    "ci95_low": f"{e['ci95'][0]:.4f}",
                    "ci95_high": f"{e['ci95'][1]:.4f}",
                }
            )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "modes": dict(modes), "rules": rules}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
