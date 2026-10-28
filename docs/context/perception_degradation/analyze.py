"""Paired seed-block analysis of the matched pedestrian-perception study.

Needs only NumPy and PyYAML; reads the compact episode rows, never the raw archive:

    uv run python docs/context/perception_degradation/analyze.py \
        --episodes docs/context/perception_degradation/evidence/episodes.jsonl.gz \
        --output <scratch dir>

Writes validity.json, rates.csv, effects.csv and summary.json (with the predeclared
decision-rule evaluation from contract.yaml).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

PACKET = Path(__file__).resolve().parent
PLANNERS = [
    "orca",
    "hybrid_v4_continuous_guard_on",
    "hybrid_v4_continuous_guard_off",
    "ppo",
]
CONDITIONS = ["oracle", "position_noise", "delay", "static_occlusion"]
ENDPOINTS = [
    "success",
    "collision",
    "pedestrian_collision",
    "wall_collision",
    "timeout",
    "min_distance_m",
]
RATE_ENDPOINTS = ENDPOINTS[:5]
PRIMARY = ["success", "collision"]
RELEVANT = 0.05
NARROW_HALF_WIDTH = 0.05
# Ignore floating-point roundoff at the preregistered boundaries, not sampling error.
NUMERIC_EPSILON = 1e-12
DRAWS = 10_000
CHUNK = 250


def require(condition: bool, message: str) -> None:
    """Fail closed even under ``python -O``."""
    if not condition:
        raise ValueError(message)


def endpoint_values(row: dict) -> list[float]:
    """Return the endpoint vector for one episode, checking outcome consistency.

    Returns:
        Values in ``ENDPOINTS`` order.
    """
    outcome, metrics = row["outcome"], row["metrics"]
    collision = bool(outcome["collision_event"])
    success = bool(outcome["route_complete"]) and not collision
    require(bool(metrics["success"]) == success, "success/outcome contradiction")
    timeout = row["termination_reason"] == "max_steps" and not collision and not success
    min_distance = metrics["min_distance"]
    return [
        float(success),
        float(collision),
        float((metrics["ped_collision_count"] or 0) > 0),
        float((metrics["obstacle_collision_count"] or 0) > 0),
        float(timeout),
        float("nan") if min_distance is None else float(min_distance),
    ]


def load(path: Path, contract: dict) -> tuple[np.ndarray, list[int], list[str], list[dict]]:
    """Build the complete paired tensor Y[seed, cell, planner, condition, endpoint].

    Returns:
        Tensor, seeds, cells and the raw rows.
    """
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    cells = list(contract["cells"])
    seeds = sorted({int(r["seed"]) for r in rows})
    index = {}
    for r in rows:
        key = (r["seed"], r["scenario_id"], r["planner_key"], r["condition"])
        require(key not in index, f"duplicate episode {key}")
        index[key] = r
    y = np.full((len(seeds), len(cells), len(PLANNERS), len(CONDITIONS), len(ENDPOINTS)), np.nan)
    missing = 0
    for si, seed in enumerate(seeds):
        for ci, cell in enumerate(cells):
            for pi, planner in enumerate(PLANNERS):
                for ki, cond in enumerate(CONDITIONS):
                    row = index.get((seed, cell, planner, cond))
                    if row is None:
                        missing += 1
                        continue
                    y[si, ci, pi, ki] = endpoint_values(row)
    require(missing == 0, f"{missing} paired episodes missing; analysis needs complete seed blocks")
    return y, seeds, cells, rows


def intervention_failures_by_cell(rows: list[dict], cells: list[str], exercise: dict) -> dict:
    """Apply O5 literally to each cell and condition, including negative controls.

    Returns:
        Cell/condition failures and their recorded causes.
    """
    failures = {}
    for cell in cells:
        for cond in CONDITIONS:
            sub = [r for r in rows if r["scenario_id"] == cell and r["condition"] == cond]
            undisturbed = [
                r
                for r in sub
                if cond != "oracle"
                and (r["perception"]["stats"] or {}).get("true_rows", 0) > 0
                and r["perception"]["stats"]["steps_perceived_differs"] == 0
            ]
            unconsumed = sum(
                1 for r in sub if not r["perception"]["all_channels_consumed_every_snapshot"]
            )
            no_occlusion = cond == "static_occlusion" and not exercise[cell]["exercised"]
            if undisturbed or unconsumed or no_occlusion:
                failures[f"{cell}|{cond}"] = {
                    "treatment_never_differs_from_truth": len(undisturbed),
                    "undisturbed_by_planner": {
                        planner: sum(r["planner_key"] == planner for r in undisturbed)
                        for planner in PLANNERS
                        if any(r["planner_key"] == planner for r in undisturbed)
                    },
                    "channel_not_consumed_every_step": unconsumed,
                    "occlusion_not_exercised": no_occlusion,
                }
    return failures


def validity(rows: list[dict], cells: list[str]) -> dict:
    """Count recorded execution markers and intervention failures by cell/condition.

    Returns:
        JSON-safe validity report.
    """
    report: dict = {"episodes": len(rows), "by_planner_condition": {}, "occlusion_exercise": {}}
    failures = []
    for planner in PLANNERS:
        for cond in CONDITIONS:
            sub = [r for r in rows if r["planner_key"] == planner and r["condition"] == cond]
            modes: dict[str, int] = {}
            for r in sub:
                modes[str(r["mode"])] = modes.get(str(r["mode"]), 0) + 1
            fallback = sum(
                1
                for r in sub
                if r["fallback_or_degraded_marker"] is not None or (r["fallback_count"] or 0) > 0
            )
            unconsumed = sum(
                1 for r in sub if not r["perception"]["all_channels_consumed_every_snapshot"]
            )
            with_peds = [r for r in sub if (r["perception"]["stats"] or {}).get("true_rows", 0) > 0]
            undisturbed = sum(
                1 for r in with_peds if r["perception"]["stats"]["steps_perceived_differs"] == 0
            )
            entry = {
                "episodes": len(sub),
                "modes": modes,
                "fallback_or_degraded": fallback,
                "fallback_count_recorded": sum(r["fallback_count"] is not None for r in sub),
                "fallback_count_missing": sum(r["fallback_count"] is None for r in sub),
                "fallback_count_zero": sum(r["fallback_count"] == 0 for r in sub),
                "status_not_ok": sum(1 for r in sub if r["status"] != "ok"),
                "checkpoint_not_loaded": sum(
                    1
                    for r in sub
                    if (
                        r["checkpoint_load_status"] != "loaded"
                        or r["checkpoint_fallback_triggered"] is not False
                    )
                )
                if planner == "ppo"
                else None,
                "effective_view_degraded": sum(1 for r in sub if r["effective_view_degraded"]),
                "channel_not_consumed_every_step": unconsumed,
                "episodes_with_pedestrians": len(with_peds),
                "treatment_never_differs_from_truth": undisturbed if cond != "oracle" else None,
                "oracle_differs_from_truth": sum(
                    1 for r in sub if r["perception"]["stats"]["steps_perceived_differs"] > 0
                )
                if cond == "oracle"
                else None,
                "mean_abs_position_error_m": float(
                    np.mean([r["perception"]["stats"]["mean_abs_position_error_m"] for r in sub])
                )
                if sub
                else None,
            }
            report["by_planner_condition"][f"{planner}|{cond}"] = entry
            if fallback or unconsumed or entry["status_not_ok"] or entry["checkpoint_not_loaded"]:
                failures.append(f"{planner}|{cond}: invalid execution rows")
            if cond in ("position_noise", "delay") and undisturbed:
                failures.append(f"{planner}|{cond}: {undisturbed} episodes never perturbed")
            if cond == "oracle" and entry["oracle_differs_from_truth"]:
                failures.append(f"{planner}|oracle: control differs from truth")
    for cell in cells:
        sub = [r for r in rows if r["condition"] == "static_occlusion" and r["scenario_id"] == cell]
        snaps = sum(r["perception"]["stats"]["snapshots"] for r in sub)
        occl = sum(r["perception"]["stats"]["steps_with_occlusion"] for r in sub)
        report["occlusion_exercise"][cell] = {
            "episodes": len(sub),
            "episodes_with_any_occlusion": sum(
                1 for r in sub if r["perception"]["stats"]["steps_with_occlusion"] > 0
            ),
            "share_of_steps_occluded": occl / snaps if snaps else None,
            "exercised": occl > 0,
        }
    report["intervention_failures"] = intervention_failures_by_cell(
        rows, cells, report["occlusion_exercise"]
    )
    report["failures"] = failures
    return report


def negative_control(y: np.ndarray, cells: list[str], occl_cells: list[str]) -> dict:
    """In cells without occlusion the inputs equal oracle, so outcomes must match exactly.

    Returns:
        Per non-analysis cell: number of seed x planner pairs whose endpoints differ.
    """
    oracle, occl = CONDITIONS.index("oracle"), CONDITIONS.index("static_occlusion")
    out = {}
    for ci, cell in enumerate(cells):
        if cell in occl_cells:
            continue
        a, b = y[:, ci, :, oracle, :5], y[:, ci, :, occl, :5]
        out[cell] = {
            "pairs": int(a.shape[0] * a.shape[1]),
            "differ": int(np.any(a != b, axis=-1).sum()),
        }
    return out


def bootstrap(y: np.ndarray, fn, seed: int = 1001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seed-block bootstrap of a statistic computed from per-seed means.

    ``fn`` maps a cell-level mean tensor [cell, planner, condition, endpoint] (with an
    optional leading draw axis) to the statistic array.

    Returns:
        Point estimate, 2.5th and 97.5th percentile arrays.
    """
    point = fn(np.nanmean(y, axis=0))
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    draws = []
    for start in range(0, DRAWS, CHUNK):
        idx = rng.integers(0, n, size=(min(CHUNK, DRAWS - start), n))
        draws.append(fn(np.nanmean(y[idx], axis=1)))
    stacked = np.concatenate(draws, axis=0)
    return point, np.percentile(stacked, 2.5, axis=0), np.percentile(stacked, 97.5, axis=0)


def pool_weights(cells: list[str], contract: dict) -> np.ndarray:
    """Return equal-cell pooling weights per condition, shape [condition, cell].

    Occlusion pools only the cells where the pilot showed it is exercised (frozen in
    the contract); its treatment and control are averaged over the same cells.

    Returns:
        Row-normalized weights.
    """
    w = np.ones((len(CONDITIONS), len(cells)))
    occl_cells = contract.get("occlusion_analysis_cells")
    if occl_cells is not None:
        w[CONDITIONS.index("static_occlusion")] = [float(c in occl_cells) for c in cells]
    return w / w.sum(axis=1, keepdims=True)


def append_pooled(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Append the pooled mean as an extra cell for ``x`` shaped [..., cell, A, cond, ep].

    Returns:
        Array with the cell axis extended by one (last = pooled).
    """
    pooled = np.einsum("...cake,kc->...ake", x, w)[..., None, :, :, :]
    return np.concatenate([x, pooled], axis=-4)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the predeclared analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = yaml.safe_load((PACKET / "contract.yaml").read_text())
    y, seeds, cells, rows = load(args.episodes, contract)
    args.output.mkdir(parents=True, exist_ok=True)
    valid = validity(rows, cells)
    (args.output / "validity.json").write_text(json.dumps(valid, indent=2) + "\n")
    cell_names = [*cells, "POOLED_EQUAL_CELL"]

    weights = pool_weights(cells, contract)
    rate, rlo, rhi = bootstrap(y, lambda m: append_pooled(m, weights))
    with (args.output / "rates.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["cell", "planner", "condition", "endpoint", "n_seeds", "estimate", "ci_low", "ci_high"]
        )
        for ci, cell in enumerate(cell_names):
            for pi, planner in enumerate(PLANNERS):
                for ki, cond in enumerate(CONDITIONS):
                    for ei, ep in enumerate(ENDPOINTS):
                        output_cell = cell
                        if cell == "POOLED_EQUAL_CELL":
                            output_cell = (
                                "POOLED_EQUAL_FOUR_ACTIVE_CELLS"
                                if cond == "static_occlusion"
                                else "POOLED_EQUAL_SIX_CELLS"
                            )
                        w.writerow(
                            [
                                output_cell,
                                planner,
                                cond,
                                ep,
                                len(seeds),
                                f"{rate[ci, pi, ki, ei]:.6f}",
                                f"{rlo[ci, pi, ki, ei]:.6f}",
                                f"{rhi[ci, pi, ki, ei]:.6f}",
                            ]
                        )

    pairs = list(combinations(range(len(PLANNERS)), 2))
    on, off = (
        PLANNERS.index("hybrid_v4_continuous_guard_on"),
        PLANNERS.index("hybrid_v4_continuous_guard_off"),
    )

    def effects(m: np.ndarray) -> np.ndarray:
        # Per-cell contrasts first, then condition-aware pooling of each contrast.
        cond_eff = m[..., :, 1:, :] - m[..., :, :1, :]  # [.., cell, planner, cond-1, ep]
        diff = np.stack([m[..., a, :, :] - m[..., b, :, :] for a, b in pairs], axis=-3)
        rank_change = diff[..., :, 1:, :] - diff[..., :, :1, :]
        guard = (m[..., on, :, :] - m[..., off, :, :])[..., None, :, :]
        guard_dep = guard[..., 1:, :] - guard[..., :1, :]
        cond_eff = append_pooled(cond_eff, weights[1:])
        diff = append_pooled(diff, weights)
        rank_change = append_pooled(rank_change, weights[1:])
        guard = append_pooled(guard, weights)[..., 0, :, :]
        guard_dep = append_pooled(guard_dep, weights[1:])[..., 0, :, :]
        return np.concatenate(
            [
                cond_eff.reshape(*cond_eff.shape[:-3], -1),
                diff.reshape(*diff.shape[:-3], -1),
                rank_change.reshape(*rank_change.shape[:-3], -1),
                guard.reshape(*guard.shape[:-2], -1),
                guard_dep.reshape(*guard_dep.shape[:-2], -1),
            ],
            axis=-1,
        )

    labels = []
    for planner in PLANNERS:
        for cond in CONDITIONS[1:]:
            labels += [("condition_effect", planner, cond, ep) for ep in ENDPOINTS]
    for a, b in pairs:
        for cond in CONDITIONS:
            labels += [
                ("planner_difference", f"{PLANNERS[a]}-{PLANNERS[b]}", cond, ep) for ep in ENDPOINTS
            ]
    for a, b in pairs:
        for cond in CONDITIONS[1:]:
            labels += [
                ("ranking_change", f"{PLANNERS[a]}-{PLANNERS[b]}", cond, ep) for ep in ENDPOINTS
            ]
    for cond in CONDITIONS:
        labels += [("guard_benefit", "guard_on-guard_off", cond, ep) for ep in ENDPOINTS]
    for cond in CONDITIONS[1:]:
        labels += [("guard_dependence", "guard_on-guard_off", cond, ep) for ep in ENDPOINTS]

    est, lo, hi = bootstrap(y, effects)
    require(est.shape[-1] == len(labels), "label mismatch")
    table = []
    with (args.output / "effects.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "cell",
                "effect",
                "comparison",
                "condition",
                "endpoint",
                "n_seeds",
                "estimate",
                "ci_low",
                "ci_high",
            ]
        )
        for ci, cell in enumerate(cell_names):
            for li, (kind, comp, cond, ep) in enumerate(labels):
                rec = (
                    cell,
                    kind,
                    comp,
                    cond,
                    ep,
                    float(est[ci, li]),
                    float(lo[ci, li]),
                    float(hi[ci, li]),
                )
                table.append(rec)
                output_cell = cell
                if cell == "POOLED_EQUAL_CELL":
                    output_cell = (
                        "POOLED_EQUAL_FOUR_ACTIVE_CELLS"
                        if cond == "static_occlusion"
                        else "POOLED_EQUAL_SIX_CELLS"
                    )
                w.writerow(
                    [
                        output_cell,
                        kind,
                        comp,
                        cond,
                        ep,
                        len(seeds),
                        f"{rec[5]:.6f}",
                        f"{rec[6]:.6f}",
                        f"{rec[7]:.6f}",
                    ]
                )

    pooled = {
        (k, c, cond, ep): (e, a, b)
        for cell, k, c, cond, ep, e, a, b in table
        if cell == "POOLED_EQUAL_CELL"
    }

    def excludes_zero(v):
        return v[1] > NUMERIC_EPSILON or v[2] < -NUMERIC_EPSILON

    def relevant(v):
        return abs(v[0]) + NUMERIC_EPSILON >= RELEVANT and excludes_zero(v)

    def narrow(v):
        return (v[2] - v[1]) / 2 <= NARROW_HALF_WIDTH + NUMERIC_EPSILON

    def o2_decision(base, change, now):
        if not excludes_zero(base):
            return "not_triggered"
        compress = (change[2] < -NUMERIC_EPSILON) if base[0] > 0 else (change[1] > NUMERIC_EPSILON)
        reverse = (now[2] < -NUMERIC_EPSILON) if base[0] > 0 else (now[1] > NUMERIC_EPSILON)
        return "reversal" if reverse else "compression" if compress else "not_triggered"

    # The general pooling clause names six cells; the occlusion-specific clause
    # names four. Evaluate each population with its own matched oracle baseline.
    occl_cells = contract.get("occlusion_analysis_cells") or cells
    populations = {"six_cells": cells, "four_active_cells": occl_cells}

    def occlusion_rankings(m: np.ndarray) -> np.ndarray:
        results = []
        for selected in populations.values():
            selected_rows = m[..., [cells.index(cell) for cell in selected], :, :, :]
            differences = np.stack(
                [selected_rows[..., a, :, :] - selected_rows[..., b, :, :] for a, b in pairs],
                axis=-3,
            )
            base = differences[..., :, :, CONDITIONS.index("oracle"), :].mean(axis=-3)
            now = differences[..., :, :, CONDITIONS.index("static_occlusion"), :].mean(axis=-3)
            results.append(np.stack([base, now, now - base], axis=-2))
        return np.stack(results, axis=-4)

    rank_est, rank_lo, rank_hi = bootstrap(y, occlusion_rankings)
    ranking_populations: dict[str, dict] = {}
    for population_index, (population, selected) in enumerate(populations.items()):
        comparisons = {}
        for pair_index, (a, b) in enumerate(pairs):
            comp = f"{PLANNERS[a]}-{PLANNERS[b]}"
            for ep in PRIMARY:
                ep_index = ENDPOINTS.index(ep)
                intervals = [
                    tuple(
                        float(array[population_index, pair_index, stat_index, ep_index])
                        for array in (rank_est, rank_lo, rank_hi)
                    )
                    for stat_index in range(3)
                ]
                base, now, change = intervals
                comparisons[f"{comp}|{ep}"] = {
                    name: [round(value, 6) for value in interval]
                    for name, interval in zip(
                        ("oracle", "static_occlusion", "ranking_change"), intervals, strict=True
                    )
                }
                comparisons[f"{comp}|{ep}"]["o2"] = o2_decision(base, change, now)
        ranking_populations[population] = {"cells": selected, "comparisons": comparisons}

    rule: dict = {
        "O1_relevant_and_narrow": [],
        "O2_ranking_compresses_or_reverses": [],
        "O3_guard": [],
        "O4_uniform_deterioration": {},
        "O4_secondary_directional_deterioration": {},
        "O5_intervention_failed": [],
        "O6_null_wide": [],
    }
    for planner in PLANNERS:
        for cond in CONDITIONS[1:]:
            for ep in PRIMARY:
                v = pooled[("condition_effect", planner, cond, ep)]
                if relevant(v) and narrow(v):
                    rule["O1_relevant_and_narrow"].append(f"{planner}|{cond}|{ep}")
                elif not relevant(v) and not narrow(v):
                    rule["O6_null_wide"].append(f"{planner}|{cond}|{ep}")
    for a, b in pairs:
        comp = f"{PLANNERS[a]}-{PLANNERS[b]}"
        for ep in PRIMARY:
            for cond in CONDITIONS[1:]:
                if cond == "static_occlusion":
                    decision = ranking_populations["six_cells"]["comparisons"][f"{comp}|{ep}"]["o2"]
                else:
                    base = pooled[("planner_difference", comp, "oracle", ep)]
                    change = pooled[("ranking_change", comp, cond, ep)]
                    now = pooled[("planner_difference", comp, cond, ep)]
                    decision = o2_decision(base, change, now)
                if decision == "reversal":
                    rule["O2_ranking_compresses_or_reverses"].append(f"REVERSAL {comp}|{cond}|{ep}")
                elif decision == "compression":
                    rule["O2_ranking_compresses_or_reverses"].append(
                        f"compression {comp}|{cond}|{ep}"
                    )
    for cond in CONDITIONS[1:]:
        coll = pooled[("condition_effect", "hybrid_v4_continuous_guard_on", cond, "collision")]
        tout = pooled[("condition_effect", "hybrid_v4_continuous_guard_on", cond, "timeout")]
        if coll[0] + NUMERIC_EPSILON >= RELEVANT and excludes_zero(coll):
            rule["O3_guard"].append(f"guard_on collision-prone under {cond}")
        if tout[0] + NUMERIC_EPSILON >= RELEVANT and excludes_zero(tout):
            rule["O3_guard"].append(f"guard_on more conservative (timeout) under {cond}")
        for ep in PRIMARY:
            if relevant(pooled[("guard_dependence", "guard_on-guard_off", cond, ep)]):
                rule["O3_guard"].append(f"guard benefit depends on exact info: {cond}|{ep}")
        for ep in PRIMARY:
            effs = [pooled[("condition_effect", p, cond, ep)] for p in PLANNERS]
            same_sign = all(relevant(v) for v in effs) and len({np.sign(v[0]) for v in effs}) == 1
            no_rank = not any(
                excludes_zero(pooled[("ranking_change", f"{PLANNERS[a]}-{PLANNERS[b]}", cond, ep)])
                for a, b in pairs
            )
            rule["O4_uniform_deterioration"][f"{cond}|{ep}"] = bool(same_sign and no_rank)
            worsening = all(v[0] < 0 if ep == "success" else v[0] > 0 for v in effs)
            rule["O4_secondary_directional_deterioration"][f"{cond}|{ep}"] = bool(
                same_sign and no_rank and worsening
            )
    negative_controls = set(contract.get("occlusion_negative_control_cells", []))
    rule["O5_intervention_failed"] = [
        {
            "cell": cell,
            "condition": cond,
            "negative_control": cond == "static_occlusion" and cell in negative_controls,
            **finding,
        }
        for key, finding in valid["intervention_failures"].items()
        for cell, cond in [key.split("|", 1)]
    ]
    rule["occlusion_negative_control"] = negative_control(y, cells, occl_cells)
    execution_evidence = {}
    for planner in PLANNERS:
        groups = [valid["by_planner_condition"][f"{planner}|{cond}"] for cond in CONDITIONS]
        execution_evidence[planner] = {
            name: sum(group[name] for group in groups)
            for name in (
                "episodes",
                "fallback_or_degraded",
                "fallback_count_recorded",
                "fallback_count_missing",
                "fallback_count_zero",
                "status_not_ok",
                "effective_view_degraded",
                "channel_not_consumed_every_step",
            )
        }
        execution_evidence[planner]["checkpoint_not_loaded"] = (
            sum(group["checkpoint_not_loaded"] for group in groups) if planner == "ppo" else None
        )
    summary = {
        "schema": "xperc-perception-degradation.summary.v1",
        "episodes": len(rows),
        "seeds": len(seeds),
        "cells": cells,
        "bootstrap": {"draws": DRAWS, "rng": "numpy default_rng(1001)", "unit": "seed block"},
        "execution_validity_failures": valid["failures"],
        "execution_evidence_by_planner": execution_evidence,
        "evidence_limitations": {
            "fallback": (
                "No fallback/degraded marker or positive counter was recorded; absent counters "
                "are missing evidence, not recorded zero counts."
            ),
            "pre_episode_push": (
                "Not independently verified from available records. The execution commit's Git "
                "timestamp is 2026-10-28T16:09:11Z; the archived environment starts at "
                "16:09:36Z and the earliest raw episode at 16:09:45.092746Z. The earliest "
                "local remote-tracking fetch observation found is 16:10:22Z. A Git timestamp "
                "does not establish the push time."
            ),
            "occlusion": (
                "O5 marks the cross-trap and overtaking occlusion cell/conditions as failures, "
                "so pooled occlusion results including them are descriptive, not robustness "
                "evidence. The wall-free negative controls are separately labelled."
            ),
            "o2_populations": (
                "The general six-cell pooling and occlusion-specific four-cell pooling clauses "
                "differ. O2 uses matched six-cell contrasts; the matched four-active-cell "
                "analysis is secondary and includes O5-failed cells."
            ),
        },
        "pooled_population_by_condition": {
            cond: occl_cells if cond == "static_occlusion" else cells for cond in CONDITIONS
        },
        "occlusion_ranking_populations": ranking_populations,
        "secondary_occlusion_cell_exercise_only": {
            "description": (
                "Post-hoc geometry-exercise description only; ignoring per-episode differences "
                "does not satisfy the preregistered O5 rule."
            ),
            "active_cells_exercised": {
                cell: valid["occlusion_exercise"][cell]["exercised"] for cell in occl_cells
            },
        },
        "pooled_equal_cell": {
            f"{k}|{c}|{cond}|{ep}": [round(v, 6) for v in val]
            for (k, c, cond, ep), val in pooled.items()
            if ep in (*PRIMARY, "timeout")
        },
        "decision_rule": rule,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(rule, indent=2))


if __name__ == "__main__":
    main()
