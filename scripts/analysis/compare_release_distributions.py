#!/usr/bin/env python3
"""Independent, descriptive 0.0.7/v1 versus 0.0.8/v2 release distributions.

This tool reads rows only. It never creates episodes, pairs evaluation seeds,
or attributes a combined release difference to a planner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import fisher_exact

from robot_sf.benchmark.infeasible_probe_safe_failure import PROBE_SCENARIO_IDS, classify_probe_row
from robot_sf.benchmark.metric_definitions import CHANGED_METRICS, require_uniform_metric_schema
from robot_sf.benchmark.release_parameter_freeze import ARM_SLOTS_0_0_7_TO_0_0_8

REGISTRY_PATH = Path(__file__).with_name("registries") / "release_distribution_metrics.json"
BOOTSTRAP_SEED = 20260930  # analysis RNG only; never an episode seed
RESAMPLES = 10_000
Q = 0.05
SUCCESS_CONDITIONING = "conditional on success; estimands differ when success rates differ"
RATES = ("success", "collision", "timeout")
SUCCESS_ONLY = {
    "path_efficiency",
    "time_to_goal_norm_success_only",
    "time_to_goal_ideal_ratio",
    "success_path_length",
}


def registry() -> dict:
    """Load the explicit reviewed registry and fail closed on definition drift."""
    result = json.loads(REGISTRY_PATH.read_text())
    if set(result["allowlist"]) & CHANGED_METRICS:
        raise ValueError("allowlist intersects CHANGED_METRICS")
    return result


def wilson(k: int, n: int) -> list[float] | None:
    """Wilson 95% score interval with z=1.959963984540054."""
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    divisor = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / divisor
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / divisor
    return [max(0.0, centre - half), min(1.0, centre + half)]


def newcombe(k0: int, n0: int, k1: int, n1: int) -> list[float] | None:
    """Newcombe hybrid score interval for independent p1 minus p0."""
    if not n0 or not n1:
        return None
    lo0, hi0 = wilson(k0, n0)
    lo1, hi1 = wilson(k1, n1)
    p0, p1 = k0 / n0, k1 / n1
    delta = p1 - p0
    return [
        max(-1.0, delta - math.hypot(p1 - lo1, hi0 - p0)),
        min(1.0, delta + math.hypot(hi1 - p1, p0 - lo0)),
    ]


def outcomes(row: dict, release: str) -> dict[str, bool]:
    """Use outcome flags plus historical authored-limit termination labels."""
    outcome = row["outcome"]
    for key in ("route_complete", "collision_event", "timeout_event"):
        if type(outcome.get(key)) is not bool:
            raise ValueError(f"outcome requires boolean {key}")
    reason = row.get("termination_reason")
    timed_out = outcome["timeout_event"] or reason == "max_steps"
    if release == "0.0.7" and reason == "terminated":
        definition = row.get("_definition", {})
        scenario = definition.get("scenario", {})
        authored_limit = scenario.get("simulation_config", {}).get("max_episode_steps")
        steps = definition.get("steps")
        # #9999: 'terminated' denotes an authored-limit timeout, not success/collision.
        timed_out |= (
            isinstance(authored_limit, (int, float))
            and isinstance(steps, (int, float))
            and steps >= authored_limit
            and not outcome["route_complete"]
            and not outcome["collision_event"]
        )
    return {
        "success": outcome["route_complete"],
        "collision": outcome["collision_event"],
        "timeout": bool(timed_out),
    }


def check_slots(rows: dict, *, release: str, diagnostic_partial: bool = False) -> None:
    """Check the exact roster and rectangular 48-scenario, 30-seed sample."""
    expected = {
        getattr(slot, "key_" + release.replace(".", "_")) for slot in ARM_SLOTS_0_0_7_TO_0_0_8
    }
    arms = {slot[0] for slot in rows}
    if arms != expected:
        raise ValueError(f"{release}: expected 14 mapped arms")
    groups = defaultdict(set)
    seed_sets = defaultdict(set)
    for arm, kin, scenario, seed, track in rows:
        groups[arm, kin, scenario, track].add(seed)
        seed_sets[arm, kin].add(seed)
    if diagnostic_partial:
        return
    scenarios = {slot[2] for slot in rows}
    arm_shapes = {arm: {(s[1], s[2], s[4]) for s in rows if s[0] == arm} for arm in arms}
    shapes = list(arm_shapes.values())
    if (
        len(rows) != 14 * 48 * 30
        or len(groups) != 14 * 48
        or len(scenarios) != 48
        or any(len(seeds) != 30 for seeds in groups.values())
        or any(len(seeds) != 30 for seeds in seed_sets.values())
        or any(shape != shapes[0] for shape in shapes)
    ):
        raise ValueError(f"{release}: expected exact 14 x 48 x 30 slots")


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in sorted(value.items()):
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        elif isinstance(item, list) and key == "rows":
            for cell in item:
                if isinstance(cell, dict) and isinstance(cell.get("metric"), str):
                    result[f"{prefix}.{cell['metric']}"] = cell.get("value")
        elif item is None or isinstance(item, (int, float)):
            result[name] = item
    return result


def _samples(rows: list[dict], metric: str, release: str) -> tuple[np.ndarray, dict]:
    success = sum(outcomes(row, release)["success"] for row in rows)
    values = []
    eligible = []
    for row in rows:
        if metric in RATES:
            eligible.append(float(outcomes(row, release)[metric]))
        elif (
            metric.split(".", maxsplit=1)[0] in SUCCESS_ONLY
            and not outcomes(row, release)["success"]
        ):
            continue
        else:
            eligible.append(_finite(_flatten(row["metrics"]).get(metric)))
    values = [value for value in eligible if value is not None]
    return np.asarray(values, dtype=float), {
        "n": len(rows),
        "n_success": success,
        "n_defined": len(values),
        "n_missing": len(eligible) - len(values),
        "n_excluded": len(rows) - len(eligible),
    }


def _bootstrap(
    groups: list[np.ndarray], rng: np.random.Generator, resamples: int, *, pooled: bool
) -> np.ndarray | None:
    # No resample matrix scales with release size: process <=256 replicates at a time.
    defined = [g for g in groups if len(g)]
    if not defined:
        return None
    draws = np.empty(resamples)
    for start in range(0, resamples, 256):
        size = min(256, resamples - start)
        # Scenarios are fixed; resample seeds independently within each scenario.
        totals = np.zeros(size)
        selected_groups = defined if pooled else defined[:1]
        for group in selected_groups:
            totals += group[rng.integers(0, len(group), (size, len(group)))].mean(axis=1)
        draws[start : start + size] = totals / len(selected_groups)
    return draws


def _joint_scenario_difference(
    old: list[np.ndarray], new: list[np.ndarray], rng: np.random.Generator, resamples: int
) -> np.ndarray:
    """Sensitivity: common scenario draws, independent within-release seed draws."""
    if len(old) != len(new):
        raise ValueError("joint sensitivity requires matched scenario groups")
    draws = np.empty(resamples)
    for start in range(0, resamples, 256):
        size = min(256, resamples - start)
        selected = rng.integers(0, len(old), (size, len(old)))
        totals = [np.zeros(size), np.zeros(size)]
        counts = [np.zeros(size), np.zeros(size)]
        for index, pair in enumerate(zip(old, new, strict=True)):
            multiplicity = (selected == index).sum(axis=1)
            for occurrence in range(int(multiplicity.max())):
                mask = multiplicity > occurrence
                n = int(mask.sum())
                for release, group in enumerate(pair):
                    if len(group):
                        sample = group[rng.integers(0, len(group), (n, len(group)))].mean(axis=1)
                        totals[release][mask] += sample
                        counts[release][mask] += 1
        means = [
            np.divide(total, count, out=np.full(size, np.nan), where=count > 0)
            for total, count in zip(totals, counts, strict=True)
        ]
        draws[start : start + size] = means[1] - means[0]
    return draws


def _ci(draws: np.ndarray | None) -> list[float] | None:
    if draws is None or not np.isfinite(draws).any():
        return None
    return np.nanquantile(draws, [0.025, 0.975]).tolist()


def _fingerprint(rows: list[dict]) -> dict:
    # A definition is seed-independent; retain all scientifically relevant controls.
    definitions = {}
    for row in rows:
        scenario = row.get("_definition", {}).get("scenario", {})
        simulation = scenario.get("simulation_config", {})
        definition = {
            key: scenario.get(key)
            for key in ("map_file", "run_horizon", "run_dt", "robot_config", "ped_density")
        }
        definition["simulation_config"] = {
            key: value for key, value in simulation.items() if key != "route_spawn_seed"
        }
        definition["authored_horizon"] = scenario.get("metadata", {}).get("scenario_horizon")
        definition["density_metadata"] = {
            key: scenario.get("metadata", {}).get(key)
            for key in ("density", "spawn_mode", "groups", "flow")
        }
        definition["map_sha256"] = row.get("_map_sha256")
        canonical = json.dumps(definition, sort_keys=True)
        definitions[hashlib.sha256(canonical.encode()).hexdigest()] = definition
    configs = sorted({str(row.get("_definition", {}).get("algo_config_hash")) for row in rows})
    return {"scenario_definitions": definitions, "algo_config_hashes": configs}


def _cell(
    old: list[list[dict]],
    new: list[list[dict]],
    metric: str,
    status: str,
    identity: dict,
    *,
    resamples: int,
) -> dict:
    pooled = identity["level"] == "arm"
    groups = []
    summaries = []
    draws = []
    key = json.dumps({**identity, "metric": metric}, sort_keys=True)
    rng = np.random.default_rng(
        [BOOTSTRAP_SEED, int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)]
    )
    for release, source in (("0.0.7", old), ("0.0.8", new)):
        arrays_and_counts = [_samples(rows, metric, release) for rows in source]
        arrays = [item[0] for item in arrays_and_counts]
        counts = {
            key: sum(item[1][key] for item in arrays_and_counts)
            for key in ("n", "n_success", "n_defined", "n_missing", "n_excluded")
        }
        sufficient = (
            metric.split(".", maxsplit=1)[0] not in SUCCESS_ONLY or counts["n_success"] >= 5
        )
        estimate = (
            float(np.mean([a.mean() for a in arrays if len(a)]))
            if pooled
            else float(arrays[0].mean())
            if arrays and len(arrays[0])
            else None
        )
        boot = _bootstrap(arrays, rng, resamples, pooled=pooled) if sufficient else None
        ci = _ci(boot)
        score = None
        if metric in RATES:
            score = wilson(int(sum(a.sum() for a in arrays)), counts["n_defined"])
            if not pooled:
                ci = score
        summaries.append(
            {
                **counts,
                "estimate": estimate if sufficient else None,
                "ci": ci,
                "wilson_ci": score,
                "sufficient": sufficient,
            }
        )
        draws.append(boot)
        groups.append(arrays)
    a, b = summaries
    difference = None
    interval = None
    sensitivity_interval = None
    p = None
    if status == "same" and all(s["sufficient"] and s["estimate"] is not None for s in summaries):
        difference = b["estimate"] - a["estimate"]
        delta_draws = draws[1] - draws[0]  # independent RNG draws, no seed pairing
        interval = _ci(delta_draws)
        if pooled:
            sensitivity_rng = np.random.default_rng(
                [BOOTSTRAP_SEED, int(hashlib.sha256(key.encode()).hexdigest()[:16], 16), 1]
            )
            sensitivity_interval = _ci(
                _joint_scenario_difference(groups[0], groups[1], sensitivity_rng, resamples)
            )
        if metric in RATES and not pooled:
            k0, k1 = int(groups[0][0].sum()), int(groups[1][0].sum())
            n0, n1 = a["n_defined"], b["n_defined"]
            interval = newcombe(k0, n0, k1, n1)
            p = float(fisher_exact([[k0, n0 - k0], [k1, n1 - k1]]).pvalue)
        else:
            centred = delta_draws - difference
            finite = centred[np.isfinite(centred)]
            p = float((1 + np.count_nonzero(np.abs(finite) >= abs(difference))) / (len(finite) + 1))
    if status == "0.0.8-only":
        a = {
            "n": 0,
            "n_success": 0,
            "n_defined": 0,
            "n_missing": 0,
            "n_excluded": 0,
            "estimate": None,
            "ci": None,
            "wilson_ci": None,
            "sufficient": False,
        }
    support_groups = groups[1:] if status == "0.0.8-only" else groups
    degenerate = any(len(array) < 2 for arrays in support_groups for array in arrays)
    return {
        **identity,
        "metric": metric,
        "degenerate": degenerate,
        "degeneracy": "degenerate: fewer than 2 seeds per scenario" if degenerate else "",
        "conditioning": SUCCESS_CONDITIONING
        if metric.split(".", maxsplit=1)[0] in SUCCESS_ONLY
        else "",
        "definition_status": status,
        "definition_changed": status == "changed",
        "sufficient": (b["sufficient"] and b["n_defined"] > 0)
        if status == "0.0.8-only"
        else all(s["sufficient"] and s["n_defined"] > 0 for s in summaries),
        "name_0_0_7": f"{metric}@v1" if status == "changed" else metric,
        "name_0_0_8": f"{metric}@v2" if status != "same" else metric,
        "release_0_0_7": a,
        "release_0_0_8": b,
        "difference": difference,
        "difference_ci": interval,
        "sensitivity_difference_ci": sensitivity_interval,
        "sensitivity_method": "paired joint scenario/independent seed percentile bootstrap"
        if pooled
        else None,
        "p_value": p,
        "q_value": None,
        "changed": False,
        "primary": pooled and metric in ("success", "collision"),
        "ci_method": "scenario-conditional seed percentile bootstrap"
        if pooled
        else "Wilson/Newcombe"
        if metric in RATES
        else "independent seed percentile bootstrap",
    }


def benjamini_hochberg(cells: list[dict]) -> None:
    """Adjust the exploratory family with BH step-up monotonicity."""
    tested = sorted((c for c in cells if c["p_value"] is not None), key=lambda c: c["p_value"])
    minimum = 1.0
    for rank in range(len(tested), 0, -1):
        cell = tested[rank - 1]
        minimum = min(minimum, cell["p_value"] * len(tested) / rank)
        cell["q_value"] = minimum
        cell["adjusted_p_value"] = minimum
        interval = cell["difference_ci"]
        cell["changed"] = bool(minimum <= Q and interval and (interval[0] > 0 or interval[1] < 0))


def holm(cells: list[dict], *, family_size: int) -> None:
    """Control family-wise error for the 28 predeclared primary contrasts."""
    tested = sorted((c for c in cells if c["p_value"] is not None), key=lambda c: c["p_value"])
    if len(tested) > family_size:
        raise ValueError("tested primary contrasts exceed the predeclared family")
    maximum = 0.0
    for rank, cell in enumerate(tested):
        maximum = max(maximum, min(1.0, cell["p_value"] * (family_size - rank)))
        cell["adjusted_p_value"] = maximum
        interval = cell["difference_ci"]
        cell["changed"] = bool(maximum <= Q and interval and (interval[0] > 0 or interval[1] < 0))


def _multiplicity(cells: list[dict]) -> None:
    primary = [c for c in cells if c["primary"]]
    exploratory = [c for c in cells if not c["primary"]]
    primary_size = 2 * len(ARM_SLOTS_0_0_7_TO_0_0_8)
    exploratory_size = sum(c["p_value"] is not None for c in exploratory)
    for cell in cells:
        cell["inference_family"] = "primary" if cell["primary"] else "exploratory"
        cell["multiplicity_method"] = "Holm alpha=0.05" if cell["primary"] else "BH q=0.05"
        cell["family_size"] = primary_size if cell["primary"] else exploratory_size
        cell["adjusted_p_value"] = None
    holm(primary, family_size=primary_size)
    benjamini_hochberg(exploratory)


def compare_samples(  # noqa: C901 - separate unit definitions, metric statuses and arm pooling
    old: dict,
    new: dict,
    *,
    provenance: dict | None = None,
    diagnostic_partial: bool = False,
    resamples: int = RESAMPLES,
) -> dict:
    """Aggregate already-admitted rows without making any release claim."""
    if resamples < RESAMPLES:
        raise ValueError("at least 10000 bootstrap resamples required")
    schemas = [require_uniform_metric_schema(rows.values()) for rows in (old, new)]
    if schemas != ["robot-sf-metrics.v1", "robot-sf-metrics.v2"]:
        raise ValueError("release schemas must be v1 for 0.0.7 and v2 for 0.0.8")
    reg = registry()
    lineage = {slot.key_0_0_8: slot.key_0_0_7 for slot in ARM_SLOTS_0_0_7_TO_0_0_8}
    grouped = [defaultdict(list), defaultdict(list)]
    for index, rows in enumerate((old, new)):
        for (arm, kin, scenario, seed, track), row in sorted(rows.items()):
            canonical = arm if index == 0 else lineage.get(arm)
            if canonical is None:
                raise ValueError(f"unmapped successor arm: {arm}")
            grouped[index][canonical, kin, scenario, track].append(row)
    cells = []
    probe_units = []
    fingerprints = {}
    pool = defaultdict(list)
    omitted = []
    for unit in sorted(set(grouped[0]) | set(grouped[1])):
        a, b = grouped[0][unit], grouped[1][unit]
        if not a or not b:
            if not diagnostic_partial:
                raise ValueError(f"missing distribution unit {unit}")
            omitted.append(unit)
            continue
        arm, kin, scenario, track = unit
        successor = next(key for key, value in lineage.items() if value == arm)
        f0, f1 = _fingerprint(a), _fingerprint(b)
        flags = {
            "arm_replaced": arm != successor,
            "scenario_changed": f0["scenario_definitions"] != f1["scenario_definitions"],
            "planner_config_changed": f0["algo_config_hashes"] != f1["algo_config_hashes"],
        }
        identity = {
            "level": "unit",
            "arm_0_0_7": arm,
            "arm_0_0_8": successor,
            "kinematics": kin,
            "scenario_id": scenario,
            "benchmark_track": track,
            **flags,
            "interpretation": "combined release difference"
            if any(flags.values())
            else "release difference",
        }
        fingerprints["|".join(unit)] = {"0.0.7": f0, "0.0.8": f1}
        if scenario in PROBE_SCENARIO_IDS:
            probe_units.append(
                {
                    **identity,
                    "release_0_0_7": {
                        "n": len(a),
                        "safe_failure_classes": dict(
                            sorted(Counter(classify_probe_row(row) for row in a).items())
                        ),
                    },
                    "release_0_0_8": {
                        "n": len(b),
                        "safe_failure_classes": dict(
                            sorted(Counter(classify_probe_row(row) for row in b).items())
                        ),
                    },
                }
            )
            continue
        pool[arm, successor, kin, track].append((a, b, flags))
        keys = set().union(*(_flatten(row["metrics"]).keys() for row in [*a, *b]))
        changed = CHANGED_METRICS | reg["additional_changed"].keys()
        metrics = dict.fromkeys((*RATES, *reg["allowlist"]), "same")
        for name in sorted(keys):
            root = name.split(".")[0]
            if (
                root.startswith(("robot_force_", "snqi_v2"))
                or root == "path_efficiency_reference_violation"
            ):
                metrics[name] = "0.0.8-only"
            elif root in changed:
                metrics[name] = "changed"
        for metric, status in sorted(metrics.items()):
            cells.append(_cell([a], [b], metric, status, identity, resamples=resamples))
    for (arm, successor, kin, track), units in sorted(pool.items()):
        flags = {key: any(unit[2][key] for unit in units) for key in units[0][2]}
        identity = {
            "level": "arm",
            "arm_0_0_7": arm,
            "arm_0_0_8": successor,
            "kinematics": kin,
            "scenario_id": "__pooled_non_probe__",
            "benchmark_track": track,
            **flags,
            "interpretation": "combined release difference"
            if any(flags.values())
            else "release difference",
        }
        for metric in RATES:
            cells.append(
                _cell(
                    [unit[0] for unit in units],
                    [unit[1] for unit in units],
                    metric,
                    "same",
                    identity,
                    resamples=resamples,
                )
            )
    _multiplicity(cells)
    return {
        "schema": "release-distribution-comparison.v1",
        "label": "diagnostic, not the release comparison"
        if diagnostic_partial
        else "descriptive release comparison",
        "provenance": provenance or {},
        "metric_schema_versions": schemas,
        "registry": reg,
        "registry_sha256": hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resamples": resamples,
        "methods": {
            "rates": "Wilson 95%; Newcombe independent hybrid score; two-sided Fisher exact p",
            "means": "seed percentile bootstrap; independent two-sample difference; centred bootstrap p",
            "pool": "equal scenario weight; fixed scenarios with independent within-scenario seed bootstrap per release; probe excluded",
            "sensitivity": "paired joint scenario resample matched by scenario_id; independent seeds per release",
            "multiplicity": "28 predeclared primaries: Holm alpha=0.05; other unit and arm cells: exploratory BH q=0.05",
            "primary": "scenario-conditional arm-level success and collision differences (fixed benchmark suite)",
            "missing": "null/NaN/infinite excluded and counted; no imputation; success-only >=5 each",
            "timeout": "0.0.7 max_steps and unsuccessful, noncollision terminated at authored step limit (#9999)",
            "jerk": "episode rows only; empty breakdown cells are missing (#10044)",
        },
        "fingerprints": fingerprints,
        "probe_units": probe_units,
        "omitted_diagnostic_units": omitted,
        "cells": cells,
    }


def write_report(report: dict, output: Path) -> None:
    """Write reproducible JSON and full unit/metric thesis tables."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "distribution.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    table = []
    for cell in report["cells"]:
        flat = {
            key: value
            for key, value in cell.items()
            if key not in ("release_0_0_7", "release_0_0_8")
        }
        for release in ("0_0_7", "0_0_8"):
            flat.update(
                {f"{key}_{release}": value for key, value in cell[f"release_{release}"].items()}
            )
        table.append(
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in flat.items()
            }
        )
    columns = list(table[0]) if table else []
    with (output / "distribution.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    lines = [
        report["label"],
        "",
        "All statistics are descriptive. Arm/scenario/planner-config flags denote combined release differences. "
        "Degeneracy flags qualify seed uncertainty. Probe excluded from pooling.",
        "",
        "28 predeclared primaries: Holm alpha=0.05. Other unit and arm cells: exploratory BH q=0.05. Fixed-scenario primary; paired joint scenario sensitivity.",
        "",
        "| Family | Level | Arm (old → new) | Scenario | Metric / definition | n defined v1 / v2 | v1 mean/rate [95% CI] | v2 mean/rate [95% CI] | Difference [95% CI] | Paired joint scenario sensitivity [95% CI] | Adjusted p / q | Changed | Flags | Conditioning |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in report["cells"]:
        a, b = cell["release_0_0_7"], cell["release_0_0_8"]
        flags = ", ".join(
            [
                key
                for key in ("arm_replaced", "scenario_changed", "planner_config_changed")
                if cell[key]
            ]
            + ([cell["degeneracy"]] if cell["degenerate"] else [])
        )
        lines.append(
            f"| {cell['inference_family']} | {cell['level']} | {cell['arm_0_0_7']} → {cell['arm_0_0_8']} | {cell['scenario_id']} | {cell['name_0_0_7']} / {cell['name_0_0_8']} ({cell['definition_status']}) | {a['n_defined']} / {b['n_defined']} | {a['estimate']} {a['ci']} | {b['estimate']} {b['ci']} | {cell['difference']} {cell['difference_ci']} | {cell['sensitivity_difference_ci']} | {cell['adjusted_p_value']} | {cell['changed']} | {flags} | {cell['conditioning']} |"
        )
    (output / "distribution.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    """CLI: identity/schema refusals are errors; unchanged distributions are not."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--successor-root", type=Path, required=True)
    parser.add_argument("--expected-tooling-commit", help="Exact clean main tooling checkout SHA")
    parser.add_argument("--snqi-v2-anchors", type=Path)
    parser.add_argument("--successor-manifest", type=Path, required=True)
    parser.add_argument("--successor-manifest-sha256", required=True)
    parser.add_argument("--successor-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-partial",
        action="store_true",
        help="Relax sample/probe coverage only; output is diagnostic, not release evidence",
    )
    args = parser.parse_args()
    from scripts.analysis._distribution_admission import admit

    try:
        from scripts.analysis.compare_release_0_0_7_to_0_0_8 import _tooling_identity

        _tooling_identity(args.expected_tooling_commit)
        old, new, provenance = admit(
            args.baseline_bundle,
            args.successor_root,
            successor_manifest=args.successor_manifest,
            successor_manifest_sha256=args.successor_manifest_sha256,
            successor_source_root=args.successor_source_root,
            snqi_v2_anchors=args.snqi_v2_anchors,
            diagnostic_partial=args.diagnostic_partial,
        )
        report = compare_samples(
            old, new, provenance=provenance, diagnostic_partial=args.diagnostic_partial
        )
        write_report(report, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"admission failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
