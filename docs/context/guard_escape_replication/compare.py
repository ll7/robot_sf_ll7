"""Compare independent development seed sets and calculate the secondary pooled estimate."""

import argparse
import csv
import gzip
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import yaml

if TYPE_CHECKING:
    from importlib.abc import Loader
    from importlib.machinery import ModuleSpec


def read_records(packet):
    """Read actual-seed compact evidence without label adaptation."""
    with gzip.open(packet / "evidence/episodes.jsonl.gz", "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def arrays(records, groups, seeds):
    """Build complete paired tensors for all four original endpoints."""
    slots = {(r["planner_key"], r["scenario_id"], r["seed"]): r for r in records}
    expected = {
        (f"G{g}E{e}", cell, seed)
        for g in (0, 1)
        for e in (0, 1)
        for cells in groups.values()
        for cell in cells
        for seed in seeds
    }
    assert len(slots) == len(records) == len(expected) and set(slots) == expected
    tensors = {}
    for group, cells in groups.items():
        values = np.zeros((len(seeds), len(cells), 2, 2, 4))
        for si, seed in enumerate(seeds):
            for ci, cell in enumerate(cells):
                for g in (0, 1):
                    for e in (0, 1):
                        row = slots[(f"G{g}E{e}", cell, seed)]
                        m, o = row["metrics"], row["outcome"]
                        success = o["route_complete"] and not o["collision_event"]
                        assert bool(m["success"]) == success
                        values[si, ci, g, e] = [
                            success,
                            m["ped_collision_count"] > 0,
                            m["obstacle_collision_count"] > 0,
                            o["timeout_event"] and not o["collision_event"] and not success,
                        ]
        tensors[group] = values
    return tensors


def independent_primary(tensors):
    """Calculate paired seed contrasts first, independently of the frozen analyzer."""
    rng = np.random.default_rng(1001)
    draws = rng.integers(0, 30, size=(10000, 30))
    result = {}
    for group, values in tensors.items():
        per_seed = values.mean(axis=1)
        vectors = {
            "guard": (per_seed[:, 1, 0] + per_seed[:, 1, 1] - per_seed[:, 0, 0] - per_seed[:, 0, 1])
            / 2,
            "escape": (
                per_seed[:, 0, 1] + per_seed[:, 1, 1] - per_seed[:, 0, 0] - per_seed[:, 1, 0]
            )
            / 2,
            "interaction": per_seed[:, 1, 1]
            - per_seed[:, 1, 0]
            - per_seed[:, 0, 1]
            + per_seed[:, 0, 0],
        }
        for factor, vector in vectors.items():
            point = vector.mean(axis=0)
            interval = np.quantile(vector[draws].mean(axis=1), [0.025, 0.975], axis=0)
            for k, endpoint in enumerate(
                ["success", "pedestrian_collision", "wall_collision", "timeout"]
            ):
                result[(group, endpoint, factor)] = [point[k], interval[0, k], interval[1, k]]
    return result


def decisions(old_effects, new_effects):
    """Apply the exact frozen decision interpretation to both independent seed sets."""
    outcomes = {}
    for label, source in [("original", old_effects), ("replication", new_effects)]:
        factors = {}
        for factor in ("guard", "escape"):
            control = source[("crowding_control", "success", factor)]
            qualifies = []
            detail = {}
            for group in ("static_constriction", "portal"):
                row = source[(group, "success", factor)]
                diff, lo, hi = [100 * float(row[k]) for k in ("difference", "ci_low", "ci_high")]
                passes = abs(diff) >= 5.0 and (lo > 0 or hi < 0)
                if passes:
                    qualifies.append(diff)
                detail[group] = {
                    "difference_pp": diff,
                    "ci_low_pp": lo,
                    "ci_high_pp": hi,
                    "qualifies": passes,
                    "positive_hypothesis_supported": passes and diff > 0,
                }
                if factor == "escape":
                    detail[group]["bounded_below_five_points"] = hi < 5.0
            mixed = bool(qualifies) and min(qualifies) < 0 < max(qualifies)
            as_much = bool(qualifies) and abs(100 * float(control["difference"])) >= max(
                abs(x) for x in qualifies
            )
            factors[factor] = {
                "groups": detail,
                "mixed": mixed,
                "control_moves_as_much": as_much,
                "qualifying_confinement_specific_effect": bool(qualifies)
                and not mixed
                and not as_much,
                "outcome": "MIXED"
                if mixed
                else "EFFECT"
                if qualifies and not as_much
                else "NO EFFECT",
            }
        overall = (
            "EFFECT"
            if any(x["qualifying_confinement_specific_effect"] for x in factors.values())
            else "MIXED"
            if any(x["mixed"] for x in factors.values())
            else "NO EFFECT"
        )
        outcomes[label] = {"overall": overall, "factors": factors}
    outcomes["replication_agrees_with_original"] = {
        "overall": outcomes["original"]["overall"] == outcomes["replication"]["overall"],
        **{
            f: outcomes["original"]["factors"][f]["outcome"]
            == outcomes["replication"]["factors"][f]["outcome"]
            for f in ("guard", "escape")
        },
    }
    return outcomes


def main():
    """Verify primary numbers, estimate the secondary pool and write comparisons."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--original", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    original = a.original or a.packet.parent / "guard_escape_ablation"
    a.output.mkdir(parents=True, exist_ok=True)
    groups = yaml.safe_load((a.packet / "inputs/study.yaml").read_text())["groups"]
    old, new = read_records(original), read_records(a.packet)
    assert {r["seed"] for r in old}.isdisjoint({r["seed"] for r in new})
    tensors = arrays(new, groups, list(range(1031, 1061)))
    independent = independent_primary(tensors)
    with (a.packet / "evidence/paired_effects.csv").open() as f:
        effects = list(csv.DictReader(f))
    max_error = max(
        abs(float(r[key]) - independent[(r["group"], r["endpoint"], r["effect"])][i])
        for r in effects
        for i, key in enumerate(("difference", "ci_low", "ci_high"))
    )
    assert max_error < 2e-14, max_error
    # Same draw shape, order of means, percentile call and contrasts as original.
    module_spec = cast(
        "ModuleSpec", importlib.util.spec_from_file_location("unchanged", a.packet / "analyze.py")
    )
    analyzer = importlib.util.module_from_spec(module_spec)
    cast("Loader", module_spec.loader).exec_module(analyzer)
    pooled = arrays(old + new, groups, list(range(1001, 1061)))
    draws = np.random.default_rng(1001).integers(0, 60, size=(10000, 60))
    pooled_rows = []
    for group, values in pooled.items():
        point = analyzer.contrasts(np.moveaxis(values.mean(axis=(0, 1)), -1, 0))
        boot = analyzer.contrasts(np.moveaxis(values[draws].mean(axis=(1, 2)), -1, 1))
        for factor in ("guard", "escape", "interaction"):
            intervals = np.quantile(boot[factor], [0.025, 0.975], axis=0)
            for k, endpoint in enumerate(analyzer.ENDPOINTS):
                pooled_rows.append(
                    {
                        "group": group,
                        "endpoint": endpoint,
                        "effect": factor,
                        "status": "secondary_not_preregistered_primary",
                        "seed_blocks": 60,
                        "paired_denominator_per_arm": 60 * len(groups[group]),
                        "difference": float(point[factor][k]),
                        "ci_low": float(intervals[0, k]),
                        "ci_high": float(intervals[1, k]),
                    }
                )
    with (a.output / "pooled_effects.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pooled_rows[0]))
        writer.writeheader()
        writer.writerows(pooled_rows)
    with (original / "evidence/paired_effects.csv").open() as f:
        old_effects = {(r["group"], r["endpoint"], r["effect"]): r for r in csv.DictReader(f)}
    new_effects = {(r["group"], r["endpoint"], r["effect"]): r for r in effects}
    comparisons = []
    for row in pooled_rows:
        if row["endpoint"] != "success":
            continue
        item = {"group": row["group"], "factor": row["effect"]}
        key = (row["group"], row["endpoint"], row["effect"])
        for label, source in [
            ("original", old_effects[key]),
            ("replication", new_effects[key]),
            ("pooled_secondary", row),
        ]:
            for metric in ("difference", "ci_low", "ci_high"):
                item[f"{label}_{metric}_pp"] = 100 * float(source[metric])
        comparisons.append(item)
    with (a.output / "replication_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    outcomes = decisions(old_effects, new_effects)
    outcomes["independent_primary_max_absolute_error_proportion"] = max_error
    outcomes["pooled_analysis_role"] = (
        "Secondary; not preregistered as primary; never used to rescue primary outcome"
    )
    (a.output / "decision.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    print(
        json.dumps(
            {
                "independent_primary_max_absolute_error": max_error,
                "decision": outcomes["replication"]["overall"],
                "agreement": outcomes["replication_agrees_with_original"],
            }
        )
    )


if __name__ == "__main__":
    main()
