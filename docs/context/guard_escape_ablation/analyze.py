"""Reproduce the reviewed diagnostic from compact episode inputs, without simulation.

Only NumPy and PyYAML are needed. Outputs go to an explicit scratch directory;
--verify checks every generated number against the frozen reviewed tables.
"""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

PACKET = Path(__file__).resolve().parent
ARMS = ["G0E0", "G0E1", "G1E0", "G1E1"]
ENDPOINTS = ["success", "pedestrian_collision", "wall_collision", "timeout"]
RATE_COLUMNS = [
    "group",
    "arm",
    "endpoint",
    "status",
    "mode",
    "planned_denominator",
    "denominator",
    "count",
    "rate",
    "ci_low",
    "ci_high",
]
EFFECT_COLUMNS = [
    "group",
    "endpoint",
    "effect",
    "status",
    "mode",
    "planned_denominator_per_arm",
    "paired_denominator_per_arm",
    "seed_blocks",
    "difference",
    "ci_low",
    "ci_high",
]


def require(condition, message):
    """Fail closed even when Python optimization disables assertions."""
    if not condition:
        raise ValueError(message)


def contrasts(values):
    """Use the original analyzer's arithmetic and factor ordering."""
    guard = ((values[..., 1, 0] - values[..., 0, 0]) + (values[..., 1, 1] - values[..., 0, 1])) / 2
    escape = ((values[..., 0, 1] - values[..., 0, 0]) + (values[..., 1, 1] - values[..., 1, 0])) / 2
    interaction = values[..., 1, 1] - values[..., 1, 0] - values[..., 0, 1] + values[..., 0, 0]
    return {"guard": guard, "escape": escape, "interaction": interaction}


def group_values(cells, slots):
    """Build one group tensor, checking endpoint and termination consistency."""
    values = np.zeros((30, len(cells), 2, 2, 4), dtype=float)
    for si, seed in enumerate(range(1001, 1031)):
        for ci, cell in enumerate(cells):
            for gi in (0, 1):
                for ei in (0, 1):
                    row = slots[(f"G{gi}E{ei}", cell, seed)]
                    metrics, outcome = row["metrics"], row["outcome"]
                    require(
                        all(
                            type(outcome[key]) is bool
                            for key in ("collision_event", "route_complete", "timeout_event")
                        ),
                        "Invalid outcome flags",
                    )
                    ped = metrics["ped_collision_count"] > 0
                    wall = metrics["obstacle_collision_count"] > 0
                    collision = outcome["collision_event"]
                    success = outcome["route_complete"] and not collision
                    require(bool(metrics["success"]) == success, "Success/outcome contradiction")
                    timeout = outcome["timeout_event"] and not collision and not success
                    require(
                        row["termination_reason"] in ("success", "collision", "max_steps"),
                        "Invalid termination",
                    )
                    require(
                        {"success": success, "collision": collision, "max_steps": timeout}[
                            row["termination_reason"]
                        ],
                        "Termination/outcome contradiction",
                    )
                    values[si, ci, gi, ei] = [success, ped, wall, timeout]
    return values


def compute(packet):
    """Validate every paired episode and calculate the original 84 estimates."""
    spec = yaml.safe_load((packet / "inputs/study.yaml").read_text())
    groups = spec["groups"]
    input_path = packet / "evidence/episodes.jsonl.gz"
    with gzip.open(input_path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    slots = {(r["planner_key"], r["scenario_id"], r["seed"]): r for r in records}
    expected = {
        (arm, cell, seed)
        for arm in ARMS
        for cells in groups.values()
        for cell in cells
        for seed in range(1001, 1031)
    }
    require(
        len(slots) == len(records) == 2160 and set(slots) == expected,
        "Incomplete, duplicate, or non-development paired design",
    )
    for row in records:
        require(
            row["mode"] == "adapter"
            and row["status"] == "ok"
            and row["fallback_count"] == 0
            and row["fallback_or_degraded_marker"] is None,
            "Invalid, fallback, or degraded execution",
        )
        require(row["git_hash"] == spec["execution_commit"], "Execution source mismatch")
        require(
            row["effective_budget_steps"] == spec["horizon"]["selected_budgets"][row["scenario_id"]]
            and 0 < row["steps"] <= row["effective_budget_steps"],
            "Invalid episode budget",
        )
        factors = row["factors"]
        require(
            factors["ablation_disable_hard_guard"] == (row["planner_key"][1] == "0"),
            "Guard binding mismatch",
        )
        require(
            all(
                factors[key] == (row["planner_key"][3] == "1")
                for key in spec["escape"]["bundle_switches"]
            ),
            "Escape binding mismatch",
        )
        require(len(row["source_line_sha256"]) == 64, "Missing producer row identity")
    rng = np.random.default_rng(1001)
    draws = rng.integers(0, 30, size=(10000, 30))
    table, effects = [], []
    for group, cells in groups.items():
        values = group_values(cells, slots)
        # Preserve the original mean order and floating-point arithmetic exactly.
        boot = values[draws].mean(axis=(1, 2))
        point = values.mean(axis=(0, 1))
        for gi in (0, 1):
            for ei in (0, 1):
                for mi, endpoint in enumerate(ENDPOINTS):
                    lo, hi = np.quantile(boot[:, gi, ei, mi], [0.025, 0.975])
                    table.append(
                        {
                            "group": group,
                            "arm": f"G{gi}E{ei}",
                            "endpoint": endpoint,
                            "status": "valid_diagnostic",
                            "mode": "adapter",
                            "planned_denominator": len(cells) * 30,
                            "denominator": len(cells) * 30,
                            "count": int(values[:, :, gi, ei, mi].sum()),
                            "rate": point[gi, ei, mi],
                            "ci_low": lo,
                            "ci_high": hi,
                        }
                    )
        for mi, endpoint in enumerate(ENDPOINTS):
            observed = contrasts(point[:, :, mi])
            bootstrap = contrasts(boot[:, :, :, mi])
            for effect in ("guard", "escape", "interaction"):
                lo, hi = np.quantile(bootstrap[effect], [0.025, 0.975])
                effects.append(
                    {
                        "group": group,
                        "endpoint": endpoint,
                        "effect": effect,
                        "status": "valid_diagnostic",
                        "mode": "adapter",
                        "planned_denominator_per_arm": len(cells) * 30,
                        "paired_denominator_per_arm": len(cells) * 30,
                        "seed_blocks": 30,
                        "difference": float(observed[effect]),
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    success_effects = {}
    for row in effects:
        if row["endpoint"] == "success":
            success_effects.setdefault(row["group"], {})[row["effect"]] = {
                "difference_pp": row["difference"] * 100,
                "ci_low_pp": row["ci_low"] * 100,
                "ci_high_pp": row["ci_high"] * 100,
                "display_pp": [
                    f"{row[key] * 100:.1f}" for key in ("difference", "ci_low", "ci_high")
                ],
            }
    summary = {
        "schema": "guard-escape-ablation.summary.v1",
        "evidence_status": "diagnostic-only",
        "repository": "ll7/robot_sf_ll7",
        "execution_commit": spec["execution_commit"],
        "release_base_commit": spec["release_base_commit"],
        "episodes": 2160,
        "arms": ARMS,
        "cells": 18,
        "seed_blocks": 30,
        "development_seeds": {"first": 1001, "last": 1030},
        "execution_census": {"adapter": 2160, "native": 0, "fallback": 0, "degraded": 0},
        "duplicates": 0,
        "missing_slots": 0,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "statistics": spec["statistics"],
        "success_effects": success_effects,
        "review_verdict": "CONFIRMED",
        "claim_boundary": "Within-controller, fixed-roster development diagnostic; no historical "
        "ranking explanation, individual-component mediation, release evidence, "
        "or manuscript admission.",
    }
    return table, effects, summary


def write_csv(path, records, columns):
    """Keep the original CSV column order and numerical serialization."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def main():
    """Write only to a caller-chosen scratch directory; optionally verify frozen outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    packet, output = args.packet.resolve(), args.output.resolve()
    require(not output.is_relative_to(packet), "Output must be outside the immutable packet")
    table, effects, summary = compute(packet)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "figure_data.csv", table, RATE_COLUMNS)
    write_csv(output / "paired_effects.csv", effects, EFFECT_COLUMNS)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.verify:
        for name in ("figure_data.csv", "paired_effects.csv", "summary.json"):
            require(
                (output / name).read_bytes() == (packet / "evidence" / name).read_bytes(),
                f"Frozen output differs: {name}",
            )
        expected = {
            "static_constriction": ["46.0", "42.5", "50.0"],
            "portal": ["34.0", "30.7", "37.1"],
            "crowding_control": ["-1.1", "-6.7", "5.0"],
        }
        for group, display in expected.items():
            require(
                summary["success_effects"][group]["guard"]["display_pp"] == display,
                f"Headline differs: {group}",
            )
        print(
            "PASS: 2160 paired episodes; 48 rates, 36 contrasts and summary byte-identical; "
            "all three headline values reproduce exactly."
        )


if __name__ == "__main__":
    main()
