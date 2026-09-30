#!/usr/bin/env python3
"""Compare diagnostic PPO episodes; requested commands are BEFORE wrapper clipping.

Linear clipping fractions use the shared release interval [0,2], also for V3
(counterfactual release clipping). Mean consecutive command differences are
componentwise absolute differences in m/s and rad/s, never across episodes.
Time to goal averages successful episodes only; no-pedestrian clearance is null.
Fallback/degraded rows remain explicit and are excluded from navigation means.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

FAMILIES = {
    "classic_urban_crossing_medium": "crossing",
    "classic_cross_trap_medium": "crossing",
    "classic_overtaking_medium": "overtaking",
    "francis2023_robot_overtaking": "overtaking",
    "classic_head_on_corridor_medium": "head-on",
    "francis2023_frontal_approach": "head-on",
    "classic_group_crossing_medium": "group",
    "classic_doorway_medium": "doorway",
    "francis2023_crowd_navigation": "crowd",
    "classic_station_platform_medium": "crowd",
    "classic_t_intersection_medium": "intersection",
    "francis2023_following_human": "following",
    **{"empty_map_8_directions_" + x: "empty-world" for x in ["east", "north", "west", "south"]},
}


def finite(x):
    """Return whether a metric is a finite numeric value."""
    return isinstance(x, (float, int)) and math.isfinite(x)


def mean(values):
    """Average finite samples, or return null when none exist."""
    values = [v for v in values if finite(v)]
    return sum(values) / len(values) if values else None


def summarize(rows):  # noqa: C901
    """Aggregate eligible episodes without crossing episode command boundaries."""
    eligible = []
    excluded = 0
    commands = []
    selected = []
    deltas = [[], []]
    selected_deltas = [[], []]
    clearance = []
    for row in rows:
        meta = row["algorithm_metadata"]
        # Runtime validity and declared guard interventions are separate from outcomes.
        guards = meta.get("guard_stats", {})
        bad = meta.get("status") in {"fallback", "failed", "degraded", "not_available"} or any(
            guards.get(k, 0) > 0
            for k in [
                "stop_best_effort",
                "fallback_best_effort",
                "uncertainty_fallback_stop",
                "uncertainty_fallback_slow_down",
                "uncertainty_fallback_configured",
            ]
        )
        if bad:
            excluded += 1
            continue
        eligible.append(row)
        steps = meta.get("simulation_step_trace", {}).get("steps")
        if not steps:
            raise ValueError("Missing required simulation step trace")
        proposal = [s["planner"]["ppoeval_proposal"]["requested_command"] for s in steps]
        chosen = [
            [s["planner"]["selected_action"][k] for k in ["linear_velocity", "angular_velocity"]]
            for s in steps
        ]
        if any(len(c) != 2 or not all(finite(v) for v in c) for c in proposal + chosen):
            raise ValueError("Invalid command trace")
        commands.extend(proposal)
        selected.extend(chosen)
        for src, dst in [(proposal, deltas), (chosen, selected_deltas)]:
            for prev, now in pairwise(src):
                for i in range(2):
                    dst[i].append(abs(now[i] - prev[i]))
        clearance.extend(p.get("surface_clearance_m") for s in steps for p in s["pedestrians"])
    metrics = [r["metrics"] for r in eligible]
    n = len(eligible)
    negative = sum(c[0] < 0 for c in commands)
    above = sum(c[0] > 2 for c in commands)
    guard_counts = defaultdict(int)
    for r in rows:
        for k, v in r["algorithm_metadata"].get("guard_stats", {}).items():
            if finite(v):
                guard_counts[k] += v
    return {
        "episodes": len(rows),
        "eligible_episodes": n,
        "excluded_fallback_degraded": excluded,
        "success": sum(m.get("success", 0) > 0 for m in metrics),
        "collision": sum(
            m.get("collisions", m.get("collision_count", 0)) > 0
            or m.get("wall_collisions", 0) > 0
            or m.get("agent_collisions", 0) > 0
            for m in metrics
        ),
        "timeout": sum(m.get("timeout", 0) > 0 for m in metrics),
        "minimum_pedestrian_clearance_m": min([x for x in clearance if finite(x)], default=None),
        "mean_path_length_m": mean([m.get("path_length") for m in metrics]),
        "mean_time_to_goal_s_success_only": mean(
            [m.get("time_to_goal") for m in metrics if m.get("success", 0) > 0]
        ),
        "commands": len(commands),
        "fraction_v_negative": negative / len(commands) if commands else None,
        "fraction_v_above_2": above / len(commands) if commands else None,
        "fraction_commands_clipped_release": (negative + above) / len(commands)
        if commands
        else None,
        "mean_abs_delta_requested_v_m_s": mean(deltas[0]),
        "mean_abs_delta_requested_omega_rad_s": mean(deltas[1]),
        "mean_abs_delta_selected_v_m_s": mean(selected_deltas[0]),
        "mean_abs_delta_selected_omega_rad_s": mean(selected_deltas[1]),
        "guard_decision_counts": dict(guard_counts),
    }


def compare(inputs):
    """Validate run identities and aggregate arm, family and scenario rows."""
    groups = defaultdict(list)
    source_heads = {}
    for variant, root in inputs.items():
        receipts = list(root.rglob("ppoeval_receipt.json"))
        if len(receipts) != 1:
            raise ValueError(f"{variant}: expected one frozen run receipt")
        receipt = json.loads(receipts[0].read_text())
        source_heads[variant] = receipt["head_sha"]
        seen = set()
        count = 0
        for path in sorted(root.rglob("*.jsonl")):
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if "scenario_id" not in r or "metrics" not in r:
                    continue
                name = r["scenario_id"]
                arm = r["algorithm_metadata"].get("algorithm")
                if (
                    arm not in {"ppo", "guarded_ppo"}
                    or name not in FAMILIES
                    or r["seed"] not in range(1001, 1011)
                ):
                    raise ValueError(f"Unexpected episode identity: {arm}/{name}/{r['seed']}")
                identity = (arm, name, r["seed"])
                if identity in seen:
                    raise ValueError(f"Duplicate episode: {identity}")
                seen.add(identity)
                count += 1
                for level, key in [("arm", "all"), ("family", FAMILIES[name]), ("scenario", name)]:
                    groups[(variant, arm, level, key)].append(r)
        if count != receipt["expected_episodes"]:
            raise ValueError(f"{variant}: incomplete matrix {count}/{receipt['expected_episodes']}")
    summaries = [
        dict(zip(["variant", "arm", "level", "group"], key, strict=True), **summarize(rows))
        for key, rows in sorted(groups.items())
    ]
    return {
        "evidence_status": "diagnostic-only",
        "source_heads": source_heads,
        "command_semantics": "requested target before clipping; V3 [0,2] clipping is counterfactual",
        "rows": summaries,
    }


def main():
    """Write diagnostic comparison JSON and CSV from frozen run directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="VARIANT=ROOT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}
    for item in args.run:
        variant, path = item.split("=", 1)
        if variant in inputs or variant not in {"v1", "v2", "v3"}:
            raise ValueError("Unique v1/v2/v3 run names required")
        inputs[variant] = Path(path)
    report = compare(inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    print(f"{len(report['rows'])} comparison rows -> {args.output}")


if __name__ == "__main__":
    main()
