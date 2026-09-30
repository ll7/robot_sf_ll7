#!/usr/bin/env python3
"""Analyze HZEV diagnostic traces; clearance is always bounded by observation.

Distance to corridor means distance to the finite start-goal segment minus its
1.5 m half-width (a capsule), then a further 3 m clearance. Thus the central-line
threshold is 4.5 m. Also report the alternative 3 m central-line interpretation.
Wait arithmetic is an optimistic timing test, not a collision-free success proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = ("stationary", "orca", "goal")
BUDGETS = (400, 500, 600)
FLOW_CLASS = "no persistent pedestrian flow (routes respawn, so the dynamics never end)"


def finite(value):
    """Reject missing and nonfinite measurements instead of replacing them with zero."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite trace geometry or time")
    return number


def point(value):
    """Require a two-dimensional finite world position."""
    if len(value) != 2:
        raise ValueError("position must have exactly two coordinates")
    return tuple(finite(v) for v in value)


def segment_distance(p, a, b):
    """Distance to a finite segment, including its endpoints."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    alpha = (
        max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2)) if length2 else 0.0
    )
    return math.dist(p, (a[0] + alpha * dx, a[1] + alpha * dy))


def validated_frames(record):
    """Validate the initial snapshot and every regularly sampled post-step frame."""
    h = record["hzev"]
    dt = finite(h["dt_s"])
    if dt <= 0:
        raise ValueError("dt must be positive")
    frames = record["algorithm_metadata"]["simulation_step_trace"]["steps"]
    if not frames or len(frames) != h["steps"]:
        raise ValueError("missing or short trace")
    initial = h["initial_frame"]
    if finite(initial["time_s"]) != 0:
        raise ValueError("missing t=0 snapshot")
    for i, frame in enumerate(frames):
        if frame["step"] != i or not math.isclose(
            finite(frame["time_s"]), (i + 1) * dt, abs_tol=1e-8
        ):
            raise ValueError("missing, duplicate, or irregular step")
    all_frames = [initial, *frames]
    for frame in all_frames:
        point(frame["robot"]["position"])
        if not isinstance(frame["pedestrians"], list):
            raise ValueError("missing pedestrian position list")
        for ped in frame["pedestrians"]:
            point(ped["position"])
    return all_frames


def clearance(frames, start, goal, threshold):
    """First observed clear sample after the final occupied sample; None if censored."""
    occupied = [
        i
        for i, f in enumerate(frames)
        if any(
            segment_distance(point(p["position"]), start, goal) <= threshold
            for p in f["pedestrians"]
        )
    ]
    if not occupied:
        return 0.0
    last = occupied[-1]
    return None if last == len(frames) - 1 else finite(frames[last + 1]["time_s"])


def episode_metrics(record, *, turning_margin_s=2.0):
    """Compute measurements with explicit recurrence and right-censoring status."""
    h = record["hzev"]
    frames = validated_frames(record)
    start, goal = point(h["start"]), point(h["goal"])
    if h["seed"] not in range(1001, 1006) or h["arm"] not in ARMS:
        raise ValueError("unexpected arm or non-dev seed")
    arm = h["arm"]
    result = {
        "scenario_id": h["scenario_id"],
        "seed": h["seed"],
        "arm": arm,
        "authored_budget_steps": h["authored_budget_steps"],
        "dt_s": h["dt_s"],
        "distance_m": math.dist(start, goal),
        "observed_duration_s": frames[-1]["time_s"],
        "recurring_flow": h["recurring_flow"],
        "respawn_count": len(h["respawns"]),
        "first_terminal": h["first_terminal"],
        "trace_steps": h["steps"],
        "smoke": h["identity"]["smoke"],
    }
    if arm == "stationary":
        if any(math.dist(point(f["robot"]["position"]), start) > 1e-8 for f in frames):
            raise ValueError("stationary trace moved")
        observed_clear = clearance(frames, start, goal, 4.5)
        result["T_clear_observed_tail_s"] = observed_clear
        result["T_clear_centerline_3m_observed_tail_s"] = clearance(frames, start, goal, 3.0)
        # Recurring controllers may return after the observed tail; no 'ever again' claim.
        t_clear = None if h["recurring_flow"] else observed_clear
        result["T_clear_s"] = t_clear
        result["T_clear_status"] = (
            "recurring_flow"
            if h["recurring_flow"]
            else ("right_censored" if t_clear is None else "clear_through_observed_horizon")
        )
        estimate = (
            None if t_clear is None else t_clear + result["distance_m"] / 2.0 + turning_margin_s
        )
        result["wait_then_go_time_s"] = estimate
        result["wait_then_go_steps"] = (
            None if estimate is None else math.ceil(estimate / h["dt_s"] - 1e-9)
        )
        result["wait_then_go_within_steps"] = {
            str(b): None if estimate is None else estimate <= b * h["dt_s"] + 1e-9 for b in BUDGETS
        }
        budget = h["authored_budget_steps"] * h["dt_s"]
        result["wait_then_go_within_authored_budget"] = (
            None if estimate is None else estimate <= budget + 1e-9
        )
        if h["recurring_flow"]:
            result["classification"] = FLOW_CLASS
        elif result["smoke"]:
            result["classification"] = "unresolved: short smoke trace"
        elif t_clear is None:
            # If still occupied at 80s, the lower bound already exceeds a <=80s budget.
            lower_bound = frames[-1]["time_s"] + result["distance_m"] / 2.0 + turning_margin_s
            result["wait_then_go_lower_bound_s"] = lower_bound
            result["classification"] = (
                "budget in the dynamic window"
                if lower_bound > budget
                else "unresolved: observation horizon too short"
            )
        else:
            result["classification"] = (
                "wait-exploitable at the authored budget"
                if estimate <= budget
                else "budget in the dynamic window"
            )
    else:
        interactions = [
            finite(f["time_s"])
            for f in frames
            if any(
                math.dist(point(p["position"]), point(f["robot"]["position"])) <= 2.5
                for p in f["pedestrians"]
            )
        ]
        result["T_last_interaction_s"] = max(interactions) if interactions else None
        result["interaction_observed"] = bool(interactions)
        result["interaction_at_horizon"] = bool(
            interactions and interactions[-1] == frames[-1]["time_s"]
        )
        terminal = h["first_terminal"]
        before = (
            interactions
            if terminal is None
            else [t for t in interactions if t <= terminal["time_s"]]
        )
        result["T_last_interaction_before_first_terminal_s"] = max(before) if before else None
    return result


def summary(values):
    """Median/max over observed seeds, with unavailable values explicitly counted."""
    observed = [v for v in values if v is not None]
    return {
        "median": statistics.median(observed) if observed else None,
        "max": max(observed) if observed else None,
        "n_observed": len(observed),
        "n_unavailable": len(values) - len(observed),
    }


def aggregate(rows):
    """Keep per-seed disagreements visible; use any exploitable seed conservatively."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario_id"]].append(row)
    scenarios = []
    for sid, group in sorted(groups.items()):
        stationary = [r for r in group if r["arm"] == "stationary"]
        result = {
            "scenario_id": sid,
            "authored_budget_steps": group[0]["authored_budget_steps"],
            "T_clear_s": summary([r["T_clear_s"] for r in stationary]),
            "T_clear_observed_tail_s": summary([r["T_clear_observed_tail_s"] for r in stationary]),
            "wait_then_go_time_s": summary([r["wait_then_go_time_s"] for r in stationary]),
            "wait_then_go_within_steps": {
                str(b): {
                    "true": sum(r["wait_then_go_within_steps"][str(b)] is True for r in stationary),
                    "false": sum(
                        r["wait_then_go_within_steps"][str(b)] is False for r in stationary
                    ),
                    "unavailable": sum(
                        r["wait_then_go_within_steps"][str(b)] is None for r in stationary
                    ),
                }
                for b in BUDGETS
            },
            "classifications_by_seed": {str(r["seed"]): r["classification"] for r in stationary},
            "movers": {},
        }
        for arm in ("orca", "goal"):
            mover = [r for r in group if r["arm"] == arm]
            result["movers"][arm] = {
                "T_last_interaction_s": summary([r["T_last_interaction_s"] for r in mover]),
                "T_last_interaction_before_first_terminal_s": summary(
                    [r["T_last_interaction_before_first_terminal_s"] for r in mover]
                ),
                "at_horizon_count": sum(r["interaction_at_horizon"] for r in mover),
            }
        classes = set(result["classifications_by_seed"].values())
        if not classes or any(c.startswith("unresolved") for c in classes):
            result["classification"] = "unresolved: incomplete observation"
        elif FLOW_CLASS in classes:
            result["classification"] = FLOW_CLASS
        elif "wait-exploitable at the authored budget" in classes:
            result["classification"] = "wait-exploitable at the authored budget"
        else:
            result["classification"] = "budget in the dynamic window"
        result["seed_disagreement"] = len(classes) > 1
        scenarios.append(result)
    return scenarios


def main():  # noqa: C901 - fail-closed roster and checksum validation
    """Check roster/checksums and write machine-readable and reviewable reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Smoke inspection only; cannot settle scenarios",
    )
    parser.add_argument("--turning-margin-s", type=float, default=2.0)
    args = parser.parse_args()
    if not math.isfinite(args.turning_margin_s) or args.turning_margin_s < 0:
        parser.error("turning margin must be finite and nonnegative")
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    identity = manifest["identity"]
    expected = {
        (s, seed, arm)
        for s in identity["scenario_ids"]
        for seed in identity["seeds"]
        for arm in identity["arms"]
    }
    full_roster = (
        len(identity["scenario_ids"]) == 48
        and sorted(identity["seeds"]) == list(range(1001, 1006))
        and set(identity["arms"]) == set(ARMS)
        and identity["steps"] == 800
        and not identity["smoke"]
    )
    if not args.allow_partial and (not manifest["complete"] or not full_roster):
        raise ValueError(
            "full 48×5×3 H800 acquisition required; --allow-partial is smoke inspection"
        )
    seen, rows = set(), []
    for member in manifest["rows"]:
        name = member["path"]
        if Path(name).name != name:
            raise ValueError("manifest path escapes input directory")
        path = args.input_dir / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
            raise ValueError(f"checksum mismatch: {name}")
        record = json.loads(path.read_text())
        h = record["hzev"]
        key = (h["scenario_id"], h["seed"], h["arm"])
        if key in seen or key not in expected or h["identity"] != identity:
            raise ValueError("duplicate, unexpected or mixed-source episode")
        seen.add(key)
        rows.append(episode_metrics(record, turning_margin_s=args.turning_margin_s))
    if not args.allow_partial and seen != expected:
        raise ValueError("missing episodes")
    scenarios = aggregate(rows)
    if args.allow_partial and not full_roster:
        for scenario in scenarios:
            scenario["classification"] = "unresolved: partial diagnostic acquisition"
    report = {
        "schema_version": "hzev-analysis.v1",
        "evidence_status": "diagnostic-only",
        "identity": identity,
        "complete": full_roster and seen == expected and manifest["complete"],
        "definitions": {
            "corridor_half_width_m": 1.5,
            "distance_from_corridor_m": 3.0,
            "centerline_threshold_m": 4.5,
            "interaction_radius_m": 2.5,
            "speed_m_s": 2.0,
            "turning_margin_s": args.turning_margin_s,
            "flow_label_note": "Requested label is internally contradictory; routes respawn means persistent/recurring flow, not absence of flow.",
            "classification_rule": "any exploitable dev seed flags scenario; seed disagreements retained",
            "caveat": "finite traces cannot prove never again; wait arithmetic ignores static obstacles, goal-zone geometry and robot-dependent pedestrian motion; continued post-terminal data is diagnostic",
        },
        "scenarios": scenarios,
        "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial")
    partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    partial.replace(args.output)
    lines = [
        "# HZEV dynamic-window analysis",
        "",
        "Diagnostic only. Clearance is bounded by the recorded horizon; wait timing is an optimistic test, not observed goal success.",
        "",
        "| Scenario | Budget | T_clear median/max (s) | ORCA last median/max (s) | Goal last median/max (s) | Classification |",
        "|---|---:|---|---|---|---|",
    ]
    for s in scenarios:

        def pair(stat):
            return f"{stat['median']}/{stat['max']} (n={stat['n_observed']})"

        lines.append(
            f"| {s['scenario_id']} | {s['authored_budget_steps']} | {pair(s['T_clear_s'])} | {pair(s['movers']['orca']['T_last_interaction_s'])} | {pair(s['movers']['goal']['T_last_interaction_s'])} | {s['classification']} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "episodes": len(rows),
                "scenarios": len(scenarios),
                "complete": report["complete"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
